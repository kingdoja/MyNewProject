#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
自动化处理脚本：监听指定文件夹，自动对新增的全图进行切片、过滤和批量预测。

功能：
1. 监听指定文件夹（如 DataWSI/），检测新增图片
2. 自动调用切片脚本，将全图切分为 640x640 的 patch
3. 自动过滤空白 patch
4. 自动调用模型进行批量预测
5. 将结果保存到指定目录

使用方法：
cd /home/ubuntu/lsn/project_new/RT-DETR-main/DataSlice2Inference/auto_process_package
python auto_process_monitor.py \
  --watch-dir /home/ubuntu/lsn/project_new/RT-DETR-main/DataWSI \
  --output-dir /home/ubuntu/lsn/project_new/RT-DETR-main/DataPatchesInference \
  --model /home/ubuntu/lsn/project_new/RT-DETR-main/DataSlice2Inference/models/rtdetr_torchscript_cuda.pt \
  --threshold 0.5 \
  --process-existing
"""

import argparse
import os
import sys
import time
import subprocess
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, Dict, Any
from collections import Counter
import threading
from queue import Queue, Empty

import numpy as np

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    print("❌ 错误：需要安装 watchdog 库")
    print("请运行: pip install watchdog")
    sys.exit(1)


from utils.minio_helper import MinioUploader

# 支持的图像格式
IMAGE_EXTENSIONS = {'.jpeg', '.jpg', '.png', '.tif', '.tiff', '.ndpi'}


class ImageProcessHandler(FileSystemEventHandler):
    """处理新增图片的事件处理器 - 支持并行处理"""
    
    def __init__(
        self,
        watch_dir: Path,           # 监听的文件夹路径
        output_base_dir: Path,       # 输出结果的根目录
        model_path: str,             # 模型文件路径
        patch_size: int = 640,       # 切片大小
        threshold: float = 0.5,      # 预测置信度阈值
        processed_files: set = None,
        bg_rgb: tuple = (238, 235, 235), # 背景色 RGB
        tolerance: int = 30,
        bg_ratio: float = 0.9,       # 背景比例阈值
        std_thresh: float = 10.0,    # 灰度标准差阈值
        auto_bg: bool = True,        # 自动背景分析
        bg_analysis_limit: int = 5,  # 自动背景分析时采样的patch数量
        save_visualization: bool = True, # 是否保存可视化
        file_wait_timeout: int = 30,  # 等待文件完全写入的秒数
        max_concurrent_tasks: int = 2,  # 最大并发任务数
        auto_downscale: bool = True,  # 是否自动缩放大图像
        downscale_threshold: int = 50000,  # 触发缩放的像素阈值
        downscale_quality: int = 95,  # 缩放后JPEG质量
        logger=None,  # 日志记录器，如果为None则使用print
        minio_config: Optional[Dict[str, Any]] = None  # MinIO 配置
    ):
        self.watch_dir = watch_dir
        self.output_base_dir = output_base_dir
        self.model_path = model_path
        self.patch_size = patch_size
        self.threshold = threshold
        self.processed_files = processed_files or set()
        self.bg_rgb = bg_rgb
        self.tolerance = tolerance
        self.bg_ratio = bg_ratio
        self.std_thresh = std_thresh
        self.auto_bg = auto_bg
        self.bg_analysis_limit = bg_analysis_limit
        self.save_visualization = save_visualization
        self.file_wait_timeout = file_wait_timeout
        self.max_concurrent_tasks = max_concurrent_tasks
        self.auto_downscale = auto_downscale
        self.downscale_threshold = downscale_threshold
        self.downscale_quality = downscale_quality
        self.logger = logger
        self.stop_event = threading.Event()  # 用于响应优雅停止
        self.minio_uploader = None
        self.minio_upload_json = False
        self.minio_upload_original = False
        
        # 创建输出目录结构
        self.patches_dir = output_base_dir.parent / "DataPatches"
        self.patches_keep_dir = output_base_dir.parent / "DataPatchesKeep"
        self.patches_trash_dir = output_base_dir.parent / "DataPatchesTrash"
        self.downscaled_wsi_dir = output_base_dir.parent / "DataWSI_downscaled"  # 存放缩小后的图像

        # MinIO 初始化
        if minio_config and minio_config.get("enabled", False):
            try:
                self.minio_upload_json = bool(minio_config.get("upload_json", True))
                self.minio_upload_original = bool(minio_config.get("upload_original_image", True))
                self.minio_uploader = MinioUploader(
                    endpoint=minio_config["endpoint"],
                    access_key=minio_config["access_key"],
                    secret_key=minio_config["secret_key"],
                    bucket=minio_config["bucket"],
                    secure=bool(minio_config.get("secure", False)),
                    base_path=minio_config.get("base_path"),
                    region=minio_config.get("region") or None,
                    enable_bucket_create=bool(minio_config.get("create_bucket_if_missing", True)),
                )
                self._log(f"✅ MinIO 上传已启用，bucket={minio_config['bucket']}, base_path={minio_config.get('base_path','')}")
            except Exception as e:  # pragma: no cover - 依赖环境
                self._log(f"⚠️ MinIO 初始化失败，将跳过上传: {e}", "warning")
                self.minio_uploader = None
                self.minio_upload_json = False
                self.minio_upload_original = False
        
        # 处理状态记录文件
        self.status_file = output_base_dir / "processing_status.json"
        self.load_status()
        
        # 并行处理相关
        self.task_queue = Queue()  # 任务队列
        self.file_enqueue_time = {}  # 记录文件首次加入队列的时间
        self.processing_lock = threading.Lock()  # 用于同步 processed_files 的访问
        self.active_workers = 0  # 当前活跃的工作线程数
        self.active_workers_lock = threading.Lock()
        
        # 启动工作线程
        self.workers = []
        for i in range(max_concurrent_tasks):
            worker = threading.Thread(
                target=self._worker_thread,
                daemon=True,
                name=f"ImageProcessor-Worker-{i+1}"
            )
            worker.start()
            self.workers.append(worker)
        
        self._log(f"✅ 已启动 {max_concurrent_tasks} 个并行处理线程")
    
    def _log(self, message: str, level: str = 'info'):
        """统一的日志输出方法"""
        if self.logger:
            if level == 'info':
                self.logger.info(message)
            elif level == 'warning':
                self.logger.warning(message)
            elif level == 'error':
                self.logger.error(message)
            elif level == 'debug':
                self.logger.debug(message)
        else:
            print(message)
    
    def load_status(self):
        """加载处理状态"""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r', encoding='utf-8') as f:
                    status = json.load(f)
                    self.processed_files = set(status.get('processed_files', []))
            except Exception as e:
                self._log(f"⚠️ 加载状态文件失败: {e}", 'warning')
    
    def save_status(self):
        """保存处理状态"""
        try:
            self.status_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.status_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'processed_files': list(self.processed_files),
                    'last_update': datetime.now().isoformat()
                }, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self._log(f"⚠️ 保存状态文件失败: {e}", 'warning')
    
    def get_file_hash(self, file_path: Path) -> str:
        """计算文件哈希值，用于判断文件是否已处理"""
        try:
            with open(file_path, 'rb') as f:
                # 只读取文件头部和大小，加快计算速度
                file_size = file_path.stat().st_size
                file_header = f.read(8192)  # 读取前8KB
                hash_obj = hashlib.md5()
                hash_obj.update(file_header)
                hash_obj.update(str(file_size).encode())
                return hash_obj.hexdigest()
        except Exception:
            return str(file_path.stat().st_mtime)  # 如果计算失败，使用修改时间
    
    def is_image_file(self, file_path: Path) -> bool:
        """判断是否为支持的图像文件"""
        return file_path.suffix.lower() in IMAGE_EXTENSIONS
    
    def _worker_thread(self):
        """工作线程：从队列中取任务并处理"""
        thread_name = threading.current_thread().name
        self._log(f"🔧 {thread_name} 已启动")
        
        while not self.stop_event.is_set():
            try:
                # 从队列获取任务（1秒超时，以便能响应停止信号）
                try:
                    file_path = self.task_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                # 增加活跃线程计数
                with self.active_workers_lock:
                    self.active_workers += 1
                
                try:
                    # 智能等待：计算实际需要等待的时间
                    enqueue_time = self.file_enqueue_time.get(str(file_path), time.time())
                    elapsed = time.time() - enqueue_time
                    remaining_wait = max(0, self.file_wait_timeout - elapsed)
                    
                    if remaining_wait > 0:
                        self._log(f"⏳ [{thread_name}] {file_path.name}: 等待文件稳定 {remaining_wait:.1f}秒")
                        # 分段等待，以便响应停止信号
                        wait_steps = int(remaining_wait) + 1
                        for _ in range(wait_steps):
                            if self.stop_event.is_set():
                                self._log(f"⏹️ [{thread_name}] 收到停止信号，取消处理 {file_path.name}", 'warning')
                                return
                            time.sleep(min(1.0, remaining_wait))
                            remaining_wait -= 1.0
                            if remaining_wait <= 0:
                                break
                    else:
                        self._log(f"✅ [{thread_name}] {file_path.name}: 文件已稳定(排队 {elapsed:.1f}秒)，直接处理")
                    
                    # 检查文件是否存在
                    if not file_path.exists():
                        self._log(f"⚠️ [{thread_name}] 文件不存在，跳过: {file_path.name}", 'warning')
                        continue
                    
                    # 检查停止信号
                    if self.stop_event.is_set():
                        self._log(f"⏹️ [{thread_name}] 收到停止信号，取消处理", 'warning')
                        return
                    
                    # 重新加载状态文件（确保使用最新的处理记录）
                    with self.processing_lock:
                        self.load_status()
                    
                    # 检查是否已处理
                    file_hash = self.get_file_hash(file_path)
                    with self.processing_lock:
                        if file_hash in self.processed_files:
                            self._log(f"⏭️ [{thread_name}] 跳过已处理的文件: {file_path.name}")
                            continue
                    
                    self._log(f"\n{'='*70}")
                    self._log(f"🚀 [{thread_name}] 开始处理: {file_path.name}")
                    self._log(f"{'='*70}\n")
                    
                    # 处理图片
                    start_time = time.time()
                    self.process_image(file_path, file_hash)
                    elapsed_time = time.time() - start_time
                    
                    self._log(f"\n{'='*70}")
                    self._log(f"✅ [{thread_name}] 处理完成: {file_path.name} (耗时: {elapsed_time/60:.2f}分钟)")
                    self._log(f"{'='*70}\n")
                    
                except KeyboardInterrupt:
                    self._log(f"\n⏹️ [{thread_name}] 收到中断信号", 'warning')
                    self.stop_event.set()
                    raise
                except Exception as e:
                    self._log(f"❌ [{thread_name}] 处理失败 {file_path.name}: {e}\n", 'error')
                    import traceback
                    if self.logger:
                        self.logger.error(traceback.format_exc())
                    else:
                        traceback.print_exc()
                finally:
                    # 减少活跃线程计数
                    with self.active_workers_lock:
                        self.active_workers -= 1
                    
                    # 标记任务完成
                    self.task_queue.task_done()
                    
                    # 清理记录
                    with self.processing_lock:
                        self.file_enqueue_time.pop(str(file_path), None)
                    
            except Exception as e:
                self._log(f"❌ [{thread_name}] 工作线程异常: {e}", 'error')
                if self.logger:
                    import traceback
                    self.logger.error(traceback.format_exc())
        
        self._log(f"🔧 {thread_name} 已停止")
    
    def on_created(self, event):
        """文件创建事件 - 快速响应，将任务加入队列"""
        if event.is_directory:
            return
        
        # 检查是否已请求停止
        if self.stop_event.is_set():
            self._log("服务已停止，忽略新文件", 'warning')
            return
        
        file_path = Path(event.src_path)
        if not self.is_image_file(file_path):
            return
        
        # 记录文件首次检测到的时间
        with self.processing_lock:
            if str(file_path) not in self.file_enqueue_time:
                self.file_enqueue_time[str(file_path)] = time.time()
        
        # 将任务快速加入队列（不阻塞）
        self.task_queue.put(file_path)
        
        # 获取队列大小和活跃线程数
        queue_size = self.task_queue.qsize()
        with self.active_workers_lock:
            active = self.active_workers
        
        self._log(f"📥 文件已加入处理队列: {file_path.name} [队列: {queue_size}, 活跃线程: {active}/{self.max_concurrent_tasks}]")
    
    def process_image(self, image_path: Path, file_hash: str):
        """处理单张图片：切片 -> 过滤 -> 预测"""
        
        # 检查停止信号
        if self.stop_event.is_set():
            self._log("收到停止信号，取消处理", 'warning')
            return
        
        # 步骤0: 检查图像尺寸，如果需要则先缩放
        actual_image_path = image_path  # 实际要处理的图像路径
        original_image_path = image_path  # 原始图像路径（用于记录）
        scale_factor = 1.0  # 缩放系数（用于坐标转换）
        
        if self.auto_downscale:
            self._log("="*70)
            self._log("步骤 0: 检查图像尺寸")
            self._log("="*70)
            
            needs_downscale, orig_width, orig_height = self.check_image_size(image_path)
            
            if needs_downscale:
                self._log(f"⚠️ 图像尺寸 {orig_width}x{orig_height} 超过阈值 {self.downscale_threshold}")
                self._log("正在执行自动缩放（50%）...")
                
                # 创建缩小后图像的保存目录
                self.downscaled_wsi_dir.mkdir(parents=True, exist_ok=True)
                
                # 生成缩小后图像的文件名（保持原扩展名）
                downscaled_filename = f"{image_path.stem}_downscaled{image_path.suffix}"
                downscaled_path = self.downscaled_wsi_dir / downscaled_filename
                
                # 执行缩放
                try:
                    orig_w, orig_h, new_w, new_h = self.downscale_image(
                        image_path,
                        downscaled_path,
                        scale_factor=0.5
                    )
                    scale_factor = 2.0  # 坐标需要乘以2才能映射回原图
                    actual_image_path = downscaled_path
                    self._log(f"✅ 图像已缩放: {orig_w}x{orig_h} -> {new_w}x{new_h}")
                    self._log(f"✅ 缩放后图像保存至: {downscaled_path}")
                    self._log(f"📊 坐标转换系数: {scale_factor}x（输出坐标将自动映射到原始图像）\n")
                except Exception as e:
                    self._log(f"❌ 缩放失败: {e}", 'error')
                    self._log("⚠️ 将使用原始图像继续处理", 'warning')
                    scale_factor = 1.0
                    actual_image_path = image_path
            else:
                self._log(f"✅ 图像尺寸 {orig_width}x{orig_height} 无需缩放\n")
        
        # 生成唯一的输出目录名（基于文件名和时间戳）
        image_name = image_path.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{image_name}_{timestamp}"
        
        # 创建输出目录
        patches_subdir = self.patches_dir / output_name
        patches_keep_subdir = self.patches_keep_dir / output_name
        patches_trash_subdir = self.patches_trash_dir / output_name
        inference_output_dir = self.output_base_dir / output_name
        json_name = f"{image_name}.json"
        
        self._log(f"📁 输出目录: {patches_subdir}")
        self._log(f"📁 预测输出目录: {inference_output_dir}\n")
        
        # 步骤1: 切片（使用actual_image_path，但记录scale_factor用于坐标转换）
        self._log("="*70)
        self._log("步骤 1/3: 图片切片")
        self._log("="*70)
        patches_subdir.mkdir(parents=True, exist_ok=True)
        self.slice_image(actual_image_path, patches_subdir, scale_factor=scale_factor)
        
        # 检查停止信号
        if self.stop_event.is_set():
            self._log("切片完成后收到停止信号，停止后续处理", 'warning')
            return
        
        # 自动背景参数分析
        if self.auto_bg:
            self._log("\n" + "="*70)
            self._log("步骤 1.5: 自动分析背景参数")
            self._log("="*70)
            self.auto_configure_background(patches_subdir)
        
        # 检查停止信号
        if self.stop_event.is_set():
            self._log("背景分析完成后收到停止信号，停止后续处理", 'warning')
            return
        
        # 步骤2: 过滤空白patch
        self._log("\n" + "="*70)
        self._log("步骤 2/3: 过滤空白patch")
        self._log("="*70)
        patches_keep_subdir.mkdir(parents=True, exist_ok=True)
        patches_trash_subdir.mkdir(parents=True, exist_ok=True)
        self.filter_patches(patches_subdir, patches_keep_subdir, patches_trash_subdir)
        
        # 检查停止信号
        if self.stop_event.is_set():
            self._log("过滤完成后收到停止信号，停止后续处理", 'warning')
            # 仍然标记为已处理，避免重复处理
            self.processed_files.add(file_hash)
            self.save_status()
            return
        
        # 检查是否有保留的patch
        keep_patches = list(patches_keep_subdir.glob("*.png"))
        if len(keep_patches) == 0:
            self._log("⚠️ 警告：没有保留的patch，跳过预测步骤", 'warning')
            self.processed_files.add(file_hash)
            self.save_status()
            return
        
        # 步骤3: 批量预测
        self._log("\n" + "="*70)
        self._log(f"步骤 3/3: 批量预测 (共 {len(keep_patches)} 个patch)")
        self._log("="*70)
        inference_output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_predict(
            patches_keep_subdir,
            inference_output_dir,
            patches_subdir,
            image_path,
            json_name=json_name
        )

        # 上传至 MinIO（如已配置）
        if self.minio_uploader:
            try:
                json_path = inference_output_dir / json_name
                self.upload_results_to_minio(
                    json_path=json_path,
                    global_image_path=original_image_path,
                    output_dir=inference_output_dir
                )
            except Exception as e:
                self._log(f"⚠️ MinIO 上传失败: {e}", "warning")
        
        # 检查停止信号
        if self.stop_event.is_set():
            self._log("预测完成后收到停止信号", 'warning')
        
        # 标记为已处理
        self.processed_files.add(file_hash)
        self.save_status()
    
    def check_image_size(self, image_path: Path) -> tuple[bool, int, int]:
        """
        检查图像尺寸是否超过阈值
        
        Returns:
            (是否需要缩放, 图像宽度, 图像高度)
        """
        from PIL import Image
        
        Image.MAX_IMAGE_PIXELS = None  # 允许处理超大图像
        
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                needs_downscale = (width > self.downscale_threshold or 
                                 height > self.downscale_threshold)
                return needs_downscale, width, height
        except Exception as e:
            self._log(f"❌ 无法读取图像尺寸: {e}", 'error')
            return False, 0, 0
    
    def downscale_image(
        self,
        src_path: Path,
        dst_path: Path,
        scale_factor: float = 0.5
    ) -> tuple[int, int, int, int]:
        """
        缩放图像
        
        Args:
            src_path: 源图像路径
            dst_path: 目标图像路径
            scale_factor: 缩放比例（默认0.5，即50%）
        
        Returns:
            (原始宽度, 原始高度, 新宽度, 新高度)
        """
        # 导入缩放函数
        import sys
        slice_tool_dir = Path(__file__).resolve().parent.parent / "sliceTool"
        if str(slice_tool_dir) not in sys.path:
            sys.path.insert(0, str(slice_tool_dir))
        
        from convert_wsi40x_to_20x import convert_single_image
        
        return convert_single_image(
            src_path=src_path,
            dst_path=dst_path,
            quality=self.downscale_quality,
            scale_factor=scale_factor,
            logger=self.logger
        )
    
    def slice_image(self, image_path: Path, output_dir: Path, scale_factor: float = 1.0):
        """
        切片图像
        
        Args:
            image_path: 图像路径（可能是原始图像或缩放后的图像）
            output_dir: 输出目录
            scale_factor: 缩放系数（例如2.0表示当前图像是原图的50%，坐标需乘以2映射回原图）
        """
        from PIL import Image, ImageFile
        from tqdm import tqdm
        import csv
        
        # 禁用 PIL 的 decompression bomb 检查
        Image.MAX_IMAGE_PIXELS = None
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 打开图像
        self._log("正在打开图像文件...")
        try:
            pil_img = Image.open(image_path)
            if pil_img.mode != 'RGB':
                self._log(f"图像模式为 {pil_img.mode}，将在处理时转换为 RGB")
        except Exception as e:
            raise FileNotFoundError(f"无法打开图片文件: {image_path}, 错误: {e}")
        
        # 获取图像尺寸
        width, height = pil_img.size
        self._log(f"当前图像尺寸: {width} x {height} 像素")
        if scale_factor != 1.0:
            orig_width = int(width * scale_factor)
            orig_height = int(height * scale_factor)
            self._log(f"原始图像尺寸: {orig_width} x {orig_height} 像素")
            self._log(f"坐标转换系数: {scale_factor}x")
        self._log(f"总像素数: {width * height:,}")
        
        # 计算可切 patch 数量
        num_patches_x = width // self.patch_size
        num_patches_y = height // self.patch_size
        total_patches = num_patches_x * num_patches_y
        self._log(f"将切分为 {num_patches_x} x {num_patches_y} = {total_patches} 个 patch")
        
        # 记录坐标（在当前图像上的坐标）
        x_r = []
        y_r = []
        patch_count = 0
        
        # 分块读取并保存 patch
        self._log("开始切分图像...")
        # tqdm 进度条始终输出到 stdout，确保用户能看到进度
        for y_idx in tqdm(range(num_patches_y), desc="处理行", unit="行"):
            if self.stop_event.is_set():
                self._log("收到停止信号，中断切片...", 'warning')
                return
            for x_idx in range(num_patches_x):
                if self.stop_event.is_set():
                    self._log("收到停止信号，中断切片...", 'warning')
                    return
                x = x_idx * self.patch_size
                y = y_idx * self.patch_size
                
                x_r.append(x)
                y_r.append(y)
                
                # 使用 crop 方法只读取当前 patch 区域
                patch_box = (x, y, x + self.patch_size, y + self.patch_size)
                try:
                    patch_pil = pil_img.crop(patch_box)
                except Exception as e:
                    self._log(f"\n警告：读取 patch ({x}, {y}) 时出错: {e}", 'warning')
                    continue
                
                # 转换为 RGB 模式
                if patch_pil.mode != 'RGB':
                    patch_pil = patch_pil.convert('RGB')
                
                # 保存 patch
                save_path = output_dir / f'patch_{patch_count}.png'
                try:
                    patch_pil.save(save_path, 'PNG')
                except Exception as e:
                    self._log(f"\n警告：保存 patch {patch_count} 时出错: {e}", 'warning')
                    continue
                
                patch_count += 1
        
        # 保存坐标信息到CSV文件（坐标映射回原始图像）
        csv_path = output_dir / 'patch_coordinates.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 添加说明：这些坐标是基于原始图像的
            if scale_factor != 1.0:
                writer.writerow(['# Coordinates are in the original image coordinate system'])
                writer.writerow([f'# Scale factor: {scale_factor}x'])
            writer.writerow(['filename', 'x_start', 'y_start', 'x_end', 'y_end'])
            for i in range(patch_count):
                filename = f'patch_{i}.png'
                # 将坐标映射回原始图像（乘以scale_factor）
                orig_x_start = int(x_r[i] * scale_factor)
                orig_y_start = int(y_r[i] * scale_factor)
                orig_x_end = int((x_r[i] + self.patch_size) * scale_factor)
                orig_y_end = int((y_r[i] + self.patch_size) * scale_factor)
                writer.writerow([filename, orig_x_start, orig_y_start, orig_x_end, orig_y_end])
        
        self._log(f"\n✓ 完成！共保存 {patch_count} 个 patch")
        if scale_factor != 1.0:
            self._log(f"✓ 坐标已自动映射到原始图像坐标系（缩放系数: {scale_factor}x）")
        self._log(f"✓ 坐标信息已保存到: {csv_path}")
    
    def auto_configure_background(self, patches_dir: Path):
        """分析patch背景信息并自动调整过滤参数"""
        image_files = sorted([p for p in patches_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
        if not image_files:
            self._log("⚠️ 没有找到可分析的patch，保持原有背景参数", 'warning')
            return
        
        sample_files = image_files[: min(self.bg_analysis_limit, len(image_files))]
        results = []
        
        for img_path in sample_files:
            bg_color, bg_percent, std = self.analyze_single_patch(img_path)
            if bg_color is not None:
                results.append((bg_color, bg_percent, std))
        
        if not results:
            self._log("⚠️ 背景分析失败，保持原有背景参数", 'warning')
            return
        
        avg_bg = np.mean([r[0] for r in results], axis=0).astype(int)
        avg_std = float(np.mean([r[2] for r in results]))
        avg_bg_percent = float(np.mean([r[1] for r in results]))
        
        recommended_tolerance = max(15, min(30, int(avg_std * 2)))
        if avg_bg_percent > 0.95:
            recommended_ratio = 0.98
        elif avg_bg_percent > 0.90:
            recommended_ratio = 0.95
        else:
            recommended_ratio = 0.90
        recommended_std_thresh = max(3.0, min(10.0, avg_std * 1.5))
        
        self._log(f"分析 {len(results)} 个patch 得到背景参数：")
        self._log(f"  背景色 (RGB): {tuple(avg_bg.tolist())}")
        self._log(f"  背景覆盖率: {avg_bg_percent*100:.1f}%")
        self._log(f"  灰度标准差: {avg_std:.2f}")
        self._log("推荐过滤参数：")
        self._log(f"  BG_RGB      = {tuple(avg_bg.tolist())}")
        self._log(f"  TOLERANCE   = {recommended_tolerance}")
        self._log(f"  BG_RATIO    = {recommended_ratio}")
        self._log(f"  STD_THRESH  = {recommended_std_thresh:.2f}")
        
        self.bg_rgb = tuple(avg_bg.tolist())
        self.tolerance = recommended_tolerance
        self.bg_ratio = recommended_ratio
        self.std_thresh = recommended_std_thresh
    
    def analyze_single_patch(self, image_path: Path) -> Tuple[Optional[Tuple[int, int, int]], Optional[float], Optional[float]]:
        """分析单个patch的背景色和灰度"""
        from PIL import Image
        try:
            with Image.open(image_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                arr = np.array(img)
        except Exception as e:
            self._log(f"⚠️ 分析 {image_path.name} 时出错: {e}", 'warning')
            return None, None, None
        
        height, width, _ = arr.shape
        edge_colors = []
        edge_colors.extend(arr[0, :, :].tolist())
        edge_colors.extend(arr[-1, :, :].tolist())
        edge_colors.extend(arr[:, 0, :].tolist())
        edge_colors.extend(arr[:, -1, :].tolist())
        
        edge_counter = Counter(map(tuple, edge_colors))
        all_colors = arr.reshape(-1, 3)
        color_counter = Counter(map(tuple, all_colors))
        
        most_common_color = color_counter.most_common(1)[0]
        background_color = most_common_color[0]
        background_percentage = most_common_color[1] / len(all_colors)
        
        gray = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.float32)
        std = float(gray.std())
        
        return background_color, background_percentage, std
    
    def filter_patches(self, src_dir: Path, keep_dir: Path, trash_dir: Path):
        """过滤空白patch"""
        import shutil
        import numpy as np
        from PIL import Image
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
        WORKERS = 8
        MOVE = False
        DELETE = False
        
        def is_blank(img: Image.Image) -> tuple[bool, float, float]:
            """判断是否为空白patch"""
            arr = np.asarray(img)
            
            if arr.ndim == 2:  # 灰度
                arr_rgb = np.stack([arr, arr, arr], axis=-1)
            else:
                if arr.shape[2] == 4:  # 透明
                    rgb, a = arr[..., :3].astype(np.float32), arr[..., 3:4].astype(np.float32) / 255.0
                    arr_rgb = (rgb * a + 255.0 * (1 - a)).astype(np.uint8)
                else:
                    arr_rgb = arr[..., :3]
            
            bg = np.array(self.bg_rgb, dtype=np.int16)[None, None, :]
            diff = np.abs(arr_rgb.astype(np.int16) - bg).sum(axis=2)
            bg_mask = diff <= self.tolerance
            ratio = bg_mask.mean()
            
            gray = (0.299 * arr_rgb[..., 0] + 0.587 * arr_rgb[..., 1] + 0.114 * arr_rgb[..., 2]).astype(np.float32)
            std = float(gray.std())
            
            blank = (ratio >= self.bg_ratio) and (std <= self.std_thresh)
            return blank, ratio, std
        
        def process_one(path: Path) -> tuple[str, bool, float, float]:
            """处理单个patch"""
            # 检查停止信号
            if self.stop_event.is_set():
                return (str(path), False, -2.0, -2.0)  # 使用特殊值表示被中断
            
            try:
                with Image.open(path) as im:
                    im = im.convert("RGB")
                    blank, ratio, std = is_blank(im)
            except Exception:
                return (str(path), False, -1.0, -1.0)
            
            dest_dir = trash_dir if blank else keep_dir
            dest_dir.mkdir(parents=True, exist_ok=True)
            
            if DELETE and blank:
                path.unlink(missing_ok=True)
            else:
                if MOVE:
                    shutil.move(str(path), str(dest_dir / path.name))
                else:
                    shutil.copy2(str(path), str(dest_dir / path.name))
            
            return (str(path), blank, ratio, std)
        
        # 获取所有patch文件
        files = [p for p in src_dir.iterdir() if p.suffix.lower() in IMG_EXTS]
        total = len(files)
        
        if total == 0:
            self._log("⚠️ 未找到patch文件", 'warning')
            return
        
        self._log(f"共发现 {total} 个 patch，开始筛选...")
        
        kept = removed = failed = 0
        keep_dir.mkdir(parents=True, exist_ok=True)
        trash_dir.mkdir(parents=True, exist_ok=True)
        
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = [executor.submit(process_one, p) for p in files]
            for i, future in enumerate(as_completed(futures), 1):
                # 检查停止信号
                if self.stop_event.is_set():
                    self._log("\n⏹️ 收到停止信号，取消剩余过滤任务...", 'warning')
                    # 取消所有未完成的任务
                    for f in futures:
                        f.cancel()
                    break
                
                path, blank, ratio, std = future.result()
                
                if ratio == -2.0:  # 被中断
                    continue
                elif ratio < 0:  # 处理失败
                    failed += 1
                    self._log(f"[FAIL] 读取失败: {path}", 'warning')
                else:
                    if blank:
                        removed += 1
                    else:
                        kept += 1
                
                if i % 500 == 0 or i == total:
                    self._log(f"[{i}/{total}] 保留={kept}, 移除={removed}, 失败={failed}")
        
        if self.stop_event.is_set():
            self._log(f"⚠️ 过滤被中断。已处理: {kept + removed + failed}/{total}", 'warning')
        else:
            self._log(f"✓ 总计: {total}, 保留(非空白): {kept}, 移除(空白): {removed}, 失败: {failed}")
    
    def batch_predict(
        self,
        patch_dir: Path,
        output_dir: Path,
        coordinates_dir: Path,
        global_image_path: Optional[Path] = None,
        json_name: Optional[str] = None
    ):
        """调用批量预测脚本"""
        predict_script = (Path(__file__).resolve().parent.parent / "inferenceTool" / "predict_batch_torchscript.py").resolve()
        
        # 查找坐标CSV文件
        coordinates_csv = coordinates_dir / "patch_coordinates.csv"
        if not coordinates_csv.exists():
            self._log(f"⚠️ 警告：坐标CSV文件不存在: {coordinates_csv}", 'warning')
            self._log("预测将无法进行坐标转换", 'warning')
            coordinates_csv = None
        
        # 构建命令
        cmd = [
            sys.executable,
            str(predict_script),
            "--model", str(self.model_path),
            "--patch-dir", str(patch_dir),
            "--output-dir", str(output_dir),
            "--threshold", str(self.threshold),
            "--pattern", "*.png"
        ]
        
        if coordinates_csv:
            cmd.extend(["--coordinates-csv", str(coordinates_csv)])
        if global_image_path:
            cmd.extend(["--global-image-url", str(global_image_path)])
        if not self.save_visualization:
            cmd.append("--no-visualization")
        if json_name:
            cmd.extend(["--json-name", json_name])
        if self.stop_event.is_set():
            self._log("收到停止信号，跳过预测步骤", 'warning')
            return
        
        # 执行命令
        self._log(f"开始批量预测，共 {len(list(patch_dir.glob('*.png')))} 个patch...")
        
        # 添加环境变量，确保Python输出无缓冲
        env = os.environ.copy()
        env['PYTHONUNBUFFERED'] = '1'
        
        proc = subprocess.Popen(
            cmd,
            cwd=str(Path(__file__).resolve().parent.parent),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # 合并stderr到stdout，避免缓冲区阻塞
            text=True,
            bufsize=1,  # 行缓冲
            env=env
        )

        # 持续读取输出，并在收到停止信号时终止子进程
        terminated_by_user = False
        
        try:
            import select
            
            # 使用非阻塞方式读取输出
            while True:
                # 检查停止信号
                if self.stop_event.is_set():
                    self._log("收到停止信号，终止预测进程...", 'warning')
                    terminated_by_user = True
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        self._log("预测进程未响应 terminate，强制 kill", 'warning')
                        proc.kill()
                        proc.wait(timeout=2)
                    break
                
                # 检查进程是否结束
                if proc.poll() is not None:
                    break
                
                # 非阻塞读取（使用 timeout）
                try:
                    # 使用较短的超时，以便能及时检查停止信号
                    if sys.platform != 'win32':
                        # Unix系统使用 select
                        ready, _, _ = select.select([proc.stdout], [], [], 0.5)
                        if ready:
                            line = proc.stdout.readline()
                            if line:
                                # 实时输出日志，不要等到最后
                                self._log(line.rstrip("\n"))
                    else:
                        # Windows 系统直接读取（会阻塞）
                        line = proc.stdout.readline()
                        if line:
                            # 实时输出日志
                            self._log(line.rstrip("\n"))
                        elif proc.poll() is not None:
                            break
                except Exception as e:
                    self._log(f"读取输出时出错: {e}", 'warning')
                    break
            
            # 读取剩余输出
            try:
                remaining_out, _ = proc.communicate(timeout=5)
                if remaining_out:
                    for line in remaining_out.strip().split("\n"):
                        if line.strip():
                            self._log(line)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
                
        except KeyboardInterrupt:
            self._log("\n收到键盘中断，终止预测进程...", 'warning')
            terminated_by_user = True
            proc.kill()
            raise
        except Exception as e:
            self._log(f"执行预测时出错: {e}", 'error')
            proc.kill()
            raise

        result_code = proc.returncode if proc.returncode is not None else -1
        
        if result_code != 0 and not terminated_by_user and not self.stop_event.is_set():
            raise RuntimeError(f"预测脚本执行失败，返回码: {result_code}")
        elif terminated_by_user or self.stop_event.is_set():
            self._log("预测被用户中断", 'warning')

    def request_stop(self):
        """外部调用以请求停止后续任务"""
        self._log("⏹️ 正在停止所有处理线程...")
        self.stop_event.set()
        
        # 等待所有工作线程完成
        for worker in self.workers:
            if worker.is_alive():
                worker.join(timeout=2.0)
        
        # 获取队列中未处理的任务数
        remaining = self.task_queue.qsize()
        if remaining > 0:
            self._log(f"⚠️ 队列中还有 {remaining} 个任务未处理", 'warning')
        
        self._log("✅ 所有处理线程已停止")

    def upload_results_to_minio(self, json_path: Path, global_image_path: Path, output_dir: Path):
        """将JSON与原始全图上传到 MinIO"""
        if not self.minio_uploader:
            return

        uploads = []
        # 直接上传到 base_path，不使用时间戳文件夹
        # prefix = output_dir.name  # 已移除，不再使用时间戳文件夹

        if self.minio_upload_json and json_path.exists():
            obj = json_path.name  # 直接使用文件名，上传到 heathycare/source/
            uploads.append(("json", json_path, obj))
        elif self.minio_upload_json:
            self._log(f"⚠️ JSON 文件不存在，无法上传: {json_path}", "warning")

        if self.minio_upload_original and global_image_path.exists():
            obj = global_image_path.name  # 直接使用文件名，上传到 heathycare/source/
            uploads.append(("image", global_image_path, obj))
        elif self.minio_upload_original:
            self._log(f"⚠️ 原始全图不存在，无法上传: {global_image_path}", "warning")

        for kind, local_path, object_name in uploads:
            try:
                remote = self.minio_uploader.upload_file(local_path, object_name=object_name)
                self._log(f"☁️ 已上传 {kind} 至 MinIO: {remote}")
            except Exception as e:  # pragma: no cover
                self._log(f"⚠️ 上传 {local_path} 失败: {e}", "warning")


def process_existing_files(watch_dir: Path, handler: ImageProcessHandler):
    """处理已存在的文件"""
    handler._log(f"\n{'='*70}")
    handler._log("🔍 扫描已存在的图片文件...")
    handler._log(f"{'='*70}\n")
    
    # 重新加载状态文件（确保使用最新的处理记录）
    handler.load_status()
    
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(watch_dir.glob(f"*{ext}"))
        image_files.extend(watch_dir.glob(f"*{ext.upper()}"))
    
    if len(image_files) == 0:
        handler._log("✓ 未找到图片文件\n")
        return
    
    handler._log(f"找到 {len(image_files)} 个图片文件\n")
    
    for image_file in sorted(image_files):
        file_hash = handler.get_file_hash(image_file)
        if file_hash not in handler.processed_files:
            handler._log(f"\n📝 处理已存在的文件: {image_file.name}")
            try:
                handler.process_image(image_file, file_hash)
                handler._log(f"✅ 处理完成: {image_file.name}\n")
            except Exception as e:
                handler._log(f"❌ 处理失败 {image_file.name}: {e}\n", 'error')
                import traceback
                if handler.logger:
                    handler.logger.error(traceback.format_exc())
                else:
                    traceback.print_exc()


def main():
    parser = argparse.ArgumentParser(
        description="自动化处理脚本：监听文件夹，自动切片和预测"
    )
    parser.add_argument(
        "--watch-dir",
        type=str,
        default="/home/ubuntu/lsn/project_new/RT-DETR-main/DataWSI",
        help="监听的文件夹路径（默认: DataWSI）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/home/ubuntu/lsn/project_new/RT-DETR-main/DataPatchesInference",
        help="预测结果输出目录（默认: DataPatchesInference）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/home/ubuntu/lsn/project_new/RT-DETR-main/DataSlice2Inference/models/rtdetr_torchscript_cuda.pt",
        help="模型文件路径",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=640,
        help="切片大小（默认: 640）",
    )
    parser.add_argument(
        "--threshold",
        
        type=float,
        default=0.5,
        help="预测置信度阈值（默认: 0.5）",
    )
    parser.add_argument(
        "--bg-rgb",
        type=str,
        default="238,235,235",
        help="背景色 RGB（默认: 238,235,235）",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=30,
        help="背景色容差（默认: 30）",
    )
    parser.add_argument(
        "--bg-ratio",
        type=float,
        default=0.9,
        help="背景比例阈值（默认: 0.9）",
    )
    parser.add_argument(
        "--std-thresh",
        type=float,
        default=10.0,
        help="灰度标准差阈值（默认: 10.0）",
    )
    parser.add_argument(
        "--disable-auto-bg",
        action="store_true",
        help="禁用自动背景参数分析（默认：开启）",
    )
    parser.add_argument(
        "--bg-analysis-limit",
        type=int,
        default=5,
        help="自动背景分析时采样的patch数量（默认: 5）",
    )
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="推理时不保存带框可视化图片，只输出JSON结果",
    )
    parser.add_argument(
        "--process-existing",
        action="store_true",
        help="处理已存在的文件",
    )
    parser.add_argument(
        "--file-wait-timeout",
        type=int,
        default=30,
        help="等待文件完全写入的秒数（默认: 30）",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径（可选）",
    )
    
    args = parser.parse_args()
    
    # 解析路径
    watch_dir = Path(args.watch_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    model_path = Path(args.model).resolve()
    
    # 检查路径
    if not watch_dir.exists():
        print(f"❌ 错误：监听目录不存在: {watch_dir}")
        sys.exit(1)
    
    if not model_path.exists():
        print(f"❌ 错误：模型文件不存在: {model_path}")
        sys.exit(1)
    
    # 解析背景色
    bg_rgb = tuple(map(int, args.bg_rgb.split(',')))
    if len(bg_rgb) != 3:
        print(f"❌ 错误：背景色格式错误，应为 R,G,B")
        sys.exit(1)
    
    # 创建事件处理器
    handler = ImageProcessHandler(
        watch_dir=watch_dir,
        output_base_dir=output_dir,
        model_path=str(model_path),
        patch_size=args.patch_size,
        threshold=args.threshold,
        bg_rgb=bg_rgb,
        tolerance=args.tolerance,
        bg_ratio=args.bg_ratio,
        std_thresh=args.std_thresh,
        auto_bg=not args.disable_auto_bg,
        bg_analysis_limit=args.bg_analysis_limit,
        save_visualization=not args.no_visualization,
        file_wait_timeout=args.file_wait_timeout
    )
    
    # 处理已存在的文件
    if args.process_existing:
        process_existing_files(watch_dir, handler)
    
    # 创建观察者
    observer = Observer()
    observer.schedule(handler, str(watch_dir), recursive=False)
    
    print(f"\n{'='*70}")
    print("🚀 自动化处理服务已启动")
    print(f"{'='*70}")
    print(f"📁 监听目录: {watch_dir}")
    print(f"📁 输出目录: {output_dir}")
    print(f"🤖 模型路径: {model_path}")
    print(f"🔧 切片大小: {args.patch_size}x{args.patch_size}")
    print(f"🎯 置信度阈值: {args.threshold}")
    print(f"🧠 自动背景分析: {'开启' if not args.disable_auto_bg else '关闭'} (采样 {args.bg_analysis_limit} 个patch)")
    print(f"🖼️ 保存可视化: {'是' if not args.no_visualization else '否'}")
    print(f"{'='*70}")
    print("\n💡 提示：将图片放入监听目录即可自动处理")
    print("💡 按 Ctrl+C 停止服务\n")
    
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n\n⏹️ 正在停止服务...")
        observer.stop()
    
    observer.join()
    print("✅ 服务已停止")


if __name__ == "__main__":
    main()

