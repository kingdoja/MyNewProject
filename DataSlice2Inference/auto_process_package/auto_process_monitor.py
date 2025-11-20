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
from typing import Optional, Tuple
from collections import Counter

import numpy as np

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    print("❌ 错误：需要安装 watchdog 库")
    print("请运行: pip install watchdog")
    sys.exit(1)


# 支持的图像格式
IMAGE_EXTENSIONS = {'.jpeg', '.jpg', '.png', '.tif', '.tiff', '.ndpi'}


class ImageProcessHandler(FileSystemEventHandler):
    """处理新增图片的事件处理器"""
    
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
        save_visualization: bool = True # 是否保存可视化    
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
        
        # 创建输出目录结构
        self.patches_dir = output_base_dir.parent / "DataPatches"
        self.patches_keep_dir = output_base_dir.parent / "DataPatchesKeep"
        self.patches_trash_dir = output_base_dir.parent / "DataPatchesTrash"
        
        # 处理状态记录文件
        self.status_file = output_base_dir / "processing_status.json"
        self.load_status()
    
    def load_status(self):
        """加载处理状态"""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r', encoding='utf-8') as f:
                    status = json.load(f)
                    self.processed_files = set(status.get('processed_files', []))
            except Exception as e:
                print(f"⚠️ 加载状态文件失败: {e}")
    
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
            print(f"⚠️ 保存状态文件失败: {e}")
    
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
    
    def on_created(self, event):
        """文件创建事件"""
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        if not self.is_image_file(file_path):
            return
        
        # 等待文件完全写入（大文件可能需要时间）
        time.sleep(2)
        
        if not file_path.exists():
            return
        
        # 检查是否已处理
        file_hash = self.get_file_hash(file_path)
        if file_hash in self.processed_files:
            print(f"⏭️ 跳过已处理的文件: {file_path.name}")
            return
        
        print(f"\n{'='*70}")
        print(f"🆕 检测到新图片: {file_path.name}")
        print(f"{'='*70}\n")
        
        try:
            # 处理图片
            self.process_image(file_path, file_hash)
            print(f"✅ 处理完成: {file_path.name}\n")
        except Exception as e:
            print(f"❌ 处理失败 {file_path.name}: {e}\n")
            import traceback
            traceback.print_exc()
    
    def process_image(self, image_path: Path, file_hash: str):
        """处理单张图片：切片 -> 过滤 -> 预测"""
        
        # 生成唯一的输出目录名（基于文件名和时间戳）
        image_name = image_path.stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_name = f"{image_name}_{timestamp}"
        
        # 创建输出目录
        patches_subdir = self.patches_dir / output_name
        patches_keep_subdir = self.patches_keep_dir / output_name
        patches_trash_subdir = self.patches_trash_dir / output_name
        inference_output_dir = self.output_base_dir / output_name
        
        print(f"📁 输出目录: {patches_subdir}")
        print(f"📁 预测输出目录: {inference_output_dir}\n")
        
        # 步骤1: 切片
        print("="*70)
        print("步骤 1/3: 图片切片")
        print("="*70)
        patches_subdir.mkdir(parents=True, exist_ok=True)
        self.slice_image(image_path, patches_subdir)
        
        # 自动背景参数分析
        if self.auto_bg:
            print("\n" + "="*70)
            print("步骤 1.5: 自动分析背景参数")
            print("="*70)
            self.auto_configure_background(patches_subdir)
        
        # 步骤2: 过滤空白patch
        print("\n" + "="*70)
        print("步骤 2/3: 过滤空白patch")
        print("="*70)
        patches_keep_subdir.mkdir(parents=True, exist_ok=True)
        patches_trash_subdir.mkdir(parents=True, exist_ok=True)
        self.filter_patches(patches_subdir, patches_keep_subdir, patches_trash_subdir)
        
        # 检查是否有保留的patch
        keep_patches = list(patches_keep_subdir.glob("*.png"))
        if len(keep_patches) == 0:
            print("⚠️ 警告：没有保留的patch，跳过预测步骤")
            self.processed_files.add(file_hash)
            self.save_status()
            return
        
        # 步骤3: 批量预测
        print("\n" + "="*70)
        print(f"步骤 3/3: 批量预测 (共 {len(keep_patches)} 个patch)")
        print("="*70)
        inference_output_dir.mkdir(parents=True, exist_ok=True)
        self.batch_predict(
            patches_keep_subdir,
            inference_output_dir,
            patches_subdir,
            image_path
        )
        
        # 标记为已处理
        self.processed_files.add(file_hash)
        self.save_status()
    
    def slice_image(self, image_path: Path, output_dir: Path):
        """切片图像"""
        from PIL import Image, ImageFile
        from tqdm import tqdm
        import csv
        
        # 禁用 PIL 的 decompression bomb 检查
        Image.MAX_IMAGE_PIXELS = None
        ImageFile.LOAD_TRUNCATED_IMAGES = True
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 打开图像
        print("正在打开图像文件...")
        try:
            pil_img = Image.open(image_path)
            if pil_img.mode != 'RGB':
                print(f"图像模式为 {pil_img.mode}，将在处理时转换为 RGB")
        except Exception as e:
            raise FileNotFoundError(f"无法打开图片文件: {image_path}, 错误: {e}")
        
        # 获取图像尺寸
        width, height = pil_img.size
        print(f"图像尺寸: {width} x {height} 像素")
        print(f"总像素数: {width * height:,}")
        
        # 计算可切 patch 数量
        num_patches_x = width // self.patch_size
        num_patches_y = height // self.patch_size
        total_patches = num_patches_x * num_patches_y
        print(f"将切分为 {num_patches_x} x {num_patches_y} = {total_patches} 个 patch")
        
        # 记录坐标
        x_r = []
        y_r = []
        patch_count = 0
        
        # 分块读取并保存 patch
        print("开始切分图像...")
        for y_idx in tqdm(range(num_patches_y), desc="处理行", unit="行"):
            for x_idx in range(num_patches_x):
                x = x_idx * self.patch_size
                y = y_idx * self.patch_size
                
                x_r.append(x)
                y_r.append(y)
                
                # 使用 crop 方法只读取当前 patch 区域
                patch_box = (x, y, x + self.patch_size, y + self.patch_size)
                try:
                    patch_pil = pil_img.crop(patch_box)
                except Exception as e:
                    print(f"\n警告：读取 patch ({x}, {y}) 时出错: {e}")
                    continue
                
                # 转换为 RGB 模式
                if patch_pil.mode != 'RGB':
                    patch_pil = patch_pil.convert('RGB')
                
                # 保存 patch
                save_path = output_dir / f'patch_{patch_count}.png'
                try:
                    patch_pil.save(save_path, 'PNG')
                except Exception as e:
                    print(f"\n警告：保存 patch {patch_count} 时出错: {e}")
                    continue
                
                patch_count += 1
        
        # 保存坐标信息到CSV文件
        csv_path = output_dir / 'patch_coordinates.csv'
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['filename', 'x_start', 'y_start', 'x_end', 'y_end'])
            for i in range(patch_count):
                filename = f'patch_{i}.png'
                writer.writerow([filename, x_r[i], y_r[i], x_r[i] + self.patch_size, y_r[i] + self.patch_size])
        
        print(f"\n✓ 完成！共保存 {patch_count} 个 patch")
        print(f"✓ 坐标信息已保存到: {csv_path}")
    
    def auto_configure_background(self, patches_dir: Path):
        """分析patch背景信息并自动调整过滤参数"""
        image_files = sorted([p for p in patches_dir.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg"}])
        if not image_files:
            print("⚠️ 没有找到可分析的patch，保持原有背景参数")
            return
        
        sample_files = image_files[: min(self.bg_analysis_limit, len(image_files))]
        results = []
        
        for img_path in sample_files:
            bg_color, bg_percent, std = self.analyze_single_patch(img_path)
            if bg_color is not None:
                results.append((bg_color, bg_percent, std))
        
        if not results:
            print("⚠️ 背景分析失败，保持原有背景参数")
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
        
        print(f"分析 {len(results)} 个patch 得到背景参数：")
        print(f"  背景色 (RGB): {tuple(avg_bg.tolist())}")
        print(f"  背景覆盖率: {avg_bg_percent*100:.1f}%")
        print(f"  灰度标准差: {avg_std:.2f}")
        print("推荐过滤参数：")
        print(f"  BG_RGB      = {tuple(avg_bg.tolist())}")
        print(f"  TOLERANCE   = {recommended_tolerance}")
        print(f"  BG_RATIO    = {recommended_ratio}")
        print(f"  STD_THRESH  = {recommended_std_thresh:.2f}")
        
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
            print(f"⚠️ 分析 {image_path.name} 时出错: {e}")
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
            print("⚠️ 未找到patch文件")
            return
        
        print(f"共发现 {total} 个 patch，开始筛选...")
        
        kept = removed = failed = 0
        keep_dir.mkdir(parents=True, exist_ok=True)
        trash_dir.mkdir(parents=True, exist_ok=True)
        
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = [executor.submit(process_one, p) for p in files]
            for i, future in enumerate(as_completed(futures), 1):
                path, blank, ratio, std = future.result()
                if ratio < 0:
                    failed += 1
                    print(f"[FAIL] 读取失败: {path}")
                else:
                    if blank:
                        removed += 1
                    else:
                        kept += 1
                if i % 500 == 0 or i == total:
                    print(f"[{i}/{total}] 保留={kept}, 移除={removed}, 失败={failed}")
        
        print(f"✓ 总计: {total}, 保留(非空白): {kept}, 移除(空白): {removed}, 失败: {failed}")
    
    def batch_predict(
        self,
        patch_dir: Path,
        output_dir: Path,
        coordinates_dir: Path,
        global_image_path: Optional[Path] = None
    ):
        """调用批量预测脚本"""
        predict_script = (Path(__file__).resolve().parent.parent / "inferenceTool" / "predict_batch_torchscript.py").resolve()
        
        # 查找坐标CSV文件
        coordinates_csv = coordinates_dir / "patch_coordinates.csv"
        if not coordinates_csv.exists():
            print(f"⚠️ 警告：坐标CSV文件不存在: {coordinates_csv}")
            print("预测将无法进行坐标转换")
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
        
        # 执行命令
        result = subprocess.run(
            cmd,
            cwd=str(Path(__file__).resolve().parent.parent),
            capture_output=True,
            text=True,
            timeout=7200  # 2小时超时
        )
        
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        
        if result.returncode != 0:
            raise RuntimeError(f"预测脚本执行失败，返回码: {result.returncode}")


def process_existing_files(watch_dir: Path, handler: ImageProcessHandler):
    """处理已存在的文件"""
    print(f"\n{'='*70}")
    print("🔍 扫描已存在的图片文件...")
    print(f"{'='*70}\n")
    
    image_files = []
    for ext in IMAGE_EXTENSIONS:
        image_files.extend(watch_dir.glob(f"*{ext}"))
        image_files.extend(watch_dir.glob(f"*{ext.upper()}"))
    
    if len(image_files) == 0:
        print("✓ 未找到图片文件\n")
        return
    
    print(f"找到 {len(image_files)} 个图片文件\n")
    
    for image_file in sorted(image_files):
        file_hash = handler.get_file_hash(image_file)
        if file_hash not in handler.processed_files:
            print(f"\n📝 处理已存在的文件: {image_file.name}")
            try:
                handler.process_image(image_file, file_hash)
                print(f"✅ 处理完成: {image_file.name}\n")
            except Exception as e:
                print(f"❌ 处理失败 {image_file.name}: {e}\n")
                import traceback
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
        save_visualization=not args.no_visualization
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

