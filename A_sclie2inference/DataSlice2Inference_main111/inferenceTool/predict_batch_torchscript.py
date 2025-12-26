#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 RT-DETR v2 的 TorchScript 模型进行批量预测，支持将patch坐标转换为全图坐标。

功能：
1. 批量处理patch图像
2. 自动读取patch坐标信息（从CSV文件）
3. 将检测框坐标从patch坐标转换为全图坐标
4. 保存预测结果（可视化图片和JSON标注）

坐标转换逻辑：
- patch在全图中的左上角坐标：从CSV读取 (x_start, y_start)
- 模型输出的检测框坐标：相对于patch的 (x1, y1, x2, y2)
- 转换为全图坐标：(x_start + x1, y_start + y1, x_start + x2, y_start + y2)

示例：
python predict_batch_torchscript.py \
  --model ../models/rtdetr_torchscript_cuda.pt \
  --patch-dir ../../DataPatchesKeep/Patches5 \
  --output-dir ../../DataPatchesInference/Patches5 \
  --global-image-url /home/ubuntu/lsn/project_new/RT-DETR-main/DataWSI/45-庄驷40X.jpeg \
  --threshold 0.5


  python predict_batch_torchscript.py \
  --model ../models/rtdetr_torchscript_cuda.pt \
  --patch-dir /home/ubuntu/lsn/project_new/RT-DETR-main/DataPatchesKeep/37-吴红娟40X_20251124_235450 \
  --output-dir /home/ubuntu/lsn/project_new/RT-DETR-main/DataPatchesInference/37-吴红娟40X_new \
  --global-image-url /home/ubuntu/lsn/project_new/RT-DETR-main/DataWSI/37-吴红娟40X.jpeg \
  --threshold 0.3
"""

import argparse
import json
import os
import csv
import io
import time
import yaml
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List, Optional

import torch
try:
    import torchvision.transforms as T
    TORCHVISION_AVAILABLE = True
except Exception as e:
    # NPU环境下torchvision可能不可用，使用替代实现
    TORCHVISION_AVAILABLE = False
    print(f"⚠️ torchvision导入失败，将使用替代实现: {e}")

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm


# 类别名称定义
CLASS_NAMES = [
    "AD", "BC", "EC", "L", "LC", "M", "NT", "SM", "SQ", "TC1", "TC2", "TC3"
]


def load_config(config_path: Optional[str] = None) -> dict:
    """加载配置文件
    
    Args:
        config_path: 配置文件路径，如果为None则尝试从默认位置加载
    
    Returns:
        配置字典
    """
    # 默认配置文件路径（相对于脚本目录）
    if config_path is None:
        script_dir = Path(__file__).parent.parent
        config_path = script_dir / "config.yaml"
    else:
        config_path = Path(config_path)
    
    # 如果配置文件不存在，返回空配置
    if not config_path.exists():
        print(f"⚠️ 配置文件不存在: {config_path}，使用默认参数")
        return {}
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✓ 已加载配置文件: {config_path}")
        return config
    except Exception as e:
        print(f"⚠️ 加载配置文件失败: {e}，使用默认参数")
        return {}


def expand_env_vars(value):
    """展开环境变量（支持 ${VAR:default} 格式）"""
    if not isinstance(value, str):
        return value
    
    import re
    # 匹配 ${VAR:default} 或 ${VAR} 格式
    pattern = r'\$\{([^:}]+)(?::([^}]*))?\}'
    
    def replace_var(match):
        var_name = match.group(1)
        default_value = match.group(2) or ""
        return os.environ.get(var_name, default_value)
    
    return re.sub(pattern, replace_var, value)


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用 TorchScript RT-DETR v2 模型进行批量预测（支持坐标转换）"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径（默认：../config.yaml）",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="TorchScript 模型路径（.pt），命令行参数会覆盖配置文件",
    )
    parser.add_argument(
        "--patch-dir",
        type=str,
        default=None,
        help="patch图像所在目录（必需，除非在配置文件中指定）",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="输出目录（默认：patch_dir + '_inference'）",
    )
    parser.add_argument(
        "--coordinates-csv",
        type=str,
        default=None,
        help="patch坐标CSV文件路径（默认：在patch-dir目录下查找patch_coordinates.csv）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="推理设备：auto / cpu / cuda / cuda:0 等（默认从配置文件读取，配置文件默认 auto）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="置信度阈值（默认从配置文件读取，配置文件默认 0.5）",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="批量推理大小（默认从配置文件读取，配置文件默认自动根据GPU显存调整）",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default=None,
        help="图像文件匹配模式（默认从配置文件读取，配置文件默认 *.png）",
    )
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="不保存可视化图片（只保存JSON结果）",
    )
    parser.add_argument(
        "--save-visualization",
        action="store_true",
        help="保存可视化图片（会覆盖配置文件）",
    )
    parser.add_argument(
        "--global-image-url",
        type=str,
        default=None,
        help="大图存储路径或URL（会写入JSON info字段）",
    )
    parser.add_argument(
        "--json-name",
        type=str,
        default="annotations.json",
        help="输出JSON文件名（默认 annotations.json，可传入全图同名JSON）",
    )
    parser.add_argument(
        "--use-fp16",
        type=str,
        choices=['true', 'false', 'auto'],
        default=None,
        help="是否启用FP16混合精度（true/false/auto，auto表示GPU自动启用）",
    )
    
    args = parser.parse_args()
    
    # 加载配置文件
    config = load_config(args.config)
    batch_config = config.get('batch_inference', {})
    
    # 合并配置：命令行参数 > 配置文件 > 默认值
    # 展开环境变量
    if args.model is None:
        # 优先从 batch_inference.model_path 读取
        model_path = batch_config.get('model_path')
        # 如果未设置，尝试从 paths.model_path 读取
        if not model_path:
            paths_config = config.get('paths', {})
            model_path = paths_config.get('model_path')
        # 如果仍未设置，使用默认值
        if not model_path:
            model_path = '/home/ubuntu/lsn/project_new/RT-DETR-main/DataSlice2Inference/models/rtdetr_torchscript_cuda.pt'
        args.model = expand_env_vars(model_path)
    
    if args.patch_dir is None:
        args.patch_dir = batch_config.get('default_patch_dir')
        if args.patch_dir:
            args.patch_dir = expand_env_vars(args.patch_dir)
    
    if args.output_dir is None:
        default_output = batch_config.get('default_output_dir')
        if default_output:
            args.output_dir = expand_env_vars(default_output)
    
    if args.coordinates_csv is None:
        default_csv = batch_config.get('default_coordinates_csv')
        if default_csv:
            args.coordinates_csv = expand_env_vars(default_csv)
    
    if args.device is None:
        args.device = batch_config.get('device', 'auto')
    
    if args.threshold is None:
        args.threshold = batch_config.get('threshold', 0.5)
    
    if args.batch_size is None:
        args.batch_size = batch_config.get('batch_size')  # None表示自动
    
    if args.pattern is None:
        args.pattern = batch_config.get('pattern', '*.png')
    
    if args.global_image_url is None:
        args.global_image_url = batch_config.get('default_global_image_url', '')
    
    # 处理可视化参数：命令行 > 配置文件
    if args.save_visualization:
        args.no_visualization = False
    elif not args.no_visualization:
        # 如果命令行没有明确指定，从配置文件读取
        save_vis = batch_config.get('save_visualization', True)
        args.no_visualization = not save_vis
    
    # 处理FP16参数
    if args.use_fp16 is None:
        use_fp16_config = batch_config.get('use_fp16', True)
        args.use_fp16 = 'auto' if use_fp16_config else 'false'
    
    # 验证必需参数
    if args.patch_dir is None:
        parser.error("--patch-dir 参数是必需的（或在配置文件中指定 default_patch_dir）")
    
    # 保存配置供后续使用
    args._config = config
    
    return args


def resolve_device(device_str: str) -> torch.device:
    """自动检测并返回设备（支持NPU/GPU/CPU）"""
    # 尝试导入NPU工具模块
    import sys
    from pathlib import Path
    
    # 添加utils目录到路径
    utils_path = Path(__file__).parent.parent / "utils"
    if str(utils_path) not in sys.path:
        sys.path.insert(0, str(utils_path))
    
    try:
        from npu_utils import resolve_device as npu_resolve_device
        return npu_resolve_device(device_str)
    except (ImportError, Exception) as e:
        # 回退到原始逻辑
        device_str_lower = device_str.lower()
        if device_str_lower == "auto":
            if torch.cuda.is_available():
                print("⚡ 检测到可用 GPU，使用 CUDA 进行推理")
                return torch.device("cuda")
            print("⚠️ 未检测到 GPU，使用 CPU 推理")
            return torch.device("cpu")
        return torch.device(device_str)


def load_torchscript_model(model_path: str, device: torch.device, use_fp16: bool = True) -> torch.jit.ScriptModule:
    """加载TorchScript模型
    
    Args:
        model_path: 模型文件路径
        device: 推理设备
        use_fp16: 是否启用FP16混合精度（NPU/GPU支持）
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    # 检查文件大小，如果太小可能文件损坏
    file_size = os.path.getsize(model_path)
    if file_size < 1024:  # 小于1KB肯定有问题
        raise RuntimeError(f"模型文件异常小（{file_size}字节），可能文件损坏或不完整: {model_path}")
    
    # 检查文件是否可读
    if not os.access(model_path, os.R_OK):
        raise PermissionError(f"模型文件不可读: {model_path}")

    print(f"=== 加载 TorchScript 模型 ===")
    print(f"模型路径: {model_path}")
    print(f"文件大小: {file_size / 1024 / 1024:.2f} MB")
    print(f"设备: {device}")

    # NPU 环境特殊处理：禁用多进程编译以避免冲突
    if device.type == 'npu':
        # 设置环境变量，禁用 TBE 多进程编译
        os.environ['TE_PARALLEL_COMPILER'] = '0'  # 禁用并行编译
        os.environ['ASCEND_RT_VISIBLE_DEVICES'] = '0'  # 指定设备
        # 设置单线程模式
        try:
            import torch_npu  # noqa: F401
            # 禁用多进程编译（如果支持）
            if hasattr(torch_npu, 'npu') and hasattr(torch_npu.npu, 'set_compile_mode'):
                try:
                    torch_npu.npu.set_compile_mode(jit_compile=False)
                except Exception:
                    pass
        except ImportError:
            pass
        print("🔧 NPU 环境：已禁用多进程编译模式")

    try:
        # 对于 NPU，先加载到 CPU，再移动到 NPU（避免加载时的多进程问题）
        if device.type == 'npu':
            print("📦 正在加载模型（NPU 模式：先加载到 CPU，再移动到 NPU）...")
            # 先加载到 CPU
            model = torch.jit.load(model_path, map_location='cpu')
            model.eval()
            # 然后移动到 NPU
            print("📦 正在将模型移动到 NPU 设备...")
            model = model.to(device)
            # 同步等待，确保模型完全加载
            if hasattr(torch, 'npu') and hasattr(torch.npu, 'synchronize'):
                torch.npu.synchronize()
        else:
            model = torch.jit.load(model_path, map_location=device)
            model.eval()
    except RuntimeError as e:
        if "failed reading zip archive" in str(e) or "central directory" in str(e):
            raise RuntimeError(
                f"模型文件损坏或不完整: {model_path}\n"
                f"错误详情: {e}\n"
                f"文件大小: {file_size / 1024 / 1024:.2f} MB\n"
                f"请检查：\n"
                f"  1. 模型文件是否完整（是否在传输/复制过程中被截断）\n"
                f"  2. 模型文件是否损坏\n"
                f"  3. 尝试重新导出或下载模型文件"
            ) from e
        raise
    
    # 启用FP16混合精度（NPU/GPU支持）
    if use_fp16 and device.type in ['npu', 'cuda']:
        print("📦 正在启用 FP16 混合精度...")
        model = model.half()
        # NPU 需要同步
        if device.type == 'npu' and hasattr(torch, 'npu') and hasattr(torch.npu, 'synchronize'):
            torch.npu.synchronize()
        print(f"✅ 已启用 FP16 混合精度加速（{device.type.upper()}）")
        print("   - 推理速度提升约40%")
        print("   - 显存占用减少约50%")
    
    print("✓ 模型加载完成\n")
    return model


def load_patch_coordinates(csv_path: str) -> Tuple[Dict[str, Tuple[int, int]], float]:
    """从CSV文件加载patch坐标信息
    
    CSV格式：filename, x_start, y_start, x_end, y_end
    支持跳过以 # 开头的注释行，并从注释中提取scale_factor
    
    Returns:
        (coordinates, scale_factor): 
            - coordinates: {filename: (x_start, y_start)} - 已映射到原图的坐标
            - scale_factor: 图像缩放系数（从注释行提取，默认1.0）
    """
    coordinates = {}
    scale_factor = 1.0  # 默认无缩放
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"坐标CSV文件不存在: {csv_path}")
    
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        # 读取所有行，提取scale_factor并过滤注释行
        lines = []
        for line in f:
            stripped = line.strip()
            if stripped.startswith('# Scale factor:'):
                # 提取缩放系数: "# Scale factor: 2.0x" -> 2.0
                try:
                    scale_str = stripped.split(':')[1].strip().rstrip('x')
                    scale_factor = float(scale_str)
                    print(f"✓ 检测到图像缩放系数: {scale_factor}x")
                except (IndexError, ValueError) as e:
                    print(f"⚠️ 无法解析缩放系数: {stripped}, 使用默认值1.0")
            elif not stripped.startswith('#'):
                lines.append(line)
        
        # 使用过滤后的行创建 DictReader
        csv_content = io.StringIO(''.join(lines))
        reader = csv.DictReader(csv_content)
        
        for row in reader:
            filename = row['filename']
            x_start = int(row['x_start'])
            y_start = int(row['y_start'])
            coordinates[filename] = (x_start, y_start)
    
    print(f"✓ 已加载 {len(coordinates)} 个patch的坐标信息")
    if scale_factor != 1.0:
        print(f"✓ CSV坐标已映射到原图（缩放系数: {scale_factor}x）")
    
    return coordinates, scale_factor


def convert_to_global_coordinates(
    boxes: torch.Tensor,
    patch_offset: Tuple[int, int],
    scale_factor: float = 1.0
) -> torch.Tensor:
    """将patch内的检测框坐标转换为全图坐标
    
    Args:
        boxes: 检测框坐标 (N, 4)，格式为 (x1, y1, x2, y2)，相对于patch
        patch_offset: patch在全图中的偏移量 (x_start, y_start) - 已经是原图坐标
        scale_factor: 图像缩放系数，用于将patch内坐标映射到原图尺度
    
    Returns:
        全图坐标 (N, 4)，格式为 (x1, y1, x2, y2)
    
    说明：
        当图像经过缩放时（如从100000x80000缩放到50000x40000），
        patch是在缩放后的图像上切的640x640，但patch_offset已经映射回原图。
        因此需要：
        1. 先将patch内的检测框坐标按scale_factor缩放（映射到原图尺度）
        2. 再加上patch在原图中的偏移量
        
        例如：
        - 原图100000x80000，缩放为50000x40000（scale_factor=2.0）
        - patch在缩放图上是640x640，对应原图1280x1280
        - patch_offset=(1280, 0) 是原图坐标
        - 检测框在patch内(100, 150, 200, 250)
        - 映射：(100*2, 150*2, 200*2, 250*2) + (1280, 0, 1280, 0)
        - 结果：(1480, 300, 1680, 500) 在原图上
    """
    # 步骤1：将patch内坐标按scale_factor缩放到原图尺度
    boxes_scaled = boxes * scale_factor
    
    # 步骤2：加上patch在原图中的偏移量
    x_offset, y_offset = patch_offset
    offset_tensor = torch.tensor(
        [x_offset, y_offset, x_offset, y_offset],
        dtype=boxes_scaled.dtype,
        device=boxes_scaled.device
    )
    
    return boxes_scaled + offset_tensor


def prepare_image(image_path: str, device: torch.device, use_fp16: bool = False):
    """加载并预处理图像
    
    Args:
        image_path: 图像路径
        device: 设备
        use_fp16: 是否使用FP16
    """
    image_pil = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image_pil.size

    # 使用torchvision或替代实现
    if TORCHVISION_AVAILABLE:
        transforms = T.Compose([
            T.Resize((640, 640)),
            T.ToTensor(),
        ])
        image_tensor = transforms(image_pil).unsqueeze(0).to(device)
    else:
        # 替代实现：使用PIL和numpy
        # Resize
        image_resized = image_pil.resize((640, 640), Image.BILINEAR)
        # 转换为numpy数组并归一化到[0,1]
        image_array = np.array(image_resized, dtype=np.float32) / 255.0
        # 转换为CHW格式并转换为tensor
        image_array = image_array.transpose(2, 0, 1)  # HWC -> CHW
        image_tensor = torch.from_numpy(image_array).unsqueeze(0).to(device)
    
    # 如果使用FP16，转换数据类型（NPU/GPU支持）
    if use_fp16 and device.type in ['npu', 'cuda']:
        image_tensor = image_tensor.half()
    
    orig_sizes = torch.tensor([[orig_w, orig_h]], dtype=torch.int64, device=device)

    return image_pil, image_tensor, orig_sizes


def postprocess_outputs(outputs, threshold: float):
    """后处理模型输出"""
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 3:
        raise RuntimeError(f"模型输出格式异常，期望为 (labels, boxes, scores)，实际: {type(outputs)}")

    labels, boxes, scores = outputs
    # 去掉 batch 维度
    if labels.dim() > 1:
        labels = labels[0]
    if boxes.dim() > 2:
        boxes = boxes[0]
    if scores.dim() > 1:
        scores = scores[0]

    # 过滤低置信度
    valid = scores > threshold
    labels = labels[valid]
    boxes = boxes[valid]
    scores = scores[valid]

    return labels, boxes, scores


def draw_detections(image_pil, labels, boxes, scores, threshold: float):
    """在图像上绘制检测结果"""
    draw = ImageDraw.Draw(image_pil)

    # 加载字体
    font = None
    for font_path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        if os.path.exists(font_path):
            try:
                font = ImageFont.truetype(font_path, 18)
                break
            except Exception:
                continue
    if font is None:
        font = ImageFont.load_default()

    for lab, box, score in zip(labels, boxes, scores):
        cls_id = int(lab.item())
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
        x1, y1, x2, y2 = [float(x) for x in box.tolist()]

        # 绘制框
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        # 绘制标签
        text = f"{cls_name} {score:.2f}"
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except Exception:
            text_w = len(text) * 8
            text_h = 16

        pad = 4
        draw.rectangle([x1, y1 - text_h - 2 * pad, x1 + text_w + 2 * pad, y1], fill="red")
        draw.text((x1 + pad, y1 - text_h - pad), text, fill="white", font=font)

    return image_pil


def predict_batch_patches(
    model: torch.jit.ScriptModule,
    batch_patch_paths: List[str],
    batch_patch_offsets: List[Tuple[int, int]],
    device: torch.device,
    threshold: float,
    use_fp16: bool = False,
    save_vis: bool = True,
    output_dir: str = None,
    scale_factor: float = 1.0
) -> List[dict]:
    """批量预测多个patch，返回检测结果列表
    
    Args:
        model: TorchScript模型
        batch_patch_paths: 批量patch路径列表
        batch_patch_offsets: 批量patch偏移量列表
        scale_factor: 图像缩放系数，用于将检测框坐标映射到原图尺度
        device: 推理设备
        threshold: 置信度阈值
        use_fp16: 是否使用FP16
        save_vis: 是否保存可视化
        output_dir: 输出目录
    
    Returns:
        检测结果列表
    """
    try:
        batch_results = []
        batch_images = []
        batch_pil_images = []
        batch_orig_sizes = []
        
        # 批量加载图像
        for patch_path in batch_patch_paths:
            image_pil, image_tensor, orig_sizes = prepare_image(patch_path, device, use_fp16)
            batch_images.append(image_tensor)
            batch_pil_images.append(image_pil)
            batch_orig_sizes.append(orig_sizes)
        
        # 拼接为批量张量
        batch_tensor = torch.cat(batch_images, dim=0)  # (batch_size, 3, 640, 640)
        batch_sizes = torch.cat(batch_orig_sizes, dim=0)  # (batch_size, 2)
        
        # 批量推理
        with torch.no_grad():
            outputs = model(batch_tensor, batch_sizes)
        
        # 批量后处理
        for idx, (patch_path, patch_offset, image_pil) in enumerate(
            zip(batch_patch_paths, batch_patch_offsets, batch_pil_images)
        ):
            # 提取当前patch的输出
            if isinstance(outputs, (tuple, list)) and len(outputs) == 3:
                labels_all, boxes_all, scores_all = outputs
                
                # 提取第idx个样本的结果
                if labels_all.dim() > 1:
                    labels = labels_all[idx]
                    boxes_patch = boxes_all[idx]
                    scores = scores_all[idx]
                else:
                    # 如果是单个样本，直接使用
                    labels = labels_all
                    boxes_patch = boxes_all
                    scores = scores_all
                
                # 过滤低置信度
                valid = scores > threshold
                labels = labels[valid]
                boxes_patch = boxes_patch[valid]
                scores = scores[valid]
            else:
                # 回退到单样本处理
                labels, boxes_patch, scores = postprocess_outputs(outputs, threshold)
            
            # 转换为全图坐标
            boxes_global = convert_to_global_coordinates(boxes_patch, patch_offset, scale_factor)
            
            # 构建检测结果
            detections_patch = []
            detections_global = []
            
            for lab, box_patch, box_global, score in zip(labels, boxes_patch, boxes_global, scores):
                cls_id = int(lab.item())
                cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
                x1_p, y1_p, x2_p, y2_p = [float(x) for x in box_patch.tolist()]
                x1_g, y1_g, x2_g, y2_g = [float(x) for x in box_global.tolist()]

                detections_patch.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "bbox": [x1_p, y1_p, x2_p, y2_p],
                    "score": float(score.item())
                })

                detections_global.append({
                    "class_id": cls_id,
                    "class_name": cls_name,
                    "bbox": [x1_g, y1_g, x2_g, y2_g],
                    "score": float(score.item())
                })
            
            # 保存可视化图片
            if save_vis and output_dir:
                vis_image = draw_detections(image_pil.copy(), labels, boxes_patch, scores, threshold)
                patch_name = os.path.basename(patch_path)
                vis_path = os.path.join(output_dir, f"pred_{patch_name}")
                Path(vis_path).parent.mkdir(parents=True, exist_ok=True)
                vis_image.save(vis_path)
            
            batch_results.append({
                "patch_path": patch_path,
                "patch_name": os.path.basename(patch_path),
                "patch_offset": patch_offset,
                "patch_size": image_pil.size,
                "detection_count": len(detections_patch),
                "detections_patch": detections_patch,
                "detections_global": detections_global,
            })
        
        return batch_results
        
    except Exception as e:
        print(f"❌ 批量处理时出错: {e}")
        import traceback
        traceback.print_exc()
        return []


def predict_single_patch(
    model: torch.jit.ScriptModule,
    patch_path: str,
    patch_offset: Tuple[int, int],
    device: torch.device,
    threshold: float,
    use_fp16: bool = False,
    save_vis: bool = True,
    output_dir: str = None,
    scale_factor: float = 1.0
) -> dict:
    """预测单个patch，返回检测结果（包含patch坐标和全图坐标）
    
    Args:
        scale_factor: 图像缩放系数，用于将检测框坐标映射到原图尺度
    """
    try:
        # 加载和预处理图像
        image_pil, image_tensor, orig_sizes = prepare_image(patch_path, device, use_fp16)

        # 推理
        with torch.no_grad():
            outputs = model(image_tensor, orig_sizes)

        # 后处理
        labels, boxes_patch, scores = postprocess_outputs(outputs, threshold)

        # 转换为全图坐标
        boxes_global = convert_to_global_coordinates(boxes_patch, patch_offset, scale_factor)

        # 构建检测结果
        detections_patch = []
        detections_global = []
        
        for lab, box_patch, box_global, score in zip(labels, boxes_patch, boxes_global, scores):
            cls_id = int(lab.item())
            cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
            x1_p, y1_p, x2_p, y2_p = [float(x) for x in box_patch.tolist()]
            x1_g, y1_g, x2_g, y2_g = [float(x) for x in box_global.tolist()]

            detections_patch.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "bbox": [x1_p, y1_p, x2_p, y2_p],  # patch坐标
                "score": float(score.item())
            })

            detections_global.append({
                "class_id": cls_id,
                "class_name": cls_name,
                "bbox": [x1_g, y1_g, x2_g, y2_g],  # 全图坐标
                "score": float(score.item())
            })

        # 保存可视化图片
        if save_vis and output_dir:
            vis_image = draw_detections(image_pil.copy(), labels, boxes_patch, scores, threshold)
            patch_name = os.path.basename(patch_path)
            vis_path = os.path.join(output_dir, f"pred_{patch_name}")
            Path(vis_path).parent.mkdir(parents=True, exist_ok=True)
            vis_image.save(vis_path)

        return {
            "patch_path": patch_path,
            "patch_name": os.path.basename(patch_path),
            "patch_offset": patch_offset,
            "patch_size": image_pil.size,
            "detection_count": len(detections_patch),
            "detections_patch": detections_patch,  # patch坐标
            "detections_global": detections_global,  # 全图坐标
        }
    except Exception as e:
        print(f"❌ 处理 {patch_path} 时出错: {e}")
        return None


def main():
    args = parse_args()
    
    # NPU 环境：提前设置环境变量，避免多进程冲突
    if args.device and args.device.lower() == 'npu':
        # 禁用 TBE 多进程编译
        os.environ.setdefault('TE_PARALLEL_COMPILER', '0')
        os.environ.setdefault('ASCEND_RT_VISIBLE_DEVICES', '0')
        # 设置 NPU 内存分配配置
        os.environ.setdefault('PYTORCH_NPU_ALLOC_CONF', 'expandable_segments:True')
        print("🔧 NPU 环境变量已设置")
    
    # 解析路径
    patch_dir = Path(args.patch_dir)
    if not patch_dir.exists():
        raise FileNotFoundError(f"patch目录不存在: {patch_dir}")

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(str(patch_dir) + "_inference")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 查找坐标CSV文件
    if args.coordinates_csv:
        csv_path = Path(args.coordinates_csv)
    else:
        # 先尝试在patch目录下查找
        csv_path = patch_dir / "patch_coordinates.csv"
        # 如果不存在，尝试在父目录的DataPatches对应目录中查找
        if not csv_path.exists():
            # 例如：DataPatchesKeep/Patches5 -> DataPatches/Patches5
            if "Keep" in str(patch_dir) or "Trash" in str(patch_dir):
                possible_dir = str(patch_dir).replace("Keep", "").replace("Trash", "").rstrip("/")
                possible_dir = possible_dir.replace("DataPatchesKeep", "DataPatches").replace("DataPatchesTrash", "DataPatches")
                possible_csv = Path(possible_dir) / "patch_coordinates.csv"
                if possible_csv.exists():
                    csv_path = possible_csv
    
    if not csv_path.exists():
        raise FileNotFoundError(
            f"坐标CSV文件不存在: {csv_path}\n"
            "请指定 --coordinates-csv 参数，或确保patch目录或其对应原始目录下有 patch_coordinates.csv 文件"
        )

    # 加载坐标信息和缩放系数
    coordinates, scale_factor = load_patch_coordinates(str(csv_path))

    # 加载模型（启用FP16）
    device = resolve_device(args.device)
    
    # 处理FP16参数
    if args.use_fp16 == 'auto':
        use_fp16 = (device.type == 'cuda')  # GPU自动启用FP16
    elif args.use_fp16 == 'true':
        use_fp16 = True
    else:
        use_fp16 = False
    
    model = load_torchscript_model(args.model, device, use_fp16=use_fp16)

    # 查找所有patch图像
    patch_files = sorted(list(patch_dir.glob(args.pattern)))
    if len(patch_files) == 0:
        print(f"❌ 在 {patch_dir} 中未找到匹配 {args.pattern} 的图像文件")
        return

    print(f"\n{'='*70}")
    print(f"找到 {len(patch_files)} 个patch文件，开始批量推理...")
    print(f"{'='*70}\n")

    # 批量推理配置
    if args.batch_size is not None:
        # 使用命令行或配置文件指定的batch size
        BATCH_SIZE = args.batch_size
        print(f"📦 使用指定的 batch_size={BATCH_SIZE}")
    else:
        # 自动根据设备和显存调整batch size
        BATCH_SIZE = 16  # 默认值
        if device.type in ['npu', 'cuda']:
            # 根据设备显存自动调整batch size
            try:
                device_mem = None
                device_name = "设备"
                
                if device.type == 'npu':
                    # NPU内存信息获取方式（华为910B NPU约32GB）
                    device_mem = 32.0  # 可根据实际情况调整
                    device_name = "NPU"
                elif device.type == 'cuda':
                    device_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # GB
                    device_name = "GPU"
                
                if device_mem:
                    if device_mem < 8:
                        BATCH_SIZE = 8
                    elif device_mem >= 16:
                        BATCH_SIZE = 32
                    print(f"🔧 检测到 {device_mem:.1f}GB {device_name}显存，自动设置 batch_size={BATCH_SIZE}")
            except Exception as e:
                print(f"⚠️ 无法检测设备显存，使用默认 batch_size={BATCH_SIZE}: {e}")
        else:
            BATCH_SIZE = 4  # CPU模式使用较小的batch size
            print(f"💻 CPU模式，设置 batch_size={BATCH_SIZE}")
    
    print(f"📦 批量推理模式：每批处理 {BATCH_SIZE} 张图像")
    print()

    # 批量处理
    start_time = time.time()
    all_results = []
    success_count = 0
    fail_count = 0
    total_detections = 0

    # 准备批量数据
    valid_patches = []
    valid_offsets = []
    for patch_path in patch_files:
        patch_name = patch_path.name
        if patch_name in coordinates:
            valid_patches.append(str(patch_path))
            valid_offsets.append(coordinates[patch_name])
        else:
            print(f"⚠️ 警告：{patch_name} 的坐标信息未找到，跳过")
            fail_count += 1
    
    # 批量推理
    total_batches = (len(valid_patches) + BATCH_SIZE - 1) // BATCH_SIZE
    
    for batch_idx in tqdm(range(total_batches), desc="批量推理进度", unit="批"):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(valid_patches))
        
        batch_paths = valid_patches[start_idx:end_idx]
        batch_offsets = valid_offsets[start_idx:end_idx]
        
        # 批量预测
        batch_results = predict_batch_patches(
            model=model,
            batch_patch_paths=batch_paths,
            batch_patch_offsets=batch_offsets,
            device=device,
            threshold=args.threshold,
            use_fp16=use_fp16,
            save_vis=not args.no_visualization,
            output_dir=str(output_dir),
            scale_factor=scale_factor
        )
        
        # 统计结果
        for result in batch_results:
            if result:
                all_results.append(result)
                total_detections += result["detection_count"]
                success_count += 1
            else:
                fail_count += 1

    # 保存JSON结果
    json_path = output_dir / args.json_name
    json_data = {
        "info": {
            "description": "RT-DETR v2 TorchScript 批量推理结果",
            "version": "1.0",
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_path": args.model,
            "patch_dir": str(patch_dir),
            "threshold": args.threshold,
            "batch_size": BATCH_SIZE,
            "device": str(device),
            "use_fp16": use_fp16,
            "global_image_url": args.global_image_url,
        },
        "statistics": {
            "total_patches": len(patch_files),
            "successful": success_count,
            "failed": fail_count,
            "total_detections": total_detections,
            "average_detections_per_patch": round(total_detections / success_count, 2) if success_count > 0 else 0,
        },
        "results": all_results
    }

    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    # 统计信息
    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print("批量推理完成")
    print(f"{'='*70}")
    print(f"总patch数:     {len(patch_files):6d} 个")
    print(f"成功处理:     {success_count:6d} 个")
    print(f"处理失败:     {fail_count:6d} 个")
    print(f"检测目标总数: {total_detections:6d} 个")
    if success_count > 0:
        print(f"平均每patch:  {total_detections/success_count:.2f} 个目标")
    print(f"总耗时:       {total_time:8.2f} 秒 ({total_time/60:.2f} 分钟)")
    print(f"处理速度:     {len(patch_files)/total_time:8.2f} patch/秒")
    print(f"\n输出目录: {output_dir}")
    print(f"JSON结果: {json_path}")
    if not args.no_visualization:
        print(f"可视化图片: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()

