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
  --patch-dir /home/ubuntu/lsn/project_new/RT-DETR-main/DataPatchesKeep/cj_20251119_154443 \
  --output-dir /home/ubuntu/lsn/project_new/RT-DETR-main/DataPatchesInference/cj_20251119_154443 \
  --global-image-url /home/ubuntu/lsn/project_new/RT-DETR-main/DataWSI/cj.jpeg \
  --threshold 0.5
"""

import argparse
import json
import os
import csv
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple, List

import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


# 类别名称定义
CLASS_NAMES = [
    "AD", "BC", "EC", "L", "LC", "M", "NT", "SM", "SQ", "TC1", "TC2", "TC3"
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用 TorchScript RT-DETR v2 模型进行批量预测（支持坐标转换）"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/home/ubuntu/lsn/project_new/RT-DETR-main/DataSlice2Inference/models/rtdetr_torchscript_cuda.pt",
        help="TorchScript 模型路径（.pt）",
    )
    parser.add_argument(
        "--patch-dir",
        type=str,
        required=True,
        help="patch图像所在目录",
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
        default="auto",
        help="推理设备：auto / cpu / cuda / cuda:0 等，默认 auto（优先 CUDA）",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="置信度阈值（默认 0.5）",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.png",
        help="图像文件匹配模式（默认 *.png）",
    )
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="不保存可视化图片（只保存JSON结果）",
    )
    parser.add_argument(
        "--global-image-url",
        type=str,
        default="",
        help="大图存储路径或URL（会写入JSON info字段）",
    )
    return parser.parse_args()


def resolve_device(device_str: str) -> torch.device:
    """自动检测并返回设备"""
    device_str = device_str.lower()
    if device_str == "auto":
        if torch.cuda.is_available():
            print("⚡ 检测到可用 GPU，使用 CUDA 进行推理")
            return torch.device("cuda")
        print("⚠️ 未检测到 GPU，使用 CPU 推理")
        return torch.device("cpu")
    return torch.device(device_str)


def load_torchscript_model(model_path: str, device: torch.device) -> torch.jit.ScriptModule:
    """加载TorchScript模型"""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    print(f"=== 加载 TorchScript 模型 ===")
    print(f"模型路径: {model_path}")
    print(f"设备: {device}")

    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    print("✓ 模型加载完成\n")
    return model


def load_patch_coordinates(csv_path: str) -> Dict[str, Tuple[int, int]]:
    """从CSV文件加载patch坐标信息
    
    CSV格式：filename, x_start, y_start, x_end, y_end
    返回：{filename: (x_start, y_start)}
    """
    coordinates = {}
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"坐标CSV文件不存在: {csv_path}")
    
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row['filename']
            x_start = int(row['x_start'])
            y_start = int(row['y_start'])
            coordinates[filename] = (x_start, y_start)
    
    print(f"✓ 已加载 {len(coordinates)} 个patch的坐标信息")
    return coordinates


def convert_to_global_coordinates(
    boxes: torch.Tensor,
    patch_offset: Tuple[int, int]
) -> torch.Tensor:
    """将patch内的检测框坐标转换为全图坐标
    
    Args:
        boxes: 检测框坐标 (N, 4)，格式为 (x1, y1, x2, y2)，相对于patch
        patch_offset: patch在全图中的偏移量 (x_start, y_start)
    
    Returns:
        全图坐标 (N, 4)，格式为 (x1, y1, x2, y2)
    """
    x_offset, y_offset = patch_offset
    offset_tensor = torch.tensor(
        [x_offset, y_offset, x_offset, y_offset],
        dtype=boxes.dtype,
        device=boxes.device
    )
    return boxes + offset_tensor


def prepare_image(image_path: str, device: torch.device):
    """加载并预处理图像"""
    image_pil = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image_pil.size

    transforms = T.Compose([
        T.Resize((640, 640)),
        T.ToTensor(),
    ])
    image_tensor = transforms(image_pil).unsqueeze(0).to(device)
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


def predict_single_patch(
    model: torch.jit.ScriptModule,
    patch_path: str,
    patch_offset: Tuple[int, int],
    device: torch.device,
    threshold: float,
    save_vis: bool = True,
    output_dir: str = None
) -> dict:
    """预测单个patch，返回检测结果（包含patch坐标和全图坐标）"""
    try:
        # 加载和预处理图像
        image_pil, image_tensor, orig_sizes = prepare_image(patch_path, device)

        # 推理
        with torch.no_grad():
            outputs = model(image_tensor, orig_sizes)

        # 后处理
        labels, boxes_patch, scores = postprocess_outputs(outputs, threshold)

        # 转换为全图坐标
        boxes_global = convert_to_global_coordinates(boxes_patch, patch_offset)

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

    # 加载坐标信息
    coordinates = load_patch_coordinates(str(csv_path))

    # 加载模型
    device = resolve_device(args.device)
    model = load_torchscript_model(args.model, device)

    # 查找所有patch图像
    patch_files = sorted(list(patch_dir.glob(args.pattern)))
    if len(patch_files) == 0:
        print(f"❌ 在 {patch_dir} 中未找到匹配 {args.pattern} 的图像文件")
        return

    print(f"\n{'='*70}")
    print(f"找到 {len(patch_files)} 个patch文件，开始批量推理...")
    print(f"{'='*70}\n")

    # 批量处理
    start_time = time.time()
    all_results = []
    success_count = 0
    fail_count = 0
    total_detections = 0

    for patch_path in tqdm(patch_files, desc="推理进度", unit="张"):
        patch_name = patch_path.name
        
        # 获取patch坐标
        if patch_name not in coordinates:
            print(f"⚠️ 警告：{patch_name} 的坐标信息未找到，跳过")
            fail_count += 1
            continue

        patch_offset = coordinates[patch_name]
        
        # 预测
        result = predict_single_patch(
            model=model,
            patch_path=str(patch_path),
            patch_offset=patch_offset,
            device=device,
            threshold=args.threshold,
            save_vis=not args.no_visualization,
            output_dir=str(output_dir)
        )

        if result:
            all_results.append(result)
            total_detections += result["detection_count"]
            success_count += 1
        else:
            fail_count += 1

    # 保存JSON结果
    json_path = output_dir / "annotations.json"
    json_data = {
        "info": {
            "description": "RT-DETR v2 TorchScript 批量推理结果",
            "version": "1.0",
            "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model_path": args.model,
            "patch_dir": str(patch_dir),
            "threshold": args.threshold,
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

