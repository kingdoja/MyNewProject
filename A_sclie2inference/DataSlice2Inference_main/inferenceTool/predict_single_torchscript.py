#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 RT-DETR v2 的 TorchScript 模型对单张patch图像进行预测，支持将patch坐标转换为全图坐标。

功能：
1. 加载TorchScript模型进行推理
2. 自动读取patch坐标信息（从CSV文件）
3. 将检测框坐标从patch坐标转换为全图坐标
4. 保存预测结果（可视化图片和JSON标注）

坐标转换逻辑：
- patch在全图中的左上角坐标：从CSV读取 (x_start, y_start)
- 模型输出的检测框坐标：相对于patch的 (x1, y1, x2, y2)
- 转换为全图坐标：(x_start + x1, y_start + y1, x_start + x2, y_start + y2)

示例：
python predict_single_torchscript.py \
  --model ../models/rtdetr_torchscript_cuda.pt \
  --patch ../../DataPatchesKeep/Patches5/patch_0.png \
  --coordinates-csv ../../DataPatches/Patches5/patch_coordinates.csv \
  --output-image pred_patch_0.png \
  --output-json pred_patch_0.json \
  --threshold 0.5
"""

import argparse
import json
import os
import csv
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont


# 类别名称定义
CLASS_NAMES = [
    "AD", "BC", "EC", "L", "LC", "M", "NT", "SM", "SQ", "TC1", "TC2", "TC3"
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用 TorchScript RT-DETR v2 模型对单张patch进行预测（支持坐标转换）"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="/home/ubuntu/lsn/project_new/RT-DETR-main/DataSlice2Inference/models/rtdetr_torchscript_cuda.pt",
        help="TorchScript 模型路径（.pt）",
    )
    parser.add_argument(
        "--patch",
        type=str,
        required=True,
        help="待预测的patch图像路径",
    )
    parser.add_argument(
        "--coordinates-csv",
        type=str,
        required=True,
        help="patch坐标CSV文件路径",
    )
    parser.add_argument(
        "--output-image",
        type=str,
        default=None,
        help="预测结果图片保存路径（默认：与原图同目录，文件名前加 pred_）",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="可选：保存 JSON 结果的路径（默认不保存）",
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
        help="置信度阈值（默认 0.5），低于该值的检测会被过滤",
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


def load_patch_offset(csv_path: str, patch_filename: str) -> tuple[int, int]:
    """从CSV文件加载指定patch的坐标偏移量
    
    Args:
        csv_path: CSV文件路径
        patch_filename: patch文件名（如 patch_0.png）
    
    Returns:
        (x_start, y_start) patch在全图中的左上角坐标
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"坐标CSV文件不存在: {csv_path}")
    
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['filename'] == patch_filename:
                x_start = int(row['x_start'])
                y_start = int(row['y_start'])
                return (x_start, y_start)
    
    raise ValueError(f"在CSV文件中未找到 {patch_filename} 的坐标信息")


def convert_to_global_coordinates(
    boxes: torch.Tensor,
    patch_offset: tuple[int, int]
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
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

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

    print(f"检测阈值: {threshold}")
    print(f"有效检测数: {len(labels)}")

    for idx, (lab, box, score) in enumerate(zip(labels, boxes, scores), 1):
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
            try:
                text_w, text_h = font.getsize(text)
            except Exception:
                text_w = len(text) * 8
                text_h = 16

        pad = 4
        draw.rectangle([x1, y1 - text_h - 2 * pad, x1 + text_w + 2 * pad, y1], fill="red")
        draw.text((x1 + pad, y1 - text_h - pad), text, fill="white", font=font)

        print(
            f"{idx:02d}. 类别: {cls_name} (ID={cls_id}), "
            f"置信度: {score:.3f}, "
            f"BBox (patch坐标): [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]"
        )

    return image_pil


def save_json_result(
    output_json: str,
    patch_path: str,
    patch_offset: tuple[int, int],
    patch_size: tuple[int, int],
    labels: torch.Tensor,
    boxes_patch: torch.Tensor,
    boxes_global: torch.Tensor,
    scores: torch.Tensor
):
    """保存JSON结果"""
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

    result = {
        "patch_path": patch_path,
        "patch_name": os.path.basename(patch_path),
        "patch_offset": {"x_start": patch_offset[0], "y_start": patch_offset[1]},
        "patch_size": {"width": patch_size[0], "height": patch_size[1]},
        "detection_count": len(detections_patch),
        "detections_patch": detections_patch,  # patch坐标
        "detections_global": detections_global,  # 全图坐标
    }

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"📝 JSON 结果已保存到: {output_json}")


def main():
    args = parse_args()
    device = resolve_device(args.device)

    # 加载模型
    model = load_torchscript_model(args.model, device)

    # 获取patch文件名
    patch_filename = os.path.basename(args.patch)
    
    # 加载patch坐标偏移量
    print(f"=== 加载patch坐标信息 ===")
    print(f"CSV文件: {args.coordinates_csv}")
    print(f"Patch文件名: {patch_filename}")
    try:
        patch_offset = load_patch_offset(args.coordinates_csv, patch_filename)
        print(f"✓ Patch在全图中的偏移量: ({patch_offset[0]}, {patch_offset[1]})\n")
    except (FileNotFoundError, ValueError) as e:
        print(f"❌ 错误: {e}")
        return

    # 加载和预处理图像
    print(f"=== 加载图像 ===")
    print(f"图像路径: {args.patch}")
    image_pil, image_tensor, orig_sizes = prepare_image(args.patch, device)
    print(f"✓ 图像尺寸: {image_pil.size[0]} x {image_pil.size[1]}\n")

    # 推理
    print(f"=== 模型推理 ===")
    with torch.no_grad():
        outputs = model(image_tensor, orig_sizes)

    # 后处理
    labels, boxes_patch, scores = postprocess_outputs(outputs, args.threshold)

    # 转换为全图坐标
    boxes_global = convert_to_global_coordinates(boxes_patch, patch_offset)

    # 打印检测结果
    print(f"\n=== 检测结果（patch坐标）===")
    for idx, (lab, box, score) in enumerate(zip(labels, boxes_patch, scores), 1):
        cls_id = int(lab.item())
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
        x1, y1, x2, y2 = [float(x) for x in box.tolist()]
        print(
            f"{idx:02d}. 类别: {cls_name} (ID={cls_id}), "
            f"置信度: {score:.3f}, "
            f"BBox: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]"
        )

    print(f"\n=== 检测结果（全图坐标）===")
    for idx, (lab, box, score) in enumerate(zip(labels, boxes_global, scores), 1):
        cls_id = int(lab.item())
        cls_name = CLASS_NAMES[cls_id] if cls_id < len(CLASS_NAMES) else f"class_{cls_id}"
        x1, y1, x2, y2 = [float(x) for x in box.tolist()]
        print(
            f"{idx:02d}. 类别: {cls_name} (ID={cls_id}), "
            f"置信度: {score:.3f}, "
            f"BBox: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]"
        )

    # 生成输出图片路径
    if args.output_image is None:
        img_dir = os.path.dirname(args.patch)
        img_name = os.path.basename(args.patch)
        out_img = os.path.join(img_dir, f"pred_{img_name}")
    else:
        out_img = args.output_image

    # 绘制并保存（使用patch坐标绘制，因为图像本身是patch）
    print(f"\n=== 保存可视化结果 ===")
    vis_image = draw_detections(image_pil.copy(), labels, boxes_patch, scores, args.threshold)
    Path(out_img).parent.mkdir(parents=True, exist_ok=True)
    vis_image.save(out_img)
    print(f"📸 预测可视化图片已保存到: {out_img}")

    # 可选：保存 JSON
    if args.output_json:
        save_json_result(
            args.output_json,
            patch_path=args.patch,
            patch_offset=patch_offset,
            patch_size=image_pil.size,
            labels=labels,
            boxes_patch=boxes_patch,
            boxes_global=boxes_global,
            scores=scores
        )

    print("\n✅ 单张patch TorchScript 推理完成")


if __name__ == "__main__":
    main()

