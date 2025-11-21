#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 RT-DETR v2 的 TorchScript 模型，对单张图片进行预测并保存结果。

支持模型：
- exported_models/rtdetr_torchscript.pt
- exported_models/rtdetr_torchscript_cuda.pt

功能：
- 自动选择设备（CPU / CUDA）
- 对输入图片进行预处理（Resize 640x640）
- 调用 TorchScript 模型：labels, boxes, scores = model(images, orig_sizes)
- 过滤低置信度目标，绘制预测框，保存预测图片
- 可选：保存 JSON 结果（包含类别、坐标、置信度）

示例：
cd /home/ubuntu/lsn/project_new/RT-DETR-main

python ZZ/torchscript_predict_single.py \
  --model rtdetrv2_pytorch/exported_models/rtdetr_torchscript.pt \
  --image ZZ/11.png \
  --output-image ZZ/pred_11_torchscript.png \
  --output-json ZZ/annotations_torchscript.json \
  --threshold 0.5 \
  --device cpu
"""

import argparse
import json
import os
from pathlib import Path

import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw, ImageFont


def parse_args():
    parser = argparse.ArgumentParser(description="使用 TorchScript RT-DETR v2 模型对单张图片进行预测")
    parser.add_argument(
        "--model",
        type=str,
        default=str(Path(__file__).parent / "exported_models" / "rtdetr_torchscript.pt"),
        help="TorchScript 模型路径（.pt），默认使用 exported_models/rtdetr_torchscript.pt",
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="待预测图片路径",
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
    device_str = device_str.lower()
    if device_str == "auto":
        if torch.cuda.is_available():
            print("⚡ 检测到可用 GPU，使用 CUDA 进行推理")
            return torch.device("cuda")
        print("⚠️ 未检测到 GPU，使用 CPU 推理")
        return torch.device("cpu")
    return torch.device(device_str)


def load_torchscript_model(model_path: str, device: torch.device) -> torch.jit.ScriptModule:
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    print(f"=== 加载 TorchScript 模型 ===")
    print(f"模型路径: {model_path}")
    print(f"设备: {device}")

    # 使用 map_location 确保模型加载到目标设备
    model = torch.jit.load(model_path, map_location=device)
    model.eval()
    print("✓ 模型加载完成\n")
    return model


def prepare_image(image_path: str, device: torch.device):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    image_pil = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image_pil.size

    transforms = T.Compose(
        [
            T.Resize((640, 640)),
            T.ToTensor(),
        ]
    )
    image_tensor = transforms(image_pil).unsqueeze(0).to(device)
    orig_sizes = torch.tensor([[orig_w, orig_h]], dtype=torch.int64, device=device)

    return image_pil, image_tensor, orig_sizes


def postprocess_outputs(outputs, threshold: float):
    """
    TorchScript DeployModel 的输出应该是 (labels, boxes, scores)
    labels: [B, N]
    boxes:  [B, N, 4]  (x1, y1, x2, y2) 像素坐标，已映射回原图尺寸
    scores: [B, N]
    """
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

    valid = scores > threshold
    labels = labels[valid]
    boxes = boxes[valid]
    scores = scores[valid]

    return labels, boxes, scores


def draw_detections(image_pil, labels, boxes, scores, threshold: float):
    draw = ImageDraw.Draw(image_pil)

    # 尝试加载一个通用字体
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

    # 根据你的数据集自定义类别名称（这里沿用 inference.py 中的定义）
    class_names = [
        "AD",
        "BC",
        "EC",
        "L",
        "LC",
        "M",
        "NT",
        "SM",
        "SQ",
        "TC1",
        "TC2",
        "TC3",
    ]

    print(f"检测阈值: {threshold}")
    print(f"有效检测数: {len(labels)}")

    for idx, (lab, box, score) in enumerate(zip(labels, boxes, scores), 1):
        cls_id = int(lab.item())
        cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        x1, y1, x2, y2 = [float(x) for x in box.tolist()]

        # 绘制框
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

        text = f"{cls_name} {score:.2f}"

        # 兼容不同版本 Pillow 的文本尺寸计算
        try:
            # 新版本推荐使用 textbbox
            bbox = draw.textbbox((0, 0), text, font=font)
            text_w = bbox[2] - bbox[0]
            text_h = bbox[3] - bbox[1]
        except Exception:
            try:
                # 旧版本可以使用字体对象的 getsize
                text_w, text_h = font.getsize(text)
            except Exception:
                # 最后兜底估算
                text_w = len(text) * 8
                text_h = 16

        pad = 4
        draw.rectangle([x1, y1 - text_h - 2 * pad, x1 + text_w + 2 * pad, y1], fill="red")
        draw.text((x1 + pad, y1 - text_h - pad), text, fill="white", font=font)

        print(
            f"{idx:02d}. 类别: {cls_name} (ID={cls_id}), "
            f"置信度: {score:.3f}, "
            f"BBox: [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]"
        )

    return image_pil


def save_json_result(output_json, image_path, image_size, labels, boxes, scores):
    detections = []

    class_names = [
        "AD",
        "BC",
        "EC",
        "L",
        "LC",
        "M",
        "NT",
        "SM",
        "SQ",
        "TC1",
        "TC2",
        "TC3",
    ]

    for lab, box, score in zip(labels, boxes, scores):
        cls_id = int(lab.item())
        cls_name = class_names[cls_id] if cls_id < len(class_names) else f"class_{cls_id}"
        x1, y1, x2, y2 = [float(x) for x in box.tolist()]
        detections.append(
            {
                "class_id": cls_id,
                "class_name": cls_name,
                "bbox": [x1, y1, x2, y2],
                "score": float(score.item()),
            }
        )

    result = {
        "image_path": image_path,
        "image_name": os.path.basename(image_path),
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "detection_count": len(detections),
        "detections": detections,
    }

    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"📝 JSON 结果已保存到: {output_json}")


def main():
    args = parse_args()
    device = resolve_device(args.device)

    model = load_torchscript_model(args.model, device)
    image_pil, image_tensor, orig_sizes = prepare_image(args.image, device)

    with torch.no_grad():
        outputs = model(image_tensor, orig_sizes)

    labels, boxes, scores = postprocess_outputs(outputs, args.threshold)

    # 生成输出图片路径
    if args.output_image is None:
        img_dir = os.path.dirname(args.image)
        img_name = os.path.basename(args.image)
        out_img = os.path.join(img_dir, f"pred_{img_name}")
    else:
        out_img = args.output_image

    # 绘制并保存
    vis_image = draw_detections(image_pil.copy(), labels, boxes, scores, args.threshold)
    Path(out_img).parent.mkdir(parents=True, exist_ok=True)
    vis_image.save(out_img)
    print(f"📸 预测可视化图片已保存到: {out_img}")

    # 可选：保存 JSON
    if args.output_json:
        save_json_result(
            args.output_json,
            image_path=args.image,
            image_size=image_pil.size,
            labels=labels,
            boxes=boxes,
            scores=scores,
        )

    print("✅ 单张图片 TorchScript 推理完成")


if __name__ == "__main__":
    main()


