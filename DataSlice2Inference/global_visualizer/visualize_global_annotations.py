#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
在大图上可视化批量推理得到的全图坐标检测框。

示例：
python visualize_global_annotations.py \
  --annotations /home/ubuntu/lsn/project_new/RT-DETR-main/DataPatchesInference/cj_20251119_154443/annotations.json \
  --output /home/ubuntu/lsn/project_new/RT-DETR-main/DataPatchesInference/cj_20251119_154443/cj_global_pred.png
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from PIL import Image, ImageDraw, ImageFont, ImageFile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="根据 annotations.json 把检测框绘制到原始大图上"
    )
    parser.add_argument(
        "--annotations",
        required=True,
        help="predict_batch_torchscript.py 生成的 annotations.json 路径",
    )
    parser.add_argument(
        "--image",
        help="大图路径，若不提供则尝试读取 JSON 中 info.global_image_url",
    )
    parser.add_argument(
        "--output",
        help="可视化结果输出路径，默认与 JSON 同目录下生成 *_global.png",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.0,
        help="只绘制置信度大于该值的框（默认不过滤）",
    )
    parser.add_argument(
        "--class-filter",
        nargs="*",
        help="只显示指定类别名称，多个名称用空格分隔（默认显示全部）",
    )
    parser.add_argument(
        "--max-draw",
        type=int,
        default=0,
        help="最多绘制前 N 个框（按遍历顺序），0 表示绘制全部",
    )
    parser.add_argument(
        "--hide-label",
        action="store_true",
        help="只绘制框，不渲染文字标签",
    )
    return parser.parse_args()


def load_annotations(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"annotations.json 不存在: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def resolve_image_path(args: argparse.Namespace, info: Dict) -> Path:
    if args.image:
        img_path = Path(args.image).expanduser()
    else:
        global_path = info.get("global_image_url") or info.get("global_image_path")
        if not global_path:
            raise ValueError("JSON 中缺少 info.global_image_url，请使用 --image 指定大图路径")
        img_path = Path(global_path)
    if not img_path.exists():
        raise FileNotFoundError(f"大图文件不存在: {img_path}")
    return img_path


def prepare_output_path(args: argparse.Namespace, annotations_path: Path, image_path: Path) -> Path:
    if args.output:
        out_path = Path(args.output)
    else:
        base_dir = annotations_path.parent
        default_name = f"{image_path.stem}_global_pred.png"
        out_path = base_dir / default_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    return out_path


def load_font(size: int = 22) -> ImageFont.ImageFont:
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for font_path in candidates:
        if os.path.exists(font_path):
            try:
                return ImageFont.truetype(font_path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def collect_global_detections(
    annotations: Dict,
    score_threshold: float,
    class_filter: Optional[List[str]],
) -> List[Dict]:
    results = []
    for patch_result in annotations.get("results", []):
        for det in patch_result.get("detections_global", []):
            if det.get("score", 0.0) < score_threshold:
                continue
            if class_filter and det.get("class_name") not in class_filter:
                continue
            det_copy = det.copy()
            det_copy["patch_name"] = patch_result.get("patch_name")
            results.append(det_copy)
    return results


def color_for_class(cls_name: str, cache: Dict[str, Tuple[int, int, int]]) -> Tuple[int, int, int]:
    if cls_name not in cache:
        random.seed(hash(cls_name) & 0xFFFFFFFF)
        cache[cls_name] = tuple(random.randint(50, 230) for _ in range(3))
    return cache[cls_name]


def draw_detections(
    image: Image.Image,
    detections: List[Dict],
    output_path: Path,
    hide_label: bool = False,
):
    draw = ImageDraw.Draw(image)
    font = load_font(20)
    color_cache: Dict[str, Tuple[int, int, int]] = {}

    for det in detections:
        bbox = det.get("bbox", [])
        if len(bbox) != 4:
            continue
        x1, y1, x2, y2 = map(float, bbox)
        cls_name = det.get("class_name", "unknown")
        score = det.get("score", 0.0)
        color = color_for_class(cls_name, color_cache)

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        if hide_label:
            continue

        label = f"{cls_name} {score:.2f}"
        try:
            text_w, text_h = draw.textbbox((0, 0), label, font=font)[2:]
        except Exception:
            text_w = len(label) * 8
            text_h = 16
        pad = 4
        text_bg = [x1, y1 - text_h - 2 * pad, x1 + text_w + 2 * pad, y1]
        draw.rectangle(text_bg, fill=color)
        draw.text((x1 + pad, y1 - text_h - pad), label, fill="white", font=font)

    image.save(output_path)


def main():
    args = parse_args()
    annotations_path = Path(args.annotations).expanduser()
    annotations = load_annotations(annotations_path)

    image_path = resolve_image_path(args, annotations.get("info", {}))
    output_path = prepare_output_path(args, annotations_path, image_path)

    # 允许处理超大图像
    Image.MAX_IMAGE_PIXELS = None
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    detections = collect_global_detections(
        annotations,
        score_threshold=args.score_threshold,
        class_filter=args.class_filter,
    )
    if args.max_draw > 0:
        detections = detections[: args.max_draw]

    if not detections:
        raise RuntimeError("未找到满足条件的检测结果，无法绘制")

    print(f"✳️ 载入大图: {image_path}")
    image = Image.open(image_path).convert("RGB")
    print(f"✳️ 读取检测框: {len(detections)} 个（过滤阈值 {args.score_threshold}）")

    draw_detections(image, detections, output_path, hide_label=args.hide_label)

    print(f"✅ 已生成可视化: {output_path}")
    print("提示：若需要输出矢量/JSON，可在此脚本基础上扩展。")


if __name__ == "__main__":
    main()


