#!/usr/bin/env python3
"""
可视化 `DATA/SputumCell/4data_augmentation.py` 中的图像增强组合。
一次性展示原图以及每种增强结果（含检测框），便于检查增强是否符合预期。

示例：
    python DATA/SputumCell/scripts/preview_augmentations.py \
        --image-dir DATA/SputumCell/patches4/images \
        --file-names patch_580.png patch_604.png \
        --annotation-json DATA/SputumCell/patches4/coco_format.json \
        --save-dir DATA/SputumCell/augmentation_previews
        

python DATA/SputumCell/scripts/preview_augmentations.py --image-dir DATA/SputumCell/patches4/images --file-names patch_1853.png --annotation-json DATA/SputumCell/patches4/coco_format.json --save-dir DATA/SputumCell/augmentation_previews --samples-per-transform 3
        
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import albumentations as A
import cv2
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="可视化痰液细胞检测数据的增强样式",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="原始图片所在目录",
    )
    parser.add_argument(
        "--file-names",
        nargs="+",
        required=True,
        help="需要预览的图片文件名（例如 patch_580.png）",
    )
    parser.add_argument(
        "--annotation-json",
        type=Path,
        default=None,
        help="COCO 标注文件路径（可选，不提供则不绘制检测框）",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=Path("DATA/SputumCell/augmentation_previews"),
        help="可视化结果输出目录",
    )
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=0.15,
        help="与增广脚本一致的可见度阈值",
    )
    parser.add_argument(
        "--samples-per-transform",
        type=int,
        default=3,
        help="每种增广重复采样次数（>=1），用于生成足够多的预览",
    )
    return parser.parse_args()


TransformSpec = Tuple[str, A.BasicTransform]


def build_transforms(min_visibility: float) -> List[TransformSpec]:
    """复制 4data_augmentation.py 中的增强组合，附加可读名称。"""
    bbox_params = A.BboxParams(
        format="coco",
        label_fields=["category_ids", "iscrowd_flags", "ignore_flags"],
        min_visibility=min_visibility,
    )
    return [
        (
            "geom_flip_light",
            A.Compose(
                [
                    A.ShiftScaleRotate(
                        shift_limit=0.10,
                        scale_limit=0.08,
                        rotate_limit=15,
                        border_mode=cv2.BORDER_REFLECT101,
                        p=0.9,
                    ),
                    A.HorizontalFlip(p=0.6),
                    A.VerticalFlip(p=0.4),
                ],
                bbox_params=bbox_params,
            ),
        ),
        (
            "geom_rotate90",
            A.Compose(
                [
                    A.RandomRotate90(p=0.5),
                    A.ShiftScaleRotate(
                        shift_limit=0.15,
                        scale_limit=0.10,
                        rotate_limit=20,
                        border_mode=cv2.BORDER_REFLECT101,
                        p=0.7,
                    ),
                ],
                bbox_params=bbox_params,
            ),
        ),
        (
            "color_clahe",
            A.Compose(
                [
                    A.RandomBrightnessContrast(
                        brightness_limit=(-0.08, 0.15),
                        contrast_limit=(-0.10, 0.18),
                        p=0.8,
                    ),
                    A.HueSaturationValue(
                        hue_shift_limit=3,
                        sat_shift_limit=10,
                        val_shift_limit=14,
                        p=0.5,
                    ),
                    A.CLAHE(clip_limit=2.5, tile_grid_size=(8, 8), p=0.4),
                ],
                bbox_params=bbox_params,
            ),
        ),
        (
            "gamma_soft",
            A.Compose(
                [
                    A.RandomGamma(gamma_limit=(90, 115), p=0.5),
                    A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.4),
                    A.RandomBrightnessContrast(
                        brightness_limit=(-0.04, 0.10),
                        contrast_limit=(-0.06, 0.12),
                        p=0.6,
                    ),
                ],
                bbox_params=bbox_params,
            ),
        ),
        (
            "blur_noise_sharp",
            A.Compose(
                [
                    A.OneOf(
                        [
                            A.GaussianBlur(blur_limit=3, p=1.0),
                            A.MedianBlur(blur_limit=3, p=1.0),
                            A.MotionBlur(blur_limit=3, p=1.0),
                        ],
                        p=0.4,
                    ),
                    A.RandomBrightnessContrast(
                    brightness_limit=(-0.08, 0.15),
                    contrast_limit=(-0.10, 0.18),
                    p=0.6,
                     ),
                    A.HueSaturationValue(
                        hue_shift_limit=3,
                        sat_shift_limit=10,
                        val_shift_limit=14,
                        p=0.3,
                    ),
                    A.CLAHE(clip_limit=2.5, tile_grid_size=(8, 8), p=0.2),
                    # A.GaussNoise(var_limit=(0.5, 3.0), mean=0, p=0.25),
                    A.UnsharpMask(p=0.1),
                ],
                bbox_params=bbox_params,
            ),
        ),
    ]


def load_annotation_map(annotation_json: Path) -> Dict[str, Dict[str, List]]:
    """将 COCO 标注转换为按文件名索引的结构。"""
    with annotation_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    image_id_to_name = {}
    for img in data.get("images", []):
        if isinstance(img, dict) and "id" in img and "file_name" in img:
            image_id_to_name[img["id"]] = img["file_name"]

    lookup: Dict[str, Dict[str, List]] = {}
    for ann in data.get("annotations", []):
        img_name = image_id_to_name.get(ann.get("image_id"))
        if not img_name:
            continue
        entry = lookup.setdefault(
            img_name,
            {"bboxes": [], "category_ids": [], "iscrowd": [], "ignore": []},
        )
        entry["bboxes"].append(ann.get("bbox", []))
        entry["category_ids"].append(ann.get("category_id", 0))
        entry["iscrowd"].append(ann.get("iscrowd", 0))
        entry["ignore"].append(ann.get("ignore", 0))
    return lookup


def draw_boxes(image_bgr, bboxes: List[List[float]], color=(0, 255, 0)):
    canvas = image_bgr.copy()
    for bbox in bboxes:
        if not bbox or len(bbox) < 4:
            continue
        x, y, w, h = bbox
        pt1 = (int(round(x)), int(round(y)))
        pt2 = (int(round(x + w)), int(round(y + h)))
        cv2.rectangle(canvas, pt1, pt2, color, thickness=2)
    return canvas


def visualize_single_image(
    image_path: Path,
    ann_lookup: Dict[str, Dict[str, List]],
    transforms: List[TransformSpec],
    save_dir: Path,
    samples_per_transform: int,
):
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[WARN] 无法读取图像 {image_path}")
        return

    ann = ann_lookup.get(image_path.name, {})
    bboxes = ann.get("bboxes", [])
    category_ids = ann.get("category_ids", [0] * len(bboxes))
    iscrowd = ann.get("iscrowd", [0] * len(bboxes))
    ignore = ann.get("ignore", [0] * len(bboxes))

    panels = [("original", draw_boxes(image, bboxes))]

    samples = max(1, samples_per_transform)
    for name, transform in transforms:
        for idx in range(samples):
            transformed = transform(
                image=image,
                bboxes=bboxes,
                category_ids=category_ids,
                iscrowd_flags=iscrowd,
                ignore_flags=ignore,
            )
            aug_image = draw_boxes(transformed["image"], transformed["bboxes"])
            panels.append((f"{name}_{idx+1}", aug_image))

    total = len(panels)
    cols = min(4, total)
    rows = math.ceil(total / cols)
    plt.figure(figsize=(4 * cols, 4 * rows))
    for idx, (title, img_bgr) in enumerate(panels, start=1):
        plt.subplot(rows, cols, idx)
        plt.imshow(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        plt.title(title)
        plt.axis("off")

    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"{image_path.stem}_augmentation_preview.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[INFO] 保存预览: {out_path}")


def main():
    args = parse_args()
    transforms = build_transforms(args.min_visibility)

    ann_lookup: Dict[str, Dict[str, List]] = {}
    if args.annotation_json:
        ann_lookup = load_annotation_map(args.annotation_json)
    else:
        print("[INFO] 未提供 annotation-json，仅展示图像整体效果（无框）。")

    for file_name in args.file_names:
        image_path = args.image_dir / file_name
        visualize_single_image(
            image_path,
            ann_lookup,
            transforms,
            args.save_dir,
            samples_per_transform=args.samples_per_transform,
        )


if __name__ == "__main__":
    main()


