#!/usr/bin/env python3
"""
利用 Albumentations 对痰液细胞 COCO 数据集进行检测增强。
- 默认会克隆 `split_dataset` 到 `split_dataset_aug`
- 支持按 split 自动计算目标倍数（默认为 train 适度放大）
- 同时允许设置单图最少的固定增强数量，保证增广充足但不过分激进

使用示例：
 cd /home/ubuntu/lsn/project_new/RT-DETR-main
 python DATA/SputumCell/4data_augmentation.py \
          --source-root DATA/SputumCell/split_dataset \
     --target-root DATA/SputumCell/split_dataset_aug \
     --train-target-multiplier 8.5

cd /home/ubuntu/lsn/project_new/RT-DETR-main
 python DATA/SputumCell/4data_augmentation.py \
     --source-root DATA/SputumCell/split45 \
     --target-root DATA/SputumCell/split45_aug \
     --train-target-multiplier 8.5
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from copy import deepcopy
from math import ceil
from pathlib import Path
from typing import Dict, List, Set, Tuple

import albumentations as A
import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="为痰液细胞 COCO 数据生成增强样本",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("DATA/SputumCell/split4"),
        help="原始 COCO 数据根目录（包含 train/val/test）",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=Path("DATA/SputumCell/split4_aug"),
        help="增强后数据输出目录",
    )
    # 只对 train 做主动增强，val/test 默认不增强（保持评估干净）
    parser.add_argument("--train-factor", type=int, default=2, help="train 每张图最少新增样本数")
    parser.add_argument("--val-factor", type=int, default=0, help="val 每张图最少新增样本数")
    parser.add_argument("--test-factor", type=int, default=0, help="test 每张图最少新增样本数")
    parser.add_argument(
        "--train-target-multiplier",
        type=float,
        default=3.0,
        help="train 目标整体增广倍数（含原图），至少满足该倍数",
    )
    parser.add_argument(
        "--val-target-multiplier",
        type=float,
        default=1.0,
        help="val 目标整体增广倍数（含原图），默认只复制不增强",
    )
    parser.add_argument(
        "--test-target-multiplier",
        type=float,
        default=1.0,
        help="test 目标整体增广倍数（含原图），默认只复制不增强",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=25.0,
        help="过滤过小框的面积阈值 (像素^2)",
    )
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=0.15,
        help="Albumentations 框可见度阈值",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=8,
        help="为单张图像寻找有效增强的最大尝试次数倍数",
    )
    parser.add_argument("--seed", type=int, default=42, help="随机数种子")
    parser.add_argument(
        "--category-filter",
        type=int,
        nargs="+",
        default=None,
        help="只增强包含这些类别 ID 的图像，默认处理全部图像",
    )
    parser.add_argument(
        "--plan-file",
        type=Path,
        default=None,
        help="class_balance_plan.json，启用后自动根据 suggested_multiplier 定向增强",
    )
    return parser.parse_args()


def build_transforms(min_visibility: float) -> List[A.BasicTransform]:
    bbox_params = A.BboxParams(
        format="coco",
        label_fields=["category_ids", "iscrowd_flags", "ignore_flags"],
        min_visibility=min_visibility,
    )
    # 针对显微痰液细胞图像设计多样但不过分激进的组合：
    # - 组合 1/2：几何扰动（已移除镜像翻转以避免细胞对称畸变）
    # - 组合 3/4：染色、曝光、对比度，模拟不同涂片/染色批次
    # - 组合 5：模糊/噪声 + 锐化，模拟不同对焦与成像质量
    # - 组合 6：局部遮挡，模拟杂质、气泡、涂片不均等伪影
    return [
        # 组合 1：水平翻转
        A.Compose(
            [
                A.HorizontalFlip(p=1.0),  # 纯水平翻转
            ],
            bbox_params=bbox_params,
        ),
        # 组合 2：垂直翻转
        A.Compose(
            [
                A.VerticalFlip(p=1.0),  # 纯垂直翻转
            ],
            bbox_params=bbox_params,
        ),
        # 组合 3：90度旋转
        A.Compose(
            [
                A.RandomRotate90(p=1.0),  # 90度旋转（包括90, 180, 270度）
            ],
            bbox_params=bbox_params,
        ),
        # # 组合 4：180度旋转
        # A.Compose(
        #     [
        #         A.Rotate(limit=180, p=1.0),  # 180度旋转
        #     ],
        #     bbox_params=bbox_params,
        # ),
        # 组合 5：亮度/对比度/色调微调 + CLAHE（染色差异）
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
        # 组合 6：Gamma + 柔和对比增强（不同曝光/显微镜设置）
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
        # 组合 7：轻微模糊/噪声 + 轻量锐化（不同对焦与传感器噪声）
        A.Compose(
            [
                A.OneOf(
                    [
                        A.GaussianBlur(blur_limit=3, p=1.0), 
                        A.MedianBlur(blur_limit=3, p=1.0),
                        A.MotionBlur(blur_limit=3, p=1.0),
                    ],
                    p=0.4, # 模糊
                ), 
                A.RandomBrightnessContrast( # 亮度/对比度/色调微调
                    brightness_limit=(-0.08, 0.15),
                    contrast_limit=(-0.10, 0.18),
                    p=0.6,
                ),
                A.HueSaturationValue( # 色调微调
                    hue_shift_limit=3,
                    sat_shift_limit=10,
                    val_shift_limit=14,
                    p=0.3,
                ),
                A.CLAHE(clip_limit=2.5, tile_grid_size=(8, 8), p=0.2),
                # A.GaussNoise(var_limit=(0.5, 3.0), mean=0, p=0.25),
                A.UnsharpMask(p=0.1), # 锐化
            ],
            bbox_params=bbox_params,
        ),
       
    ]


def copy_original_images(src: Path, dst: Path) -> int:
    dst.mkdir(parents=True, exist_ok=True)
    copied = 0
    for image_path in src.glob("*"):
        if not image_path.is_file():
            continue
        target = dst / image_path.name
        if target.exists():
            continue
        shutil.copy2(image_path, target)
        copied += 1
    return copied


def load_annotation(annotation_path: Path) -> Dict:
    with open(annotation_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_annotation_index(annotations: List[Dict]) -> Dict[int, List[Dict]]:
    index: Dict[int, List[Dict]] = {}
    for ann in annotations:
        index.setdefault(ann["image_id"], []).append(ann)
    return index


def ensure_valid_boxes(
    bboxes: List[Tuple[float, float, float, float]],
    min_area: float,
) -> List[int]:
    keep_idx: List[int] = []
    for idx, bbox in enumerate(bboxes):
        _, _, w, h = bbox
        if w <= 0 or h <= 0:
            continue
        if w * h < min_area:
            continue
        keep_idx.append(idx)
    return keep_idx


def augment_split(
    split: str,
    factor: int,
    target_multiplier: float,
    args: argparse.Namespace,
    transforms: List[A.BasicTransform],
    category_filter: Set[int] | None = None,
    category_multiplier: Dict[int, int] | None = None,
) -> Dict[str, int]:
    stats = {"copied_images": 0, "augmented_images": 0, "augmented_boxes": 0}
    src_split = args.source_root / split
    if not src_split.exists():
        print(f"[WARN] 跳过 {split}: {src_split} 不存在")
        return stats

    src_images = src_split / "images"
    src_ann = src_split / "annotations" / f"instances_{split}.json"
    if not src_images.exists() or not src_ann.exists():
        print(f"[WARN] 跳过 {split}: 缺少 images 或 annotations")
        return stats

    dst_split = args.target_root / split
    dst_images = dst_split / "images"
    dst_annotations_dir = dst_split / "annotations"
    dst_annotations_dir.mkdir(parents=True, exist_ok=True)

    stats["copied_images"] = copy_original_images(src_images, dst_images)

    ann_data = load_annotation(src_ann)
    annotations_index = build_annotation_index(ann_data.get("annotations", []))
    original_images = list(ann_data.get("images", []))
    base_count = len(original_images)

    auto_factor = 0
    if target_multiplier > 1.0 and base_count > 0:
        desired_total = ceil(base_count * target_multiplier)
        extra_needed = max(desired_total - base_count, 0)
        if extra_needed > 0:
            auto_factor = ceil(extra_needed / base_count)

    factor = max(factor, auto_factor)

    augmented_ann = {
        "info": deepcopy(ann_data.get("info")),
        "licenses": deepcopy(ann_data.get("licenses")),
        "categories": deepcopy(ann_data.get("categories", [])),
        "images": list(ann_data.get("images", [])),
        "annotations": list(ann_data.get("annotations", [])),
    }

    next_image_id = (max((img["id"] for img in augmented_ann["images"]), default=0) + 1)
    next_ann_id = (max((ann["id"] for ann in augmented_ann["annotations"]), default=0) + 1)

    if factor <= 0:
        out_path = dst_annotations_dir / src_ann.name
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(augmented_ann, f, indent=2, ensure_ascii=False)
        print(f"[INFO] {split}: 只复制原始数据，无增强。")
        return stats

    print(
        f"[INFO] {split}: 原始 {base_count} 张，目标倍数 {target_multiplier}，"
        f"单图最少生成 {factor} 张增强样本。"
    )
    for image_meta in original_images:
        image_id = image_meta["id"]
        anns = annotations_index.get(image_id)
        if not anns:
            continue

        src_path = src_images / image_meta["file_name"]
        if not src_path.exists():
            print(f"[WARN] 找不到图像 {src_path}，跳过。")
            continue

        image = cv2.imread(str(src_path))
        if image is None:
            print(f"[WARN] 无法读取 {src_path}，跳过。")
            continue

        original_suffix = Path(image_meta["file_name"]).suffix or ".png"
        bboxes = [ann["bbox"] for ann in anns]
        category_ids = [ann["category_id"] for ann in anns]
        iscrowd_flags = [ann.get("iscrowd", 0) for ann in anns]
        ignore_flags = [ann.get("ignore", 0) for ann in anns]

        categories_in_image = set(category_ids)
        if category_filter and categories_in_image.isdisjoint(category_filter):
            continue

        image_factor = factor
        if category_multiplier:
            boosted = max((category_multiplier.get(cat, 1) for cat in categories_in_image), default=1)
            image_factor = max(image_factor, boosted)

        produced = 0
        attempts = 0
        max_attempts = max(1, image_factor * args.max_retries)

        while produced < image_factor and attempts < max_attempts:
            attempts += 1
            transform = random.choice(transforms)
            transformed = transform(
                image=image,
                bboxes=bboxes,
                category_ids=category_ids,
                iscrowd_flags=iscrowd_flags,
                ignore_flags=ignore_flags,
            )

            keep_idx = ensure_valid_boxes(transformed["bboxes"], args.min_area)
            if not keep_idx:
                continue

            new_filename = (
                f"{Path(image_meta['file_name']).stem}_aug{produced+1}_{attempts}{original_suffix}"
            )
            output_path = dst_images / new_filename
            cv2.imwrite(str(output_path), transformed["image"])

            new_image = {
                "id": next_image_id,
                "file_name": new_filename,
                "width": transformed["image"].shape[1],
                "height": transformed["image"].shape[0],
            }
            augmented_ann["images"].append(new_image)
            next_image_id += 1
            stats["augmented_images"] += 1

            for idx in keep_idx:
                bbox = transformed["bboxes"][idx]
                annotation = {
                    "id": next_ann_id,
                    "image_id": new_image["id"],
                    "category_id": transformed["category_ids"][idx],
                    "bbox": [float(x) for x in bbox],
                    "area": float(bbox[2] * bbox[3]),
                    "segmentation": [],
                    "iscrowd": transformed["iscrowd_flags"][idx],
                    "ignore": transformed["ignore_flags"][idx],
                }
                augmented_ann["annotations"].append(annotation)
                next_ann_id += 1
                stats["augmented_boxes"] += 1

            produced += 1

    out_path = dst_annotations_dir / src_ann.name
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(augmented_ann, f, indent=2, ensure_ascii=False)

    print(
        f"[INFO] {split}: "
        f"新增 {stats['augmented_images']} 张图像 / {stats['augmented_boxes']} 个标注，"
        f"写入 {out_path}"
    )
    return stats


def load_minority_plan(plan_path: Path) -> Tuple[str, Set[int], Dict[int, int]]:
    with plan_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    plan_split = data.get("split", "train")
    plan_entries = data.get("plan", {})
    filter_set: Set[int] = set()
    multiplier: Dict[int, int] = {}
    for cat_id, info in plan_entries.items():
        cat_int = int(cat_id)
        mult = int(info.get("suggested_multiplier", 1))
        multiplier[cat_int] = max(1, mult)
        if mult > 1:
            filter_set.add(cat_int)
    return plan_split, filter_set, multiplier


def main() -> None:
    args = parse_args()
    args.source_root = args.source_root.resolve()
    args.target_root = args.target_root.resolve()
    args.target_root.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    transforms = build_transforms(args.min_visibility)
    factors = {
        "train": args.train_factor,
        "val": args.val_factor,
        "test": args.test_factor,
    }
    target_multipliers = {
        "train": args.train_target_multiplier,
        "val": args.val_target_multiplier,
        "test": args.test_target_multiplier,
    }

    user_filter = set(args.category_filter or [])
    if not user_filter:
        user_filter = None

    plan_split = None
    plan_filter: Set[int] | None = None
    plan_multiplier: Dict[int, int] | None = None
    if args.plan_file:
        if not args.plan_file.exists():
            raise FileNotFoundError(f"plan 文件 {args.plan_file} 不存在")
        plan_split, plan_filter, plan_multiplier = load_minority_plan(args.plan_file)
        print(
            f"[INFO] 使用定向增强 plan: split={plan_split}, targets={sorted(plan_filter)}"
        )

    summary = {}
    for split in factors.keys():
        active_filter = user_filter
        active_multiplier = None
        if plan_multiplier and (plan_split is None or plan_split == split):
            active_multiplier = plan_multiplier
            if not active_filter:
                active_filter = plan_filter
        summary[split] = augment_split(
            split=split,
            factor=factors[split],
            target_multiplier=target_multipliers[split],
            args=args,
            transforms=transforms,
            category_filter=active_filter,
            category_multiplier=active_multiplier,
        )

    print("\n===== 数据增强完成 =====")
    for split, stats in summary.items():
        print(
            f"{split}: 复制 {stats['copied_images']} 张 | "
            f"增强 {stats['augmented_images']} 张 | "
            f"新增框 {stats['augmented_boxes']}"
        )


if __name__ == "__main__":
    main()

