#!/usr/bin/env python3
"""
利用 Albumentations 对痰液细胞 COCO 数据集进行“弱类定向增强”。
- 默认只增强包含弱类的图像（BC/M/TC1）
- 可按类别配置增强倍数，优先补齐少数类
- 增强完成后自动输出重采样与类别权重文件，可直接用于训练

使用示例：
 cd /home/ubuntu/lsn/project_new/RT-DETR-main
python DATA/SputumCell/4data_augmentation.py \
          --source-root DATA/SputumCell/split_dataset \
     --target-root DATA/SputumCell/split_dataset_aug \
     --weak-class-multipliers 1:3 5:6 9:3


#定向增强
 cd /home/ubuntu/lsn/project_new/RT-DETR-main
     python DATA/SputumCell/4data_augmentation.py \
  --source-root DATA/SputumCell/split_dataset \
  --target-root DATA/SputumCell/split_dataset_aug \
  --category-filter 1 5 9 \
  --weak-class-multipliers 1:3 5:6 9:3 \
  --train-factor 1   

  cd /home/ubuntu/lsn/project_new/RT-DETR-main
python DATA/SputumCell/4data_augmentation.py \
  --source-root DATA/SputumCell/split_dataset \
  --target-root DATA/SputumCell/split_dataset_aug \
  --category-filter 1 5 9 \
  --weak-class-multipliers 6:16 7:16 5:13 1:9 4:2 \
  --train-factor 1 \
  --train-target-multiplier 1.0 \
  --max-retries 12 \
  --min-area 20 \
  --min-visibility 0.12 \
  --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
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
        default=Path("DATA/SputumCell/split_dataset"),
        help="原始 COCO 数据根目录（包含 train/val/test）",
    )
    parser.add_argument(
        "--target-root",
        type=Path,
        default=Path("DATA/SputumCell/split_dataset_aug"),
        help="增强后数据输出目录",
    )
    # 定向增强默认只处理弱类图像，避免对头部类别继续放大。
    parser.add_argument("--train-factor", type=int, default=0, help="train 基础每图最少新增样本数")
    parser.add_argument("--val-factor", type=int, default=0, help="val 每张图最少新增样本数")
    parser.add_argument("--test-factor", type=int, default=0, help="test 每张图最少新增样本数")
    parser.add_argument(
        "--train-target-multiplier",
        type=float,
        default=1.0,
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
        default=[1, 5, 9],
        help="只增强包含这些类别 ID 的图像（默认 BC/M/TC1）",
    )
    parser.add_argument(
        "--weak-class-multipliers",
        type=str,
        nargs="+",
        default=["1:3", "5:6", "9:3"],
        help="弱类增强倍数，格式为 '类别ID:倍数'（示例: 1:3 5:6 9:3）",
    )
    parser.add_argument(
        "--plan-file",
        type=Path,
        default=None,
        help="class_balance_plan.json，启用后自动根据 suggested_multiplier 定向增强",
    )
    parser.add_argument(
        "--skip-balance-files",
        action="store_true",
        help="跳过生成 class_balance_class_weights.json 与 class_balance_image_weights.json",
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


def parse_class_multiplier_specs(specs: List[str] | None) -> Dict[int, int]:
    if not specs:
        return {}
    parsed: Dict[int, int] = {}
    for item in specs:
        if ":" not in item:
            raise ValueError(f"weak class multiplier 格式错误: {item}，应为 类别ID:倍数")
        cat_str, mul_str = item.split(":", 1)
        cat_id = int(cat_str.strip())
        multiplier = int(mul_str.strip())
        if multiplier < 1:
            raise ValueError(f"类别 {cat_id} 的倍数必须 >=1，当前为 {multiplier}")
        parsed[cat_id] = multiplier
    return parsed


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

    # 兼容两类计划文件格式：
    # 1) class_balance_plan: {"plan": {"1": {"suggested_multiplier": 3}, ...}}
    # 2) count_distribution plan: {"plan": [{"id": 1, "recommended_multiplier": 3}, ...]}
    if isinstance(plan_entries, dict):
        for cat_id, info in plan_entries.items():
            cat_int = int(cat_id)
            mult = int(info.get("suggested_multiplier", info.get("recommended_multiplier", 1)))
            multiplier[cat_int] = max(1, mult)
            if mult > 1:
                filter_set.add(cat_int)
    elif isinstance(plan_entries, list):
        for item in plan_entries:
            if not isinstance(item, dict):
                continue
            if "id" not in item:
                continue
            cat_int = int(item["id"])
            mult = int(item.get("recommended_multiplier", item.get("suggested_multiplier", 1)))
            multiplier[cat_int] = max(1, mult)
            if mult > 1:
                filter_set.add(cat_int)
    else:
        raise ValueError("plan 文件格式错误：'plan' 必须是 dict 或 list")
    return plan_split, filter_set, multiplier


def generate_balance_files(dataset_root: Path, split: str = "train", smoothing: float = 1.0) -> None:
    ann_path = dataset_root / split / "annotations" / f"instances_{split}.json"
    if not ann_path.exists():
        print(f"[WARN] 跳过平衡文件导出：{ann_path} 不存在")
        return

    with ann_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    category_names = {cat["id"]: cat["name"] for cat in data.get("categories", [])}

    class_counter: Counter = Counter()
    image_to_cats: Dict[int, List[int]] = defaultdict(list)
    for ann in annotations:
        cat_id = ann["category_id"]
        image_id = ann["image_id"]
        class_counter[cat_id] += 1
        image_to_cats[image_id].append(cat_id)

    total = sum(class_counter.values())
    if total == 0:
        print("[WARN] 标注为空，无法生成平衡文件。")
        return

    raw_weights = {
        cat_id: total / (count + smoothing)
        for cat_id, count in class_counter.items()
    }
    max_weight = max(raw_weights.values())
    class_weights = {cat_id: value / max_weight for cat_id, value in raw_weights.items()}

    image_weights: Dict[str, Dict[str, float]] = {}
    for img in images:
        img_id = img["id"]
        cats = image_to_cats.get(img_id)
        if not cats:
            continue
        img_weight = max(class_weights.get(cat_id, 1.0) for cat_id in cats)
        image_weights[str(img_id)] = {
            "file_name": img["file_name"],
            "weight": round(float(img_weight), 6),
        }

    output_dir = dataset_root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    class_out = output_dir / "class_balance_class_weights.json"
    image_out = output_dir / "class_balance_image_weights.json"

    with class_out.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "split": split,
                "total_annotations": total,
                "weights": {
                    str(cat_id): {
                        "name": category_names.get(cat_id, str(cat_id)),
                        "count": class_counter[cat_id],
                        "ratio": class_counter[cat_id] / total,
                        "weight": class_weights[cat_id],
                    }
                    for cat_id in sorted(class_weights.keys())
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    with image_out.open("w", encoding="utf-8") as f:
        json.dump({"split": split, "image_weights": image_weights}, f, indent=2, ensure_ascii=False)

    print(f"[INFO] 类别权重文件写入: {class_out}")
    print(f"[INFO] 图像重采样权重写入: {image_out}")


def main() -> None:
    args = parse_args()
    args.source_root = args.source_root.resolve()
    args.target_root = args.target_root.resolve()
    args.target_root.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    transforms = build_transforms(args.min_visibility)
    weak_class_multipliers = parse_class_multiplier_specs(args.weak_class_multipliers)
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
        active_multiplier = weak_class_multipliers if split == "train" else None
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

    if not args.skip_balance_files:
        generate_balance_files(args.target_root, split="train", smoothing=1.0)


if __name__ == "__main__":
    main()

