#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
移动配对的patch数据集和转换后的COCO输出文件夹，并将其移动/复制到DATA/SputumCell目录下。
Pair patch dataset folders with converted COCO output folders and move/copy only:

1) patch image files from DataPatches/<patch_dir>
2) coco_format.json from annotationConverter/output/<output_dir>

into:
    new-<id>/

In each target folder:
  - images go to new-<id>/images/
  - coco_format.json goes to new-<id>/coco_format.json

Visualization files are not moved/copied.
"""

import argparse
import json
import os
import re
import shutil
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

try:
    from pypinyin import lazy_pinyin  # type: ignore[import-not-found]
except ImportError:
    lazy_pinyin = None


DEFAULT_PATCH_ROOT = (
    "/home/ubuntu/lsn/project_new/RT-DETR-main/"
    "A_sclie2inference/DataPatches"
)
DEFAULT_OUTPUT_ROOT = (
    "/home/ubuntu/lsn/project_new/RT-DETR-main/"
    "A_sclie2inference/DataSlice2Inference_main/annotationConverter/output"
)
DEFAULT_TARGET_ROOT = (
    "/home/ubuntu/lsn/project_new/RT-DETR-main/DATA/SputumCell"
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class PatchEntry:
    path: Path
    dirname: str
    id_prefix: str
    identity_raw: str
    identity_pinyin: str
    identity_raw_key: str
    identity_pinyin_key: str
    image_names: Set[str]


@dataclass
class OutputEntry:
    path: Path
    dirname: str
    id_prefix: str
    name_key: str
    coco_image_names: Set[str]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Pair DataPatches folders and converted output folders, then move them to DATA/SputumCell."
    )
    parser.add_argument(
        "--patch-root",
        type=str,
        default=DEFAULT_PATCH_ROOT,
        help="Root directory containing patch dataset folders.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root directory containing converted output folders.",
    )
    parser.add_argument(
        "--target-root",
        type=str,
        default=DEFAULT_TARGET_ROOT,
        help="Destination root directory (e.g. DATA/SputumCell).",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="Copy instead of move.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print planned operations, do not modify files.",
    )
    return parser.parse_args()


def normalize_key(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def normalize_dash_chars(value: str) -> str:
    """统一各类横线字符，避免 '21﹣xxx' 这类目录名无法按 '-' 分割。"""
    normalized = unicodedata.normalize("NFKC", value)
    return re.sub(r"[‐‑‒–—―﹣－]", "-", normalized)


def extract_id_prefix(value: str) -> str:
    """提取前导数字编码，如 '21-凌学余'/'21﹣凌学余' -> '21'。"""
    normalized = normalize_dash_chars(value).strip().lower()
    m = re.match(r"^(\d+)", normalized)
    if m:
        return m.group(1)
    # 兜底：沿用原有按 '-' 分割逻辑，防止极端命名回归
    return normalized.split("-")[0]


def to_pinyin_text(value: str) -> str:
    if lazy_pinyin is None:
        return value
    return "".join(lazy_pinyin(value))


def parse_patch_identity(dirname: str) -> Tuple[str, str]:
    # Example: 23-达明珠（复3）_20260304_174522 -> 23-达明珠（复3）, 23
    identity_raw = dirname.split("_")[0]
    id_prefix = extract_id_prefix(identity_raw)
    return identity_raw, id_prefix


def parse_output_identity(dirname: str) -> Tuple[str, str]:
    # Example: 23-damingzhu-3 -> 23-damingzhu-3, 23
    id_prefix = extract_id_prefix(dirname)
    return dirname, id_prefix


def load_coco_image_names(output_dir: Path) -> Set[str]:
    """读取output目录中的coco_format.json，返回images里的file_name集合。"""
    coco_path = output_dir / "coco_format.json"
    if not coco_path.exists():
        return set()
    try:
        with open(coco_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        image_names: Set[str] = set()
        for img in data.get("images", []):
            if isinstance(img, dict):
                name = img.get("file_name")
                if isinstance(name, str) and name:
                    image_names.add(Path(name).name)
        return image_names
    except Exception:
        return set()


def list_patch_entries(patch_root: Path) -> List[PatchEntry]:
    entries: List[PatchEntry] = []
    for child in sorted(patch_root.iterdir()):
        if not child.is_dir():
            continue
        identity_raw, id_prefix = parse_patch_identity(child.name)
        identity_pinyin = to_pinyin_text(identity_raw)
        entries.append(
            PatchEntry(
                path=child,
                dirname=child.name,
                id_prefix=id_prefix,
                identity_raw=identity_raw,
                identity_pinyin=identity_pinyin,
                identity_raw_key=normalize_key(identity_raw),
                identity_pinyin_key=normalize_key(identity_pinyin),
                image_names={
                    x.name for x in child.iterdir()
                    if x.is_file() and x.suffix.lower() in IMAGE_EXTENSIONS
                },
            )
        )
    return entries


def list_output_entries(output_root: Path) -> List[OutputEntry]:
    entries: List[OutputEntry] = []
    for child in sorted(output_root.iterdir()):
        if not child.is_dir():
            continue
        identity, id_prefix = parse_output_identity(child.name)
        entries.append(
            OutputEntry(
                path=child,
                dirname=child.name,
                id_prefix=id_prefix,
                name_key=normalize_key(identity),
                coco_image_names=load_coco_image_names(child),
            )
        )
    return entries


def similarity_score(patch_entry: PatchEntry, output_entry: OutputEntry) -> float:
    # Weighted best-match score from raw and pinyin forms.
    score_raw = SequenceMatcher(None, patch_entry.identity_raw_key, output_entry.name_key).ratio()
    score_pinyin = SequenceMatcher(None, patch_entry.identity_pinyin_key, output_entry.name_key).ratio()
    return max(score_raw, score_pinyin)


def pair_entries(
    patch_entries: List[PatchEntry],
    output_entries: List[OutputEntry],
) -> Tuple[List[Tuple[PatchEntry, OutputEntry]], List[PatchEntry], List[OutputEntry]]:
    output_by_id: Dict[str, List[OutputEntry]] = {}
    for out in output_entries:
        output_by_id.setdefault(out.id_prefix, []).append(out)

    used_output_paths: Set[Path] = set()
    pairs: List[Tuple[PatchEntry, OutputEntry]] = []
    unmatched_patches: List[PatchEntry] = []

    for patch in patch_entries:
        candidates = [
            out for out in output_by_id.get(patch.id_prefix, [])
            if out.path not in used_output_paths
        ]
        if not candidates:
            unmatched_patches.append(patch)
            continue

        def pair_score(out: OutputEntry):
            # 先按COCO图片名与patch文件名重叠度匹配（更可靠），再用名称相似度兜底
            overlap_count = len(patch.image_names & out.coco_image_names)
            coverage = (
                overlap_count / len(out.coco_image_names)
                if out.coco_image_names else 0.0
            )
            name_score = similarity_score(patch, out)
            return (overlap_count, coverage, name_score)

        best = max(candidates, key=pair_score)
        pairs.append((patch, best))
        used_output_paths.add(best.path)

    unmatched_outputs = [out for out in output_entries if out.path not in used_output_paths]
    return pairs, unmatched_patches, unmatched_outputs


def build_target_dir(target_root: Path, id_prefix: str) -> Path:
    # Create new-<id>, and append -2/-3 for duplicates.
    base_name = f"new-{id_prefix}"
    candidate = target_root / base_name
    if not candidate.exists():
        return candidate

    idx = 2
    while True:
        candidate = target_root / f"{base_name}-{idx}"
        if not candidate.exists():
            return candidate
        idx += 1


def move_or_copy_dir(src: Path, dst: Path, copy_mode: bool):
    if copy_mode:
        shutil.copytree(src, dst)
    else:
        shutil.move(str(src), str(dst))


def move_or_copy_file(src: Path, dst: Path, copy_mode: bool):
    if copy_mode:
        shutil.copy2(src, dst)
    else:
        shutil.move(str(src), str(dst))


def list_patch_images(patch_dir: Path) -> List[Path]:
    images: List[Path] = []
    for child in sorted(patch_dir.iterdir()):
        if child.is_file() and child.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(child)
    return images


def prune_coco_by_existing_images(coco_path: Path, image_dir: Path):
    """清理COCO：仅保留实际存在的图片及其对应标注。"""
    if not coco_path.exists():
        return

    with open(coco_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    images = data.get("images", [])
    annotations = data.get("annotations", [])
    existing_files = {p.name for p in image_dir.iterdir() if p.is_file()}

    kept_images = []
    kept_image_ids = set()
    for img in images:
        file_name = img.get("file_name")
        image_id = img.get("id")
        if file_name in existing_files:
            kept_images.append(img)
            kept_image_ids.add(image_id)

    kept_annotations = [
        ann for ann in annotations
        if ann.get("image_id") in kept_image_ids
    ]

    removed_images = len(images) - len(kept_images)
    removed_annotations = len(annotations) - len(kept_annotations)
    if removed_images == 0 and removed_annotations == 0:
        return

    data["images"] = kept_images
    data["annotations"] = kept_annotations
    with open(coco_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(
        f"  ✓ 已清理COCO缺失图片引用: "
        f"移除images={removed_images}, annotations={removed_annotations}"
    )


def execute(
    pairs: List[Tuple[PatchEntry, OutputEntry]],
    target_root: Path,
    copy_mode: bool,
    dry_run: bool,
):
    target_root.mkdir(parents=True, exist_ok=True)

    for patch_entry, output_entry in pairs:
        target_dir = build_target_dir(target_root, patch_entry.id_prefix)
        coco_src = output_entry.path / "coco_format.json"
        patch_images = list_patch_images(patch_entry.path)

        print("-" * 70)
        print(f"Target: {target_dir}")
        print(f"  Patch images source: {patch_entry.path} (count={len(patch_images)})")
        print(f"  COCO json source   : {coco_src}")

        if not patch_images:
            print("  ⚠️ 跳过：patch目录中未找到图片文件")
            continue
        if not coco_src.exists():
            print("  ⚠️ 跳过：未找到coco_format.json")
            continue

        if dry_run:
            print(f"  [DRY-RUN] 图片将放到: {target_dir / 'images'}")
            print(f"  [DRY-RUN] coco_format.json将放到: {target_dir}")
            continue

        target_dir.mkdir(parents=True, exist_ok=False)
        target_images_dir = target_dir / "images"
        target_images_dir.mkdir(parents=True, exist_ok=False)
        for image_path in patch_images:
            dst_path = target_images_dir / image_path.name
            move_or_copy_file(image_path, dst_path, copy_mode)
        target_coco_path = target_dir / "coco_format.json"
        move_or_copy_file(coco_src, target_coco_path, copy_mode)
        prune_coco_by_existing_images(target_coco_path, target_images_dir)


def main():
    args = parse_args()

    patch_root = Path(args.patch_root)
    output_root = Path(args.output_root)
    target_root = Path(args.target_root)

    if not patch_root.exists():
        raise FileNotFoundError(f"Patch root not found: {patch_root}")
    if not output_root.exists():
        raise FileNotFoundError(f"Output root not found: {output_root}")

    patch_entries = list_patch_entries(patch_root)
    output_entries = list_output_entries(output_root)
    pairs, unmatched_patches, unmatched_outputs = pair_entries(patch_entries, output_entries)

    print("=" * 70)
    print("Pairing summary")
    print(f"Patch folders : {len(patch_entries)}")
    print(f"Output folders: {len(output_entries)}")
    print(f"Paired        : {len(pairs)}")
    print(f"Unmatched patch folders : {len(unmatched_patches)}")
    print(f"Unmatched output folders: {len(unmatched_outputs)}")
    if lazy_pinyin is None:
        print("Note: pypinyin not installed, matching uses raw text similarity only.")
    print("=" * 70)

    if unmatched_patches:
        print("\n[Unmatched patch folders]")
        for item in unmatched_patches:
            print(f"  - {item.dirname}")

    if unmatched_outputs:
        print("\n[Unmatched output folders]")
        for item in unmatched_outputs:
            print(f"  - {item.dirname}")

    if not pairs:
        print("\nNo pairs found. Exit.")
        return

    print(
        f"\nMode: {'COPY' if args.copy else 'MOVE'} "
        f"{'(DRY-RUN)' if args.dry_run else ''}"
    )
    execute(
        pairs=pairs,
        target_root=target_root,
        copy_mode=args.copy,
        dry_run=args.dry_run,
    )
    print("\nDone.")


if __name__ == "__main__":
    main()

