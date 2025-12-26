#!/usr/bin/env python3
"""
Downsample WSI-level JPEG images from 40x to 20x (50% resolution).

Example:
    python convert_wsi40x_to_20x.py \
        --src /home/ubuntu/lsn/project_new/RT-DETR-main/DataWSI_copy/wsi40x \
        --dst /home/ubuntu/lsn/project_new/RT-DETR-main/DataWSI_copy/wsi20x
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS = None  # allow very large slides
ImageFile.LOAD_TRUNCATED_IMAGES = True  # best-effort load for slightly corrupted JPEGs


def parse_args() -> argparse.Namespace:
    root = Path("/home/ubuntu/lsn/project_new/RT-DETR-main")
    default_src = root / "DataWSI_copy" / "wsi40x"
    default_dst = root / "DataWSI_copy" / "wsi20x"

    parser = argparse.ArgumentParser(
        description="Convert 40x JPEG WSIs to 20x by halving the resolution."
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=default_src,
        help=f"Directory containing 40x JPEGs (default: {default_src})",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=default_dst,
        help=f"Directory to save 20x JPEGs (default: {default_dst})",
    )
    parser.add_argument(
        "--formats",
        type=str,
        default=".jpeg,.jpg,.png",
        help="Comma-separated list of extensions to process (default: .jpeg,.jpg,.png)",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=95,
        help="JPEG quality for output files (default: 95). Ignored for PNG.",
    )
    return parser.parse_args()


def iter_images(directory: Path, exts: Iterable[str]) -> Iterable[Path]:
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() in exts:
            yield path


def convert_image(img_path: Path, dst_dir: Path, quality: int) -> None:
    dst_path = dst_dir / img_path.name
    if dst_path.exists():
        print(f"[skip] {dst_path} already exists")
        return

    with Image.open(img_path) as img:
        new_size = (img.width // 2, img.height // 2)
        if min(new_size) == 0:
            raise ValueError(f"{img_path} is too small to be halved (size={img.size})")
        resized = img.resize(new_size, resample=Image.Resampling.LANCZOS)
        save_kwargs = {}
        if dst_path.suffix.lower() in {".jpg", ".jpeg"}:
            save_kwargs.update({"quality": quality, "optimize": True})
        resized.save(dst_path, **save_kwargs)

    orig_mb = img_path.stat().st_size / (1024 * 1024)
    new_mb = dst_path.stat().st_size / (1024 * 1024)
    print(f"[done] {img_path.name}: {new_size} ({orig_mb:.1f}MB -> {new_mb:.1f}MB)")


def convert_single_image(
    src_path: Path,
    dst_path: Path,
    quality: int = 95,
    scale_factor: float = 0.5,
    logger=None
) -> tuple[int, int, int, int]:
    """
    转换单个图像（按指定比例缩放）
    
    Args:
        src_path: 源图像路径
        dst_path: 目标图像路径
        quality: JPEG质量（默认95）
        scale_factor: 缩放比例（默认0.5，即50%）
        logger: 日志记录器（可选）
    
    Returns:
        (原始宽度, 原始高度, 新宽度, 新高度)
    """
    def _log(msg):
        if logger:
            logger.info(msg)
        else:
            print(msg)
    
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    
    with Image.open(src_path) as img:
        orig_size = (img.width, img.height)
        new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
        
        if min(new_size) == 0:
            raise ValueError(f"{src_path} 缩放后尺寸过小 (原始={orig_size}, 缩放后={new_size})")
        
        _log(f"正在缩放图像: {orig_size} -> {new_size} (缩放比例={scale_factor})")
        resized = img.resize(new_size, resample=Image.Resampling.LANCZOS)
        
        save_kwargs = {}
        if dst_path.suffix.lower() in {".jpg", ".jpeg"}:
            save_kwargs.update({"quality": quality, "optimize": True})
        resized.save(dst_path, **save_kwargs)
    
    orig_mb = src_path.stat().st_size / (1024 * 1024)
    new_mb = dst_path.stat().st_size / (1024 * 1024)
    _log(f"缩放完成: {orig_size} -> {new_size} ({orig_mb:.1f}MB -> {new_mb:.1f}MB)")
    
    return (*orig_size, *new_size)


def main() -> None:
    args = parse_args()

    if not args.src.exists():
        raise FileNotFoundError(f"Source directory not found: {args.src}")
    args.dst.mkdir(parents=True, exist_ok=True)

    extensions = tuple(ext.strip().lower() for ext in args.formats.split(",") if ext.strip())
    if not extensions:
        raise ValueError("No valid extensions provided in --formats")

    files = list(iter_images(args.src, extensions))
    if not files:
        raise FileNotFoundError(
            f"No files with extensions {extensions} found in {args.src}"
        )

    print(f"Converting {len(files)} slide(s) from {args.src} -> {args.dst}")
    for path in files:
        try:
            convert_image(path, args.dst, args.quality)
        except Exception as exc:
            print(f"[error] {path.name}: {exc}")


if __name__ == "__main__":
    main()

