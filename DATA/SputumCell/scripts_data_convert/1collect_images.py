#!/usr/bin/env python3
"""
Collect all images from the patches3 dataset into a single images/ directory.

By default the script copies *.png files from:
    /home/ubuntu/lsn/project_new/RT-DETR-main/DATA/SputumCell/patches3
into:
    /home/ubuntu/lsn/project_new/RT-DETR-main/DATA/SputumCell/images

The paths can be overridden via --source and --dest. Use --move to move instead
of copy. The script skips files that already exist at the destination unless
--overwrite is specified.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Iterable


PROJECT_ROOT = Path("/home/ubuntu/lsn/project_new/RT-DETR-main")
DATA_ROOT = PROJECT_ROOT / "DATA" / "SputumCell"
DEFAULT_SOURCE = DATA_ROOT / "new37"
DEFAULT_DEST = DEFAULT_SOURCE / "images"


def iter_image_files(source_dir: Path) -> Iterable[Path]:
    """Yield all png files in source_dir (non-recursive)."""
    for path in sorted(source_dir.glob("*.png")):
        if path.is_file():
            yield path


def transfer_file(
    src: Path,
    dest_dir: Path,
    overwrite: bool = False,
) -> bool:
    """Move src into dest_dir. Returns True if a transfer occurred."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name

    if dest.exists() and not overwrite:
        return False

    shutil.move(str(src), dest)

    return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Move patches3 images into a dedicated images directory.",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help=f"Directory containing patch images (default: {DEFAULT_SOURCE})",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Target images directory (default: {DEFAULT_DEST})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite files that already exist at the destination.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source: Path = args.source
    dest: Path = args.dest

    if not source.exists():
        raise FileNotFoundError(f"Source directory not found: {source}")
    if not source.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {source}")

    transferred = 0
    skipped = 0
    for img_path in iter_image_files(source):
        if transfer_file(img_path, dest, overwrite=args.overwrite):
            transferred += 1
        else:
            skipped += 1

    print(
        f"Done. Moved {transferred} files to {dest}. "
        f"Skipped {skipped} existing files."
    )


if __name__ == "__main__":
    main()

