import os, sys, shutil
from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# 你可以在这里直接修改路径
SRC_DIR = Path("/home/ubuntu/lsn/project_new/RT-DETR-main/DataPatches/Patches5")  # patch所在目录
KEEP_DIR = Path("/home/ubuntu/lsn/project_new/RT-DETR-main/DataPatchesKeep/Patches5")  # 保留输出目录
TRASH_DIR = Path("/home/ubuntu/lsn/project_new/RT-DETR-main/DataPatchesTrash/Patches5")  # 空白输出目录

BG_RGB = (238,235,235)  # 背景色
TOLERANCE =30           # 容差
BG_RATIO = 0.9         # 背景比例阈值
STD_THRESH = 10.0         # 灰度标准差阈值
MOVE = False             # True=移动，False=复制
DELETE = False            # True=直接删除空白patch
WORKERS = 8               # 并行线程数

def is_blank(img: Image.Image, bg_rgb=BG_RGB, tolerance=TOLERANCE, bg_ratio=BG_RATIO, std_thresh=STD_THRESH) -> bool:
    arr = np.asarray(img)#转换为数组

    if arr.ndim == 2:  # 灰度
        arr_rgb = np.stack([arr, arr, arr], axis=-1)
    else:
        if arr.shape[2] == 4:#判断是否为透明
            rgb, a = arr[..., :3].astype(np.float32), arr[..., 3:4].astype(np.float32) / 255.0
            arr_rgb = (rgb * a + 255.0 * (1 - a)).astype(np.uint8)
        else:
            arr_rgb = arr[..., :3]#rgb

    bg = np.array(bg_rgb, dtype=np.int16)[None, None, :]#背景色
    diff = np.abs(arr_rgb.astype(np.int16) - bg).sum(axis=2)#计算差异
    bg_mask = diff <= tolerance#判断是否为背景
    ratio = bg_mask.mean()

    gray = (0.299 * arr_rgb[...,0] + 0.587 * arr_rgb[...,1] + 0.114 * arr_rgb[...,2]).astype(np.float32)
    std = float(gray.std())

    return (ratio >= bg_ratio) and (std <= std_thresh)

def process_one(path: Path) -> Tuple[str, bool, float, float]:#处理单个patch
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            blank = is_blank(im, BG_RGB, TOLERANCE, BG_RATIO, STD_THRESH)
    except Exception:
        return (str(path), False, -1.0, -1.0)

    arr = np.asarray(im.convert("RGB"))
    bg = np.array(BG_RGB, dtype=np.int16)[None, None, :]
    diff = np.abs(arr.astype(np.int16) - bg).sum(axis=2)
    bg_mask = diff <= TOLERANCE
    ratio = bg_mask.mean()
    gray = (0.299 * arr[...,0] + 0.587 * arr[...,1] + 0.114 * arr[...,2]).astype(np.float32)#灰度
    std = float(gray.std())

    dest_dir = TRASH_DIR if blank else KEEP_DIR
    dest_dir.mkdir(parents=True, exist_ok=True)

    if DELETE and blank:
        path.unlink(missing_ok=True)
    else:
        if MOVE:
            shutil.move(str(path), str(dest_dir / path.name))
        else:
            shutil.copy2(str(path), str(dest_dir / path.name))

    return (str(path), blank, ratio, std)

def main():
    if not SRC_DIR.exists():
        print(f"[ERR] src_dir 不存在: {SRC_DIR}")
        sys.exit(1)

    files = [p for p in SRC_DIR.iterdir() if p.suffix.lower() in IMG_EXTS]
    total = len(files)
    print(f"[INFO] 共发现 {total} 个 patch，开始筛选...")

    kept = removed = failed = 0
    KEEP_DIR.mkdir(parents=True, exist_ok=True)
    TRASH_DIR.mkdir(parents=True, exist_ok=True)

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(process_one, p) for p in files]
        for i, f in enumerate(as_completed(futs), 1):
            path, blank, ratio, std = f.result()
            if ratio < 0:
                failed += 1
                print(f"[FAIL] 读取失败: {path}")
            else:
                if blank:
                    removed += 1
                else:
                    kept += 1
            if i % 500 == 0 or i == total:
                print(f"[{i}/{total}] kept={kept}, removed={removed}, failed={failed}")

    print(f"[DONE] 总计: {total}, 保留(非空白): {kept}, 移除(空白): {removed}, 失败: {failed}")

if __name__ == "__main__":
    main()
