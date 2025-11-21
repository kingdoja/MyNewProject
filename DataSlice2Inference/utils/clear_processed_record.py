#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
删除指定图像在 processing_status.json 中的“已处理”记录。

示例：
    cd /home/ubuntu/lsn/project_new/RT-DETR-main/DataSlice2Inference/utils
    python clear_processed_record.py --image /home/ubuntu/lsn/project_new/RT-DETR-main/DataWSI/cj.jpeg

需要与 auto_process_monitor.py 使用同一份 config.yaml，默认位置：
    DataSlice2Inference/config.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

UTILS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = UTILS_DIR.parent

# 确保项目根目录优先于 utils 目录，避免与标准库模块（如 queue）冲突
sys.path = [str(PROJECT_ROOT)] + [
    p for p in sys.path if p not in {str(UTILS_DIR), str(PROJECT_ROOT)}
]

from utils.config import Config
from utils.validation import calculate_file_hash, backup_status_file

DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description="从 processing_status.json 中删除指定文件的处理记录"
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="原始全图路径（会计算其哈希以匹配记录）",
    )
    parser.add_argument(
        "--status-file",
        type=Path,
        default=None,
        help="processing_status.json 路径（默认：根据 config.yaml 的 output_dir 推断）",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="配置文件路径（仅在未指定 --status-file 时用于推断 output_dir）",
    )
    return parser.parse_args()


def resolve_status_file(args) -> Path:
    if args.status_file:
        return args.status_file.expanduser().resolve()
    
    config_path = args.config.expanduser().resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    config = Config(config_path)
    output_dir = config.get_path("paths.output_dir")
    return output_dir / "processing_status.json"


def load_status(status_file: Path) -> dict:
    if status_file.exists():
        with open(status_file, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, dict):
                return data
    return {"processed_files": []}


def save_status(status_file: Path, data: dict):
    status_file.parent.mkdir(parents=True, exist_ok=True)
    backup_status_file(status_file)
    with open(status_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    
    image_path = args.image.expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(f"待清除的图像不存在: {image_path}")
    
    status_file = resolve_status_file(args)
    status_data = load_status(status_file)
    processed = set(status_data.get("processed_files", []))
    
    if not processed:
        print(f"⚠️ 状态文件中没有任何记录: {status_file}")
        return
    
    try:
        file_hash = calculate_file_hash(image_path, algorithm="md5")
    except Exception as e:
        raise RuntimeError(f"计算文件哈希失败: {e}") from e
    
    if file_hash not in processed:
        print(f"ℹ️ 记录中未找到该文件的哈希，无需删除 ({image_path.name})")
        return
    
    processed.remove(file_hash)
    status_data["processed_files"] = sorted(processed)
    status_data["last_update"] = datetime.now().isoformat()
    
    save_status(status_file, status_data)
    print(f"✅ 已从 {status_file} 移除 {image_path.name} 的处理记录")


if __name__ == "__main__":
    main()


