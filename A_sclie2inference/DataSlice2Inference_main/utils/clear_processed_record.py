#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
删除指定图像在 processing_status.json 中的"已处理"记录。

支持功能：
1. 清除单个图像记录
2. 批量清除多个图像记录
3. 从目录中清除所有图像记录
4. 从文件列表中清除记录
5. 清除所有记录（危险操作）
6. 预览模式（--dry-run）查看将要清除的记录

需要与 auto_process_monitor.py 使用同一份 config.yaml，默认位置：
    DataSlice2Inference/config.yaml


cd /home/ubuntu/lsn/project_new/RT-DETR-main/A_sclie2inference/DataSlice2Inference_main/utils
python clear_processed_record.py --image /home/ubuntu/lsn/project_new/RT-DETR-main/A_sclie2inference/DataWSI/45-庄驷40X.jpeg
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
from utils.validation import backup_status_file

DEFAULT_CONFIG = PROJECT_ROOT / "config.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description="从 processing_status.json 中删除指定文件的处理记录",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 清除单个图像记录
  python clear_processed_record.py --image /path/to/image.jpg
  
  # 使用调试模式查看详细信息（推荐用于诊断问题）
  python clear_processed_record.py --image /path/to/image.jpg --debug
  
  # 批量清除多个图像记录
  python clear_processed_record.py --images /path/to/image1.jpg /path/to/image2.jpg
  
  # 从目录中清除所有图像记录
  python clear_processed_record.py --dir /path/to/imges/
  
  # 清除所有记录（危险操作）
  python clear_processed_record.py --clear-all
  
  # 从文件列表清除
  python clear_processed_record.py --list /path/to/file_list.txt
  
  # 预览模式（不实际删除，仅查看）
  python clear_processed_record.py --image /path/to/image.jpg --dry-run --debug
        """
    )
    
    # 互斥组：只能选择一种清除方式
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--image",
        type=Path,
        help="单个原始全图路径（会计算其哈希以匹配记录）",
    )
    group.add_argument(
        "--images",
        type=Path,
        nargs="+",
        help="多个原始全图路径（批量清除）",
    )
    group.add_argument(
        "--dir",
        type=Path,
        help="图像目录路径（清除该目录下所有图像的处理记录）",
    )
    group.add_argument(
        "--list",
        type=Path,
        help="包含图像路径的文件列表（每行一个路径）",
    )
    group.add_argument(
        "--clear-all",
        action="store_true",
        help="清除所有处理记录（危险操作，请谨慎使用）",
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅显示将要清除的记录，不实际删除（用于预览）",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="显示详细的调试信息（包括哈希值、文件信息等）",
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


def get_image_paths_from_dir(directory: Path) -> list[Path]:
    """从目录中获取所有图像文件路径"""
    IMAGE_EXTENSIONS = {'.jpeg', '.jpg', '.png', '.tif', '.tiff', '.ndpi', '.svs'}
    image_paths = []
    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(directory.glob(f"*{ext}"))
        image_paths.extend(directory.glob(f"*{ext.upper()}"))
    return sorted(image_paths)


def get_image_paths_from_list(list_file: Path) -> list[Path]:
    """从文件列表中读取图像路径"""
    if not list_file.exists():
        raise FileNotFoundError(f"文件列表不存在: {list_file}")
    
    image_paths = []
    with open(list_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):  # 忽略空行和注释
                path = Path(line).expanduser().resolve()
                if path.exists():
                    image_paths.append(path)
                else:
                    print(f"⚠️ 警告：文件不存在，跳过: {path}")
    return image_paths


def compute_processing_hash(image_path: Path, debug: bool = False) -> tuple[str, dict]:
    """使用与自动处理一致的哈希算法（8KB文件头 + 文件大小）
    
    Returns:
        tuple: (hash_value, debug_info)
    """
    debug_info = {}
    try:
        with open(image_path, "rb") as f:
            file_size = image_path.stat().st_size
            file_header = f.read(8192)
            import hashlib

            hash_obj = hashlib.md5()
            hash_obj.update(file_header)
            hash_obj.update(str(file_size).encode())
            hash_value = hash_obj.hexdigest()
            
            debug_info = {
                "method": "md5_hash",
                "file_size": file_size,
                "header_bytes": len(file_header),
                "hash": hash_value
            }
            return hash_value, debug_info
    except Exception as e:
        # 兼容自动处理中的降级策略：使用修改时间
        mtime = image_path.stat().st_mtime
        hash_value = str(mtime)
        debug_info = {
            "method": "mtime_fallback",
            "error": str(e),
            "mtime": mtime,
            "hash": hash_value
        }
        return hash_value, debug_info


def clear_single_image(image_path: Path, processed: set, status_file: Path, dry_run: bool = False, debug: bool = False) -> bool:
    """清除单个图像的处理记录
    
    Returns:
        bool: 是否成功清除（True=已清除，False=未找到记录）
    """
    if not image_path.exists():
        print(f"⚠️ 警告：图像不存在，跳过: {image_path}")
        return False
    
    try:
        file_hash, debug_info = compute_processing_hash(image_path, debug)
    except Exception as e:
        print(f"❌ 计算文件哈希失败 {image_path}: {e}")
        return False
    
    # 如果哈希不匹配，尝试检查时间戳哈希（降级策略）
    original_hash = file_hash
    if file_hash not in processed:
        # 检查是否有时间戳格式的哈希（可能是记录时使用了降级策略，或现在使用了降级策略）
        try:
            file_mtime = image_path.stat().st_mtime
            mtime_str = str(file_mtime)
            if mtime_str in processed:
                if debug:
                    method_note = "记录时使用了降级策略" if debug_info.get('method') == 'md5_hash' else "现在使用了降级策略"
                    print(f"✅ 找到匹配的时间戳哈希（{method_note}）: {mtime_str}")
                file_hash = mtime_str  # 使用时间戳哈希
        except:
            pass
    
    # 调试信息输出
    if debug:
        print(f"\n{'='*70}")
        print(f"🔍 调试信息 - {image_path.name}")
        print(f"{'='*70}")
        print(f"文件路径: {image_path}")
        print(f"文件大小: {debug_info.get('file_size', 'N/A')} 字节")
        print(f"哈希方法: {debug_info.get('method', 'unknown')}")
        print(f"计算出的哈希: {original_hash}")
        if file_hash != original_hash:
            print(f"使用的哈希（时间戳匹配）: {file_hash}")
        if debug_info.get('method') == 'mtime_fallback':
            print(f"⚠️ 警告：使用了降级策略（修改时间），原因: {debug_info.get('error', 'unknown')}")
        print(f"状态文件中的记录数: {len(processed)}")
        
        # 检查是否有相似的哈希值（可能是文件被修改了）
        if file_hash not in processed:
            # 检查是否有以相同前缀开头的哈希（MD5前8位）
            if debug_info.get('method') == 'md5_hash':
                prefix = original_hash[:8]
                similar = [h for h in processed if len(h) >= 8 and h.startswith(prefix)]
                if similar:
                    print(f"⚠️ 发现 {len(similar)} 个以相同前缀开头的哈希值（可能是文件被修改）:")
                    for h in similar[:5]:  # 只显示前5个
                        print(f"  - {h[:32]}...")
                else:
                    # 显示一些示例哈希值用于对比
                    print(f"状态文件中的示例哈希值（前5个）:")
                    for i, h in enumerate(list(processed)[:5]):
                        print(f"  [{i+1}] {h[:32]}...")
    
    if file_hash not in processed:
        if not dry_run:
            print(f"ℹ️ 记录中未找到该文件的哈希，无需删除 ({image_path.name})")
            if debug:
                print(f"   计算的哈希: {file_hash}")
                print(f"   提示：如果文件被修改过，哈希值会改变，需要手动删除记录")
        return False
    
    if dry_run:
        print(f"🔍 [预览] 将清除: {image_path.name} (哈希: {file_hash[:8]}...)")
        return True
    
    processed.remove(file_hash)
    print(f"✅ 已移除: {image_path.name}")
    return True


def main():
    args = parse_args()
    
    status_file = resolve_status_file(args)
    status_data = load_status(status_file)
    processed = set(status_data.get("processed_files", []))
    
    if not processed:
        print(f"⚠️ 状态文件中没有任何记录: {status_file}")
        return
    
    initial_count = len(processed)
    
    if args.clear_all:
        # 清除所有记录
        if args.dry_run:
            print(f"🔍 [预览] 将清除所有 {initial_count} 条处理记录")
            print("⚠️ 这是预览模式，实际未删除任何记录")
        else:
            # 确认操作
            print(f"⚠️ 警告：您即将清除所有 {initial_count} 条处理记录！")
            response = input("请输入 'YES' 确认此操作: ")
            if response != 'YES':
                print("❌ 操作已取消")
                return
            
            processed.clear()
            status_data["processed_files"] = []
            status_data["last_update"] = datetime.now().isoformat()
            save_status(status_file, status_data)
            print(f"✅ 已清除所有 {initial_count} 条处理记录")
        return
    
    # 收集要清除的图像路径
    image_paths = []
    
    if args.image:
        image_paths = [args.image.expanduser().resolve()]
    elif args.images:
        image_paths = [p.expanduser().resolve() for p in args.images]
    elif args.dir:
        image_paths = get_image_paths_from_dir(args.dir.expanduser().resolve())
        print(f"📁 从目录中找到 {len(image_paths)} 个图像文件")
    elif args.list:
        image_paths = get_image_paths_from_list(args.list)
        print(f"📄 从文件列表读取到 {len(image_paths)} 个图像路径")
    
    if not image_paths:
        print("⚠️ 未找到任何图像文件")
        return
    
    # 清除记录
    cleared_count = 0
    not_found_count = 0
    error_count = 0
    
    print(f"\n开始处理 {len(image_paths)} 个图像...")
    if args.dry_run:
        print("🔍 [预览模式] 以下记录将被清除：\n")
    
    for image_path in image_paths:
        try:
            if clear_single_image(image_path, processed, status_file, args.dry_run, args.debug):
                cleared_count += 1
            else:
                not_found_count += 1
        except Exception as e:
            error_count += 1
            print(f"❌ 处理失败 {image_path}: {e}")
            if args.debug:
                import traceback
                traceback.print_exc()
    
    # 保存结果
    if not args.dry_run and cleared_count > 0:
        status_data["processed_files"] = sorted(processed)
        status_data["last_update"] = datetime.now().isoformat()
        save_status(status_file, status_data)
        print(f"\n✅ 状态文件已更新: {status_file}")
    
    # 输出统计
    print(f"\n{'='*70}")
    print("处理结果统计：")
    print(f"  总计: {len(image_paths)} 个图像")
    print(f"  成功清除: {cleared_count} 条记录")
    print(f"  未找到记录: {not_found_count} 个")
    if error_count > 0:
        print(f"  处理失败: {error_count} 个")
    print(f"  剩余记录数: {len(processed)} (原 {initial_count} 条)")
    print(f"{'='*70}")
    
    if args.dry_run:
        print("\n⚠️ 这是预览模式，实际未删除任何记录")
        print("   如需实际删除，请移除 --dry-run 参数")
    elif cleared_count > 0:
        print("\n💡 提示：")
        print("   记录已从状态文件中删除。")
        print("   如果自动处理脚本正在运行，它会在下次检测到文件变化时自动重新加载状态。")
        print("   您可以通过以下方式触发重新处理：")
        print("   1. 等待文件系统事件（文件被修改或重新保存）")
        print("   2. 或者重启自动处理脚本以确保立即生效")


if __name__ == "__main__":
    main()


