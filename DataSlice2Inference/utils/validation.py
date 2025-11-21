#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""数据验证模块"""

import hashlib
import logging
from pathlib import Path
from typing import Optional
from contextlib import contextmanager
import tempfile
import shutil

logger = logging.getLogger(__name__)


def calculate_file_hash(file_path: Path, algorithm: str = 'md5', chunk_size: int = 4096) -> str:
    """计算文件哈希值
    
    Args:
        file_path: 文件路径
        algorithm: 哈希算法（md5, sha256等）
        chunk_size: 读取块大小
        
    Returns:
        文件哈希值（十六进制字符串）
    """
    hash_obj = hashlib.new(algorithm)
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(chunk_size), b''):
                hash_obj.update(chunk)
        return hash_obj.hexdigest()
    except Exception as e:
        logger.error(f"计算文件哈希失败 {file_path}: {e}")
        raise


def verify_file_integrity(file_path: Path, expected_hash: str, algorithm: str = 'md5') -> bool:
    """验证文件完整性
    
    Args:
        file_path: 文件路径
        expected_hash: 期望的哈希值
        algorithm: 哈希算法
        
    Returns:
        是否匹配
    """
    try:
        actual_hash = calculate_file_hash(file_path, algorithm)
        return actual_hash == expected_hash
    except Exception:
        return False


def verify_processing_completeness(output_dir: Path, expected_files: int, pattern: str = "*.json") -> bool:
    """验证处理完整性
    
    Args:
        output_dir: 输出目录
        expected_files: 期望的文件数量
        pattern: 文件匹配模式
        
    Returns:
        是否完整
        
    Raises:
        RuntimeError: 处理不完整
    """
    actual_files = len(list(output_dir.glob(pattern)))
    if actual_files != expected_files:
        raise RuntimeError(
            f"处理不完整: 期望 {expected_files} 个文件，实际 {actual_files} 个"
        )
    return True


@contextmanager
def atomic_output(output_dir: Path):
    """原子性输出上下文管理器，失败时自动回滚
    
    Args:
        output_dir: 最终输出目录
        
    Yields:
        临时输出目录
    """
    temp_dir = output_dir.parent / f"{output_dir.name}.tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        yield temp_dir
        # 处理成功，移动到最终位置
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.move(str(temp_dir), str(output_dir))
        logger.info(f"原子性输出完成: {output_dir}")
    except Exception as e:
        # 处理失败，清理临时目录
        logger.error(f"处理失败，清理临时目录: {e}")
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise


def backup_status_file(status_file: Path, max_backups: int = 5):
    """备份状态文件
    
    Args:
        status_file: 状态文件路径
        max_backups: 最大备份数量
    """
    if not status_file.exists():
        return
    
    try:
        backup_file = status_file.with_suffix('.json.bak')
        shutil.copy2(status_file, backup_file)
        logger.debug(f"状态文件已备份: {backup_file}")
        
        # 只保留最近 N 个备份
        backups = sorted(status_file.parent.glob('*.json.bak'))
        for old_backup in backups[:-max_backups]:
            old_backup.unlink()
            logger.debug(f"删除旧备份: {old_backup}")
    except Exception as e:
        logger.warning(f"备份状态文件失败: {e}")

