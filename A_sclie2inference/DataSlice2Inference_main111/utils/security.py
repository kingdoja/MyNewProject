#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""安全模块"""

import os
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def validate_path(path: Path, base_dir: Path) -> Path:
    """验证路径安全性，防止路径遍历攻击
    
    Args:
        path: 要验证的路径
        base_dir: 基础目录
        
    Returns:
        解析后的安全路径
        
    Raises:
        ValueError: 路径超出允许范围
    """
    try:
        resolved = path.resolve()
        base_resolved = base_dir.resolve()
        
        # 检查路径是否在基础目录内
        if not str(resolved).startswith(str(base_resolved)):
            raise ValueError(f"路径超出允许范围: {path}")
        
        return resolved
    except Exception as e:
        logger.error(f"路径验证失败: {path}, 错误: {e}")
        raise


def validate_file_type(file_path: Path, allowed_types: List[str]) -> bool:
    """验证文件类型
    
    Args:
        file_path: 文件路径
        allowed_types: 允许的 MIME 类型列表
        
    Returns:
        是否为允许的类型
        
    Note:
        如果没有安装 python-magic，使用扩展名验证
    """
    try:
        import magic
        mime = magic.Magic(mime=True)
        file_type = mime.from_file(str(file_path))
        
        if file_type not in allowed_types:
            logger.warning(f"文件类型不允许: {file_path}, 类型: {file_type}")
            return False
        
        return True
    except ImportError:
        # 如果没有安装 python-magic，使用扩展名验证
        logger.debug("python-magic 未安装，使用扩展名验证")
        ext_to_mime = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.tif': 'image/tiff',
            '.tiff': 'image/tiff',
            '.ndpi': 'image/tiff'
        }
        ext = file_path.suffix.lower()
        mime_type = ext_to_mime.get(ext)
        if mime_type and mime_type in allowed_types:
            return True
        logger.warning(f"文件扩展名不允许: {file_path}, 扩展名: {ext}")
        return False
    except Exception as e:
        logger.error(f"验证文件类型失败: {file_path}, 错误: {e}")
        return False


def check_permissions(path: Path, required_perms: str = 'rw') -> bool:
    """检查文件权限
    
    Args:
        path: 文件或目录路径
        required_perms: 需要的权限 ('r'=读, 'w'=写, 'x'=执行)
        
    Returns:
        是否有权限
        
    Raises:
        PermissionError: 权限不足
    """
    if 'r' in required_perms and not os.access(path, os.R_OK):
        raise PermissionError(f"无读取权限: {path}")
    if 'w' in required_perms and not os.access(path, os.W_OK):
        raise PermissionError(f"无写入权限: {path}")
    if 'x' in required_perms and not os.access(path, os.X_OK):
        raise PermissionError(f"无执行权限: {path}")
    return True

