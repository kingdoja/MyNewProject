#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""资源管理模块"""

import os
import shutil
import logging
import resource
from pathlib import Path
from typing import Optional
import psutil

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


def set_memory_limit(max_memory_gb: float):
    """设置内存限制
    
    Args:
        max_memory_gb: 最大内存使用（GB）
    """
    try:
        max_bytes = int(max_memory_gb * 1024 * 1024 * 1024)
        resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))
        logger.info(f"设置内存限制: {max_memory_gb}GB")
    except Exception as e:
        logger.warning(f"设置内存限制失败: {e}")


def check_memory_usage(threshold: float = 90.0) -> dict:
    """检查内存使用情况
    
    Args:
        threshold: 警告阈值（百分比）
        
    Returns:
        内存使用信息字典
    """
    try:
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_percent = process.memory_percent()
        
        info = {
            'rss_mb': memory_info.rss / 1024 / 1024,
            'vms_mb': memory_info.vms / 1024 / 1024,
            'percent': memory_percent
        }
        
        if memory_percent > threshold:
            logger.warning(
                f"内存使用率过高: {memory_percent:.1f}% "
                f"(RSS: {info['rss_mb']:.1f}MB)"
            )
        
        return info
    except Exception as e:
        logger.error(f"检查内存使用失败: {e}")
        return {}


def clear_gpu_cache():
    """清理设备缓存（NPU/GPU）"""
    if not TORCH_AVAILABLE:
        return
    
    try:
        # 尝试使用NPU工具模块
        from npu_utils import is_npu_available, clear_device_cache
        import torch
        
        device = None
        if is_npu_available():
            device = torch.device("npu")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
        
        if device:
            clear_device_cache(device)
            return
    except (ImportError, Exception):
        pass
    
    # 回退到原始CUDA逻辑
    if torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("GPU 缓存已清理")
        except Exception as e:
            logger.warning(f"清理 GPU 缓存失败: {e}")


def monitor_gpu_memory() -> dict:
    """监控设备内存使用（NPU/GPU）
    
    Returns:
        设备内存信息字典
    """
    if not TORCH_AVAILABLE:
        return {}
    
    try:
        # 尝试使用NPU工具模块
        from npu_utils import is_npu_available, get_device_memory_info
        import torch
        
        device = None
        device_name = None
        if is_npu_available():
            device = torch.device("npu")
            device_name = "NPU"
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            device_name = "GPU"
        else:
            return {}
        
        info = get_device_memory_info(device)
        if info:
            logger.debug(
                f"{device_name} 内存: 已分配 {info.get('allocated_gb', 0):.2f}GB, "
                f"已保留 {info.get('reserved_gb', 0):.2f}GB, "
                f"峰值 {info.get('max_allocated_gb', 0):.2f}GB"
            )
            return info
    except (ImportError, Exception) as e:
        logger.debug(f"使用NPU工具模块失败，回退到CUDA逻辑: {e}")
    
    # 回退到原始CUDA逻辑
    if torch.cuda.is_available():
        try:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            max_allocated = torch.cuda.max_memory_allocated() / 1024**3
            
            info = {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'max_allocated_gb': max_allocated
            }
            
            logger.debug(
                f"GPU 内存: 已分配 {allocated:.2f}GB, "
                f"已保留 {reserved:.2f}GB, "
                f"峰值 {max_allocated:.2f}GB"
            )
            
            return info
        except Exception as e:
            logger.warning(f"监控 GPU 内存失败: {e}")
    
    return {}


def check_disk_space(path: Path, min_free_gb: float = 10.0) -> dict:
    """检查磁盘空间
    
    Args:
        path: 检查的路径
        min_free_gb: 最小可用空间（GB）
        
    Returns:
        磁盘空间信息字典
        
    Raises:
        RuntimeError: 磁盘空间不足
    """
    try:
        stat = shutil.disk_usage(path)
        free_gb = stat.free / (1024**3)
        total_gb = stat.total / (1024**3)
        used_gb = stat.used / (1024**3)
        used_percent = (stat.used / stat.total) * 100
        
        info = {
            'total_gb': total_gb,
            'used_gb': used_gb,
            'free_gb': free_gb,
            'used_percent': used_percent
        }
        
        if free_gb < min_free_gb:
            raise RuntimeError(
                f"磁盘空间不足: {path} 仅剩 {free_gb:.2f}GB "
                f"(需要至少 {min_free_gb}GB)"
            )
        
        if free_gb < min_free_gb * 1.5:
            logger.warning(
                f"磁盘空间不足: {path} 仅剩 {free_gb:.2f}GB "
                f"(建议至少 {min_free_gb * 1.5:.2f}GB)"
            )
        
        return info
    except Exception as e:
        logger.error(f"检查磁盘空间失败: {e}")
        raise


def get_system_resources() -> dict:
    """获取系统资源使用情况
    
    Returns:
        系统资源信息字典
    """
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        
        info = {
            'cpu_percent': cpu_percent,
            'memory_total_gb': memory.total / 1024**3,
            'memory_used_gb': memory.used / 1024**3,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / 1024**3
        }
        
        # 设备信息（NPU/GPU）
        device_info = monitor_gpu_memory()
        if device_info:
            info['device'] = device_info  # 统一使用device作为键名，兼容NPU/GPU
            info['gpu'] = device_info  # 保留gpu键名以保持向后兼容
        
        return info
    except Exception as e:
        logger.error(f"获取系统资源失败: {e}")
        return {}

