#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""NPU设备工具模块 - 用于华为昇腾910B NPU支持"""

import logging

logger = logging.getLogger(__name__)

# 尝试导入torch_npu
try:
    import torch
    import torch_npu
    NPU_AVAILABLE = True
except ImportError:
    NPU_AVAILABLE = False
    logger.warning("torch_npu未安装，NPU功能不可用。请安装适配华为910B的torch_npu。")


def is_npu_available() -> bool:
    """检查NPU是否可用
    
    Returns:
        bool: NPU是否可用
    """
    if not NPU_AVAILABLE:
        return False
    try:
        return torch.npu.is_available()
    except Exception as e:
        logger.warning(f"检查NPU可用性失败: {e}")
        return False


def resolve_device(device_str: str):
    """解析设备字符串，自动检测NPU/GPU/CPU
    
    Args:
        device_str: 设备字符串，支持: auto/npu/cuda/cpu/npu:0等
        
    Returns:
        torch.device: 设备对象
    """
    if not NPU_AVAILABLE:
        # 如果没有torch_npu，回退到原始逻辑
        import torch
        device_str = device_str.lower()
        if device_str == "auto":
            if torch.cuda.is_available():
                logger.info("⚡ 检测到可用 GPU，使用 CUDA 进行推理")
                return torch.device("cuda")
            logger.info("⚠️ 未检测到 GPU，使用 CPU 推理")
            return torch.device("cpu")
        return torch.device(device_str)
    
    import torch
    device_str = device_str.lower()
    
    if device_str == "auto":
        # 优先使用NPU
        if is_npu_available():
            logger.info("⚡ 检测到可用 NPU，使用 NPU 进行推理")
            return torch.device("npu")
        elif torch.cuda.is_available():
            logger.info("⚡ 检测到可用 GPU，使用 CUDA 进行推理")
            return torch.device("cuda")
        else:
            logger.info("⚠️ 未检测到 NPU/GPU，使用 CPU 推理")
            return torch.device("cpu")
    
    # 显式指定设备
    return torch.device(device_str)


def get_device_memory_info(device):
    """获取设备内存信息
    
    Args:
        device: torch.device对象
        
    Returns:
        dict: 内存信息字典，包含allocated_gb, reserved_gb等
    """
    if not NPU_AVAILABLE:
        return {}
    
    import torch
    
    try:
        if device.type == "npu":
            allocated = torch.npu.memory_allocated(device) / 1024**3
            reserved = torch.npu.memory_reserved(device) / 1024**3
            max_allocated = torch.npu.max_memory_allocated(device) / 1024**3
            
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'max_allocated_gb': max_allocated
            }
        elif device.type == "cuda":
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            reserved = torch.cuda.memory_reserved(device) / 1024**3
            max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
            
            return {
                'allocated_gb': allocated,
                'reserved_gb': reserved,
                'max_allocated_gb': max_allocated
            }
    except Exception as e:
        logger.warning(f"获取设备内存信息失败: {e}")
        return {}
    
    return {}


def clear_device_cache(device):
    """清理设备缓存
    
    Args:
        device: torch.device对象
    """
    if not NPU_AVAILABLE:
        return
    
    import torch
    
    try:
        if device.type == "npu":
            torch.npu.empty_cache()
            torch.npu.synchronize()
            logger.debug("NPU 缓存已清理")
        elif device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            logger.debug("GPU 缓存已清理")
    except Exception as e:
        logger.warning(f"清理设备缓存失败: {e}")


def supports_fp16(device) -> bool:
    """检查设备是否支持FP16
    
    Args:
        device: torch.device对象
        
    Returns:
        bool: 是否支持FP16
    """
    if not NPU_AVAILABLE:
        return False
    
    # NPU和GPU都支持FP16
    return device.type in ["npu", "cuda"]
