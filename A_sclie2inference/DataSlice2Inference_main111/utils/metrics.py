#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""性能指标收集模块"""

import time
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional
import logging

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class ProcessingMetrics:
    """处理指标数据类"""
    timestamp: datetime = field(default_factory=datetime.now)
    total_files_processed: int = 0
    total_files_successful: int = 0
    total_files_failed: int = 0
    total_patches_created: int = 0
    total_patches_filtered: int = 0
    total_patches_kept: int = 0
    total_detections: int = 0
    avg_processing_time: float = 0.0
    total_processing_time: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    memory_usage_mb: float = 0.0
    gpu_usage: float = 0.0
    gpu_memory_usage: float = 0.0
    disk_usage: float = 0.0
    disk_free_gb: float = 0.0
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'files': {
                'total_processed': self.total_files_processed,
                'successful': self.total_files_successful,
                'failed': self.total_files_failed,
                'success_rate': (
                    self.total_files_successful / self.total_files_processed
                    if self.total_files_processed > 0 else 0.0
                )
            },
            'patches': {
                'created': self.total_patches_created,
                'filtered': self.total_patches_filtered,
                'kept': self.total_patches_kept,
                'keep_rate': (
                    self.total_patches_kept / self.total_patches_created
                    if self.total_patches_created > 0 else 0.0
                )
            },
            'detections': {
                'total': self.total_detections
            },
            'performance': {
                'avg_processing_time': self.avg_processing_time,
                'total_processing_time': self.total_processing_time
            },
            'system': {
                'cpu_usage': self.cpu_usage,
                'memory_usage': self.memory_usage,
                'memory_usage_mb': self.memory_usage_mb,
                'gpu_usage': self.gpu_usage,
                'gpu_memory_usage': self.gpu_memory_usage,
                'disk_usage': self.disk_usage,
                'disk_free_gb': self.disk_free_gb
            }
        }


class MetricsCollector:
    """指标收集器"""
    
    def __init__(self):
        self.metrics = ProcessingMetrics()
        self.lock = threading.Lock()
        self.start_time = datetime.now()
        self.processing_times = []  # 存储每次处理的时间
    
    def record_file_processed(self, successful: bool = True, patches_created: int = 0, 
                             patches_kept: int = 0, detections: int = 0, 
                             processing_time: float = 0.0):
        """记录文件处理结果
        
        Args:
            successful: 是否成功
            patches_created: 创建的 patch 数量
            patches_kept: 保留的 patch 数量
            detections: 检测数量
            processing_time: 处理时间（秒）
        """
        with self.lock:
            self.metrics.total_files_processed += 1
            if successful:
                self.metrics.total_files_successful += 1
            else:
                self.metrics.total_files_failed += 1
            
            self.metrics.total_patches_created += patches_created
            self.metrics.total_patches_kept += patches_kept
            self.metrics.total_patches_filtered = patches_created - patches_kept
            self.metrics.total_detections += detections
            self.metrics.total_processing_time += processing_time
            
            if processing_time > 0:
                self.processing_times.append(processing_time)
                self.metrics.avg_processing_time = sum(self.processing_times) / len(self.processing_times)
    
    def update_system_metrics(self, disk_path: Optional[Path] = None):
        """更新系统指标
        
        Args:
            disk_path: 检查磁盘空间的路径
        """
        with self.lock:
            self.metrics.timestamp = datetime.now()
            
            if PSUTIL_AVAILABLE:
                try:
                    self.metrics.cpu_usage = psutil.cpu_percent(interval=0.1)
                    memory = psutil.virtual_memory()
                    self.metrics.memory_usage = memory.percent
                    self.metrics.memory_usage_mb = memory.used / 1024 / 1024
                    
                    if disk_path:
                        disk = psutil.disk_usage(str(disk_path))
                        self.metrics.disk_usage = (disk.used / disk.total) * 100
                        self.metrics.disk_free_gb = disk.free / 1024 / 1024 / 1024
                except Exception as e:
                    logger.warning(f"更新系统指标失败: {e}")
            
            # 监控NPU/GPU内存
            if TORCH_AVAILABLE:
                try:
                    # 尝试导入NPU工具
                    from npu_utils import is_npu_available, get_device_memory_info
                    import torch
                    
                    device = None
                    if is_npu_available():
                        device = torch.device("npu")
                    elif torch.cuda.is_available():
                        device = torch.device("cuda")
                    
                    if device:
                        self.metrics.gpu_usage = 0.0  # 需要额外的库获取设备使用率
                        mem_info = get_device_memory_info(device)
                        self.metrics.gpu_memory_usage = mem_info.get('allocated_gb', 0.0)
                except (ImportError, Exception):
                    # 回退到原始CUDA逻辑
                    try:
                        if torch.cuda.is_available():
                            self.metrics.gpu_usage = 0.0
                            self.metrics.gpu_memory_usage = torch.cuda.memory_allocated() / 1024**3
                    except Exception:
                        pass
    
    def get_metrics(self) -> ProcessingMetrics:
        """获取当前指标
        
        Returns:
            当前指标对象
        """
        with self.lock:
            return ProcessingMetrics(**self.metrics.__dict__)
    
    def get_uptime(self) -> float:
        """获取运行时间（秒）
        
        Returns:
            运行时间（秒）
        """
        return (datetime.now() - self.start_time).total_seconds()
    
    def reset(self):
        """重置指标（保留运行时间）"""
        with self.lock:
            self.metrics = ProcessingMetrics()
            self.processing_times = []
            self.start_time = datetime.now()

