#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""优雅关闭模块"""

import signal
import logging
import threading
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class GracefulShutdown:
    """优雅关闭管理器"""
    
    def __init__(self, on_shutdown: Optional[Callable] = None):
        """初始化优雅关闭管理器
        
        Args:
            on_shutdown: 关闭时的回调函数
        """
        self.shutdown_requested = False
        self.on_shutdown = on_shutdown
        self.lock = threading.Lock()
        
        # 注册信号处理器
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """信号处理器
        
        Args:
            signum: 信号编号
            frame: 当前栈帧
        """
        signal_name = signal.Signals(signum).name
        logger.info(f"收到关闭信号 {signal_name} ({signum})，开始优雅关闭...")
        
        with self.lock:
            self.shutdown_requested = True
        
        # 调用关闭回调
        if self.on_shutdown:
            try:
                self.on_shutdown()
            except Exception as e:
                logger.error(f"执行关闭回调失败: {e}", exc_info=True)
    
    def should_continue(self) -> bool:
        """检查是否应该继续运行
        
        Returns:
            是否应该继续运行
        """
        with self.lock:
            return not self.shutdown_requested
    
    def request_shutdown(self):
        """请求关闭"""
        with self.lock:
            self.shutdown_requested = True
        
        if self.on_shutdown:
            try:
                self.on_shutdown()
            except Exception as e:
                logger.error(f"执行关闭回调失败: {e}", exc_info=True)

