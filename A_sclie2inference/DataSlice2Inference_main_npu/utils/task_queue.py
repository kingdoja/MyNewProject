#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""任务队列模块"""

import time
import logging
from queue import Queue
from threading import Thread
from typing import Callable, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class ProcessingQueue:
    """处理任务队列"""
    
    def __init__(self, max_workers: int = 2, worker_func: Optional[Callable] = None):
        """初始化任务队列
        
        Args:
            max_workers: 最大工作线程数
            worker_func: 工作函数，接收 task 参数
        """
        self.queue = Queue()
        self.max_workers = max_workers
        self.worker_func = worker_func
        self.workers = []
        self.running = False
        self.processed_count = 0
        self.failed_count = 0
    
    def start(self):
        """启动工作线程"""
        if self.running:
            logger.warning("任务队列已在运行")
            return
        
        if not self.worker_func:
            raise ValueError("工作函数未设置")
        
        self.running = True
        for i in range(self.max_workers):
            worker = Thread(target=self._worker, daemon=True, name=f"Worker-{i+1}")
            worker.start()
            self.workers.append(worker)
        
        logger.info(f"任务队列已启动，工作线程数: {self.max_workers}")
    
    def stop(self, wait: bool = True):
        """停止工作线程
        
        Args:
            wait: 是否等待队列中的所有任务完成
        """
        if not self.running:
            return
        
        logger.info("正在停止任务队列...")
        self.running = False
        
        # 向队列发送停止信号
        for _ in range(self.max_workers):
            self.queue.put(None)
        
        if wait:
            # 等待所有任务完成
            self.queue.join()
            # 等待所有工作线程退出
            for worker in self.workers:
                worker.join(timeout=5)
        
        logger.info(f"任务队列已停止，处理: {self.processed_count}, 失败: {self.failed_count}")
    
    def _worker(self):
        """工作线程主函数"""
        while self.running:
            try:
                task = self.queue.get(timeout=1)
                
                if task is None:
                    break
                
                try:
                    self.worker_func(task)
                    self.processed_count += 1
                    logger.debug(f"任务处理成功: {task}")
                except Exception as e:
                    self.failed_count += 1
                    logger.error(f"处理任务失败: {task}, 错误: {e}", exc_info=True)
                finally:
                    self.queue.task_done()
                    
            except Exception as e:
                if self.running:
                    logger.error(f"工作线程异常: {e}", exc_info=True)
                    time.sleep(1)
    
    def add_task(self, task: Any):
        """添加任务到队列
        
        Args:
            task: 任务对象
        """
        if not self.running:
            logger.warning("任务队列未运行，无法添加任务")
            return
        
        self.queue.put(task)
        logger.debug(f"任务已添加到队列: {task}")
    
    def get_queue_size(self) -> int:
        """获取队列大小
        
        Returns:
            队列中待处理的任务数
        """
        return self.queue.qsize()
    
    def wait_for_completion(self, timeout: Optional[float] = None):
        """等待所有任务完成
        
        Args:
            timeout: 超时时间（秒），None 表示无限等待
        """
        start_time = time.time()
        while self.queue.qsize() > 0:
            if timeout and (time.time() - start_time) > timeout:
                raise TimeoutError(f"等待任务完成超时: {timeout}秒")
            time.sleep(0.1)
        self.queue.join()

