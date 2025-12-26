#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""日志工具模块"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str,
    log_dir: Optional[Path] = None,
    level: str = "INFO",
    console: bool = True,
    max_bytes: int = 100 * 1024 * 1024,  # 100MB
    backup_count: int = 10
) -> logging.Logger:
    """配置日志系统
    
    Args:
        name: 日志器名称
        log_dir: 日志文件目录，None 则不写入文件
        level: 日志级别 (DEBUG/INFO/WARNING/ERROR)
        console: 是否输出到控制台
        max_bytes: 单个日志文件最大大小
        backup_count: 保留的备份文件数量
    
    Returns:
        配置好的日志器
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()  # 清除已有处理器
    
    # 日志格式
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # 控制台处理器
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # 文件处理器
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
        
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def log_structured(logger: logging.Logger, level: str, event_type: str, **kwargs):
    """记录结构化日志
    
    Args:
        logger: 日志器
        level: 日志级别
        event_type: 事件类型
        **kwargs: 其他字段
    """
    import json
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'event_type': event_type,
        **kwargs
    }
    log_message = json.dumps(log_data, ensure_ascii=False)
    getattr(logger, level.lower())(log_message)

