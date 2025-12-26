#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重试工具模块"""

from functools import wraps
import time
from typing import Callable, Type, Tuple, Any, Optional
import logging

logger = logging.getLogger(__name__)


def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Optional[Callable] = None
):
    """重试装饰器
    
    Args:
        max_attempts: 最大尝试次数
        delay: 初始延迟（秒）
        backoff: 延迟倍数
        exceptions: 需要重试的异常类型
        on_retry: 重试时的回调函数，接收 (attempt, exception) 参数
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            attempt = 0
            current_delay = delay
            
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        logger.error(
                            f"{func.__name__} 失败，已重试 {max_attempts} 次: {e}",
                            exc_info=True
                        )
                        raise
                    
                    logger.warning(
                        f"{func.__name__} 失败 (尝试 {attempt}/{max_attempts}): {e}, "
                        f"{current_delay:.1f}秒后重试"
                    )
                    
                    if on_retry:
                        try:
                            on_retry(attempt, e)
                        except Exception:
                            pass
                    
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            # 理论上不会到达这里
            raise RuntimeError(f"{func.__name__} 重试失败")
        
        return wrapper
    return decorator

