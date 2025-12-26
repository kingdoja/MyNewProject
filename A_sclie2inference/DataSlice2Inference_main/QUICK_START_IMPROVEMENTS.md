# 快速改进实施指南

本文档提供最关键的改进步骤，帮助快速将系统提升到生产可用状态。

## 🚀 第一步：日志系统改造（1-2天）

### 1.1 创建日志工具模块

创建文件：`DataSlice2Inference/utils/logger.py`

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""日志工具模块"""

import logging
import sys
from pathlib import Path
from logging.handlers import RotatingFileHandler
from datetime import datetime

def setup_logger(
    name: str,
    log_dir: Path = None,
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
```

### 1.2 在 auto_process_monitor.py 中使用

在文件开头添加：
```python
from utils.logger import setup_logger

# 初始化日志
logger = setup_logger(
    'auto_processor',
    log_dir=Path(__file__).parent.parent / 'logs',
    level='INFO'
)
```

然后替换所有 `print()` 为 `logger.info()` / `logger.warning()` / `logger.error()`

---

## 🔧 第二步：配置管理（1天）

### 2.1 创建配置文件

创建文件：`DataSlice2Inference/config.yaml`

```yaml
# RT-DETR 自动处理配置

# 路径配置（支持环境变量）
paths:
  watch_dir: "${WATCH_DIR:/home/ubuntu/lsn/project_new/RT-DETR-main/DataWSI}"
  output_dir: "${OUTPUT_DIR:/home/ubuntu/lsn/project_new/RT-DETR-main/DataPatchesInference}"
  model_path: "${MODEL_PATH:/home/ubuntu/lsn/project_new/RT-DETR-main/DataSlice2Inference/models/rtdetr_torchscript_cuda.pt}"
  log_dir: "${LOG_DIR:/home/ubuntu/lsn/project_new/RT-DETR-main/DataSlice2Inference/logs}"

# 处理参数
processing:
  patch_size: 640
  threshold: 0.5
  file_wait_timeout: 30  # 等待文件完全写入的秒数
  max_processing_time: 7200  # 单个文件最大处理时间（秒）

# 过滤参数
filtering:
  bg_rgb: [238, 235, 235]
  tolerance: 30
  bg_ratio: 0.9
  std_thresh: 10.0
  auto_bg: true
  bg_analysis_limit: 5

# 模型参数
model:
  device: "auto"  # auto/cpu/cuda/cuda:0

# 日志配置
logging:
  level: "INFO"
  console: true
  file_max_bytes: 104857600  # 100MB
  file_backup_count: 10

# 重试配置
retry:
  max_attempts: 3
  delay: 1.0
  backoff: 2.0
```

### 2.2 创建配置加载模块

创建文件：`DataSlice2Inference/utils/config.py`

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""配置管理模块"""

import os
import yaml
import re
from pathlib import Path
from typing import Any, Dict

class Config:
    def __init__(self, config_path: Path):
        """加载配置文件"""
        with open(config_path, 'r', encoding='utf-8') as f:
            self._raw_config = yaml.safe_load(f)
        
        self.config = self._resolve_env_vars(self._raw_config)
        self._validate()
    
    def _resolve_env_vars(self, config: Dict) -> Dict:
        """解析环境变量 ${VAR:default}"""
        if isinstance(config, dict):
            return {k: self._resolve_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._resolve_env_vars(item) for item in config]
        elif isinstance(config, str):
            # 匹配 ${VAR:default} 格式
            pattern = r'\$\{([^:]+):([^}]+)\}'
            match = re.match(pattern, config)
            if match:
                var_name, default_value = match.groups()
                return os.getenv(var_name, default_value)
            # 匹配 ${VAR} 格式（无默认值）
            pattern = r'\$\{([^}]+)\}'
            match = re.match(pattern, config)
            if match:
                var_name = match.group(1)
                value = os.getenv(var_name)
                if value is None:
                    raise ValueError(f"环境变量 {var_name} 未设置")
                return value
        return config
    
    def _validate(self):
        """验证配置"""
        # 检查必需路径
        watch_dir = Path(self.get('paths.watch_dir'))
        if not watch_dir.exists():
            raise FileNotFoundError(f"监听目录不存在: {watch_dir}")
        
        model_path = Path(self.get('paths.model_path'))
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """获取配置值，支持点号分隔的路径"""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
                if value is None:
                    return default
            else:
                return default
        return value
    
    def get_path(self, key_path: str) -> Path:
        """获取路径配置并转换为 Path 对象"""
        path_str = self.get(key_path)
        if path_str is None:
            raise ValueError(f"配置项不存在: {key_path}")
        return Path(path_str).expanduser().resolve()
```

### 2.3 修改 auto_process_monitor.py 使用配置

```python
from utils.config import Config

# 加载配置
config_path = Path(__file__).parent.parent / 'config.yaml'
config = Config(config_path)

# 使用配置
watch_dir = config.get_path('paths.watch_dir')
output_dir = config.get_path('paths.output_dir')
model_path = config.get_path('paths.model_path')
patch_size = config.get('processing.patch_size', 640)
threshold = config.get('processing.threshold', 0.5)
```

---

## 🛡️ 第三步：错误处理和重试（1天）

### 3.1 创建重试工具

创建文件：`DataSlice2Inference/utils/retry.py`

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""重试工具模块"""

from functools import wraps
import time
from typing import Callable, Type, Tuple, Any
import logging

logger = logging.getLogger(__name__)

def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,),
    on_retry: Callable = None
):
    """重试装饰器
    
    Args:
        max_attempts: 最大尝试次数
        delay: 初始延迟（秒）
        backoff: 延迟倍数
        exceptions: 需要重试的异常类型
        on_retry: 重试时的回调函数
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
                        on_retry(attempt, e)
                    
                    time.sleep(current_delay)
                    current_delay *= backoff
            
            # 理论上不会到达这里
            raise RuntimeError(f"{func.__name__} 重试失败")
        
        return wrapper
    return decorator
```

### 3.2 在关键函数上使用

```python
from utils.retry import retry

@retry(max_attempts=3, delay=2.0, exceptions=(RuntimeError, FileNotFoundError))
def load_model_with_retry(model_path: str, device):
    """带重试的模型加载"""
    return torch.jit.load(model_path, map_location=device)
```

---

## 📊 第四步：基础监控（1-2天）

### 4.1 添加处理统计

在 `ImageProcessHandler` 类中添加：

```python
class ImageProcessHandler(FileSystemEventHandler):
    def __init__(self, ...):
        # ... 现有代码 ...
        
        # 统计信息
        self.stats = {
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'total_patches': 0,
            'total_detections': 0,
            'start_time': datetime.now()
        }
    
    def process_image(self, image_path: Path, file_hash: str):
        """处理单张图片"""
        self.stats['total_files'] += 1
        start_time = time.time()
        
        try:
            # ... 现有处理逻辑 ...
            self.stats['successful'] += 1
            logger.info(
                f"处理完成: {image_path.name}, "
                f"耗时: {time.time() - start_time:.2f}秒"
            )
        except Exception as e:
            self.stats['failed'] += 1
            logger.error(f"处理失败: {image_path.name}, 错误: {e}", exc_info=True)
            raise
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        uptime = (datetime.now() - self.stats['start_time']).total_seconds()
        return {
            **self.stats,
            'uptime_seconds': uptime,
            'success_rate': (
                self.stats['successful'] / self.stats['total_files']
                if self.stats['total_files'] > 0 else 0
            )
        }
```

### 4.2 定期输出统计信息

```python
def print_stats_periodically(handler: ImageProcessHandler, interval: int = 300):
    """定期输出统计信息"""
    while True:
        time.sleep(interval)
        stats = handler.get_stats()
        logger.info(f"统计信息: {stats}")
```

---

## 🔄 第五步：优雅关闭（半天）

### 5.1 添加信号处理

```python
import signal

class GracefulShutdown:
    def __init__(self):
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        logger.info(f"收到关闭信号 {signum}，开始优雅关闭...")
        self.shutdown_requested = True
    
    def should_continue(self):
        return not self.shutdown_requested

# 在 main() 中使用
def main():
    # ... 现有代码 ...
    
    shutdown = GracefulShutdown()
    observer.start()
    
    try:
        while shutdown.should_continue():
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("正在停止服务...")
        observer.stop()
        observer.join()
        
        # 输出最终统计
        stats = handler.get_stats()
        logger.info(f"最终统计: {stats}")
        logger.info("服务已停止")
```

---

## 📦 第六步：创建依赖文件

创建文件：`DataSlice2Inference/requirements.txt`

```
watchdog>=3.0.0
pillow>=10.0.0
numpy>=1.24.0
torch>=2.0.0
torchvision>=0.15.0
tqdm>=4.65.0
pyyaml>=6.0
psutil>=5.9.0
```

安装依赖：
```bash
pip install -r requirements.txt
```

---

## 🚀 第七步：创建 systemd 服务（可选但推荐）

创建文件：`/etc/systemd/system/rtdetr-processor.service`

```ini
[Unit]
Description=RT-DETR Auto Processor Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/lsn/project_new/RT-DETR-main/DataSlice2Inference/auto_process_package
ExecStart=/usr/bin/python3 /home/ubuntu/lsn/project_new/RT-DETR-main/DataSlice2Inference/auto_process_package/auto_process_monitor.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# 资源限制
LimitNOFILE=65536
MemoryMax=16G

[Install]
WantedBy=multi-user.target
```

启用服务：
```bash
sudo systemctl daemon-reload
sudo systemctl enable rtdetr-processor
sudo systemctl start rtdetr-processor
sudo systemctl status rtdetr-processor
```

---

## ✅ 检查清单

完成以上步骤后，检查以下项目：

- [ ] 日志系统已替换所有 print
- [ ] 配置文件已创建并生效
- [ ] 关键操作已添加重试机制
- [ ] 统计信息正常输出
- [ ] 优雅关闭功能正常
- [ ] systemd 服务（如使用）正常运行
- [ ] 日志文件正常生成和轮转

---

## 📝 后续优化建议

完成基础改进后，可以考虑：

1. **健康检查接口**：添加简单的 HTTP 接口查看状态
2. **并发处理**：支持同时处理多个文件
3. **通知机制**：处理完成/失败时发送通知
4. **性能监控**：集成 Prometheus 或类似工具
5. **单元测试**：为关键函数添加测试

详细说明请参考 `PRODUCTION_IMPROVEMENTS.md`

