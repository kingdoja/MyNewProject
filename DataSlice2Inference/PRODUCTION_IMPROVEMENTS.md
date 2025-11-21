# DataSlice2Inference 生产环境改进建议

本文档列出了将自动监听切片和推理模块用于生产环境落地时需要的修改和维护工作。

## 📋 目录

1. [日志系统改进](#1-日志系统改进)
2. [错误处理和重试机制](#2-错误处理和重试机制)
3. [配置管理](#3-配置管理)
4. [监控和健康检查](#4-监控和健康检查)
5. [进程管理和服务化](#5-进程管理和服务化)
6. [资源管理和优化](#6-资源管理和优化)
7. [数据完整性和备份](#7-数据完整性和备份)
8. [并发处理优化](#8-并发处理优化)
9. [通知和告警机制](#9-通知和告警机制)
10. [安全和权限管理](#10-安全和权限管理)
11. [测试和验证](#11-测试和验证)
12. [文档和部署](#12-文档和部署)

---

## 1. 日志系统改进

### 当前问题
- ❌ 全部使用 `print()` 输出，无法控制日志级别
- ❌ 日志无法持久化到文件
- ❌ 缺少日志轮转和清理机制
- ❌ 无法区分不同模块的日志

### 改进建议

#### 1.1 引入标准 logging 模块
```python
import logging
from logging.handlers import RotatingFileHandler
import sys

def setup_logger(name: str, log_file: Path, level=logging.INFO):
    """配置日志系统"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # 文件处理器（带轮转）
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=100*1024*1024,  # 100MB
        backupCount=10,
        encoding='utf-8'
    )
    file_handler.setFormatter(
        logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(
        logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    )
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger
```

#### 1.2 替换所有 print 语句
- 使用 `logger.info()` 替代普通信息输出
- 使用 `logger.warning()` 替代警告
- 使用 `logger.error()` 替代错误
- 使用 `logger.debug()` 替代调试信息

#### 1.3 结构化日志
考虑使用 JSON 格式日志，便于后续分析和监控：
```python
import json
from datetime import datetime

def log_structured(logger, level, event_type, **kwargs):
    """记录结构化日志"""
    log_data = {
        'timestamp': datetime.now().isoformat(),
        'event_type': event_type,
        **kwargs
    }
    getattr(logger, level)(json.dumps(log_data, ensure_ascii=False))
```

---

## 2. 错误处理和重试机制

### 当前问题
- ❌ 错误处理不够完善，缺少重试机制
- ❌ 文件写入失败时没有重试
- ❌ 模型加载失败时没有重试
- ❌ 子进程执行失败时缺少详细错误信息

### 改进建议

#### 2.1 添加重试装饰器
```python
from functools import wraps
import time
from typing import Callable, Type, Tuple

def retry(
    max_attempts: int = 3,
    delay: float = 1.0,
    backoff: float = 2.0,
    exceptions: Tuple[Type[Exception], ...] = (Exception,)
):
    """重试装饰器"""
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempt = 0
            current_delay = delay
            while attempt < max_attempts:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    attempt += 1
                    if attempt >= max_attempts:
                        raise
                    logger.warning(
                        f"{func.__name__} 失败 (尝试 {attempt}/{max_attempts}): {e}, "
                        f"{current_delay}秒后重试"
                    )
                    time.sleep(current_delay)
                    current_delay *= backoff
        return wrapper
    return decorator
```

#### 2.2 改进文件处理错误处理
- 文件写入时添加文件锁，避免并发写入冲突
- 大文件处理时添加进度检查点，支持断点续传
- 添加文件完整性校验（MD5/SHA256）

#### 2.3 模型加载错误处理
```python
@retry(max_attempts=3, delay=2.0, exceptions=(RuntimeError, FileNotFoundError))
def load_model_with_retry(model_path: str, device: torch.device):
    """带重试的模型加载"""
    # 检查模型文件完整性
    if not verify_model_file(model_path):
        raise FileNotFoundError(f"模型文件损坏或不完整: {model_path}")
    return torch.jit.load(model_path, map_location=device)
```

---

## 3. 配置管理

### 当前问题
- ❌ 硬编码路径较多（如 `/home/ubuntu/lsn/project_new/RT-DETR-main/`）
- ❌ 配置分散在代码中，难以维护
- ❌ 缺少环境变量支持
- ❌ 缺少配置验证

### 改进建议

#### 3.1 创建配置文件（YAML/JSON）
```yaml
# config.yaml
paths:
  watch_dir: "${WATCH_DIR:/home/ubuntu/lsn/project_new/RT-DETR-main/DataWSI}"
  output_dir: "${OUTPUT_DIR:/home/ubuntu/lsn/project_new/RT-DETR-main/DataPatchesInference}"
  model_path: "${MODEL_PATH:/home/ubuntu/lsn/project_new/RT-DETR-main/DataSlice2Inference/models/rtdetr_torchscript_cuda.pt}"
  log_dir: "${LOG_DIR:/var/log/rtdetr}"

processing:
  patch_size: 640
  threshold: 0.5
  max_concurrent_tasks: 2
  file_wait_timeout: 30  # 等待文件完全写入的秒数

filtering:
  bg_rgb: [238, 235, 235]
  tolerance: 30
  bg_ratio: 0.9
  std_thresh: 10.0
  auto_bg: true
  bg_analysis_limit: 5

model:
  device: "auto"  # auto/cpu/cuda/cuda:0
  batch_size: 1
  num_workers: 4

logging:
  level: "INFO"  # DEBUG/INFO/WARNING/ERROR
  file_max_bytes: 104857600  # 100MB
  file_backup_count: 10

retry:
  max_attempts: 3
  delay: 1.0
  backoff: 2.0
```

#### 3.2 配置加载类
```python
import yaml
import os
from pathlib import Path
from typing import Any, Dict

class Config:
    def __init__(self, config_path: Path):
        with open(config_path, 'r', encoding='utf-8') as f:
            raw_config = yaml.safe_load(f)
        self.config = self._resolve_env_vars(raw_config)
        self._validate()
    
    def _resolve_env_vars(self, config: Dict) -> Dict:
        """解析环境变量 ${VAR:default}"""
        # 实现环境变量替换逻辑
        pass
    
    def _validate(self):
        """验证配置有效性"""
        # 检查路径是否存在、参数范围等
        pass
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """获取配置值，支持点号分隔的路径"""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            value = value.get(key)
            if value is None:
                return default
        return value
```

---

## 4. 监控和健康检查

### 当前问题
- ❌ 缺少系统监控指标
- ❌ 无法了解处理进度和性能
- ❌ 缺少健康检查接口
- ❌ 无法追踪资源使用情况

### 改进建议

#### 4.1 添加性能指标收集
```python
from dataclasses import dataclass
from datetime import datetime
import psutil
import torch

@dataclass
class ProcessingMetrics:
    """处理指标"""
    timestamp: datetime
    total_files_processed: int
    total_patches_created: int
    total_patches_filtered: int
    total_detections: int
    avg_processing_time: float
    cpu_usage: float
    memory_usage: float
    gpu_usage: float
    gpu_memory_usage: float
    disk_usage: float
    
    def to_dict(self):
        return {
            'timestamp': self.timestamp.isoformat(),
            'total_files_processed': self.total_files_processed,
            'total_patches_created': self.total_patches_created,
            'total_patches_filtered': self.total_patches_filtered,
            'total_detections': self.total_detections,
            'avg_processing_time': self.avg_processing_time,
            'system': {
                'cpu_usage': self.cpu_usage,
                'memory_usage': self.memory_usage,
                'gpu_usage': self.gpu_usage,
                'gpu_memory_usage': self.gpu_memory_usage,
                'disk_usage': self.disk_usage,
            }
        }
```

#### 4.2 健康检查端点
```python
from flask import Flask, jsonify
import threading

app = Flask(__name__)

@app.route('/health')
def health_check():
    """健康检查接口"""
    status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'uptime': get_uptime(),
        'processing_queue': get_queue_size(),
        'system': get_system_status()
    }
    return jsonify(status)

@app.route('/metrics')
def metrics():
    """性能指标接口"""
    return jsonify(get_current_metrics().to_dict())

def start_monitoring_server(port=8080):
    """启动监控服务器"""
    threading.Thread(
        target=lambda: app.run(host='0.0.0.0', port=port),
        daemon=True
    ).start()
```

#### 4.3 集成 Prometheus 指标（可选）
如果使用 Prometheus 监控，可以添加指标导出：
```python
from prometheus_client import Counter, Histogram, Gauge

files_processed = Counter('files_processed_total', 'Total files processed')
processing_time = Histogram('processing_time_seconds', 'Processing time')
queue_size = Gauge('processing_queue_size', 'Current queue size')
```

---

## 5. 进程管理和服务化

### 当前问题
- ❌ 缺少守护进程支持
- ❌ 进程崩溃后无法自动重启
- ❌ 缺少 systemd 服务配置
- ❌ 无法优雅关闭

### 改进建议

#### 5.1 添加信号处理
```python
import signal
import sys

class GracefulShutdown:
    def __init__(self):
        self.shutdown_requested = False
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        logger.info(f"收到信号 {signum}，开始优雅关闭...")
        self.shutdown_requested = True
    
    def should_continue(self):
        return not self.shutdown_requested
```

#### 5.2 创建 systemd 服务文件
```ini
# /etc/systemd/system/rtdetr-processor.service
[Unit]
Description=RT-DETR Auto Processor Service
After=network.target

[Service]
Type=simple
User=ubuntu
WorkingDirectory=/home/ubuntu/lsn/project_new/RT-DETR-main/DataSlice2Inference/auto_process_package
ExecStart=/usr/bin/python3 /home/ubuntu/lsn/project_new/RT-DETR-main/DataSlice2Inference/auto_process_package/auto_process_monitor.py --config /etc/rtdetr/config.yaml
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
```

#### 5.3 使用 supervisor 或 systemd 管理
- 配置自动重启策略
- 配置日志输出到系统日志
- 配置资源限制（CPU、内存）

---

## 6. 资源管理和优化

### 当前问题
- ❌ 缺少内存使用监控
- ❌ 大文件处理可能导致内存溢出
- ❌ GPU 内存未优化
- ❌ 磁盘空间未监控

### 改进建议

#### 6.1 内存监控和限制
```python
import psutil
import resource

def set_memory_limit(max_memory_gb: float):
    """设置内存限制"""
    max_bytes = int(max_memory_gb * 1024 * 1024 * 1024)
    resource.setrlimit(resource.RLIMIT_AS, (max_bytes, max_bytes))

def check_memory_usage():
    """检查内存使用"""
    process = psutil.Process()
    memory_info = process.memory_info()
    memory_percent = process.memory_percent()
    
    if memory_percent > 90:
        logger.warning(f"内存使用率过高: {memory_percent:.1f}%")
        # 触发清理或告警
```

#### 6.2 GPU 内存管理
```python
def clear_gpu_cache():
    """清理 GPU 缓存"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def monitor_gpu_memory():
    """监控 GPU 内存"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"GPU 内存: 已分配 {allocated:.2f}GB, 已保留 {reserved:.2f}GB")
```

#### 6.3 磁盘空间检查
```python
def check_disk_space(path: Path, min_free_gb: float = 10.0):
    """检查磁盘空间"""
    stat = shutil.disk_usage(path)
    free_gb = stat.free / (1024**3)
    
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"磁盘空间不足: {path} 仅剩 {free_gb:.2f}GB "
            f"(需要至少 {min_free_gb}GB)"
        )
```

---

## 7. 数据完整性和备份

### 当前问题
- ❌ 缺少数据校验
- ❌ 处理失败时数据可能不完整
- ❌ 缺少备份机制
- ❌ 状态文件可能损坏

### 改进建议

#### 7.1 添加数据校验
```python
import hashlib

def calculate_file_hash(file_path: Path, algorithm='md5') -> str:
    """计算文件哈希值"""
    hash_obj = hashlib.new(algorithm)
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()

def verify_processing_completeness(output_dir: Path, expected_files: int):
    """验证处理完整性"""
    actual_files = len(list(output_dir.glob("*.json")))
    if actual_files != expected_files:
        raise RuntimeError(
            f"处理不完整: 期望 {expected_files} 个文件，实际 {actual_files} 个"
        )
```

#### 7.2 事务性处理
```python
from contextlib import contextmanager
import tempfile
import shutil

@contextmanager
def atomic_output(output_dir: Path):
    """原子性输出，失败时回滚"""
    temp_dir = output_dir.parent / f"{output_dir.name}.tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        yield temp_dir
        # 处理成功，移动到最终位置
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.move(str(temp_dir), str(output_dir))
    except Exception:
        # 处理失败，清理临时目录
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
        raise
```

#### 7.3 状态文件备份
```python
def backup_status_file(status_file: Path):
    """备份状态文件"""
    if status_file.exists():
        backup_file = status_file.with_suffix('.json.bak')
        shutil.copy2(status_file, backup_file)
        # 只保留最近 5 个备份
        backups = sorted(status_file.parent.glob('*.json.bak'))
        for old_backup in backups[:-5]:
            old_backup.unlink()
```

---

## 8. 并发处理优化

### 当前问题
- ❌ 单线程处理，效率较低
- ❌ 无法同时处理多个文件
- ❌ 缺少任务队列

### 改进建议

#### 8.1 添加任务队列
```python
from queue import Queue
from threading import Thread
import time

class ProcessingQueue:
    def __init__(self, max_workers: int = 2):
        self.queue = Queue()
        self.max_workers = max_workers
        self.workers = []
        self.running = False
    
    def start(self):
        """启动工作线程"""
        self.running = True
        for i in range(self.max_workers):
            worker = Thread(target=self._worker, daemon=True)
            worker.start()
            self.workers.append(worker)
    
    def _worker(self):
        """工作线程"""
        while self.running:
            try:
                task = self.queue.get(timeout=1)
                if task is None:
                    break
                self._process_task(task)
                self.queue.task_done()
            except Exception as e:
                logger.error(f"处理任务失败: {e}", exc_info=True)
    
    def add_task(self, task):
        """添加任务"""
        self.queue.put(task)
    
    def _process_task(self, task):
        """处理单个任务"""
        # 实现任务处理逻辑
        pass
```

#### 8.2 使用进程池处理 CPU 密集型任务
```python
from multiprocessing import Pool, cpu_count

def process_image_parallel(image_paths: List[Path], num_workers: int = None):
    """并行处理多个图像"""
    if num_workers is None:
        num_workers = min(cpu_count(), 4)  # 限制最大进程数
    
    with Pool(num_workers) as pool:
        results = pool.map(process_single_image, image_paths)
    return results
```

---

## 9. 通知和告警机制

### 当前问题
- ❌ 处理完成/失败时无通知
- ❌ 系统异常时无告警
- ❌ 无法及时了解系统状态

### 改进建议

#### 9.1 邮件通知
```python
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

def send_notification(subject: str, message: str, recipients: List[str]):
    """发送邮件通知"""
    msg = MIMEMultipart()
    msg['From'] = config.get('notification.email.from')
    msg['To'] = ', '.join(recipients)
    msg['Subject'] = subject
    msg.attach(MIMEText(message, 'plain', 'utf-8'))
    
    with smtplib.SMTP(config.get('notification.email.smtp_host')) as server:
        server.send_message(msg)
```

#### 9.2 Webhook 通知
```python
import requests

def send_webhook(url: str, data: dict):
    """发送 Webhook 通知"""
    try:
        response = requests.post(url, json=data, timeout=5)
        response.raise_for_status()
    except Exception as e:
        logger.error(f"发送 Webhook 失败: {e}")
```

#### 9.3 告警规则
- 处理失败率超过阈值
- 队列积压超过阈值
- 系统资源使用率过高
- 磁盘空间不足

---

## 10. 安全和权限管理

### 当前问题
- ❌ 文件路径未验证，可能存在路径遍历风险
- ❌ 缺少输入文件类型验证
- ❌ 缺少权限检查

### 改进建议

#### 10.1 路径验证
```python
def validate_path(path: Path, base_dir: Path) -> Path:
    """验证路径安全性"""
    resolved = path.resolve()
    base_resolved = base_dir.resolve()
    
    if not str(resolved).startswith(str(base_resolved)):
        raise ValueError(f"路径超出允许范围: {path}")
    
    return resolved
```

#### 10.2 文件类型验证
```python
import magic

def validate_image_file(file_path: Path) -> bool:
    """验证文件类型"""
    mime = magic.Magic(mime=True)
    file_type = mime.from_file(str(file_path))
    
    allowed_types = ['image/jpeg', 'image/png', 'image/tiff']
    return file_type in allowed_types
```

#### 10.3 权限检查
```python
def check_permissions(path: Path, required_perms: str = 'rw'):
    """检查文件权限"""
    if 'r' in required_perms and not os.access(path, os.R_OK):
        raise PermissionError(f"无读取权限: {path}")
    if 'w' in required_perms and not os.access(path, os.W_OK):
        raise PermissionError(f"无写入权限: {path}")
```

---

## 11. 测试和验证

### 改进建议

#### 11.1 单元测试
- 为关键函数添加单元测试
- 使用 pytest 框架
- 测试覆盖率目标：>80%

#### 11.2 集成测试
- 端到端测试流程
- 模拟各种异常情况
- 性能基准测试

#### 11.3 压力测试
- 大量文件并发处理
- 大文件处理测试
- 长时间运行稳定性测试

---

## 12. 文档和部署

### 改进建议

#### 12.1 部署文档
- 环境准备步骤
- 依赖安装指南
- 配置说明
- 启动和停止方法

#### 12.2 运维文档
- 常见问题排查
- 日志查看方法
- 性能调优指南
- 故障恢复流程

#### 12.3 依赖管理
创建 `requirements.txt`：
```
watchdog>=3.0.0
pillow>=10.0.0
numpy>=1.24.0
torch>=2.0.0
torchvision>=0.15.0
tqdm>=4.65.0
pyyaml>=6.0
psutil>=5.9.0
flask>=2.3.0
requests>=2.31.0
```

---

## 优先级建议

### 🔴 高优先级（必须）
1. 日志系统改进（替换 print）
2. 错误处理和重试机制
3. 配置管理（移除硬编码）
4. 进程管理和服务化（systemd）
5. 资源监控（内存、磁盘）

### 🟡 中优先级（重要）
6. 数据完整性校验
7. 并发处理优化
8. 健康检查接口
9. 通知和告警机制

### 🟢 低优先级（可选）
10. Prometheus 集成
11. 单元测试
12. 安全加固

---

## 实施计划

1. **第一阶段（1-2周）**：日志系统、配置管理、基础错误处理
2. **第二阶段（2-3周）**：服务化、监控、资源管理
3. **第三阶段（1-2周）**：并发优化、通知机制、测试
4. **第四阶段（持续）**：性能调优、文档完善、安全加固

---

## 总结

当前代码已经实现了核心功能，但在生产环境落地前需要进行以上改进。建议按照优先级逐步实施，确保系统的稳定性、可维护性和可观测性。

