#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
RT-DETR 自动切片和推理服务 - 主服务入口

这是一个完整的服务包装，提供：
1. 文件监听和自动处理
2. 健康检查接口
3. 优雅关闭
4. 日志记录
5. 监控和统计
"""

import os
import sys
import signal
import time
import threading
from pathlib import Path
from typing import Optional

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "auto_process_package"))

try:
    from flask import Flask, jsonify
    from watchdog.observers import Observer
except ImportError as e:
    print(f"❌ 缺少依赖: {e}")
    print("请运行: pip install -r requirements.txt")
    sys.exit(1)

# 导入项目模块
from auto_process_package.auto_process_monitor import ImageProcessHandler, process_existing_files

# 尝试导入工具模块，如果不存在则使用简化版本
try:
    from utils.logger import setup_logger
except ImportError:
    import logging
    def setup_logger(name, log_dir=None, level="INFO", console=True, max_bytes=100*1024*1024, backup_count=10):
        logger = logging.getLogger(name)
        logger.setLevel(getattr(logging, level))
        logger.handlers.clear()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        if console:
            handler = logging.StreamHandler()
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        if log_dir:
            from pathlib import Path
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            from datetime import datetime
            log_file = log_dir / f"{name}_{datetime.now().strftime('%Y%m%d')}.log"
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(log_file, maxBytes=max_bytes, backupCount=backup_count, encoding='utf-8')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        return logger

try:
    from utils.monitor import HealthMonitor
except ImportError:
    class HealthMonitor:
        def __init__(self, metrics_collector=None, port=8081, enabled=True):
            self.metrics_collector = metrics_collector
            self.port = port
            self.enabled = enabled
        def start(self):
            pass
        def stop(self):
            pass

try:
    from utils.shutdown import ShutdownHandler
except ImportError:
    class ShutdownHandler:
        def __init__(self):
            self.callbacks = []
            self._shutdown_requested = False
            
        def register_callback(self, callback):
            self.callbacks.append(callback)
            signal.signal(signal.SIGINT, self._handle_signal)
            signal.signal(signal.SIGTERM, self._handle_signal)
            
        def _handle_signal(self, signum, frame):
            """信号处理器 - 立即响应中断信号"""
            if self._shutdown_requested:
                # 如果已经请求过关闭但还在运行，强制退出
                print("\n⚠️ 强制退出...")
                sys.exit(1)
            
            self._shutdown_requested = True
            print(f"\n收到中断信号 (信号 {signum})，正在停止服务...")
            
            # 执行所有回调
            for cb in self.callbacks:
                try:
                    cb()
                except Exception as e:
                    print(f"执行停止回调时出错: {e}")
            
            # 给一点时间让清理完成，然后强制退出
            threading.Timer(5.0, lambda: sys.exit(0)).start()

try:
    from utils.metrics import MetricsCollector
except ImportError:
    from dataclasses import dataclass
    from datetime import datetime
    
    @dataclass
    class SimpleMetrics:
        total_files_processed: int = 0
        total_files_successful: int = 0
        total_files_failed: int = 0
        total_patches_created: int = 0
        total_patches_kept: int = 0
        total_detections: int = 0
        avg_processing_time: float = 0.0
        disk_free_gb: float = 100.0
        memory_usage: float = 0.0
        
        def to_dict(self):
            return {
                'files': {
                    'total_processed': self.total_files_processed,
                    'successful': self.total_files_successful,
                    'failed': self.total_files_failed
                },
                'patches': {
                    'created': self.total_patches_created,
                    'kept': self.total_patches_kept
                },
                'detections': {'total': self.total_detections}
            }
    
    class MetricsCollector:
        def __init__(self):
            self.metrics = SimpleMetrics()
            self.start_time = datetime.now()
        
        def get_metrics(self):
            return self.metrics
        
        def get_uptime(self):
            return (datetime.now() - self.start_time).total_seconds()
        
        def update_system_metrics(self):
            pass
        
        def get_all_metrics(self):
            return self.metrics.to_dict()
        
        def get_stats(self):
            return {
                'uptime_seconds': self.get_uptime(),
                'files': {
                    'total': self.metrics.total_files_processed,
                    'successful': self.metrics.total_files_successful,
                    'failed': self.metrics.total_files_failed
                }
            }


class RTDETRService:
    """RT-DETR 处理服务主类"""
    
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or (PROJECT_ROOT / "config.yaml")
        self.config = None
        self.logger = None
        self.observer = None
        self.handler = None
        self.health_monitor = None
        self.metrics = None
        self.shutdown_handler = None
        self._running = False
        
    def load_configuration(self):
        """加载配置"""
        try:
            import yaml
            import re
            import os
            
            # 直接加载YAML配置
            if not self.config_path.exists():
                raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
            
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 展开环境变量
            def expand_env_vars(value: str) -> str:
                if not isinstance(value, str):
                    return value
                def replacer(match):
                    var_expr = match.group(1)
                    if ':' in var_expr:
                        var_name, default = var_expr.split(':', 1)
                        return os.getenv(var_name, default)
                    else:
                        return os.getenv(var_expr, match.group(0))
                pattern = r'\$\{([^}]+)\}'
                return re.sub(pattern, replacer, value)
            
            def expand_dict(d):
                if isinstance(d, dict):
                    return {k: expand_dict(v) for k, v in d.items()}
                elif isinstance(d, list):
                    return [expand_dict(item) for item in d]
                elif isinstance(d, str):
                    return expand_env_vars(d)
                return d
            
            self.config = expand_dict(config)
            
            # 设置日志
            log_dir = Path(self.config.get("paths", {}).get("log_dir", PROJECT_ROOT / "logs"))
            logging_config = self.config.get("logging", {})
            self.logger = setup_logger(
                name="rtdetr_service",
                log_dir=log_dir,
                level=logging_config.get("level", "INFO"),
                console=logging_config.get("console", True),
                max_bytes=logging_config.get("file_max_bytes", 500 * 1024 * 1024), # 单个日志文件最大大小（默认500MB，处理大文件时建议增大）
                backup_count=logging_config.get("file_backup_count", 20) # 保留的备份日志文件数量（默认20个） 
            )
            self.logger.info("配置加载成功")
            return True
        except Exception as e:
            print(f"❌ 配置加载失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def initialize_components(self):
        """初始化各个组件"""
        try:
            # 初始化指标收集器
            self.metrics = MetricsCollector()
            
            # 初始化健康监控
            monitoring_config = self.config.get("monitoring", {})
            if monitoring_config.get("enabled", True):
                port = monitoring_config.get("health_check_port", 8081)
                self.health_monitor = HealthMonitor(
                    metrics_collector=self.metrics,
                    port=port,
                    enabled=port > 0
                )
                # 显式输出 Flask 健康检查服务的启动提示，便于观察
                self.logger.info(f"Flask 健康检查服务即将启动，监听端口 {port}")
                # 启动健康检查服务器
                self.health_monitor.start()
            
            # 初始化优雅关闭处理器
            self.shutdown_handler = ShutdownHandler()
            self.shutdown_handler.register_callback(self.stop)
            
            # 初始化文件处理器
            paths_config = self.config.get("paths", {})
            watch_dir = Path(paths_config.get("watch_dir")).expanduser().resolve()
            output_dir = Path(paths_config.get("output_dir")).expanduser().resolve()
            model_path = Path(paths_config.get("model_path")).expanduser().resolve()
            
            # 检查并创建必要的目录
            # 1. 检查监听目录是否存在，如果不存在则创建
            if not watch_dir.exists():
                self.logger.warning(f"监听目录不存在: {watch_dir}，将自动创建")
                try:
                    watch_dir.mkdir(parents=True, exist_ok=True)
                    # 创建后立即验证目录确实存在且可访问
                    if not watch_dir.exists() or not watch_dir.is_dir():
                        raise RuntimeError(f"目录创建后验证失败: {watch_dir}")
                    self.logger.info(f"✅ 已创建监听目录: {watch_dir}")
                except Exception as e:
                    self.logger.error(f"❌ 无法创建监听目录 {watch_dir}: {e}")
                    self.logger.error(f"   请检查权限或手动创建目录: mkdir -p {watch_dir}")
                    raise
            else:
                # 即使目录存在，也验证它是目录而不是文件
                if not watch_dir.is_dir():
                    error_msg = f"监听路径存在但不是目录: {watch_dir}"
                    self.logger.error(error_msg)
                    raise NotADirectoryError(error_msg)
                self.logger.debug(f"✓ 监听目录已存在: {watch_dir}")
            
            # 2. 确保输出目录存在
            if not output_dir.exists():
                self.logger.info(f"输出目录不存在: {output_dir}，将自动创建")
                try:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    self.logger.info(f"✅ 已创建输出目录: {output_dir}")
                except Exception as e:
                    self.logger.error(f"❌ 无法创建输出目录 {output_dir}: {e}")
                    raise
            
            # 3. 检查模型文件是否存在
            if not model_path.exists():
                self.logger.error(f"❌ 模型文件不存在: {model_path}")
                raise FileNotFoundError(f"模型文件不存在: {model_path}")
            
            processing_config = self.config.get("processing", {})
            filtering_config = self.config.get("filtering")
            
            # 如果 filtering 配置不存在或为 None，使用默认值
            if filtering_config is None:
                filtering_config = {
                    "bg_rgb": [238, 235, 235],
                    "tolerance": 30,
                    "bg_ratio": 0.9,
                    "std_thresh": 10.0,
                    "auto_bg": True,
                    "bg_analysis_limit": 5
                }
                self.logger.info("未找到 filtering 配置，使用默认值")
            
            # 解析背景色
            bg_rgb_str = filtering_config.get("bg_rgb", [238, 235, 235])
            if isinstance(bg_rgb_str, list):
                bg_rgb = tuple(bg_rgb_str)
            else:
                bg_rgb = tuple(map(int, str(bg_rgb_str).split(',')))
            
            # 获取图像缩放配置
            downscale_config = self.config.get("downscale", {})
            
            self.handler = ImageProcessHandler(
                watch_dir=watch_dir,
                output_base_dir=output_dir,
                model_path=str(model_path),
                patch_size=processing_config.get("patch_size", 640),
                threshold=processing_config.get("threshold", 0.5),
                bg_rgb=bg_rgb,
                tolerance=filtering_config.get("tolerance", 30),
                bg_ratio=filtering_config.get("bg_ratio", 0.9),
                std_thresh=filtering_config.get("std_thresh", 10.0),
                auto_bg=filtering_config.get("auto_bg", True),
                bg_analysis_limit=filtering_config.get("bg_analysis_limit", 5),
                save_visualization=processing_config.get("save_visualization", True),
                file_wait_timeout=processing_config.get("file_wait_timeout", 30),
                max_concurrent_tasks=processing_config.get("max_concurrent_tasks", 2),  # 并发任务数
                auto_downscale=downscale_config.get("enabled", True),  # 是否启用自动缩放
                downscale_threshold=downscale_config.get("threshold", 50000),  # 缩放阈值（像素）
                downscale_quality=downscale_config.get("quality", 95),  # 缩放质量
                logger=self.logger,  # 传递 logger 以便输出处理过程
                minio_config=self.config.get("minio")
            )
            
            # 初始化观察者之前，再次确认目录存在（双重检查）
            # 重新解析路径，确保使用绝对路径
            watch_dir_abs = watch_dir.resolve()
            
            # 添加详细的路径信息日志，便于调试
            self.logger.info(f"🔍 检查监听目录: {watch_dir_abs}")
            self.logger.info(f"   原始配置: {paths_config.get('watch_dir')}")
            self.logger.info(f"   解析后路径: {watch_dir_abs}")
            self.logger.info(f"   路径存在: {watch_dir_abs.exists()}")
            if watch_dir_abs.exists():
                self.logger.info(f"   是目录: {watch_dir_abs.is_dir()}")
            
            if not watch_dir_abs.exists() or not watch_dir_abs.is_dir():
                error_msg = f"监听目录不存在或不是目录: {watch_dir_abs} (在observer.schedule前检查失败)"
                self.logger.error(error_msg)
                self.logger.error(f"   原始配置值: {paths_config.get('watch_dir')}")
                self.logger.error(f"   请检查配置文件中的路径或设置环境变量 WATCH_DIR")
                raise FileNotFoundError(error_msg)
            
            # 初始化观察者
            self.observer = Observer()
            # 使用绝对路径字符串，确保路径正确
            watch_dir_str = str(watch_dir_abs)
            self.logger.info(f"📁 准备监听目录: {watch_dir_str}")
            try:
                self.observer.schedule(self.handler, watch_dir_str, recursive=False)
                self.logger.info(f"✓ observer.schedule 成功: {watch_dir_str}")
            except Exception as e:
                self.logger.error(f"❌ observer.schedule 失败: {e}")
                self.logger.error(f"   目录路径: {watch_dir_str}")
                self.logger.error(f"   目录存在: {Path(watch_dir_str).exists()}")
                raise
            
            self.logger.info("组件初始化完成")
            return True
            
        except Exception as e:
            if self.logger:
                self.logger.error(f"组件初始化失败: {e}", exc_info=True)
            else:
                print(f"❌ 组件初始化失败: {e}")
            return False
    
    def start_health_server(self):
        """启动健康检查服务器（由 HealthMonitor 自己管理，这里只是占位）"""
        # HealthMonitor 已经在 initialize_components 中启动了自己的服务器
        # 这个方法保留是为了兼容性，实际不需要做任何事情
        pass
    
    def start(self):
        """启动服务"""
        if self._running:
            self.logger.warning("服务已在运行")
            return
        
        if not self.load_configuration():
            return False
        
        if not self.initialize_components():
            return False
        
        # 处理已存在的文件
        processing_config = self.config.get("processing", {})
        if processing_config.get("process_existing", False):
            self.logger.info("开始处理已存在的文件...")
            paths_config = self.config.get("paths", {})
            watch_dir = Path(paths_config.get("watch_dir")).expanduser().resolve()
            process_existing_files(watch_dir, self.handler)
        
        # 健康检查服务器已在 initialize_components 中由 HealthMonitor 启动
        
        # 启动文件观察者之前，再次确认监听目录存在
        paths_config = self.config.get("paths", {})
        watch_dir = Path(paths_config.get("watch_dir")).expanduser().resolve()
        if not watch_dir.exists() or not watch_dir.is_dir():
            error_msg = f"监听目录不存在或不是目录: {watch_dir}（在启动observer前检查失败）"
            self.logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        self.logger.info(f"✓ 确认监听目录存在: {watch_dir}")
        # 启动文件观察者
        self.observer.start()
        self._running = True
        
        paths_config = self.config.get("paths", {})
        self.logger.info("=" * 70)
        self.logger.info("🚀 RT-DETR 自动处理服务已启动")
        self.logger.info("=" * 70)
        self.logger.info(f"📁 监听目录: {paths_config.get('watch_dir')}")
        self.logger.info(f"📁 输出目录: {paths_config.get('output_dir')}")
        self.logger.info(f"🤖 模型路径: {paths_config.get('model_path')}")
        self.logger.info("=" * 70)
        
        return True
    
    def stop(self):
        """停止服务"""
        if not self._running:
            return
        
        self.logger.info("正在停止服务...")
        self._running = False
        
        if self.health_monitor:
            self.health_monitor.stop()
        
        if self.observer:
            self.observer.stop()
            self.observer.join(timeout=10)
        
        if self.handler:
            # 通知处理器停止，避免长耗时任务继续运行
            if hasattr(self.handler, "request_stop"):
                self.handler.request_stop()
            self.handler.save_status()
        
        self.logger.info("✅ 服务已停止")
    
    def run(self):
        """运行服务主循环"""
        if not self.start():
            sys.exit(1)
        
        try:
            while self._running:
                # 使用短暂的 sleep 来提高响应性
                time.sleep(0.1)
                
                # 定期输出统计信息（如果需要，可以在这里添加）
                # HealthMonitor 已经通过 HTTP 接口提供统计信息
                    
        except KeyboardInterrupt:
            self.logger.info("\n收到 Ctrl+C 中断信号")
        except Exception as e:
            self.logger.error(f"服务运行时出错: {e}", exc_info=True)
        finally:
            self.stop()


def main():
    """主入口"""
    import argparse
    
    parser = argparse.ArgumentParser(description="RT-DETR 自动处理服务")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径（默认: config.yaml）"
    )
    
    args = parser.parse_args()
    
    config_path = Path(args.config) if args.config else None
    service = RTDETRService(config_path)
    service.run()


if __name__ == "__main__":
    main()

