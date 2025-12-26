#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""监控和健康检查模块"""

import threading
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any

try:
    from flask import Flask, jsonify
    FLASK_AVAILABLE = True
except ImportError:
    FLASK_AVAILABLE = False

logger = logging.getLogger(__name__)


class HealthMonitor:
    """健康监控类"""
    
    def __init__(self, metrics_collector, port: int = 8081, enabled: bool = True):
        """初始化健康监控
        
        Args:
            metrics_collector: 指标收集器
            port: HTTP 服务端口（0 表示不启动）
            enabled: 是否启用监控
        """
        self.metrics_collector = metrics_collector
        self.port = port
        self.enabled = enabled
        self.app = None
        self.server_thread = None
        self.shutdown_requested = False
    
    def start(self):
        """启动监控服务"""
        if not self.enabled or self.port == 0:
            logger.info("健康检查服务未启用")
            return
        
        if not FLASK_AVAILABLE:
            logger.warning("Flask 未安装，无法启动健康检查服务")
            return
        
        self.app = Flask(__name__)
        
        @self.app.route('/health')
        def health_check():
            """健康检查接口"""
            try:
                uptime = self.metrics_collector.get_uptime()
                metrics = self.metrics_collector.get_metrics()
                
                # 判断健康状态
                is_healthy = True
                issues = []
                
                # 检查磁盘空间
                if metrics.disk_free_gb < 5.0:
                    is_healthy = False
                    issues.append("磁盘空间不足")
                
                # 检查内存使用
                if metrics.memory_usage > 95.0:
                    is_healthy = False
                    issues.append("内存使用率过高")
                
                # 检查处理失败率
                if metrics.total_files_processed > 0:
                    fail_rate = metrics.total_files_failed / metrics.total_files_processed
                    if fail_rate > 0.5:
                        is_healthy = False
                        issues.append("处理失败率过高")
                
                status = {
                    'status': 'healthy' if is_healthy else 'unhealthy',
                    'timestamp': datetime.now().isoformat(),
                    'uptime_seconds': uptime,
                    'issues': issues,
                    'summary': {
                        'files_processed': metrics.total_files_processed,
                        'files_successful': metrics.total_files_successful,
                        'files_failed': metrics.total_files_failed,
                        'success_rate': (
                            metrics.total_files_successful / metrics.total_files_processed
                            if metrics.total_files_processed > 0 else 0.0
                        )
                    }
                }
                
                return jsonify(status), 200 if is_healthy else 503
            except Exception as e:
                logger.error(f"健康检查失败: {e}", exc_info=True)
                return jsonify({
                    'status': 'error',
                    'error': str(e)
                }), 500
        
        @self.app.route('/metrics')
        def metrics():
            """性能指标接口"""
            try:
                self.metrics_collector.update_system_metrics()
                metrics_data = self.metrics_collector.get_metrics()
                return jsonify(metrics_data.to_dict())
            except Exception as e:
                logger.error(f"获取指标失败: {e}", exc_info=True)
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/stats')
        def stats():
            """统计信息接口"""
            try:
                metrics = self.metrics_collector.get_metrics()
                return jsonify({
                    'uptime_seconds': self.metrics_collector.get_uptime(),
                    'files': {
                        'total': metrics.total_files_processed,
                        'successful': metrics.total_files_successful,
                        'failed': metrics.total_files_failed
                    },
                    'patches': {
                        'created': metrics.total_patches_created,
                        'kept': metrics.total_patches_kept
                    },
                    'detections': metrics.total_detections,
                    'avg_processing_time': metrics.avg_processing_time
                })
            except Exception as e:
                logger.error(f"获取统计信息失败: {e}", exc_info=True)
                return jsonify({'error': str(e)}), 500
        
        # 启动 Flask 服务器
        def run_server():
            try:
                self.app.run(host='0.0.0.0', port=self.port, debug=False, use_reloader=False)
            except OSError as e:
                # 更友好的端口占用提示，便于排查
                if getattr(e, "errno", None) == 98:
                    logger.error(f"监控服务器启动失败: 端口 {self.port} 已被占用，请修改配置或释放端口")
                else:
                    logger.error(f"监控服务器启动失败: {e}", exc_info=True)
            except Exception as e:
                logger.error(f"监控服务器启动失败: {e}", exc_info=True)
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        logger.info(f"健康检查服务已启动，端口: {self.port}")
    
    def stop(self):
        """停止监控服务"""
        self.shutdown_requested = True
        if self.server_thread and self.server_thread.is_alive():
            logger.info("正在停止健康检查服务...")
            # Flask 服务器会在主线程退出时自动停止（daemon=True）

