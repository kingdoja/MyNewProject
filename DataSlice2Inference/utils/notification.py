#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通知模块"""

import logging
from typing import List, Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)


def send_email_notification(
    smtp_host: str,
    smtp_port: int,
    from_addr: str,
    to_addrs: List[str],
    subject: str,
    message: str,
    username: Optional[str] = None,
    password: Optional[str] = None
):
    """发送邮件通知
    
    Args:
        smtp_host: SMTP 服务器地址
        smtp_port: SMTP 端口
        from_addr: 发件人地址
        to_addrs: 收件人地址列表
        subject: 邮件主题
        message: 邮件内容
        username: SMTP 用户名（可选）
        password: SMTP 密码（可选）
    """
    try:
        import smtplib
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        
        msg = MIMEMultipart()
        msg['From'] = from_addr
        msg['To'] = ', '.join(to_addrs)
        msg['Subject'] = subject
        msg.attach(MIMEText(message, 'plain', 'utf-8'))
        
        with smtplib.SMTP(smtp_host, smtp_port) as server:
            if username and password:
                server.starttls()
                server.login(username, password)
            server.send_message(msg)
        
        logger.info(f"邮件通知已发送: {subject}")
    except ImportError:
        logger.warning("smtplib 不可用，无法发送邮件通知")
    except Exception as e:
        logger.error(f"发送邮件通知失败: {e}", exc_info=True)


def send_webhook_notification(url: str, data: Dict[str, Any], timeout: int = 5):
    """发送 Webhook 通知
    
    Args:
        url: Webhook URL
        data: 要发送的数据
        timeout: 超时时间（秒）
    """
    try:
        import requests
        
        response = requests.post(url, json=data, timeout=timeout)
        response.raise_for_status()
        logger.info(f"Webhook 通知已发送: {url}")
    except ImportError:
        logger.warning("requests 库不可用，无法发送 Webhook 通知")
    except Exception as e:
        logger.error(f"发送 Webhook 通知失败: {e}", exc_info=True)


class NotificationManager:
    """通知管理器"""
    
    def __init__(self, config: Dict[str, Any]):
        """初始化通知管理器
        
        Args:
            config: 通知配置字典
        """
        self.config = config
        self.enabled = config.get('enabled', False)
    
    def send_processing_complete(self, file_name: str, success: bool, 
                                 patches_count: int = 0, detections: int = 0,
                                 processing_time: float = 0.0, error: Optional[str] = None):
        """发送处理完成通知
        
        Args:
            file_name: 文件名
            success: 是否成功
            patches_count: patch 数量
            detections: 检测数量
            processing_time: 处理时间（秒）
            error: 错误信息（如果失败）
        """
        if not self.enabled:
            return
        
        status = "成功" if success else "失败"
        message = f"""
文件处理{status}通知

文件名: {file_name}
状态: {status}
处理时间: {processing_time:.2f} 秒

统计信息:
- Patch 数量: {patches_count}
- 检测数量: {detections}
"""
        
        if error:
            message += f"\n错误信息: {error}"
        
        message += f"\n时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        
        subject = f"文件处理{status}: {file_name}"
        
        # 发送邮件
        if self.config.get('email', {}).get('enabled', False):
            email_config = self.config['email']
            send_email_notification(
                smtp_host=email_config.get('smtp_host', ''),
                smtp_port=email_config.get('smtp_port', 587),
                from_addr=email_config.get('from', ''),
                to_addrs=email_config.get('to', []),
                subject=subject,
                message=message,
                username=email_config.get('username'),
                password=email_config.get('password')
            )
        
        # 发送 Webhook
        if self.config.get('webhook', {}).get('enabled', False):
            webhook_config = self.config['webhook']
            webhook_data = {
                'event': 'processing_complete',
                'file_name': file_name,
                'success': success,
                'patches_count': patches_count,
                'detections': detections,
                'processing_time': processing_time,
                'timestamp': datetime.now().isoformat()
            }
            if error:
                webhook_data['error'] = error
            
            send_webhook_notification(
                url=webhook_config.get('url', ''),
                data=webhook_data,
                timeout=webhook_config.get('timeout', 5)
            )
    
    def send_system_alert(self, alert_type: str, message: str):
        """发送系统告警
        
        Args:
            alert_type: 告警类型
            message: 告警消息
        """
        if not self.enabled:
            return
        
        subject = f"系统告警: {alert_type}"
        full_message = f"""
系统告警

类型: {alert_type}
消息: {message}
时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # 发送邮件
        if self.config.get('email', {}).get('enabled', False):
            email_config = self.config['email']
            send_email_notification(
                smtp_host=email_config.get('smtp_host', ''),
                smtp_port=email_config.get('smtp_port', 587),
                from_addr=email_config.get('from', ''),
                to_addrs=email_config.get('to', []),
                subject=subject,
                message=full_message,
                username=email_config.get('username'),
                password=email_config.get('password')
            )
        
        # 发送 Webhook
        if self.config.get('webhook', {}).get('enabled', False):
            webhook_config = self.config['webhook']
            send_webhook_notification(
                url=webhook_config.get('url', ''),
                data={
                    'event': 'system_alert',
                    'alert_type': alert_type,
                    'message': message,
                    'timestamp': datetime.now().isoformat()
                },
                timeout=webhook_config.get('timeout', 5)
            )

