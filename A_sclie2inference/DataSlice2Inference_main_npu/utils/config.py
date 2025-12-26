#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""配置管理模块"""

import os
import yaml
import re
import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class Config:
    """配置管理类"""
    
    def __init__(self, config_path: Path):
        """加载配置文件
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = Path(config_path)
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._raw_config = yaml.safe_load(f)
        
        self.config = self._resolve_env_vars(self._raw_config)
        self._validate()
    
    def _resolve_env_vars(self, config: Dict) -> Dict:
        """解析环境变量 ${VAR:default} 或 ${VAR}
        
        Args:
            config: 配置字典
            
        Returns:
            解析后的配置字典
        """
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
        """验证配置有效性"""
        # 检查必需路径
        watch_dir = Path(self.get('paths.watch_dir'))
        if not watch_dir.exists():
            raise FileNotFoundError(f"监听目录不存在: {watch_dir}")
        
        model_path = Path(self.get('paths.model_path'))
        if not model_path.exists():
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 验证数值范围
        patch_size = self.get('processing.patch_size')
        if not isinstance(patch_size, int) or patch_size <= 0:
            raise ValueError(f"patch_size 必须为正整数，当前值: {patch_size}")
        
        threshold = self.get('processing.threshold')
        if not isinstance(threshold, (int, float)) or not (0 <= threshold <= 1):
            raise ValueError(f"threshold 必须在 0-1 之间，当前值: {threshold}")
        
        logger.info(f"配置文件验证通过: {self.config_path}")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """获取配置值，支持点号分隔的路径
        
        Args:
            key_path: 配置键路径，如 'paths.watch_dir'
            default: 默认值
            
        Returns:
            配置值
        """
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
        """获取路径配置并转换为 Path 对象
        
        Args:
            key_path: 配置键路径
            
        Returns:
            Path 对象
        """
        path_str = self.get(key_path)
        if path_str is None:
            raise ValueError(f"配置项不存在: {key_path}")
        return Path(path_str).expanduser().resolve()
    
    def get_dict(self, key_path: str, default: Optional[Dict] = None) -> Dict:
        """获取字典配置
        
        Args:
            key_path: 配置键路径
            default: 默认值
            
        Returns:
            配置字典
        """
        value = self.get(key_path, default)
        if value is None:
            return default or {}
        if not isinstance(value, dict):
            raise ValueError(f"配置项 {key_path} 不是字典类型")
        return value

