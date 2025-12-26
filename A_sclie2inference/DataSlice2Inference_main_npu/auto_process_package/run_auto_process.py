#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键配置并启动 auto_process_monitor.py 的脚本。

从 config.yaml 读取配置，执行：
    python run_auto_process.py
即可带着当前配置启动自动处理服务。

修改配置请编辑 config.yaml 文件。
"""

from __future__ import annotations

import os
import re
import shlex
import sys
from pathlib import Path

import yaml

from auto_process_monitor import main as monitor_main


def expand_env_vars(value: str) -> str:
    """展开环境变量，支持 ${VAR:default} 格式"""
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


def load_config(config_path: Path) -> dict:
    """从 config.yaml 加载配置并转换为命令行参数格式"""
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 展开环境变量
    def expand_dict(d):
        if isinstance(d, dict):
            return {k: expand_dict(v) for k, v in d.items()}
        elif isinstance(d, list):
            return [expand_dict(item) for item in d]
        elif isinstance(d, str):
            return expand_env_vars(d)
        return d
    
    config = expand_dict(config)
    
    # 转换为命令行参数格式
    result = {}
    
    # 路径配置
    paths = config.get("paths", {})
    result["watch_dir"] = paths.get("watch_dir")
    result["output_dir"] = paths.get("output_dir")
    result["model"] = paths.get("model_path")
    result["log_dir"] = paths.get("log_dir")
    
    # 处理参数
    processing = config.get("processing", {})
    result["patch_size"] = processing.get("patch_size", 640)
    result["threshold"] = processing.get("threshold", 0.5)
    result["no_visualization"] = not processing.get("save_visualization", True)
    result["process_existing"] = processing.get("process_existing", False)
    result["file_wait_timeout"] = processing.get("file_wait_timeout", 30)
    
    # 过滤参数
    filtering = config.get("filtering", {})
    bg_rgb = filtering.get("bg_rgb", [238, 235, 235])
    if isinstance(bg_rgb, list):
        result["bg_rgb"] = ",".join(map(str, bg_rgb))
    else:
        result["bg_rgb"] = str(bg_rgb)
    result["tolerance"] = filtering.get("tolerance", 30)
    result["bg_ratio"] = filtering.get("bg_ratio", 0.9)
    result["std_thresh"] = filtering.get("std_thresh", 10.0)
    result["disable_auto_bg"] = not filtering.get("auto_bg", True)
    result["bg_analysis_limit"] = filtering.get("bg_analysis_limit", 5)
    
    # 保存完整配置用于service_main.py
    result["_full_config"] = config
    
    return result


def build_cli_args(config: dict) -> list[str]:
    """把配置转换为 auto_process_monitor.py 所需的命令行参数"""
    args = ["auto_process_monitor.py"]

    def add_arg(flag: str, value):
        args.extend([flag, str(value)])

    add_arg("--watch-dir", config["watch_dir"])
    add_arg("--output-dir", config["output_dir"])
    add_arg("--model", config["model"])
    add_arg("--patch-size", config["patch_size"])
    add_arg("--threshold", config["threshold"])
    add_arg("--bg-rgb", config["bg_rgb"])
    add_arg("--tolerance", config["tolerance"])
    add_arg("--bg-ratio", config["bg_ratio"])
    add_arg("--std-thresh", config["std_thresh"])
    add_arg("--bg-analysis-limit", config["bg_analysis_limit"])
    add_arg("--file-wait-timeout", config.get("file_wait_timeout", 30))

    if config.get("disable_auto_bg"):
        args.append("--disable-auto-bg")
    if config.get("no_visualization"):
        args.append("--no-visualization")
    if config.get("process_existing"):
        args.append("--process-existing")

    return args


def main():
    # 获取配置文件路径（相对于当前脚本的目录）
    script_dir = Path(__file__).parent
    config_path = script_dir.parent / "config.yaml"
    
    # 加载配置
    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"❌ 加载配置文件失败: {e}")
        sys.exit(1)

    # 确保路径存在
    watch_dir = Path(config["watch_dir"]).expanduser().resolve()
    model_path = Path(config["model"]).expanduser().resolve()
    if not watch_dir.exists():
        raise FileNotFoundError(f"监听目录不存在: {watch_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    cli_args = build_cli_args(config)
    print("🚀 正在启动 auto_process_monitor.py")
    print(f"📄 配置文件: {config_path}")
    print("🔧 当前参数:")
    print("   " + " ".join(shlex.quote(arg) for arg in cli_args[1:]))
    print("（修改 config.yaml 后重新运行即可生效）\n")

    # 用配置参数覆盖 sys.argv 并调用原脚本入口
    sys.argv = cli_args
    monitor_main()


if __name__ == "__main__":
    main()


