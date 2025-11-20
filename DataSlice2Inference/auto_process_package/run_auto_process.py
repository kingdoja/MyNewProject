#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
一键配置并启动 auto_process_monitor.py 的脚本。

修改 CONFIG 即可调整参数，执行：
    python run_auto_process.py
即可带着当前配置启动自动处理服务。
"""

from __future__ import annotations

import shlex
import sys
from pathlib import Path

from auto_process_monitor import main as monitor_main


# === 在此处修改配置 ===
CONFIG: dict = {
    "watch_dir": "/home/ubuntu/lsn/project_new/RT-DETR-main/DataWSI",  # 监听的全图目录
    "output_dir": "/home/ubuntu/lsn/project_new/RT-DETR-main/DataPatchesInference",  # 推理结果输出根目录
    "model": "/home/ubuntu/lsn/project_new/RT-DETR-main/DataSlice2Inference/models/rtdetr_torchscript_cuda.pt",  # 模型文件路径
    "patch_size": 640,  # 切片大小
    "threshold": 0.5,  # 预测置信度阈值
    "bg_rgb": "238,235,235",  # 背景色 RGB
    "tolerance": 30,  # 背景色容差
    "bg_ratio": 0.9,  # 背景比例阈值
    "std_thresh": 10.0,  # 灰度标准差阈值
    "disable_auto_bg": False,  # 是否禁用自动背景参数分析
    "bg_analysis_limit": 5,    # 自动背景分析时采样的patch数量
    "no_visualization": False,  # 是否不保存带框可视化图片，只输出JSON结果
    "process_existing": False,  # 是否处理已存在的文件
}
# =====================


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

    if config.get("disable_auto_bg"):
        args.append("--disable-auto-bg")
    if config.get("no_visualization"):
        args.append("--no-visualization")
    if config.get("process_existing"):
        args.append("--process-existing")

    return args


def main():
    config = CONFIG.copy()

    # 确保路径存在
    watch_dir = Path(config["watch_dir"]).expanduser().resolve()
    model_path = Path(config["model"]).expanduser().resolve()
    if not watch_dir.exists():
        raise FileNotFoundError(f"监听目录不存在: {watch_dir}")
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    cli_args = build_cli_args(config)
    print("🚀 正在启动 auto_process_monitor.py")
    print("🔧 当前参数:")
    print("   " + " ".join(shlex.quote(arg) for arg in cli_args[1:]))
    print("（修改 run_auto_process.py 顶部的 CONFIG 后重新运行即可生效）\n")

    # 用配置参数覆盖 sys.argv 并调用原脚本入口
    sys.argv = cli_args
    monitor_main()


if __name__ == "__main__":
    main()


