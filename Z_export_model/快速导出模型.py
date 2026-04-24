#!/usr/bin/env python3
"""
RT-DETR 模型导出快速脚本
使用说明：修改配置参数，然后运行此脚本
"""

import os
import sys
import subprocess
from pathlib import Path

# ===== 配置区域 - 请根据你的实际情况修改 =====

# 选择版本：rtdetr_pytorch 或 rtdetrv2_pytorch
VERSION = "/home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch"

# 配置文件路径（相对于VERSION目录）
CONFIG_FILE = "/home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_cancer_detection.yml"

# 训练好的模型checkpoint路径（相对于VERSION目录）
CHECKPOINT = "/home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/output/rtdetrv2_r18vd_cancer_detection/best.pth"

# 输出目录
OUTPUT_DIR = "exported_models"

# 导出模式：state_dict / full_model / torchscript / deploy
# 后端部署推理推荐使用 torchscript（高性能，稳定，易于部署）
EXPORT_MODE = "torchscript"

# 输入图像尺寸（仅用于torchscript模式）
INPUT_SIZE = 640

# 导出设备（仅用于torchscript模式）
# 重要说明：
# - None: 自动检测，如果有CUDA则在CUDA上导出（主要用于CUDA推理，可通过map_location加载到CPU）
# - 'cuda': 强制在CUDA上导出（主要用于CUDA推理）
# - 'cpu': 强制在CPU上导出（主要用于CPU推理，推荐用于跨平台部署）
# 
# 注意：导出的模型包含 orig_target_sizes 参数，调用方式：model(images, orig_target_sizes)
# TorchScript模型理论上可以在不同设备间切换，但最佳实践是在目标设备上导出
EXPORT_DEVICE = None  # None=自动检测, 'cuda'=强制CUDA, 'cpu'=强制CPU

# ============================================


def print_header():
    """打印标题"""
    print("=" * 60)
    print("RT-DETR 模型导出工具")
    print("=" * 60)
    print()


def print_config(config):
    """打印配置信息"""
    print("配置信息：")
    for key, value in config.items():
        print(f"  {key}: {value}")
    print()


def get_output_filename(checkpoint_path, export_mode):
    """根据导出模式生成输出文件名"""
    basename = Path(checkpoint_path).stem
    
    suffix_map = {
        'state_dict': 'weights',
        'full_model': 'full',
        'torchscript': 'torchscript',
        'deploy': 'deploy'
    }
    
    suffix = suffix_map.get(export_mode, 'exported')
    return f"{basename}_{suffix}.pt"


def main():
    """主函数"""
    print_header()
    
    # 将路径转换为Path对象（支持绝对路径和相对路径）
    version_dir = Path(VERSION).absolute()
    config_path = Path(CONFIG_FILE).absolute()
    checkpoint_path = Path(CHECKPOINT).absolute()
    
    # 如果OUTPUT_DIR是相对路径，则相对于version_dir
    if Path(OUTPUT_DIR).is_absolute():
        output_dir = Path(OUTPUT_DIR)
    else:
        output_dir = version_dir / OUTPUT_DIR
    
    # 检查版本目录是否存在
    if not version_dir.exists():
        print(f"❌ 错误: 找不到目录 {version_dir}")
        print(f"   请确认VERSION配置正确")
        sys.exit(1)
    
    # 检查文件是否存在
    if not config_path.exists():
        print(f"❌ 错误: 找不到配置文件 {config_path}")
        sys.exit(1)
    
    if not checkpoint_path.exists():
        print(f"❌ 错误: 找不到checkpoint文件 {checkpoint_path}")
        sys.exit(1)
    
    # 创建输出目录
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 生成输出文件名
    output_filename = get_output_filename(str(checkpoint_path), EXPORT_MODE)
    output_path = output_dir / output_filename
    
    # 打印配置
    device_info = "自动检测"
    if EXPORT_MODE == "torchscript":
        if EXPORT_DEVICE:
            device_info = EXPORT_DEVICE
        else:
            device_info = "自动检测（优先CUDA）"
    
    config_info = {
        "版本目录": str(version_dir),
        "配置文件": str(config_path),
        "Checkpoint": str(checkpoint_path),
        "导出模式": EXPORT_MODE,
        "导出设备": device_info if EXPORT_MODE == "torchscript" else "N/A",
        "输出文件": str(output_path),
        "模型参数": "包含 orig_target_sizes 参数" if EXPORT_MODE == "torchscript" else "N/A"
    }
    print_config(config_info)
    
    # 构建命令
    export_script = version_dir / "tools" / "export_pt.py"
    
    if not export_script.exists():
        print(f"❌ 错误: 找不到导出脚本 {export_script}")
        sys.exit(1)
    
    cmd = [
        sys.executable,
        str(export_script),
        "--config", str(config_path),
        "--resume", str(checkpoint_path),
        "--mode", EXPORT_MODE,
        "--output", str(output_path),
        "--input-size", str(INPUT_SIZE)
    ]
    
    # 如果是torchscript模式，添加设备参数
    if EXPORT_MODE == "torchscript" and EXPORT_DEVICE:
        cmd.extend(["--device", EXPORT_DEVICE])
    
    print("开始导出...")
    print()
    
    # 执行命令
    try:
        result = subprocess.run(cmd, check=True)
        
        print()
        print("=" * 60)
        print("✅ 导出成功!")
        print("=" * 60)
        print(f"输出文件位置: {output_path}")
        print()
        
        # 显示文件信息
        if output_path.exists():
            file_size = output_path.stat().st_size / 1024 / 1024  # MB
            print(f"文件大小: {file_size:.2f} MB")
        
    except subprocess.CalledProcessError as e:
        print()
        print("=" * 60)
        print("❌ 导出失败，请检查错误信息")
        print("=" * 60)
        sys.exit(1)
    except KeyboardInterrupt:
        print()
        print("⚠️  用户中断操作")
        sys.exit(1)


if __name__ == "__main__":
    main()

