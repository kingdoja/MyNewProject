#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
检查模型文件是否完整和可用
"""

import sys
import os
from pathlib import Path

def check_model_file(model_path: str):
    """检查模型文件"""
    model_file = Path(model_path)
    
    print("=" * 70)
    print("模型文件检查工具")
    print("=" * 70)
    print()
    
    # 检查文件是否存在
    if not model_file.exists():
        print(f"❌ 模型文件不存在: {model_path}")
        return False
    
    print(f"✓ 模型文件存在: {model_path}")
    
    # 检查文件大小
    file_size = model_file.stat().st_size
    file_size_mb = file_size / 1024 / 1024
    print(f"✓ 文件大小: {file_size_mb:.2f} MB ({file_size:,} 字节)")
    
    if file_size < 1024:
        print(f"❌ 文件异常小，可能损坏")
        return False
    
    # 检查文件权限
    if not os.access(model_file, os.R_OK):
        print(f"❌ 文件不可读")
        return False
    print(f"✓ 文件可读")
    
    # 尝试读取文件头（TorchScript文件是zip格式）
    try:
        with open(model_file, 'rb') as f:
            header = f.read(4)
            # ZIP文件通常以PK开头（0x50 0x4B）
            if header[:2] == b'PK':
                print(f"✓ 文件格式: ZIP格式（TorchScript）")
            else:
                print(f"⚠️  文件头不是ZIP格式: {header.hex()}")
                print(f"   这可能是正常的，取决于PyTorch版本")
    except Exception as e:
        print(f"⚠️  无法读取文件头: {e}")
    
    # 尝试加载模型
    print(f"\n尝试加载模型...")
    try:
        import torch
        print(f"  PyTorch版本: {torch.__version__}")
        
        # 尝试加载（使用CPU，因为只是测试）
        print(f"  正在加载模型...")
        model = torch.jit.load(str(model_file), map_location='cpu')
        model.eval()
        print(f"  ✅ 模型加载成功！")
        
        # 显示模型信息
        print(f"\n模型信息:")
        print(f"  类型: {type(model)}")
        
        # 尝试获取输入和输出信息（如果可用）
        try:
            graph = model.graph
            print(f"  有计算图: 是")
        except:
            print(f"  有计算图: 无法获取")
        
        return True
        
    except RuntimeError as e:
        error_msg = str(e)
        if "failed reading zip archive" in error_msg or "central directory" in error_msg:
            print(f"  ❌ 模型文件损坏或不完整")
            print(f"  错误: {error_msg}")
            print(f"\n可能的原因:")
            print(f"  1. 文件在传输/复制过程中被截断")
            print(f"  2. 文件在写入时被中断")
            print(f"  3. 存储设备错误导致文件损坏")
            print(f"\n建议:")
            print(f"  1. 检查原始模型文件是否完整")
            print(f"  2. 重新导出或下载模型文件")
            print(f"  3. 使用 md5sum 或 sha256sum 验证文件完整性")
            return False
        else:
            print(f"  ❌ 模型加载失败: {error_msg}")
            return False
    except Exception as e:
        print(f"  ❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("使用方法: python check_model_file.py <model_path>")
        print("示例: python check_model_file.py models/best_torchscript_cuda_newAll.pt")
        sys.exit(1)
    
    model_path = sys.argv[1]
    success = check_model_file(model_path)
    
    print()
    print("=" * 70)
    if success:
        print("检查完成：模型文件正常")
    else:
        print("检查完成：模型文件有问题")
    print("=" * 70)
    
    sys.exit(0 if success else 1)
