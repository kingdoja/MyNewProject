#!/usr/bin/env python3
"""
Check GPU availability and PyTorch CUDA setup
"""

import torch
import sys

def check_gpu():
    """Check GPU availability and PyTorch CUDA setup"""
    
    print("=== GPU and PyTorch CUDA Check ===\n")
    
    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        # Check CUDA version
        print(f"CUDA version: {torch.version.cuda}")
        
        # Check GPU count
        gpu_count = torch.cuda.device_count()
        print(f"GPU count: {gpu_count}")
        
        # Check current device
        current_device = torch.cuda.current_device()
        print(f"Current device: {current_device}")
        
        # Check device name
        device_name = torch.cuda.get_device_name(current_device)
        print(f"Device name: {device_name}")
        
        # Check GPU memory
        memory_allocated = torch.cuda.memory_allocated(current_device) / 1024**3
        memory_cached = torch.cuda.memory_reserved(current_device) / 1024**3
        print(f"GPU memory allocated: {memory_allocated:.2f} GB")
        print(f"GPU memory cached: {memory_cached:.2f} GB")
        
        # Test tensor operations on GPU
        try:
            test_tensor = torch.randn(1000, 1000).cuda()
            result = torch.mm(test_tensor, test_tensor)
            print("✓ GPU tensor operations working")
        except Exception as e:
            print(f"✗ GPU tensor operations failed: {e}")
            
    else:
        print("✗ CUDA not available - PyTorch was likely installed without CUDA support")
        print("  You may need to reinstall PyTorch with CUDA support")
        
    print("\n=== Environment Check ===")
    
    # Check if running in conda environment
    if 'conda' in sys.prefix:
        print(f"Running in conda environment: {sys.prefix}")
    else:
        print("Not running in conda environment")
    
    # Check Python executable
    print(f"Python executable: {sys.executable}")

if __name__ == "__main__":
    check_gpu()
