#!/usr/bin/env python3
"""
Check validation dataset and run evaluation
"""

import os
import json
import torch
from PIL import Image

def check_validation_data():
    """Check validation dataset"""
    
    print("=== Validation Dataset Check ===\n")
    
    # Check validation directory
    val_dir = "/home/ubuntu/lsn/project_new/RT-DETR-main/DATA/APCData cervical cytology cells/split_YOLO/images/val"
    val_ann_file = "/home/ubuntu/lsn/project_new/RT-DETR-main/DATA/APCData cervical cytology cells/split_YOLO/annotations/val.json"
    
    print(f"Validation directory: {val_dir}")
    print(f"Validation annotations: {val_ann_file}")
    
    # Check if files exist
    if not os.path.exists(val_dir):
        print("❌ Validation directory not found!")
        return False
        
    if not os.path.exists(val_ann_file):
        print("❌ Validation annotations not found!")
        return False
    
    print("✅ Validation files found")
    
    # Check annotations
    try:
        with open(val_ann_file, 'r', encoding='utf-8') as f:
            val_data = json.load(f)
        
        print(f"Images count: {len(val_data.get('images', []))}")
        print(f"Annotations count: {len(val_data.get('annotations', []))}")
        print(f"Categories count: {len(val_data.get('categories', []))}")
        
        if 'categories' in val_data:
            print("\nCategories:")
            for cat in val_data['categories']:
                print(f"  - ID {cat['id']}: {cat['name']}")
        
        if 'images' in val_data and len(val_data['images']) > 0:
            sample_img = val_data['images'][0]
            img_path = os.path.join(val_dir, sample_img['file_name'])
            print(f"\nSample image: {img_path}")
            print(f"Size: {sample_img.get('width', 'N/A')}x{sample_img.get('height', 'N/A')}")
            
            if os.path.exists(img_path):
                print("✅ Sample image exists")
            else:
                print("❌ Sample image not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Error reading validation data: {e}")
        return False

def check_pytorch_cuda():
    """Check PyTorch CUDA availability"""
    
    print("\n=== PyTorch CUDA Check ===\n")
    
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        return True
    else:
        print("❌ CUDA not available!")
        return False

if __name__ == "__main__":
    print("Checking validation setup...\n")
    
    # Check validation data
    val_ok = check_validation_data()
    
    # Check PyTorch CUDA
    cuda_ok = check_pytorch_cuda()
    
    print("\n=== Summary ===")
    if val_ok and cuda_ok:
        print("✅ All checks passed! Validation should work.")
        print("\nIf you still don't see MAP values, try:")
        print("1. Look carefully for 'Test:' output in training logs")
        print("2. Run: python train.py -c configs/rtdetrv2/rtdetrv2_r18vd_cancer_detection.yml --test-only")
    else:
        print("❌ Some checks failed. Fix the issues first.")

