#!/usr/bin/env python3
"""
Quick script to check COCO annotation file information
"""

import json
import os

def check_coco_info(coco_file_path):
    """Check basic information of a COCO annotation file"""
    
    if not os.path.exists(coco_file_path):
        print(f"File not found: {coco_file_path}")
        return
    
    try:
        with open(coco_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"=== COCO File: {os.path.basename(coco_file_path)} ===")
        print(f"Images count: {len(data.get('images', []))}")
        print(f"Annotations count: {len(data.get('annotations', []))}")
        print(f"Categories count: {len(data.get('categories', []))}")
        
        if 'categories' in data:
            print("\nCategories:")
            for cat in data['categories']:
                print(f"  - ID {cat['id']}: {cat['name']}")
        
        if 'images' in data and len(data['images']) > 0:
            sample_img = data['images'][0]
            print(f"\nSample image info:")
            print(f"  - Filename: {sample_img.get('file_name', 'N/A')}")
            print(f"  - Size: {sample_img.get('width', 'N/A')}x{sample_img.get('height', 'N/A')}")
        
        if 'annotations' in data and len(data['annotations']) > 0:
            sample_ann = data['annotations'][0]
            print(f"\nSample annotation:")
            print(f"  - Category ID: {sample_ann.get('category_id', 'N/A')}")
            print(f"  - Bbox: {sample_ann.get('bbox', 'N/A')}")
            print(f"  - Area: {sample_ann.get('area', 'N/A')}")
            
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    print("Checking your cancer detection dataset...\n")
    
    # Check train annotations
    train_ann = "/home/ubuntu/lsn/project_new/RT-DETR-main/DATA/APCData cervical cytology cells/split_YOLO/annotations/train.json"
    check_coco_info(train_ann)
    
    print("\n" + "="*50 + "\n")
    
    # Check validation annotations
    val_ann = "/home/ubuntu/lsn/project_new/RT-DETR-main/DATA/APCData cervical cytology cells/split_YOLO/annotations/val.json"
    check_coco_info(val_ann)
    
    print("\n" + "="*50 + "\n")
    
    # Check test annotations
    test_ann = "/home/ubuntu/lsn/project_new/RT-DETR-main/DATA/APCData cervical cytology cells/split_YOLO/annotations/test.json"
    if os.path.exists(test_ann):
        check_coco_info(test_ann)
    else:
        print("Test set annotations not found.")