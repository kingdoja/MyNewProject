#!/usr/bin/env python3
"""
诊断bbox绘制问题
"""
import json
from PIL import Image, ImageDraw, ImageFont
import os

# 加载标注文件
annotation_file = '/home/ubuntu/lsn/project_new/RT-DETR-main/DATA/SputumCell/p4_split_aug/val/annotations/instances_val.json'
image_dir = '/home/ubuntu/lsn/project_new/RT-DETR-main/DATA/SputumCell/p4_split_aug/val/images'

with open(annotation_file, 'r') as f:
    data = json.load(f)

# 找到patch_1069.png的标注
target_file = 'patch_1069.png'
image_id = None
for img in data['images']:
    if img['file_name'] == target_file:
        image_id = img['id']
        print(f"找到图像: {target_file}, image_id: {image_id}")
        print(f"  标注中的尺寸: {img['width']}x{img['height']}")
        break

if image_id is None:
    print(f"未找到图像: {target_file}")
    exit(1)

# 获取该图像的标注
annotations = [ann for ann in data['annotations'] if ann['image_id'] == image_id]
print(f"\n找到 {len(annotations)} 个标注:")

for i, ann in enumerate(annotations[:3]):
    bbox = ann['bbox']
    print(f"\n标注 {i+1}:")
    print(f"  bbox (原始): {bbox}")
    print(f"  bbox类型: {type(bbox)}, 元素类型: {[type(x) for x in bbox]}")
    print(f"  category_id: {ann['category_id']}")
    
    # 转换为绘制坐标
    if isinstance(bbox, list) and len(bbox) == 4:
        x, y, w, h = bbox
        x1, y1, x2, y2 = x, y, x + w, y + h
        print(f"  转换后: x1={x1:.2f}, y1={y1:.2f}, x2={x2:.2f}, y2={y2:.2f}")

# 加载实际图像
image_path = os.path.join(image_dir, target_file)
if os.path.exists(image_path):
    img = Image.open(image_path)
    print(f"\n实际图像尺寸: {img.size}")
    
    # 绘制一个测试标注
    draw = ImageDraw.Draw(img)
    if annotations:
        ann = annotations[0]
        bbox = ann['bbox']
        
        # 确保是列表且长度为4
        if isinstance(bbox, list) and len(bbox) == 4:
            x, y, w, h = [float(v) for v in bbox]  # 确保是float
            x1, y1, x2, y2 = x, y, x + w, y + h
            
            print(f"\n绘制第一个标注:")
            print(f"  坐标: [{x1:.2f}, {y1:.2f}, {x2:.2f}, {y2:.2f}]")
            print(f"  宽度: {w:.2f}, 高度: {h:.2f}")
            
            # 检查坐标是否在图像范围内
            img_width, img_height = img.size
            print(f"  图像范围: [0, 0, {img_width}, {img_height}]")
            print(f"  坐标是否有效: x1>=0={x1>=0}, y1>=0={y1>=0}, x2<={img_width}={x2<=img_width}, y2<={img_height}={y2<=img_height}")
            
            # 绘制矩形
            draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
            
            # 保存测试图像
            output_path = '/tmp/test_bbox_drawing.png'
            img.save(output_path)
            print(f"\n测试图像已保存到: {output_path}")
else:
    print(f"\n图像文件不存在: {image_path}")



