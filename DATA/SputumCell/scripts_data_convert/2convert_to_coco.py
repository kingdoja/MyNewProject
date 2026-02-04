#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 265.json 转换为 COCO 格式的脚本
根据图像名称来定位图像，忽略错乱的 id
"""

import json
import os
from pathlib import Path
from collections import defaultdict

# COCO 格式的类别定义（与 merged_dataset1/result.json 保持一致）
CATEGORIES = [
    {"id": 0, "name": "AD"},
    {"id": 1, "name": "BC"},
    {"id": 2, "name": "EC"},
    {"id": 3, "name": "L"},
    {"id": 4, "name": "LC"},
    {"id": 5, "name": "M"},
    {"id": 6, "name": "NT"},
    {"id": 7, "name": "SM"},
    {"id": 8, "name": "SQ"},
    {"id": 9, "name": "TC1"},
    {"id": 10, "name": "TC2"},
    {"id": 11, "name": "TC3"},
]

# 创建类别名称到 ID 的映射
CATEGORY_NAME_TO_ID = {cat["name"]: cat["id"] for cat in CATEGORIES}


def percent_to_pixel(value_percent, image_size):
    """将百分比坐标转换为像素坐标"""
    return value_percent * image_size / 100.0


def label_studio_to_bbox(x, y, width, height, img_width, img_height):
    """
    将 Label Studio 格式的矩形标注转换为 COCO 格式的 bbox [x, y, width, height]
    Label Studio 格式：x, y 是左上角坐标（百分比），width, height 是宽度和高度（百分比）
    COCO 格式：[x, y, width, height]，其中 (x, y) 是左上角坐标（像素）
    """
    # 将百分比转换为像素
    # x, y 是左上角坐标（百分比）
    x_pixel = percent_to_pixel(x, img_width)
    y_pixel = percent_to_pixel(y, img_height)
    w_pixel = percent_to_pixel(width, img_width)
    h_pixel = percent_to_pixel(height, img_height)
    
    # 确保坐标在图像范围内
    x_pixel = max(0, min(x_pixel, img_width - 1))
    y_pixel = max(0, min(y_pixel, img_height - 1))
    w_pixel = min(w_pixel, img_width - x_pixel)
    h_pixel = min(h_pixel, img_height - y_pixel)
    
    return [x_pixel, y_pixel, w_pixel, h_pixel]


def polygon_to_segmentation(points, img_width, img_height):
    """
    将多边形点（百分比）转换为 COCO 格式的 segmentation
    COCO 格式：[[x1, y1, x2, y2, ...]]
    """
    segmentation = []
    for point in points:
        x = percent_to_pixel(point[0], img_width)
        y = percent_to_pixel(point[1], img_height)
        segmentation.extend([x, y])
    return [segmentation]


def remove_duplicate_ids(data):
    """
    去除重复的 id，只保留 updated_at 最新的条目
    
    Args:
        data: 原始数据列表
        
    Returns:
        去重后的数据列表
    """
    # 按 id 分组
    id_to_entries = defaultdict(list)
    entries_without_id = []
    
    for entry in data:
        entry_id = entry.get("id")
        if entry_id is not None:
            id_to_entries[entry_id].append(entry)
        else:
            # 没有 id 的条目直接保留
            entries_without_id.append(entry)
    
    # 对于每个 id，只保留 updated_at 最新的条目
    deduplicated_data = []
    duplicate_count = 0
    
    for entry_id, entries in id_to_entries.items():
        if len(entries) > 1:
            # 有重复，找到 updated_at 最新的
            duplicate_count += len(entries) - 1
            # 使用 updated_at 字符串比较（ISO 8601 格式可以直接字符串比较）
            latest_entry = max(entries, key=lambda e: e.get("updated_at", "1970-01-01T00:00:00Z"))
            deduplicated_data.append(latest_entry)
        else:
            # 没有重复，直接添加
            deduplicated_data.append(entries[0])
    
    # 添加没有 id 的条目
    deduplicated_data.extend(entries_without_id)
    
    print(f"去重完成: 原始条目数 {len(data)}, 去重后条目数 {len(deduplicated_data)}, 移除重复条目数 {duplicate_count}")
    
    return deduplicated_data


def convert_to_coco(input_json_path, output_json_path, images_dir):
    """
    将 265.json 转换为 COCO 格式
    
    Args:
        input_json_path: 输入的 265.json 文件路径
        output_json_path: 输出的 COCO 格式 JSON 文件路径
        images_dir: 图像文件所在的目录
    """
    # 读取输入 JSON
    print(f"正在读取 {input_json_path}...")
    with open(input_json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 去除重复的 id，只保留 updated_at 最新的条目
    print("正在处理重复的 id...")
    data = remove_duplicate_ids(data)
    
    # 获取所有图像文件
    images_dir = Path(images_dir)
    image_files = {f.name: f for f in images_dir.glob("patch_*.png")}
    print(f"找到 {len(image_files)} 个图像文件")
    
    # 构建 COCO 格式数据
    coco_data = {
        "info": {
            "description": "Converted from 265.json",
            "version": "1.0",
            "year": 2024
        },
        "licenses": [],
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }
    
    # 用于跟踪图像 ID 和文件名映射
    image_name_to_id = {}
    image_id_counter = 0
    annotation_id_counter = 0
    
    # 统计信息
    stats = {
        "total_entries": len(data),
        "images_with_annotations": 0,
        "images_without_annotations": 0,
        "total_bbox_annotations": 0,
        "total_polygon_annotations": 0,
        "missing_images": []
    }
    
    # 处理每个条目
    for entry in data:
        image_path = entry.get("image", "")
        if not image_path:
            continue
        
        # 从路径中提取文件名
        image_filename = os.path.basename(image_path)
        
        # 检查图像文件是否存在
        if image_filename not in image_files:
            stats["missing_images"].append(image_filename)
            continue
        
        # 获取或创建图像 ID
        if image_filename not in image_name_to_id:
            image_id = image_id_counter
            image_id_counter += 1
            image_name_to_id[image_filename] = image_id
            
            # 添加图像信息（假设所有图像都是 640x640）
            coco_data["images"].append({
                "id": image_id,
                "width": 640,
                "height": 640,
                "file_name": image_filename
            })
        else:
            image_id = image_name_to_id[image_filename]
        
        # 处理矩形框标注 (label)
        has_annotations = False
        if "label" in entry and entry["label"]:
            has_annotations = True
            for label_item in entry["label"]:
                if not label_item.get("rectanglelabels"):
                    continue
                
                category_name = label_item["rectanglelabels"][0]
                if category_name not in CATEGORY_NAME_TO_ID:
                    print(f"警告: 未知类别 '{category_name}'，跳过")
                    continue
                
                # 检查必需的字段是否存在
                if "x" not in label_item or "y" not in label_item or "width" not in label_item or "height" not in label_item:
                    print(f"警告: 标注缺少必需的坐标字段 (x, y, width, height)，跳过")
                    continue
                
                category_id = CATEGORY_NAME_TO_ID[category_name]
                img_width = label_item.get("original_width", 640)
                img_height = label_item.get("original_height", 640)
                
                # 转换 bbox（Label Studio 格式：x, y 是左上角坐标百分比）
                bbox = label_studio_to_bbox(
                    label_item["x"],
                    label_item["y"],
                    label_item["width"],
                    label_item["height"],
                    img_width,
                    img_height
                )
                
                # 计算面积
                area = bbox[2] * bbox[3]
                
                # 添加标注
                coco_data["annotations"].append({
                    "id": annotation_id_counter,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": [],
                    "bbox": bbox,
                    "ignore": 0,
                    "iscrowd": 0,
                    "area": area
                })
                
                annotation_id_counter += 1
                stats["total_bbox_annotations"] += 1
        
        # 处理多边形标注 (label2)
        if "label2" in entry and entry["label2"]:
            has_annotations = True
            for label2_item in entry["label2"]:
                if not label2_item.get("polygonlabels"):
                    continue
                
                category_name = label2_item["polygonlabels"][0]
                if category_name not in CATEGORY_NAME_TO_ID:
                    print(f"警告: 未知类别 '{category_name}'，跳过")
                    continue
                
                category_id = CATEGORY_NAME_TO_ID[category_name]
                img_width = label2_item.get("original_width", 640)
                img_height = label2_item.get("original_height", 640)
                
                # 转换多边形
                points = label2_item.get("points", [])
                if not points:
                    continue
                
                # 验证点的格式是否正确（每个点应该是 [x, y] 格式）
                try:
                    # 检查点的格式
                    for p in points:
                        if not isinstance(p, (list, tuple)) or len(p) < 2:
                            raise ValueError(f"点的格式不正确: {p}")
                except (ValueError, TypeError) as e:
                    print(f"警告: 多边形点格式错误，跳过: {e}")
                    continue
                
                segmentation = polygon_to_segmentation(points, img_width, img_height)
                
                # 计算 bbox（从多边形点计算边界框）
                x_coords = [percent_to_pixel(p[0], img_width) for p in points]
                y_coords = [percent_to_pixel(p[1], img_height) for p in points]
                x_min = max(0, min(x_coords))
                y_min = max(0, min(y_coords))
                x_max = min(img_width, max(x_coords))
                y_max = min(img_height, max(y_coords))
                bbox = [x_min, y_min, x_max - x_min, y_max - y_min]
                
                # 计算面积（使用多边形面积公式）
                area = 0
                n = len(points)
                for i in range(n):
                    j = (i + 1) % n
                    area += x_coords[i] * y_coords[j]
                    area -= x_coords[j] * y_coords[i]
                area = abs(area) / 2.0
                
                # 添加标注
                coco_data["annotations"].append({
                    "id": annotation_id_counter,
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": segmentation,
                    "bbox": bbox,
                    "ignore": 0,
                    "iscrowd": 0,
                    "area": area
                })
                
                annotation_id_counter += 1
                stats["total_polygon_annotations"] += 1
        
        if has_annotations:
            stats["images_with_annotations"] += 1
        else:
            stats["images_without_annotations"] += 1
    
    # 保存 COCO 格式 JSON
    print(f"正在保存到 {output_json_path}...")
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, indent=2, ensure_ascii=False)
    
    # 打印统计信息
    print("\n转换完成！统计信息：")
    print(f"  总条目数: {stats['total_entries']}")
    print(f"  有标注的图像: {stats['images_with_annotations']}")
    print(f"  无标注的图像: {stats['images_without_annotations']}")
    print(f"  矩形框标注数: {stats['total_bbox_annotations']}")
    print(f"  多边形标注数: {stats['total_polygon_annotations']}")
    print(f"  总标注数: {len(coco_data['annotations'])}")
    print(f"  总图像数: {len(coco_data['images'])}")
    if stats["missing_images"]:
        print(f"  缺失的图像文件数: {len(stats['missing_images'])}")
        if len(stats["missing_images"]) <= 10:
            print(f"  缺失的图像: {stats['missing_images']}")
        else:
            print(f"  缺失的图像（前10个）: {stats['missing_images'][:10]}")
    
    return coco_data


if __name__ == "__main__":
    # 设置路径
    script_dir = Path(__file__).parent  # DATA/SputumCell/scripts/
    patches_dir = script_dir.parent / "new37"  # DATA/SputumCell/p4_100/
    input_json = patches_dir / "37.json"
    output_json = patches_dir / "coco_format.json"
    images_dir = patches_dir / "images"  # 图像文件在 images/ 子目录下
    
    # 执行转换
    convert_to_coco(str(input_json), str(output_json), str(images_dir))
    print(f"\n转换完成！输出文件: {output_json}")

