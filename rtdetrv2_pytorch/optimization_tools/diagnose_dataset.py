#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据集诊断脚本 - 用于排查精度和召回率低的问题
检查：类别分布、目标大小、标注质量、数据增强效果等
"""

import json
import os
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

def load_coco_annotations(ann_file):
    """加载COCO格式的标注文件"""
    with open(ann_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def analyze_class_distribution(data):
    """分析类别分布"""
    print("\n" + "="*60)
    print("1. 类别分布分析")
    print("="*60)
    
    category_counts = Counter()
    category_bbox_counts = defaultdict(int)
    
    for ann in data.get('annotations', []):
        cat_id = ann['category_id']
        category_counts[cat_id] += 1
        category_bbox_counts[cat_id] += 1
    
    # 获取类别名称映射
    cat_id_to_name = {cat['id']: cat['name'] for cat in data.get('categories', [])}
    
    total_annotations = len(data.get('annotations', []))
    total_images = len(data.get('images', []))
    
    print(f"\n总图像数: {total_images}")
    print(f"总标注数: {total_annotations}")
    print(f"平均每张图像标注数: {total_annotations/total_images:.2f}")
    
    print("\n类别分布（按标注数量排序）:")
    print(f"{'类别ID':<8} {'类别名称':<20} {'标注数':<10} {'占比':<10} {'图像数':<10}")
    print("-" * 60)
    
    sorted_cats = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    for cat_id, count in sorted_cats:
        cat_name = cat_id_to_name.get(cat_id, f"Unknown_{cat_id}")
        percentage = (count / total_annotations * 100) if total_annotations > 0 else 0
        # 计算包含该类别的图像数
        images_with_cat = len(set(ann['image_id'] for ann in data.get('annotations', []) 
                                  if ann['category_id'] == cat_id))
        print(f"{cat_id:<8} {cat_name:<20} {count:<10} {percentage:>6.2f}% {images_with_cat:<10}")
    
    # 计算类别不平衡度
    if len(category_counts) > 1:
        counts = list(category_counts.values())
        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        print(f"\n类别不平衡度（最大/最小）: {imbalance_ratio:.2f}")
        if imbalance_ratio > 10:
            print("⚠️  警告: 类别严重不平衡，建议使用类别权重或过采样")
    
    return category_counts, cat_id_to_name

def analyze_bbox_sizes(data, img_folder):
    """分析边界框大小分布"""
    print("\n" + "="*60)
    print("2. 目标大小分析")
    print("="*60)
    
    bbox_areas = []
    bbox_widths = []
    bbox_heights = []
    bbox_ratios = []
    
    # 获取图像尺寸信息
    img_id_to_size = {img['id']: (img['width'], img['height']) 
                      for img in data.get('images', [])}
    
    small_objects = 0  # 面积 < 32*32
    medium_objects = 0  # 32*32 <= 面积 < 96*96
    large_objects = 0  # 面积 >= 96*96
    
    for ann in data.get('annotations', []):
        bbox = ann['bbox']  # [x, y, width, height]
        x, y, w, h = bbox
        
        # 获取图像尺寸
        img_id = ann['image_id']
        img_w, img_h = img_id_to_size.get(img_id, (640, 640))
        
        # 计算归一化尺寸
        norm_w = w / img_w
        norm_h = h / img_h
        area = w * h
        norm_area = norm_w * norm_h
        
        bbox_areas.append(area)
        bbox_widths.append(w)
        bbox_heights.append(h)
        if h > 0:
            bbox_ratios.append(w / h)
        
        # 分类目标大小
        if area < 32 * 32:
            small_objects += 1
        elif area < 96 * 96:
            medium_objects += 1
        else:
            large_objects += 1
    
    if not bbox_areas:
        print("⚠️  没有找到标注框")
        return
    
    bbox_areas = np.array(bbox_areas)
    bbox_widths = np.array(bbox_widths)
    bbox_heights = np.array(bbox_heights)
    
    print(f"\n总目标数: {len(bbox_areas)}")
    print(f"\n目标大小分布:")
    print(f"  小目标 (面积 < 32×32): {small_objects} ({small_objects/len(bbox_areas)*100:.1f}%)")
    print(f"  中目标 (32×32 ≤ 面积 < 96×96): {medium_objects} ({medium_objects/len(bbox_areas)*100:.1f}%)")
    print(f"  大目标 (面积 ≥ 96×96): {large_objects} ({large_objects/len(bbox_areas)*100:.1f}%)")
    
    print(f"\n边界框尺寸统计:")
    print(f"  宽度: 平均={bbox_widths.mean():.1f}, 中位数={np.median(bbox_widths):.1f}, "
          f"最小={bbox_widths.min():.1f}, 最大={bbox_widths.max():.1f}")
    print(f"  高度: 平均={bbox_heights.mean():.1f}, 中位数={np.median(bbox_heights):.1f}, "
          f"最小={bbox_heights.min():.1f}, 最大={bbox_heights.max():.1f}")
    print(f"  面积: 平均={bbox_areas.mean():.1f}, 中位数={np.median(bbox_areas):.1f}, "
          f"最小={bbox_areas.min():.1f}, 最大={bbox_areas.max():.1f}")
    
    if np.median(bbox_areas) < 32 * 32:
        print("\n⚠️  警告: 超过50%的目标是小目标，建议:")
        print("  - 增加num_queries（当前300可能不足）")
        print("  - 使用更小的anchor或更密集的特征图")
        print("  - 调整损失函数权重，更关注小目标")
    
    return {
        'areas': bbox_areas,
        'widths': bbox_widths,
        'heights': bbox_heights,
        'ratios': bbox_ratios,
        'small': small_objects,
        'medium': medium_objects,
        'large': large_objects
    }

def analyze_image_annotations(data):
    """分析每张图像的标注数量"""
    print("\n" + "="*60)
    print("3. 图像标注密度分析")
    print("="*60)
    
    img_ann_count = defaultdict(int)
    for ann in data.get('annotations', []):
        img_ann_count[ann['image_id']] += 1
    
    counts = list(img_ann_count.values())
    if not counts:
        print("⚠️  没有找到标注")
        return
    
    print(f"\n每张图像标注数统计:")
    print(f"  平均: {np.mean(counts):.2f}")
    print(f"  中位数: {np.median(counts):.2f}")
    print(f"  最小: {min(counts)}")
    print(f"  最大: {max(counts)}")
    print(f"  标准差: {np.std(counts):.2f}")
    
    # 统计空图像
    total_images = len(data.get('images', []))
    images_with_ann = len(img_ann_count)
    empty_images = total_images - images_with_ann
    
    print(f"\n图像标注覆盖:")
    print(f"  有标注图像: {images_with_ann} ({images_with_ann/total_images*100:.1f}%)")
    print(f"  空图像: {empty_images} ({empty_images/total_images*100:.1f}%)")
    
    if empty_images > total_images * 0.1:
        print("\n⚠️  警告: 超过10%的图像没有标注，可能影响训练")

def check_annotation_quality(data, img_folder):
    """检查标注质量"""
    print("\n" + "="*60)
    print("4. 标注质量检查")
    print("="*60)
    
    issues = []
    img_id_to_size = {img['id']: (img['width'], img['height']) 
                      for img in data.get('images', [])}
    
    invalid_bboxes = 0
    out_of_bounds = 0
    very_small = 0
    
    for ann in data.get('annotations', []):
        bbox = ann['bbox']
        x, y, w, h = bbox
        img_id = ann['image_id']
        img_w, img_h = img_id_to_size.get(img_id, (640, 640))
        
        # 检查无效框
        if w <= 0 or h <= 0:
            invalid_bboxes += 1
            continue
        
        # 检查超出边界
        if x < 0 or y < 0 or x + w > img_w or y + h > img_h:
            out_of_bounds += 1
        
        # 检查极小框
        if w * h < 4:  # 面积小于4像素
            very_small += 1
    
    total = len(data.get('annotations', []))
    if total > 0:
        print(f"\n标注质量统计:")
        print(f"  总标注数: {total}")
        print(f"  无效框 (w≤0 或 h≤0): {invalid_bboxes} ({invalid_bboxes/total*100:.2f}%)")
        print(f"  超出边界: {out_of_bounds} ({out_of_bounds/total*100:.2f}%)")
        print(f"  极小框 (面积<4): {very_small} ({very_small/total*100:.2f}%)")
        
        if invalid_bboxes > 0 or out_of_bounds > total * 0.05:
            print("\n⚠️  警告: 发现标注质量问题，建议清理数据")

def generate_recommendations(data, bbox_stats):
    """生成优化建议"""
    print("\n" + "="*60)
    print("5. 优化建议")
    print("="*60)
    
    recommendations = []
    
    # 检查类别不平衡
    category_counts = Counter(ann['category_id'] for ann in data.get('annotations', []))
    if len(category_counts) > 1:
        counts = list(category_counts.values())
        imbalance_ratio = max(counts) / min(counts)
        if imbalance_ratio > 10:
            recommendations.append({
                'issue': '类别严重不平衡',
                'solutions': [
                    '启用类别权重 (class_weight_file)',
                    '使用加权采样器 (CocoImageWeightedRandomSampler)',
                    '对少数类别进行过采样'
                ]
            })
    
    # 检查小目标比例
    if bbox_stats and bbox_stats['small'] / (bbox_stats['small'] + bbox_stats['medium'] + bbox_stats['large']) > 0.5:
        recommendations.append({
            'issue': '小目标占比超过50%',
            'solutions': [
                '增加num_queries从300到500-900',
                '调整损失权重，增加loss_vfl权重',
                '使用更密集的特征金字塔',
                '减小SanitizeBoundingBoxes的min_size阈值'
            ]
        })
    
    # 检查数据量
    total_images = len(data.get('images', []))
    if total_images > 30000:
        recommendations.append({
            'issue': '数据集较大（>3万张）',
            'solutions': [
                '考虑使用更大的batch size（如果显存允许）',
                '增加训练轮数',
                '使用更长的warmup',
                '考虑使用学习率衰减策略'
            ]
        })
    
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            print(f"\n{i}. {rec['issue']}:")
            for solution in rec['solutions']:
                print(f"   - {solution}")
    else:
        print("\n✅ 未发现明显问题，建议检查训练配置和超参数")
    
    print("\n通用优化建议:")
    print("  1. 检查训练日志，观察loss是否正常下降")
    print("  2. 验证数据增强策略是否合适（避免过度增强）")
    print("  3. 尝试不同的学习率（当前0.00005可能偏小）")
    print("  4. 增加训练轮数，使用早停机制")
    print("  5. 检查验证集是否与训练集分布一致")

def main():
    import argparse
    parser = argparse.ArgumentParser(description='诊断数据集')
    parser.add_argument('--ann-file', type=str, required=True,
                       help='COCO标注文件路径')
    parser.add_argument('--img-folder', type=str, default=None,
                       help='图像文件夹路径（可选）')
    parser.add_argument('--output-dir', type=str, default='./diagnosis_output',
                       help='输出目录')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("数据集诊断工具")
    print("="*60)
    print(f"\n标注文件: {args.ann_file}")
    
    # 加载数据
    data = load_coco_annotations(args.ann_file)
    
    # 执行分析
    category_counts, cat_id_to_name = analyze_class_distribution(data)
    bbox_stats = analyze_bbox_sizes(data, args.img_folder)
    analyze_image_annotations(data)
    check_annotation_quality(data, args.img_folder)
    generate_recommendations(data, bbox_stats)
    
    print("\n" + "="*60)
    print("诊断完成！")
    print("="*60)

if __name__ == '__main__':
    main()

