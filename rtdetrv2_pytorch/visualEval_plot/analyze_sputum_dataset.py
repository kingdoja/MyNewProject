#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
痰液细胞数据集分析脚本
分析数据集特点，为训练提供指导
"""

import json
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from collections import Counter
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class SputumDatasetAnalyzer:
    def __init__(self, data_root):
        self.data_root = data_root
        self.classes = self.load_classes()
        self.train_annotations = self.load_annotations('train')
        self.val_annotations = self.load_annotations('val')
        
    def load_classes(self):
        """加载类别信息"""
        classes_file = os.path.join(self.data_root, 'classes.txt')
        with open(classes_file, 'r') as f:
            classes = [line.strip() for line in f.readlines() if line.strip()]
        return classes
    
    def load_annotations(self, split):
        """加载标注文件"""
        ann_file = os.path.join(self.data_root, split, 'annotations', f'instances_{split}.json')
        with open(ann_file, 'r') as f:
            return json.load(f)
    
    def analyze_class_distribution(self):
        """分析类别分布"""
        print("=" * 50)
        print("类别分布分析")
        print("=" * 50)
        
        # 训练集类别分布
        train_categories = {}
        for ann in self.train_annotations['annotations']:
            cat_id = ann['category_id']
            if cat_id not in train_categories:
                train_categories[cat_id] = 0
            train_categories[cat_id] += 1
        
        # 验证集类别分布
        val_categories = {}
        for ann in self.val_annotations['annotations']:
            cat_id = ann['category_id']
            if cat_id not in val_categories:
                val_categories[cat_id] = 0
            val_categories[cat_id] += 1
        
        # 创建类别分布图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # 训练集 - 修复：类别索引从0开始
        train_counts = [train_categories.get(i, 0) for i in range(len(self.classes))]
        ax1.bar(range(len(self.classes)), train_counts)
        ax1.set_title('Training Set Class Distribution')
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Number of Samples')
        ax1.set_xticks(range(len(self.classes)))
        ax1.set_xticklabels(self.classes, rotation=45)
        
        # 验证集 - 修复：类别索引从0开始
        val_counts = [val_categories.get(i, 0) for i in range(len(self.classes))]
        ax2.bar(range(len(self.classes)), val_counts)
        ax2.set_title('Validation Set Class Distribution')
        ax2.set_xlabel('Class')
        ax2.set_ylabel('Number of Samples')
        ax2.set_xticks(range(len(self.classes)))
        ax2.set_xticklabels(self.classes, rotation=45)
        
        plt.tight_layout()
        # 确保保存目录存在
        output_dir = '/home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/plot/dataset'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'sputum_class_distribution.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 打印统计信息
        print(f"训练集总样本数: {len(self.train_annotations['images'])}")
        print(f"验证集总样本数: {len(self.val_annotations['images'])}")
        print(f"训练集总标注数: {len(self.train_annotations['annotations'])}")
        print(f"验证集总标注数: {len(self.val_annotations['annotations'])}")
        
        print("\n训练集各类别样本数:")
        for i, class_name in enumerate(self.classes):
            count = train_categories.get(i, 0)
            print(f"  {class_name}: {count}")
        
        print("\n验证集各类别样本数:")
        for i, class_name in enumerate(self.classes):
            count = val_categories.get(i, 0)
            print(f"  {class_name}: {count}")
    
    def analyze_object_sizes(self):
        """分析目标尺寸分布"""
        print("\n" + "=" * 50)
        print("目标尺寸分析")
        print("=" * 50)
        
        def get_object_sizes(annotations):
            sizes = []
            areas = []
            for ann in annotations['annotations']:
                bbox = ann['bbox']  # [x, y, w, h]
                w, h = bbox[2], bbox[3]
                area = w * h
                sizes.append([w, h])
                areas.append(area)
            return np.array(sizes), np.array(areas)
        
        train_sizes, train_areas = get_object_sizes(self.train_annotations)
        val_sizes, val_areas = get_object_sizes(self.val_annotations)
        
        # 创建尺寸分析图
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 宽度分布
        ax1.hist(train_sizes[:, 0], bins=50, alpha=0.7, label='Training Set', color='blue')
        ax1.hist(val_sizes[:, 0], bins=50, alpha=0.7, label='Validation Set', color='red')
        ax1.set_title('Object Width Distribution')
        ax1.set_xlabel('Width (pixels)')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        
        # 高度分布
        ax2.hist(train_sizes[:, 1], bins=50, alpha=0.7, label='Training Set', color='blue')
        ax2.hist(val_sizes[:, 1], bins=50, alpha=0.7, label='Validation Set', color='red')
        ax2.set_title('Object Height Distribution')
        ax2.set_xlabel('Height (pixels)')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        
        # 面积分布
        ax3.hist(train_areas, bins=50, alpha=0.7, label='Training Set', color='blue')
        ax3.hist(val_areas, bins=50, alpha=0.7, label='Validation Set', color='red')
        ax3.set_title('Object Area Distribution')
        ax3.set_xlabel('Area (pixels²)')
        ax3.set_ylabel('Frequency')
        ax3.legend()
        
        # 宽高比分布
        aspect_ratios = train_sizes[:, 0] / (train_sizes[:, 1] + 1e-6)
        ax4.hist(aspect_ratios, bins=50, alpha=0.7, color='blue')
        ax4.set_title('Object Aspect Ratio Distribution')
        ax4.set_xlabel('Aspect Ratio')
        ax4.set_ylabel('Frequency')
        
        plt.tight_layout()
        # 确保保存目录存在
        output_dir = '/home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/plot/dataset'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'sputum_object_sizes.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # 打印统计信息
        print(f"训练集目标统计:")
        print(f"  平均宽度: {np.mean(train_sizes[:, 0]):.2f} ± {np.std(train_sizes[:, 0]):.2f}")
        print(f"  平均高度: {np.mean(train_sizes[:, 1]):.2f} ± {np.std(train_sizes[:, 1]):.2f}")
        print(f"  平均面积: {np.mean(train_areas):.2f} ± {np.std(train_areas):.2f}")
        print(f"  最小面积: {np.min(train_areas):.2f}")
        print(f"  最大面积: {np.max(train_areas):.2f}")
        
        # 小目标分析
        small_objects = train_areas < 32 * 32  # 小于32x32像素的目标
        print(f"  小目标数量: {np.sum(small_objects)} ({np.sum(small_objects)/len(train_areas)*100:.1f}%)")
        
        medium_objects = (train_areas >= 32 * 32) & (train_areas < 96 * 96)
        print(f"  中等目标数量: {np.sum(medium_objects)} ({np.sum(medium_objects)/len(train_areas)*100:.1f}%)")
        
        large_objects = train_areas >= 96 * 96
        print(f"  大目标数量: {np.sum(large_objects)} ({np.sum(large_objects)/len(train_areas)*100:.1f}%)")
    
    def analyze_images_per_class(self):
        """分析每张图片的类别分布"""
        print("\n" + "=" * 50)
        print("图片类别分布分析")
        print("=" * 50)
        
        def get_images_per_class(annotations):
            image_classes = {}
            for ann in annotations['annotations']:
                image_id = ann['image_id']
                cat_id = ann['category_id']
                if image_id not in image_classes:
                    image_classes[image_id] = []
                image_classes[image_id].append(cat_id)
            return image_classes
        
        train_image_classes = get_images_per_class(self.train_annotations)
        val_image_classes = get_images_per_class(self.val_annotations)
        
        # 统计每张图片的目标数量
        train_objects_per_image = [len(classes) for classes in train_image_classes.values()]
        val_objects_per_image = [len(classes) for classes in val_image_classes.values()]
        
        # 创建分布图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        ax1.hist(train_objects_per_image, bins=30, alpha=0.7, color='blue', label='Training Set')
        ax1.set_title('Objects Per Image Distribution (Training Set)')
        ax1.set_xlabel('Number of Objects')
        ax1.set_ylabel('Number of Images')
        ax1.legend()
        
        ax2.hist(val_objects_per_image, bins=30, alpha=0.7, color='red', label='Validation Set')
        ax2.set_title('Objects Per Image Distribution (Validation Set)')
        ax2.set_xlabel('Number of Objects')
        ax2.set_ylabel('Number of Images')
        ax2.legend()
        
        plt.tight_layout()
        # 确保保存目录存在
        output_dir = '/home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/plot/dataset'
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(os.path.join(output_dir, 'sputum_objects_per_image.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"训练集平均每张图片目标数: {np.mean(train_objects_per_image):.2f}")
        print(f"验证集平均每张图片目标数: {np.mean(val_objects_per_image):.2f}")
    
    def generate_training_recommendations(self):
        """生成训练建议"""
        print("\n" + "=" * 50)
        print("训练建议")
        print("=" * 50)
        
        # 分析目标尺寸
        train_sizes, train_areas = self.get_object_sizes(self.train_annotations)
        small_objects = train_areas < 32 * 32
        small_ratio = np.sum(small_objects) / len(train_areas)
        
        print("基于数据集分析的建议:")
        print(f"1. 小目标比例: {small_ratio*100:.1f}% - 建议使用小目标检测优化策略")
        print("2. 数据增强建议:")
        print("   - 启用Mosaic数据增强")
        print("   - 使用多尺度训练")
        print("   - 增加颜色增强和几何变换")
        print("3. 模型配置建议:")
        print("   - 增加查询数量到500-800")
        print("   - 使用更深的特征金字塔")
        print("   - 调整损失函数权重")
        print("4. 训练策略建议:")
        print("   - 使用较小的学习率")
        print("   - 增加训练轮数")
        print("   - 使用学习率预热")
        print("   - 启用早停机制")
    
    def get_object_sizes(self, annotations):
        """获取目标尺寸"""
        sizes = []
        areas = []
        for ann in annotations['annotations']:
            bbox = ann['bbox']
            w, h = bbox[2], bbox[3]
            area = w * h
            sizes.append([w, h])
            areas.append(area)
        return np.array(sizes), np.array(areas)

def main():
    """主函数"""
    data_root = "/home/ubuntu/lsn/project_new/RT-DETR-main/DATA/SputumCell/split_dataset"
    
    if not os.path.exists(data_root):
        print(f"错误: 数据目录不存在 - {data_root}")
        return
    
    print("痰液细胞数据集分析")
    print("=" * 50)
    
    # 创建分析器
    analyzer = SputumDatasetAnalyzer(data_root)
    
    # 执行分析
    analyzer.analyze_class_distribution()
    analyzer.analyze_object_sizes()
    analyzer.analyze_images_per_class()
    analyzer.generate_training_recommendations()
    
    print("\n分析完成! 图表已保存到指定目录。")

if __name__ == "__main__":
    main()