#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将基于全图的标注框坐标转换为基于640x640切片的COCO格式标注文件

功能：
1. 读取基于全图的标注文件（标记.json格式）
2. 读取patch坐标信息（patch_coordinates.csv）
3. 将全图坐标转换为patch内的相对坐标
4. 生成COCO格式的JSON文件
5. 在patch上可视化标注框

使用方法：
python convert_global_to_patch.py \
    --annotation-file /path/to/标记.json \
    --patch-dir /path/to/patch_dir \
    --output-dir /path/to/output \
    --wsi-image /path/to/wsi.jpeg
"""

import argparse
import json
import os
import csv
import io
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

try:
    from pypinyin import lazy_pinyin  # type: ignore[import-not-found]
except ImportError:
    lazy_pinyin = None


# 类别名称到ID的映射（与COCO格式中的categories对应）
CLASS_NAME_TO_ID = {
    "AD": 0,
    "BC": 1,
    "EC": 2,
    "L": 3,
    "LC": 4,
    "M": 5,
    "NT": 6,
    "SM": 7,
    "SQ": 8,
    "TC1": 9,
    "TC2": 10,
    "TC3": 11,
}

# COCO格式的类别定义
COCO_CATEGORIES = [
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


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="将基于全图的标注框转换为基于切片的COCO格式标注"
    )
    parser.add_argument(
        "--annotation-file",
        type=str,
        default=None,
        help="输入的标注文件路径(标记.json格式)",
    )
    parser.add_argument(
        "--patch-dir",
        type=str,
        default=None,
        help="patch图像所在目录",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="输出目录",
    )
    parser.add_argument(
        "--wsi-image",
        type=str,
        default=None,
        help="全图路径(可选，用于验证)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=640,
        help="patch尺寸(默认640)",
    )
    parser.add_argument(
        "--coordinates-csv",
        type=str,
        default=None,
        help="patch坐标CSV文件路径(默认: patch_dir/patch_coordinates.csv)",
    )
    parser.add_argument(
        "--min-overlap-ratio",
        type=float,
        default=0.1,
        help="标注框与patch的最小重叠比例阈值(默认0.1，即10%%)",
    )
    parser.add_argument(
        "--no-visualization",
        action="store_true",
        help="不生成可视化图片",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="批量处理模式：自动处理input目录下所有json文件",
    )
    parser.add_argument(
        "--input-annotation-dir",
        type=str,
        default="/home/ubuntu/lsn/project_new/RT-DETR-main/A_sclie2inference/DataSlice2Inference_main/annotationConverter/input",
        help="批量模式下annotation json目录",
    )
    parser.add_argument(
        "--patch-root-dir",
        type=str,
        default="/home/ubuntu/lsn/project_new/RT-DETR-main/A_sclie2inference/DataPatches",
        help="批量模式下patch根目录（其下每个子目录对应一个样本）",
    )
    
    return parser.parse_args()


def normalize_key(value: str) -> str:
    """将字符串规范化为便于匹配的key。"""
    return re.sub(r"[^a-z0-9]", "", value.lower())


def to_pinyin_text(value: str) -> str:
    """将中文转换为不带声调拼音；非中文字符保留。"""
    if lazy_pinyin is None:
        return value
    return "".join(lazy_pinyin(value))


def parse_patch_identity(patch_dir_name: str) -> Tuple[str, str]:
    """解析patch目录标识和编号前缀。
    
    目录名示例：7-邱训宾_20260304_175758
    - identity_raw: 7-邱训宾
    - id_prefix: 7
    """
    identity_raw = patch_dir_name.split("_")[0]
    id_prefix = identity_raw.split("-")[0].strip().lower()
    return identity_raw, id_prefix


def resolve_patch_dir_for_json(json_stem: str, patch_dir_entries: List[dict]) -> Optional[dict]:
    """根据json名匹配patch目录。
    
    优先级：
    1) identity原文精确匹配
    2) identity拼音精确匹配
    3) 编号前缀匹配（如7-xxx -> 7）
    """
    json_key = normalize_key(json_stem)

    # 1) 原文/拼音精确匹配
    exact_candidates = []
    for entry in patch_dir_entries:
        if json_key == entry["identity_raw_key"] or json_key == entry["identity_pinyin_key"]:
            exact_candidates.append(entry)
    if len(exact_candidates) == 1:
        return exact_candidates[0]
    if len(exact_candidates) > 1:
        # 多个精确候选时按相似度取最高
        best = max(
            exact_candidates,
            key=lambda x: max(
                SequenceMatcher(None, json_key, x["identity_raw_key"]).ratio(),
                SequenceMatcher(None, json_key, x["identity_pinyin_key"]).ratio(),
            ),
        )
        return best

    # 2) 编号前缀匹配
    json_id_prefix = json_stem.split("-")[0].strip().lower()
    id_candidates = [x for x in patch_dir_entries if x["id_prefix"] == json_id_prefix]
    if len(id_candidates) == 1:
        return id_candidates[0]
    if len(id_candidates) > 1:
        best = max(
            id_candidates,
            key=lambda x: max(
                SequenceMatcher(None, json_key, x["identity_raw_key"]).ratio(),
                SequenceMatcher(None, json_key, x["identity_pinyin_key"]).ratio(),
            ),
        )
        return best

    return None


def load_annotations(annotation_file: str) -> List[dict]:
    """加载标注文件
    
    Args:
        annotation_file: 标注文件路径
        
    Returns:
        标注列表，每个标注包含box、cellType等信息
    """
    with open(annotation_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    print(f"✓ 已加载 {len(annotations)} 个标注框")
    return annotations


def load_patch_coordinates(csv_path: str) -> Tuple[Dict[str, Tuple[int, int, int, int]], float]:
    """加载patch坐标信息
    
    Args:
        csv_path: CSV文件路径
        
    Returns:
        (coordinates, scale_factor):
            - coordinates: {filename: (x_start, y_start, x_end, y_end)} 字典
            - scale_factor: 图像缩放系数（从注释行提取，默认1.0）
    """
    coordinates = {}
    scale_factor = 1.0  # 默认无缩放
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"坐标CSV文件不存在: {csv_path}")
    
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        # 读取所有行，提取scale_factor并过滤注释行
        lines = []
        for line in f:
            stripped = line.strip()
            if stripped.startswith('# Scale factor:'):
                # 提取缩放系数: "# Scale factor: 2.0x" -> 2.0
                try:
                    scale_str = stripped.split(':')[1].strip().rstrip('x')
                    scale_factor = float(scale_str)
                    print(f"✓ 检测到图像缩放系数: {scale_factor}x")
                except (IndexError, ValueError) as e:
                    print(f"⚠️ 无法解析缩放系数: {stripped}, 使用默认值1.0")
            elif not stripped.startswith('#'):
                lines.append(line)
        
        # 使用过滤后的行创建 DictReader
        csv_content = io.StringIO(''.join(lines))
        reader = csv.DictReader(csv_content)
        
        for row in reader:
            filename = row['filename']
            x_start = int(row['x_start'])
            y_start = int(row['y_start'])
            x_end = int(row['x_end'])
            y_end = int(row['y_end'])
            coordinates[filename] = (x_start, y_start, x_end, y_end)
    
    print(f"✓ 已加载 {len(coordinates)} 个patch的坐标信息")
    if scale_factor != 1.0:
        print(f"✓ CSV坐标已映射到原图（缩放系数: {scale_factor}x）")
    return coordinates, scale_factor


def parse_box(box_str: str) -> Tuple[float, float, float, float]:
    """解析标注框坐标
    
    Args:
        box_str: 字符串格式的坐标，如 "[21905.254, 13596.98, 21967.307, 13656.157]"
        
    Returns:
        (x1, y1, x2, y2) 坐标元组
    """
    # 移除方括号和空格，然后分割
    box_str = box_str.strip('[]')
    parts = [float(x.strip()) for x in box_str.split(',')]
    if len(parts) != 4:
        raise ValueError(f"无效的box格式: {box_str}")
    return tuple(parts)


def calculate_iou(box1: Tuple[float, float, float, float], 
                  box2: Tuple[float, float, float, float]) -> float:
    """计算两个框的IoU（交并比）
    
    Args:
        box1: (x1, y1, x2, y2)
        box2: (x1, y1, x2, y2)
        
    Returns:
        IoU值（0-1之间）
    """
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    
    # 计算交集
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # 计算并集
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def calculate_overlap_ratio(box: Tuple[float, float, float, float],
                           patch_box: Tuple[float, float, float, float]) -> float:
    """计算标注框与patch的重叠比例（基于标注框的面积）
    
    Args:
        box: 标注框 (x1, y1, x2, y2)
        patch_box: patch框 (x1, y1, x2, y2)
        
    Returns:
        重叠部分占标注框面积的比例
    """
    x1, y1, x2, y2 = box
    px1, py1, px2, py2 = patch_box
    
    # 计算交集
    x1_i = max(x1, px1)
    y1_i = max(y1, py1)
    x2_i = min(x2, px2)
    y2_i = min(y2, py2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    box_area = (x2 - x1) * (y2 - y1)
    
    if box_area == 0:
        return 0.0
    
    return intersection / box_area


def convert_to_patch_coordinates(
    global_box: Tuple[float, float, float, float],
    patch_offset: Tuple[int, int],
    scale_factor: float = 1.0
) -> Tuple[float, float, float, float]:
    """将全图坐标转换为patch内的相对坐标
    
    Args:
        global_box: 全图坐标 (x1, y1, x2, y2) - 原图坐标系
        patch_offset: patch在全图中的偏移量 (x_start, y_start) - 原图坐标系
        scale_factor: 图像缩放系数，用于将原图坐标映射到patch图像尺度
        
    Returns:
        patch内的相对坐标 (x1, y1, x2, y2) - 相对于640x640的patch图像
        
    说明：
        当图像经过缩放时（如从100000x80000缩放到50000x40000，scale_factor=2.0），
        patch是在缩放后的图像上切的640x640，但CSV中的patch_offset已经映射回原图。
        因此需要：
        1. 先计算原图坐标差值：(global_box - patch_offset)
        2. 再除以scale_factor，映射到640x640的patch图像尺度
        
        例如：
        - 原图100000x80000，缩放为50000x40000（scale_factor=2.0）
        - patch在原图: (1280, 0, 2560, 1280) - CSV保存
        - patch图像: 640x640（在缩放图上切的）
        - 标注框在原图: (1480, 300, 1680, 500)
        - 步骤1: (1480-1280, 300-0, 1680-1280, 500-0) = (200, 300, 400, 500) - 原图尺度
        - 步骤2: (200/2, 300/2, 400/2, 500/2) = (100, 150, 200, 250) - patch图像尺度 ✅
    """
    x1_g, y1_g, x2_g, y2_g = global_box
    x_offset, y_offset = patch_offset
    
    # 步骤1：计算原图坐标差值
    x1_diff = x1_g - x_offset
    y1_diff = y1_g - y_offset
    x2_diff = x2_g - x_offset
    y2_diff = y2_g - y_offset
    
    # 步骤2：除以scale_factor，映射到patch图像尺度（640x640）
    x1_p = x1_diff / scale_factor
    y1_p = y1_diff / scale_factor
    x2_p = x2_diff / scale_factor
    y2_p = y2_diff / scale_factor
    
    return (x1_p, y1_p, x2_p, y2_p)


def clip_box_to_patch(
    box: Tuple[float, float, float, float],
    patch_size: int
) -> Optional[Tuple[float, float, float, float]]:
    """将框裁剪到patch范围内
    
    Args:
        box: patch内的坐标 (x1, y1, x2, y2)
        patch_size: patch尺寸
        
    Returns:
        裁剪后的坐标，如果框完全在patch外则返回None
    """
    x1, y1, x2, y2 = box
    
    # 裁剪到patch范围内
    x1 = max(0, min(x1, patch_size))
    y1 = max(0, min(y1, patch_size))
    x2 = max(0, min(x2, patch_size))
    y2 = max(0, min(y2, patch_size))
    
    # 检查是否有效
    if x2 <= x1 or y2 <= y1:
        return None
    
    return (x1, y1, x2, y2)


def convert_bbox_to_coco_format(
    box: Tuple[float, float, float, float]
) -> List[float]:
    """将 (x1, y1, x2, y2) 格式转换为COCO格式 [x, y, width, height]
    
    Args:
        box: (x1, y1, x2, y2) 格式的坐标
        
    Returns:
        [x, y, width, height] 格式的坐标
    """
    x1, y1, x2, y2 = box
    x = x1
    y = y1
    width = x2 - x1
    height = y2 - y1
    return [x, y, width, height]


def assign_annotations_to_patches(
    annotations: List[dict],
    patch_coordinates: Dict[str, Tuple[int, int, int, int]],
    patch_size: int,
    min_overlap_ratio: float,
    scale_factor: float = 1.0
) -> Dict[str, List[dict]]:
    """将标注框分配到对应的patch
    
    Args:
        annotations: 标注列表
        patch_coordinates: patch坐标字典（原图坐标系）
        patch_size: patch尺寸（实际图像尺寸，如640）
        min_overlap_ratio: 最小重叠比例阈值
        scale_factor: 图像缩放系数
        
    Returns:
        {patch_filename: [annotation_dict, ...]} 字典
    """
    patch_annotations = defaultdict(list)
    
    print(f"\n开始分配标注框到patch...")
    print(f"最小重叠比例阈值: {min_overlap_ratio}")
    if scale_factor != 1.0:
        print(f"缩放系数: {scale_factor}x（标注框坐标将除以{scale_factor}映射到patch图像）")
    
    for ann_idx, ann in enumerate(tqdm(annotations, desc="处理标注框")):
        # 解析标注框坐标
        if 'boxList' in ann:
            box_global = tuple(ann['boxList'])
        elif 'box' in ann:
            box_global = parse_box(ann['box'])
        else:
            print(f"⚠️ 警告：标注 {ann_idx} 没有box或boxList字段，跳过")
            continue
        
        # 获取类别
        cell_type = ann.get('cellType', 'UNKNOWN')
        category_id = CLASS_NAME_TO_ID.get(cell_type, -1)
        if category_id == -1:
            print(f"⚠️ 警告：未知类别 {cell_type}，跳过")
            continue
        
        # 查找与这个标注框有重叠的patch
        for patch_filename, (px1, py1, px2, py2) in patch_coordinates.items():
            patch_box = (px1, py1, px2, py2)
            
            # 计算重叠比例
            overlap_ratio = calculate_overlap_ratio(box_global, patch_box)
            
            if overlap_ratio >= min_overlap_ratio:
                # 转换为patch内的相对坐标（考虑scale_factor）
                patch_offset = (px1, py1)
                box_patch = convert_to_patch_coordinates(box_global, patch_offset, scale_factor)
                
                # 裁剪到patch范围内
                box_patch_clipped = clip_box_to_patch(box_patch, patch_size)
                
                if box_patch_clipped is not None:
                    # 创建patch内的标注信息
                    patch_ann = {
                        'original_annotation': ann,
                        'global_box': box_global,
                        'patch_box': box_patch_clipped,
                        'category_id': category_id,
                        'category_name': cell_type,
                        'overlap_ratio': overlap_ratio,
                    }
                    patch_annotations[patch_filename].append(patch_ann)
    
    # 统计信息
    total_patches_with_annotations = len(patch_annotations)
    total_annotations_assigned = sum(len(anns) for anns in patch_annotations.values())
    
    print(f"\n✓ 分配完成:")
    print(f"  - 有标注的patch数量: {total_patches_with_annotations}")
    print(f"  - 分配的标注框总数: {total_annotations_assigned}")
    print(f"  - 平均每个patch的标注数: {total_annotations_assigned / total_patches_with_annotations:.2f}" if total_patches_with_annotations > 0 else "  - 平均每个patch的标注数: 0")
    
    return patch_annotations


def generate_coco_format(
    patch_annotations: Dict[str, List[dict]],
    patch_dir: str,
    patch_size: int
) -> dict:
    """生成COCO格式的JSON数据
    
    Args:
        patch_annotations: patch标注字典
        patch_dir: patch目录
        patch_size: patch尺寸
        
    Returns:
        COCO格式的字典
    """
    # 获取所有有标注的patch文件名，并排序
    patch_filenames = sorted(patch_annotations.keys())
    
    # 创建images列表
    images = []
    filename_to_image_id = {}
    
    for image_id, patch_filename in enumerate(patch_filenames):
        images.append({
            "id": image_id,
            "width": patch_size,
            "height": patch_size,
            "file_name": patch_filename,
        })
        filename_to_image_id[patch_filename] = image_id
    
    # 创建annotations列表
    annotations = []
    annotation_id = 0
    
    for patch_filename, anns in patch_annotations.items():
        image_id = filename_to_image_id[patch_filename]
        
        for ann in anns:
            box_patch = ann['patch_box']
            bbox_coco = convert_bbox_to_coco_format(box_patch)
            area = bbox_coco[2] * bbox_coco[3]  # width * height
            
            annotations.append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": ann['category_id'],
                "segmentation": [],
                "bbox": bbox_coco,
                "ignore": 0,
                "iscrowd": 0,
                "area": area,
            })
            annotation_id += 1
    
    # 构建COCO格式数据
    coco_data = {
        "info": {
            "description": "Converted from global annotations to patch-based COCO format",
            "version": "1.0",
            "year": 2024,
        },
        "licenses": [],
        "categories": COCO_CATEGORIES,
        "images": images,
        "annotations": annotations,
    }
    
    return coco_data


def visualize_annotations(
    patch_annotations: Dict[str, List[dict]],
    patch_dir: str,
    output_dir: str,
    patch_size: int
):
    """在patch上可视化标注框
    
    Args:
        patch_annotations: patch标注字典
        patch_dir: patch目录
        output_dir: 输出目录
        patch_size: patch尺寸
    """
    vis_dir = os.path.join(output_dir, "visualization")
    os.makedirs(vis_dir, exist_ok=True)
    
    # 颜色映射（为不同类别分配不同颜色）
    colors = [
        (255, 0, 0),    # AD - 红色
        (0, 255, 0),    # BC - 绿色
        (0, 0, 255),    # EC - 蓝色
        (255, 255, 0),  # L - 黄色
        (255, 0, 255),  # LC - 洋红
        (0, 255, 255),  # M - 青色
        (128, 0, 0),    # NT - 深红
        (0, 128, 0),    # SM - 深绿
        (0, 0, 128),    # SQ - 深蓝
        (128, 128, 0),  # TC1 - 橄榄
        (128, 0, 128),  # TC2 - 紫色
        (0, 128, 128),  # TC3 - 青绿
    ]
    
    print(f"\n开始生成可视化图片...")
    
    for patch_filename, anns in tqdm(patch_annotations.items(), desc="可视化"):
        patch_path = os.path.join(patch_dir, patch_filename)
        
        if not os.path.exists(patch_path):
            print(f"⚠️ 警告：patch文件不存在: {patch_path}，跳过可视化")
            continue
        
        try:
            # 加载patch图像
            img = Image.open(patch_path).convert("RGB")
            draw = ImageDraw.Draw(img)
            
            # 加载字体
            font = None
            for font_path in [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
            ]:
                if os.path.exists(font_path):
                    try:
                        font = ImageFont.truetype(font_path, 16)
                        break
                    except Exception:
                        continue
            if font is None:
                font = ImageFont.load_default()
            
            # 绘制每个标注框
            for ann in anns:
                box = ann['patch_box']
                category_id = ann['category_id']
                category_name = ann['category_name']
                overlap_ratio = ann['overlap_ratio']
                
                x1, y1, x2, y2 = box
                color = colors[category_id % len(colors)]
                
                # 绘制矩形框
                draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
                
                # 绘制标签
                label = f"{category_name} ({overlap_ratio:.2f})"
                try:
                    bbox = draw.textbbox((0, 0), label, font=font)
                    text_w = bbox[2] - bbox[0]
                    text_h = bbox[3] - bbox[1]
                except Exception:
                    text_w = len(label) * 8
                    text_h = 16
                
                pad = 2
                # 绘制标签背景
                draw.rectangle(
                    [x1, y1 - text_h - 2 * pad, x1 + text_w + 2 * pad, y1],
                    fill=color
                )
                # 绘制标签文字
                draw.text((x1 + pad, y1 - text_h - pad), label, fill="white", font=font)
            
            # 保存可视化图片
            vis_path = os.path.join(vis_dir, f"vis_{patch_filename}")
            img.save(vis_path)
            
        except Exception as e:
            print(f"⚠️ 警告：处理 {patch_filename} 时出错: {e}")
            continue
    
    print(f"✓ 可视化图片已保存到: {vis_dir}")


def process_single_annotation(
    annotation_file: str,
    patch_dir: str,
    output_dir: str,
    patch_size: int,
    min_overlap_ratio: float,
    no_visualization: bool,
    coordinates_csv: Optional[str] = None,
) -> dict:
    """处理单个annotation json并输出COCO结果。"""
    output_dir_path = Path(output_dir)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    # 确定坐标CSV文件路径
    if coordinates_csv:
        csv_path = coordinates_csv
    else:
        csv_path = os.path.join(patch_dir, "patch_coordinates.csv")

    # 加载数据
    print("=" * 70)
    print(f"开始转换标注框坐标: {annotation_file}")
    print(f"对应patch目录: {patch_dir}")
    print("=" * 70)

    annotations = load_annotations(annotation_file)
    patch_coordinates, scale_factor = load_patch_coordinates(csv_path)

    # 分配标注框到patch
    patch_annotations = assign_annotations_to_patches(
        annotations,
        patch_coordinates,
        patch_size,
        min_overlap_ratio,
        scale_factor
    )

    # 生成COCO格式
    print(f"\n生成COCO格式JSON...")
    coco_data = generate_coco_format(patch_annotations, patch_dir, patch_size)

    # 保存COCO格式JSON
    output_json_path = output_dir_path / "coco_format.json"
    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(coco_data, f, ensure_ascii=False, indent=2)

    print(f"✓ COCO格式JSON已保存到: {output_json_path}")
    print(f"  - Images数量: {len(coco_data['images'])}")
    print(f"  - Annotations数量: {len(coco_data['annotations'])}")

    # 生成可视化
    if not no_visualization:
        visualize_annotations(
            patch_annotations,
            patch_dir,
            str(output_dir_path),
            patch_size
        )

    print("\n" + "=" * 70)
    print("转换完成！")
    print("=" * 70)

    return {
        "output_json": str(output_json_path),
        "images_count": len(coco_data["images"]),
        "annotations_count": len(coco_data["annotations"]),
    }


def run_batch_mode(args):
    """批量处理input目录下全部json。"""
    input_annotation_dir = Path(args.input_annotation_dir)
    patch_root_dir = Path(args.patch_root_dir)
    output_root_dir = Path(args.output_dir)
    output_root_dir.mkdir(parents=True, exist_ok=True)

    if not input_annotation_dir.exists():
        raise FileNotFoundError(f"annotation输入目录不存在: {input_annotation_dir}")
    if not patch_root_dir.exists():
        raise FileNotFoundError(f"patch根目录不存在: {patch_root_dir}")

    json_files = sorted(input_annotation_dir.glob("*.json"))
    if not json_files:
        print(f"⚠️ 在目录中未找到json文件: {input_annotation_dir}")
        return

    patch_dirs = [p for p in patch_root_dir.iterdir() if p.is_dir()]
    patch_dir_entries = []
    for patch_dir in patch_dirs:
        identity_raw, id_prefix = parse_patch_identity(patch_dir.name)
        identity_pinyin = to_pinyin_text(identity_raw)
        patch_dir_entries.append(
            {
                "patch_dir": str(patch_dir),
                "patch_dir_name": patch_dir.name,
                "id_prefix": id_prefix,
                "identity_raw": identity_raw,
                "identity_pinyin": identity_pinyin,
                "identity_raw_key": normalize_key(identity_raw),
                "identity_pinyin_key": normalize_key(identity_pinyin),
            }
        )

    print("=" * 70)
    print("批量处理模式")
    print(f"annotation目录: {input_annotation_dir}")
    print(f"patch根目录: {patch_root_dir}")
    print(f"输出根目录: {output_root_dir}")
    print(f"待处理json数量: {len(json_files)}")
    print(f"可匹配patch目录数量: {len(patch_dir_entries)}")
    if lazy_pinyin is None:
        print("⚠️ 未安装pypinyin，将跳过中文转拼音，仅使用原文/编号匹配。")
    print("=" * 70)

    success_count = 0
    failed_count = 0
    skipped_count = 0

    for json_file in json_files:
        json_stem = json_file.stem
        print(f"\n处理: {json_file.name}")

        matched_entry = resolve_patch_dir_for_json(json_stem, patch_dir_entries)
        if matched_entry is None:
            print(f"⚠️ 未找到对应patch目录，跳过: {json_file.name}")
            skipped_count += 1
            continue

        patch_dir = matched_entry["patch_dir"]
        print(f"✓ 匹配到patch目录: {matched_entry['patch_dir_name']}")
        output_subdir = output_root_dir / json_stem

        try:
            process_single_annotation(
                annotation_file=str(json_file),
                patch_dir=patch_dir,
                output_dir=str(output_subdir),
                patch_size=args.patch_size,
                min_overlap_ratio=args.min_overlap_ratio,
                no_visualization=args.no_visualization,
                coordinates_csv=None,
            )
            success_count += 1
        except Exception as e:
            print(f"❌ 处理失败: {json_file.name}, 错误: {e}")
            failed_count += 1

    print("\n" + "=" * 70)
    print("批量处理完成")
    print(f"成功: {success_count}")
    print(f"失败: {failed_count}")
    print(f"跳过(未匹配): {skipped_count}")
    print("=" * 70)


def main():
    args = parse_args()

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.batch:
        run_batch_mode(args)
        return

    if not args.annotation_file or not args.patch_dir:
        raise ValueError("非批量模式下，必须提供 --annotation-file 和 --patch-dir")

    process_single_annotation(
        annotation_file=args.annotation_file,
        patch_dir=args.patch_dir,
        output_dir=str(output_dir),
        patch_size=args.patch_size,
        min_overlap_ratio=args.min_overlap_ratio,
        no_visualization=args.no_visualization,
        coordinates_csv=args.coordinates_csv,
    )


if __name__ == "__main__":
    main()

