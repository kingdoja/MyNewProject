#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
可视化 COCO 格式标注结果的脚本
"""

import json
import os
import argparse
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import numpy as np

# 类别颜色映射（使用不同颜色区分不同类别）
COLORS = [
    '#FF0000',  # AD - 红色
    '#00FF00',  # BC - 绿色
    '#0000FF',  # EC - 蓝色
    '#FFFF00',  # L - 黄色
    '#FF00FF',  # LC - 洋红色
    '#00FFFF',  # M - 青色
    '#FFA500',  # NT - 橙色
    '#800080',  # SM - 紫色
    '#FFC0CB',  # SQ - 粉色
    '#A52A2A',  # TC1 - 棕色
    '#808080',  # TC2 - 灰色
    '#000080',  # TC3 - 海军蓝
]


def load_coco_json(json_path):
    """加载 COCO 格式的 JSON 文件"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def get_category_info(coco_data):
    """获取类别信息"""
    categories = {}
    for cat in coco_data["categories"]:
        categories[cat["id"]] = {
            "name": cat["name"],
            "color": COLORS[cat["id"] % len(COLORS)]
        }
    return categories


def visualize_image(image_path, annotations, categories, output_path=None, show=True):
    """
    可视化单个图像的标注
    
    Args:
        image_path: 图像文件路径
        annotations: 该图像的标注列表
        categories: 类别信息字典
        output_path: 输出图像路径（可选）
        show: 是否显示图像
    """
    # 读取图像
    try:
        from PIL import Image
        img = Image.open(image_path)
        img_array = np.array(img)
    except ImportError:
        print("警告: 未安装 PIL，尝试使用 matplotlib 读取图像")
        img_array = plt.imread(image_path)
    except Exception as e:
        print(f"错误: 无法读取图像 {image_path}: {e}")
        return
    
    # 创建图形
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.imshow(img_array)
    ax.axis('off')
    
    # 绘制标注
    for ann in annotations:
        category_id = ann["category_id"]
        category_info = categories.get(category_id, {"name": f"Unknown_{category_id}", "color": "#FFFFFF"})
        color = category_info["color"]
        label = category_info["name"]
        
        # 绘制 bbox
        bbox = ann["bbox"]
        x, y, w, h = bbox
        rect = patches.Rectangle(
            (x, y), w, h,
            linewidth=2,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # 添加标签文本
        ax.text(
            x, y - 5,
            label,
            color=color,
            fontsize=10,
            weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7, edgecolor=color)
        )
        
        # 如果有分割标注，绘制多边形
        if ann.get("segmentation") and len(ann["segmentation"]) > 0:
            seg = ann["segmentation"][0]
            if len(seg) >= 6:  # 至少3个点（每个点2个坐标）
                points = np.array(seg).reshape(-1, 2)
                polygon = Polygon(
                    points,
                    linewidth=2,
                    edgecolor=color,
                    facecolor=color,
                    alpha=0.3
                )
                ax.add_patch(polygon)
    
    # 设置标题
    image_name = os.path.basename(image_path)
    ax.set_title(f"{image_name} ({len(annotations)} 个标注)", fontsize=14, pad=10)
    
    # 保存或显示
    if output_path:
        plt.savefig(output_path, bbox_inches='tight', dpi=150)
        print(f"已保存可视化结果到: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def visualize_multiple_images(coco_data, images_dir, output_dir=None, max_images=10, image_ids=None):
    """
    可视化多个图像
    
    Args:
        coco_data: COCO 格式数据
        images_dir: 图像文件目录
        output_dir: 输出目录（可选）
        max_images: 最多可视化图像数量
        image_ids: 要可视化的图像 ID 列表（可选）
    """
    images_dir = Path(images_dir)
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取类别信息
    categories = get_category_info(coco_data)
    
    # 构建图像 ID 到标注的映射
    image_id_to_annotations = defaultdict(list)
    for ann in coco_data["annotations"]:
        image_id_to_annotations[ann["image_id"]].append(ann)
    
    # 确定要可视化的图像
    if image_ids is None:
        # 选择有标注的图像
        images_with_anns = [img for img in coco_data["images"] if img["id"] in image_id_to_annotations]
        images_to_visualize = images_with_anns[:max_images]
    else:
        # 根据指定的 image_ids
        image_id_dict = {img["id"]: img for img in coco_data["images"]}
        images_to_visualize = [image_id_dict[iid] for iid in image_ids if iid in image_id_dict]
    
    print(f"准备可视化 {len(images_to_visualize)} 个图像...")
    
    # 可视化每个图像
    for img_info in images_to_visualize:
        image_id = img_info["id"]
        image_filename = img_info["file_name"]
        image_path = images_dir / image_filename
        
        if not image_path.exists():
            print(f"警告: 图像文件不存在: {image_path}")
            continue
        
        annotations = image_id_to_annotations.get(image_id, [])
        
        output_path = None
        if output_dir:
            output_path = output_dir / f"vis_{image_filename}"
        
        visualize_image(
            str(image_path),
            annotations,
            categories,
            output_path=output_path,
            show=False
        )
    
    print(f"可视化完成！")


def main():
    parser = argparse.ArgumentParser(description="可视化 COCO 格式的标注结果")
    parser.add_argument(
        "--json",
        type=str,
        default="../new37/coco_format.json",
        help="COCO 格式 JSON 文件路径（默认: ../new37/coco_format.json）"
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="../new37/images",
        help="图像文件目录（默认: ../new37/images）"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="../new37/visualizations",
        help="输出可视化结果目录（默认: ../new37/visualizations）"
    )
    parser.add_argument(
        "--max_images",
        type=int,
        default=10,
        help="最多可视化图像数量（默认: 10）"
    )
    parser.add_argument(
        "--image_ids",
        type=str,
        default=None,
        help="要可视化的图像 ID 列表（逗号分隔，例如: 0,1,2）"
    )
    parser.add_argument(
        "--single",
        type=str,
        default=None,
        help="可视化单个图像（指定图像文件名）"
    )
    
    args = parser.parse_args()
    
    # 加载 COCO 数据
    script_dir = Path(__file__).parent
    json_path = script_dir / args.json if not os.path.isabs(args.json) else Path(args.json)
    images_dir = script_dir / args.images_dir if not os.path.isabs(args.images_dir) else Path(args.images_dir)
    
    print(f"正在加载 {json_path}...")
    coco_data = load_coco_json(str(json_path))
    print(f"加载完成: {len(coco_data['images'])} 个图像, {len(coco_data['annotations'])} 个标注")
    
    # 获取类别信息
    categories = get_category_info(coco_data)
    print(f"类别: {[cat['name'] for cat in coco_data['categories']]}")
    
    # 可视化单个图像
    if args.single:
        image_filename = args.single
        image_path = images_dir / image_filename
        
        # 找到对应的图像 ID
        image_id = None
        for img in coco_data["images"]:
            if img["file_name"] == image_filename:
                image_id = img["id"]
                break
        
        if image_id is None:
            print(f"错误: 未找到图像 {image_filename}")
            return
        
        # 获取该图像的标注
        annotations = [ann for ann in coco_data["annotations"] if ann["image_id"] == image_id]
        
        visualize_image(str(image_path), annotations, categories, show=True)
    else:
        # 可视化多个图像
        image_ids = None
        if args.image_ids:
            image_ids = [int(x.strip()) for x in args.image_ids.split(",")]
        
        output_dir = script_dir / args.output_dir if not os.path.isabs(args.output_dir) else Path(args.output_dir)
        visualize_multiple_images(
            coco_data,
            str(images_dir),
            output_dir=str(output_dir),
            max_images=args.max_images,
            image_ids=image_ids
        )


if __name__ == "__main__":
    main()

