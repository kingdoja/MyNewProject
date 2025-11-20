#!/usr/bin/env python3
"""
Inference visualization script.
Generates visual outputs based on batch inference JSON results.
"""
import json
import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random
from collections import Counter
import argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-darkgrid")

# COCO数据集类别名称（与inference.py保持一致）
COCO_CLASSES = ['AD', 'BC', 'EC', 'L', 'LC', 'M', 'NT', 'SM', 'SQ', 'TC1', 'TC2', 'TC3']

# 类别颜色映射（高对比度调色板）
CLASS_COLORS = {
    'AD': (230, 57, 70),    # vivid red
    'BC': (29, 53, 87),     # deep navy
    'EC': (33, 158, 188),   # bright cyan
    'L':  (251, 133, 0),    # neon orange
    'LC': (255, 183, 3),    # bold yellow
    'M':  (102, 205, 170),  # mint green
    'NT': (106, 13, 173),   # royal purple
    'SM': (0, 0, 0),        # black
    'SQ': (255, 0, 255),    # magenta
    'TC1': (0, 191, 255),   # electric blue
    'TC2': (60, 179, 113),  # medium sea green
    'TC3': (255, 105, 180), # hot pink
}


def load_font(font_size=16):
    """Load font from common locations."""
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Helvetica.ttc",  # macOS
        "C:/Windows/Fonts/arial.ttf",  # Windows
        "C:/Windows/Fonts/arialbd.ttf",  # Windows Bold
    ]
    
    for font_path in font_paths:
        try:
            if os.path.exists(font_path):
                return ImageFont.truetype(font_path, font_size)
        except:
            continue
    
    try:
        return ImageFont.load_default()
    except:
        return None


def draw_detections_on_image(image_path, detections, output_path=None, 
                             font_size=20, box_width=2, show_score=True):
    """
    在图片上绘制检测结果
    
    Args:
        image_path: 原始图片路径
        detections: 检测结果列表，每个元素包含 class_name, bbox, score
        output_path: 输出图片路径
        font_size: 字体大小
        box_width: 边界框宽度
        show_score: 是否显示置信度分数
    """
    try:
        image = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"❌ Failed to open image {image_path}: {e}")
        return None
    
    draw = ImageDraw.Draw(image)
    img_width, img_height = image.size
    font = load_font(font_size)
    
    # 绘制每个检测框
    for det in detections:
        class_name = det['class_name']
        bbox = det['bbox']  # [x1, y1, x2, y2]
        score = det['score']
        
        # 获取类别颜色，如果不存在则使用默认颜色
        color = CLASS_COLORS.get(class_name, (255, 0, 0))
        
        x1, y1, x2, y2 = bbox
        
        # 修正可能的坐标异常（确保左上 < 右下，并限制在图像范围内）
        x1, x2 = sorted([float(x1), float(x2)])
        y1, y2 = sorted([float(y1), float(y2)])
        
        # 将坐标限制在图像范围内
        x1 = max(0, min(img_width - 1, x1))
        y1 = max(0, min(img_height - 1, y1))
        x2 = max(0, min(img_width - 1, x2))
        y2 = max(0, min(img_height - 1, y2))
        
        # 如果框的尺寸太小（可能是无效框），跳过
        if x2 - x1 < 1 or y2 - y1 < 1:
            continue
        
        # 绘制边界框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=box_width)
        
        # 准备标签文本
        if show_score:
            label_text = f"{class_name} {score:.2f}"
        else:
            label_text = class_name
        
        # 获取文本尺寸
        if font:
            try:
                bbox_text = draw.textbbox((0, 0), label_text, font=font)
                text_width = bbox_text[2] - bbox_text[0]
                text_height = bbox_text[3] - bbox_text[1]
            except:
                text_width = len(label_text) * (font_size // 2)
                text_height = font_size
        else:
            text_width = len(label_text) * (font_size // 2)
            text_height = font_size
        
        # 绘制标签背景
        padding = 3
        label_y = max(0, y1 - text_height - padding * 2)
        draw.rectangle(
            [x1, label_y, x1 + text_width + padding * 2, y1],
            fill=color
        )
        
        # 绘制标签文字
        if font:
            draw.text(
                (x1 + padding, label_y + padding),
                label_text,
                fill='white',
                font=font
            )
        else:
            draw.text(
                (x1 + padding, label_y + padding),
                label_text,
                fill='white'
            )
    
    # 保存图片
    if output_path:
        image.save(output_path)
    
    return image


def select_representative_images(json_data, num_images=9):
    """
    从推理结果中选择代表性的图片
    
    Args:
        json_data: JSON数据
        num_images: 要选择的图片数量
    
    Returns:
        选中的图片数据列表
    """
    images = json_data.get('images', [])
    
    if len(images) == 0:
        return []
    
    # 按检测数量分类
    images_with_detections = [img for img in images if img['detection_count'] > 0]
    images_without_detections = [img for img in images if img['detection_count'] == 0]
    
    selected = []
    
    # 1. 选择检测数量最多的图片（2-3张）
    if images_with_detections:
        sorted_by_count = sorted(images_with_detections, 
                                key=lambda x: x['detection_count'], 
                                reverse=True)
        selected.extend(sorted_by_count[:min(3, len(sorted_by_count))])
    
    # 2. 选择检测数量中等的图片（2-3张）
    if images_with_detections and len(selected) < num_images:
        medium_count = [img for img in images_with_detections 
                        if img not in selected]
        if medium_count:
            medium_count.sort(key=lambda x: x['detection_count'])
            mid_idx = len(medium_count) // 2
            selected.extend(medium_count[max(0, mid_idx-1):mid_idx+2])
    
    # 3. 选择不同类别的代表性图片（2-3张）
    if images_with_detections and len(selected) < num_images:
        # 统计每个类别出现的频率
        class_images = {}
        for img in images_with_detections:
            if img not in selected:
                for det in img['detections']:
                    cls = det['class_name']
                    if cls not in class_images:
                        class_images[cls] = []
                    class_images[cls].append(img)
        
        # 选择包含不同类别的图片
        for cls, img_list in class_images.items():
            if len(selected) >= num_images:
                break
            if img_list:
                selected.append(img_list[0])
    
    # 4. 如果还需要更多，随机选择
    if len(selected) < num_images:
        remaining = [img for img in images_with_detections if img not in selected]
        if remaining:
            random.shuffle(remaining)
            selected.extend(remaining[:num_images - len(selected)])
    
    # 5. 如果还需要，添加无检测的图片（1-2张）
    if len(selected) < num_images and images_without_detections:
        selected.extend(images_without_detections[:min(2, num_images - len(selected))])
    
    # 6. 如果还不够，随机补充
    if len(selected) < num_images:
        all_remaining = [img for img in images if img not in selected]
        if all_remaining:
            random.shuffle(all_remaining)
            selected.extend(all_remaining[:num_images - len(selected)])
    
    return selected[:num_images]


def create_visualization_grid(selected_images, input_dir, output_dir, 
                             grid_cols=3, grid_rows=3, 
                             image_size=(640, 640), spacing=10,
                             font_size=18, box_width=3):
    """
    创建网格布局的可视化效果图
    
    Args:
        selected_images: 选中的图片数据列表
        input_dir: 输入图片目录
        output_dir: 输出目录
        grid_cols: 网格列数
        grid_rows: 网格行数
        image_size: 每张图片的尺寸
        spacing: 图片之间的间距
    """
    num_images = len(selected_images)
    grid_cols = min(grid_cols, num_images)
    grid_rows = (num_images + grid_cols - 1) // grid_cols
    
    # 计算画布大小
    canvas_width = grid_cols * image_size[0] + (grid_cols + 1) * spacing
    canvas_height = grid_rows * image_size[1] + (grid_rows + 1) * spacing + 50  # 额外空间用于标题
    
    # 创建画布
    canvas = Image.new('RGB', (canvas_width, canvas_height), color='white')
    draw = ImageDraw.Draw(canvas)
    font = load_font(20)
    
    # 绘制标题
    title = "推理结果可视化效果图"
    if font:
        try:
            bbox = draw.textbbox((0, 0), title, font=font)
            title_width = bbox[2] - bbox[0]
        except:
            title_width = len(title) * 10
    else:
        title_width = len(title) * 10
    
    title_x = (canvas_width - title_width) // 2
    if font:
        draw.text((title_x, 10), title, fill='black', font=font)
    else:
        draw.text((title_x, 10), title, fill='black')
    
    # 绘制每张图片
    for idx, img_data in enumerate(selected_images):
        row = idx // grid_cols
        col = idx % grid_cols
        
        # 计算位置
        x = spacing + col * (image_size[0] + spacing)
        y = 50 + spacing + row * (image_size[1] + spacing)
        
        # 加载并绘制图片（带检测框）
        image_path = os.path.join(input_dir, img_data['image_name'])
        if not os.path.exists(image_path):
            # 尝试在输出目录中查找原推理输出（pred_*.png）
            image_path = os.path.join(output_dir, f"pred_{img_data['image_name']}")
        
        if os.path.exists(image_path):
            try:
                annotated_image = draw_detections_on_image(
                    image_path,
                    img_data['detections'],
                    output_path=None,
                    font_size=font_size,
                    box_width=box_width,
                    show_score=True
                )
                if annotated_image is None:
                    raise RuntimeError("annotated_image is None")
                
                # 调整图片大小
                annotated_image = annotated_image.resize(image_size, Image.Resampling.LANCZOS)
                canvas.paste(annotated_image, (x, y))
                
                # 在图片下方添加信息
                info_text = f"{img_data['image_name'][:20]} ({img_data['detection_count']}个)"
                info_y = y + image_size[1] + 2
                if font:
                    draw.text((x, info_y), info_text, fill='black', font=font)
                else:
                    draw.text((x, info_y), info_text, fill='black')
            except Exception as e:
                print(f"⚠️  Failed to load or render image {image_path}: {e}")
        else:
            # 绘制占位符
            placeholder = Image.new('RGB', image_size, color='lightgray')
            draw_placeholder = ImageDraw.Draw(placeholder)
            draw_placeholder.text((10, 10), "Image not found", fill='black')
            canvas.paste(placeholder, (x, y))
    
    return canvas


def visualize_inference_results(json_path, input_dir, output_dir, 
                               num_images=9, create_grid=True, 
                               create_individual=True):
    """
    可视化推理结果
    
    Args:
        json_path: JSON结果文件路径
        input_dir: 输入图片目录
        output_dir: 输出目录（包含推理后的图片）
        num_images: 要可视化的图片数量
        create_grid: 是否创建网格布局图
        create_individual: 是否创建单独的标注图片
    """
    print("="*70)
    print(" " * 20 + "Inference Results Visualization")
    print("="*70)
    
    # 读取JSON文件
    if not os.path.exists(json_path):
        print(f"❌ JSON file not found: {json_path}")
        return False
    
    print(f"📁 Reading JSON file: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        json_data = json.load(f)
    
    # 选择代表性图片
    print(f"📊 Selecting {num_images} representative images from {len(json_data['images'])} samples...")
    selected_images = select_representative_images(json_data, num_images)
    print(f"✅ Selected {len(selected_images)} images")
    
    # 创建输出目录
    vis_output_dir = os.path.join(output_dir, "visualization")
    os.makedirs(vis_output_dir, exist_ok=True)
    
    # 创建单独的标注图片
    if create_individual:
        print("\n📝 Generating individual annotated images...")
        individual_dir = os.path.join(vis_output_dir, "individual")
        os.makedirs(individual_dir, exist_ok=True)
        
        for img_data in selected_images:
            image_name = img_data['image_name']
            image_path = os.path.join(input_dir, image_name)
            
            # 如果输入目录没有，尝试输出目录
            if not os.path.exists(image_path):
                image_path = os.path.join(output_dir, f"pred_{image_name}")
            
            if os.path.exists(image_path):
                output_path = os.path.join(individual_dir, f"vis_{image_name}")
                draw_detections_on_image(
                    image_path,
                    img_data['detections'],
                    output_path,
                    font_size=20,
                    box_width=3,
                    show_score=True
                )
                print(f"  ✓ {image_name}")
            else:
                print(f"  ⚠️  Image not found: {image_name}")
    
    # 创建网格布局图
    if create_grid:
        print("\n📐 Building grid visualization...")
        grid_image = create_visualization_grid(
            selected_images,
            input_dir,
            output_dir,
            grid_cols=3,
            grid_rows=3,
            image_size=(640, 640),
            spacing=10
        )
        
        grid_output_path = os.path.join(vis_output_dir, "visualization_grid.png")
        grid_image.save(grid_output_path)
        print(f"✅ Grid image saved: {grid_output_path}")
    
    # 生成统计信息
    print("\n📊 Generating statistics...")
    all_detections = []
    for img_data in json_data['images']:
        all_detections.extend([d['class_name'] for d in img_data['detections']])
    
    class_counts = Counter(all_detections)
    
    stats_text = f"""
Inference Summary
=================
Total images: {json_data['statistics']['total_images']}
Successful: {json_data['statistics']['successful']}
Total detections: {json_data['statistics']['total_detections']}
Average per image: {json_data['statistics']['average_detections_per_image']:.2f}

Detections per class:
"""
    for cls in COCO_CLASSES:
        count = class_counts.get(cls, 0)
        if count > 0:
            stats_text += f"  {cls}: {count}\n"
    
    stats_path = os.path.join(vis_output_dir, "statistics.txt")
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write(stats_text)
    print(f"✅ Statistics saved: {stats_path}")

    # 绘制类别分布柱状图
    if class_counts:
        print("📈 Rendering class distribution chart...")
        classes = [cls for cls in COCO_CLASSES if class_counts.get(cls, 0) > 0]
        counts = [class_counts.get(cls, 0) for cls in classes]
        if counts:
            plt.figure(figsize=(10, 6))
            colors = []
            for cls in classes:
                color = CLASS_COLORS.get(cls, (100, 100, 100))
                colors.append(tuple(c / 255.0 for c in color))
            bars = plt.bar(classes, counts, color=colors)
            plt.title("Detections per Class")
            plt.xlabel("Class")
            plt.ylabel("Count")
            for bar, count in zip(bars, counts):
                plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{count}", ha='center', va='bottom')
            plt.tight_layout()
            bar_chart_path = os.path.join(vis_output_dir, "class_distribution.png")
            plt.savefig(bar_chart_path, dpi=200)
            plt.close()
            print(f"✅ Class distribution chart saved: {bar_chart_path}")
        else:
            print("⚠️  No class statistics available for plotting")

    # 绘制每张图片检测数量分布
    detection_counts = [img['detection_count'] for img in json_data['images']]
    if detection_counts:
        print("📈 Rendering detection count distribution chart...")
        plt.figure(figsize=(10, 6))
        counts, bins, patches = plt.hist(
            detection_counts,
            bins=25,
            color="#1f77b4",
            edgecolor='white'
        )
        plt.title("Detections per Image")
        plt.xlabel("Number of detections")
        plt.ylabel("Number of images")
        # 在柱状图上方标注数量
        for count, patch in zip(counts, patches):
            if count <= 0:
                continue
            x = patch.get_x() + patch.get_width() / 2
            y = patch.get_height()
            plt.text(x, y, f"{int(count)}", ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        hist_path = os.path.join(vis_output_dir, "detection_count_distribution.png")
        plt.savefig(hist_path, dpi=200)
        plt.close()
        print(f"✅ Detection count distribution saved: {hist_path}")
    else:
        print("⚠️  No detection-count data available")
    
    print("\n" + "="*70)
    print(" " * 25 + "Visualization Complete")
    print("="*70)
    print(f"📁 Output directory: {vis_output_dir}")
    if create_individual:
        print(f"   ├── individual/        (annotated images)")
    if create_grid:
        print(f"   ├── visualization_grid.png  (grid view)")
    print(f"   ├── statistics.txt      (text summary)")
    print(f"   ├── class_distribution.png (class bar chart)")
    print(f"   └── detection_count_distribution.png (detections histogram)")
    print("="*70 + "\n")
    
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="推理结果可视化工具")
    parser.add_argument("--json", type=str, 
                       help="JSON结果文件路径（annotations.json）")
    parser.add_argument("--input_dir", type=str,
                       help="输入图片目录（原始图片）")
    parser.add_argument("--output_dir", type=str,
                       help="输出目录（推理结果目录，包含pred_*.png）")
    parser.add_argument("--num_images", type=int, default=9,
                       help="要可视化的图片数量（默认9）")
    parser.add_argument("--no_grid", action="store_true",
                       help="不创建网格布局图")
    parser.add_argument("--no_individual", action="store_true",
                       help="不创建单独的标注图片")
    
    args = parser.parse_args()
    
    # 如果使用配置区域，可以在这里设置默认值
    USE_CONFIG = True
    
    if USE_CONFIG:
        # 配置区域 - 直接在这里设置路径
        CONFIG_JSON_PATH = "/home/ubuntu/lsn/project_new/RT-DETR-main/DataPatchesInference/Patches1/annotations.json"
        CONFIG_INPUT_DIR = "/home/ubuntu/lsn/project_new/RT-DETR-main/DataPatchesKeep/Patches1"
        CONFIG_OUTPUT_DIR = "/home/ubuntu/lsn/project_new/RT-DETR-main/DataPatchesInference/Patches1"
        CONFIG_NUM_IMAGES = 9
        
        json_path = args.json if args.json else CONFIG_JSON_PATH
        input_dir = args.input_dir if args.input_dir else CONFIG_INPUT_DIR
        output_dir = args.output_dir if args.output_dir else CONFIG_OUTPUT_DIR
        num_images = args.num_images if args.num_images != 9 else CONFIG_NUM_IMAGES
    else:
        if not args.json or not args.input_dir or not args.output_dir:
            print("❌ Error: provide --json, --input_dir, and --output_dir parameters")
            print("   or set USE_CONFIG = True and update the configuration block.")
            sys.exit(1)
        json_path = args.json
        input_dir = args.input_dir
        output_dir = args.output_dir
        num_images = args.num_images
    
    # 执行可视化
    success = visualize_inference_results(
        json_path=json_path,
        input_dir=input_dir,
        output_dir=output_dir,
        num_images=num_images,
        create_grid=not args.no_grid,
        create_individual=not args.no_individual
    )
    
    if success:
        print("\n🎉 Visualization finished!")
    else:
        print("\n❌ Visualization failed!")
        sys.exit(1)

