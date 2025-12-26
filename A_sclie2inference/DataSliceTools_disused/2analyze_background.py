import argparse
import os
from pathlib import Path
import numpy as np
from PIL import Image
from collections import Counter


# ==================== 配置区域 ====================
# 在这里直接设置你的文件路径，然后直接运行脚本即可

# 要分析的图像文件路径（单个图像）
SINGLE_IMAGE_PATH = "E:/AIMed/keep_patches/patches2/patch_52.png"

# 要分析的图像目录路径（批量分析）
IMAGE_DIRECTORY = "/home/ubuntu/lsn/project_new/RT-DETR-main/DataPatches/Patches5"

# 批量分析时的文件数量限制
ANALYSIS_LIMIT = 5

# 是否使用配置区域（True=使用配置，False=使用命令行参数）
USE_CONFIG = True
# ================================================

def analyze_image_colors(image_path):
    """分析图像的背景色和颜色分布"""
    try:
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            arr = np.array(img)
            height, width, channels = arr.shape
            print(f"图像尺寸: {width} x {height}")
            
            # 分析边缘颜色
            edge_colors = []
            edge_colors.extend(arr[0, :, :].tolist())
            edge_colors.extend(arr[-1, :, :].tolist())
            edge_colors.extend(arr[:, 0, :].tolist())
            edge_colors.extend(arr[:, -1, :].tolist())
            
            edge_counter = Counter(map(tuple, edge_colors))
            print(f"\n边缘颜色统计 (前10个):")
            for color, count in edge_counter.most_common(10):
                percentage = (count / len(edge_colors)) * 100
                print(f"  RGB{color}: {count}次 ({percentage:.1f}%)")
            
            # 分析整体颜色分布
            all_colors = arr.reshape(-1, 3)
            color_counter = Counter(map(tuple, all_colors))
            
            print(f"\n整体颜色统计 (前10个):")
            for color, count in color_counter.most_common(10):
                percentage = (count / len(all_colors)) * 100
                print(f"  RGB{color}: {count}次 ({percentage:.1f}%)")
            
            # 推测背景色
            most_common_color = color_counter.most_common(1)[0]
            background_color = most_common_color[0]
            background_percentage = (most_common_color[1] / len(all_colors)) * 100
            
            print(f"\n推测背景色: RGB{background_color} ({background_percentage:.1f}%)")
            
            # 计算与背景色的差异
            color_diff = np.abs(all_colors.astype(np.int16) - np.array(background_color, dtype=np.int16))
            total_diff = color_diff.sum(axis=1)
            
            # 统计不同容差下的背景像素比例
            tolerance_levels = [5, 10, 15, 20, 25, 30]
            print(f"\n不同容差下的背景像素比例:")
            for tolerance in tolerance_levels:
                close_to_bg = (total_diff <= tolerance * 3).sum()
                percentage = (close_to_bg / len(all_colors)) * 100
                print(f"  容差{tolerance}: {percentage:.1f}%")
            
            # 计算灰度标准差
            gray = (0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]).astype(np.float32)
            std = float(gray.std())
            print(f"\n灰度标准差: {std:.2f}")
            
            return background_color, background_percentage, std
            
    except Exception as e:
        print(f"分析图像时出错: {e}")
        return None, None, None

def analyze_multiple_images(directory, limit=5):
    """分析目录中的多个图像"""
    directory = Path(directory)
    image_files = [f for f in directory.iterdir() if f.suffix.lower() in {'.png', '.jpg', '.jpeg'}]
    
    if not image_files:
        print(f"在 {directory} 中没有找到图像文件")
        return
    
    print(f"找到 {len(image_files)} 个图像文件")
    print(f"分析前 {min(limit, len(image_files))} 个文件...\n")
    
    results = []
    for i, image_file in enumerate(image_files[:limit]):
        print(f"=== 分析 {image_file.name} ===")
        bg_color, bg_percent, std = analyze_image_colors(image_file)
        if bg_color:
            results.append({
                'file': image_file.name,
                'background_color': bg_color,
                'background_percentage': bg_percent,
                'std': std
            })
        print("-" * 50)
    
    # 总结分析结果
    if results:
        print("\n=== 分析总结 ===")
        print("推荐的过滤参数:")
        
        avg_bg = np.mean([r['background_color'] for r in results], axis=0).astype(int)
        print(f"--bg \"{avg_bg[0]},{avg_bg[1]},{avg_bg[2]}\"")
        
        avg_std = np.mean([r['std'] for r in results])
        recommended_tolerance = max(15, min(30, int(avg_std * 2)))
        print(f"--tolerance {recommended_tolerance}")
        
        avg_bg_percent = np.mean([r['background_percentage'] for r in results])
        if avg_bg_percent > 95:
            recommended_ratio = 0.98
        elif avg_bg_percent > 90:
            recommended_ratio = 0.95
        else:
            recommended_ratio = 0.90
        print(f"--bg_ratio {recommended_ratio}")
        
        recommended_std_thresh = max(3.0, min(10.0, avg_std * 1.5))
        print(f"--std_thresh {recommended_std_thresh}")
        
        print(f"\n完整命令示例:")
        print(f"python scripts/filter_patch.py --src_dir {directory} --keep_dir {directory}_filtered --bg \"{avg_bg[0]},{avg_bg[1]},{avg_bg[2]}\" --tolerance {recommended_tolerance} --bg_ratio {recommended_ratio} --std_thresh {recommended_std_thresh}")

def main():
    if USE_CONFIG:
        print("=== 使用配置区域设置 ===")
        
        if SINGLE_IMAGE_PATH and Path(SINGLE_IMAGE_PATH).exists():
            print(f"分析单个图像: {SINGLE_IMAGE_PATH}")
            analyze_image_colors(SINGLE_IMAGE_PATH)
        elif IMAGE_DIRECTORY and Path(IMAGE_DIRECTORY).exists():
            print(f"分析图像目录: {IMAGE_DIRECTORY}")
            analyze_multiple_images(IMAGE_DIRECTORY, ANALYSIS_LIMIT)
        else:
            print("配置错误：请检查路径是否正确")
            print(f"当前设置:")
            print(f"  SINGLE_IMAGE_PATH: {SINGLE_IMAGE_PATH}")
            print(f"  IMAGE_DIRECTORY: {IMAGE_DIRECTORY}")
    else:
        parser = argparse.ArgumentParser(description="分析PNG图像的背景色")
        parser.add_argument("--image", help="单个图像文件路径")
        parser.add_argument("--directory", help="图像目录路径")
        parser.add_argument("--limit", type=int, default=5, help="分析目录时的文件数量限制")
        
        args = parser.parse_args()
        
        if args.image:
            analyze_image_colors(args.image)
        elif args.directory:
            analyze_multiple_images(args.directory, args.limit)
        else:
            default_dir = "scripts/patches_tif"
            if Path(default_dir).exists():
                analyze_multiple_images(default_dir, 5)
            else:
                print("请指定图像文件或目录")

if __name__ == "__main__":
    main()
