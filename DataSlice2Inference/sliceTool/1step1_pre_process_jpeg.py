import os
import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from PIL import ImageFile
import csv

# 禁用 PIL 的 decompression bomb 检查，允许读取超大图像
Image.MAX_IMAGE_PIXELS = None
# 允许读取截断的图像（对于大文件可能出现的误报）
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 参数设置
jpeg_path = '/home/ubuntu/lsn/project_new/RT-DETR-main/DataWSI/45-庄驷40X.jpeg'  # 替换为你的jpeg文件路径
patch_size = 640
output_dir = 'DataPatches/Patches5'
os.makedirs(output_dir, exist_ok=True)

# 使用 PIL 打开图像（不加载到内存，只获取尺寸信息）
print("正在打开图像文件...")
try:
    pil_img = Image.open(jpeg_path)
    # 确保图像是 RGB 模式
    if pil_img.mode != 'RGB':
        print(f"图像模式为 {pil_img.mode}，将转换为 RGB")
        # 注意：这里不立即转换，而是在读取每个 patch 时转换
except Exception as e:
    raise FileNotFoundError(f"无法打开图片文件: {jpeg_path}, 错误: {e}")

# 获取图像尺寸（宽，高）
width, height = pil_img.size
print(f"图像尺寸: {width} x {height} 像素")
print(f"总像素数: {width * height:,}")

# 计算可切 patch 数量（整除部分）
num_patches_x = width // patch_size
num_patches_y = height // patch_size
total_patches = num_patches_x * num_patches_y
print(f"将切分为 {num_patches_x} x {num_patches_y} = {total_patches} 个 patch")

# 用于记录每个patch左上角在大图中的坐标
x_r = []
y_r = []
patch_count = 0

# 分块读取并保存 patch
print("开始切分图像...")
for y_idx in tqdm(range(num_patches_y), desc="处理行", unit="行"):
    for x_idx in range(num_patches_x):
        x = x_idx * patch_size
        y = y_idx * patch_size

        # 记录坐标
        x_r.append(x)
        y_r.append(y)

        # 使用 crop 方法只读取当前 patch 区域（不加载整个图像）
        patch_box = (x, y, x + patch_size, y + patch_size)
        try:
            patch_pil = pil_img.crop(patch_box)
        except Exception as e:
            print(f"\n警告：读取 patch ({x}, {y}) 时出错: {e}")
            continue
        
        # 转换为 RGB 模式（如果需要）
        if patch_pil.mode != 'RGB':
            patch_pil = patch_pil.convert('RGB')
        
        # 保存 patch 为 PNG 格式
        save_path = os.path.join(output_dir, f'patch_{patch_count}.png')
        try:
            patch_pil.save(save_path, 'PNG')
        except Exception as e:
            print(f"\n警告：保存 patch {patch_count} 时出错: {e}")
            continue
        
        patch_count += 1

# 保存坐标信息到CSV文件
csv_path = os.path.join(output_dir, 'patch_coordinates.csv')
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['filename', 'x_start', 'y_start', 'x_end', 'y_end'])
    for i in range(patch_count):
        filename = f'patch_{i}.png'
        writer.writerow([filename, x_r[i], y_r[i], x_r[i] + patch_size, y_r[i] + patch_size])

print(f"\n完成！共保存 {patch_count} 个 patch，保存于目录：{output_dir}")
print(f"每个 patch 尺寸: {patch_size} x {patch_size} 像素")
print(f"坐标信息已保存到: {csv_path}")
