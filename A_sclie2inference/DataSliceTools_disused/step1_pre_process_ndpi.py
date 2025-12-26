import openslide
import numpy as np
import cv2
import os
from tqdm import tqdm
import csv

# 参数设置
ndpi_path = '/home/ubuntu/lsn/project_new/RT-DETR-main/DataWSI/1.mdsx'  # 替换为你的文件路径
patch_size = 640
output_dir = 'DataPatches'
os.makedirs(output_dir, exist_ok=True)

# 打开 ndpi 文件
slide = openslide.OpenSlide(ndpi_path)
width, height = slide.dimensions
print(f"图像尺寸: {width} x {height}")

# 计算 patch 数量
num_patches_x = width // patch_size
num_patches_y = height // patch_size

# 坐标记录文件路径
coord_csv_path = os.path.join(output_dir, 'patch_coordinates.csv')

# 写入 CSV 文件头
with open(coord_csv_path, mode='w', newline='') as csv_file:
    writer = csv.writer(csv_file)
    writer.writerow(['patch_id', 'filename', 'x_start', 'y_start', 'x_end', 'y_end', 'width', 'height'])

    patch_count = 0

    for y_idx in tqdm(range(num_patches_y), desc="保存 Patch", unit="行"):
        for x_idx in range(num_patches_x):
            x = x_idx * patch_size
            y = y_idx * patch_size
            x_end = min(x + patch_size, width)
            y_end = min(y + patch_size, height)

            # 读取 patch
            region = slide.read_region((x, y), 0, (patch_size, patch_size)).convert('RGB')
            region_np = np.array(region)

            # 预处理（可选）
            blurred = cv2.GaussianBlur(region_np, (5, 5), 0)
            normalized = (blurred.astype(np.float32) / 255.0)

            # 保存图像
            filename = f'patch_{patch_count:06d}.png'
            save_path = os.path.join(output_dir, filename)
            cv2.imwrite(save_path, (normalized * 255).astype(np.uint8))

            # 写入坐标记录
            writer.writerow([
                patch_count,
                filename,
                x, y,
                x_end, y_end,
                x_end - x, y_end - y
            ])

            patch_count += 1

print(f"✅ 完成！共保存 {patch_count} 个 patch，坐标信息已保存至：{coord_csv_path}")