import openslide
import numpy as np
import cv2
import os
from tqdm import tqdm

# ===== 参数设置 =====
tif_path = r'C:/Users/layyy/Desktop/kfb_output/25001.tif'  # 修改为你的 .tif 文件路径
patch_size = 256
output_dir = 'patches_tif'
os.makedirs(output_dir, exist_ok=True)

# ===== 打开 .tif 文件 =====
slide = openslide.OpenSlide(tif_path)
width, height = slide.dimensions
print(f"图像尺寸: {width} x {height}")

# ===== 计算可切 patch 数量（整除部分） =====
num_patches_x = width // patch_size
num_patches_y = height // patch_size

# ===== 遍历每个 patch 坐标，并保存 =====
patch_count = 0
for y_idx in tqdm(range(num_patches_y), desc="保存 Patch", unit="行"):
    for x_idx in range(num_patches_x):
        x = x_idx * patch_size
        y = y_idx * patch_size

        # 读取该 patch 区域
        region = slide.read_region((x, y), 0, (patch_size, patch_size)).convert('RGB')
        region_np = np.array(region)

        # 转灰度图（可选）
        #gray = cv2.cvtColor(region_np, cv2.COLOR_RGB2GRAY)

        # 高斯模糊 + 归一化（可选）
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        normalized = (blurred.astype(np.float32) / 255.0)

        # 保存图像（归一化转回 uint8）
        save_path = os.path.join(output_dir, f'patch_{patch_count}.png')
        cv2.imwrite(save_path, (normalized * 255).astype(np.uint8))
        patch_count += 1

print(f"\n完成！共保存 {patch_count} 个 patch，保存于目录：{output_dir}")
