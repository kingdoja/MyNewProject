# 图像自动缩放功能说明

## 功能概述

本功能实现了在自动监听和切片之前，智能检查图像尺寸并自动缩放超大图像的能力。

### 主要特性

1. **自动尺寸检测**：监听到图像文件后，自动检查图像的宽度和高度
2. **智能缩放触发**：当图像宽或高超过设定阈值（默认50000像素）时，自动进行50%缩放
3. **坐标自动映射**：缩放后的图像切片坐标会自动映射回原始图像坐标系
4. **无缝集成**：对现有流程完全透明，后续推理和可视化无需修改

## 工作流程

```
原始图像 → 尺寸检查 → [需要缩放?]
                          ↓ 是
                      自动缩放50%
                          ↓
                      保存到DataWSI_downscaled/
                          ↓
                      使用缩放后图像进行切片
                          ↓
                      坐标×2映射回原图
                          ↓
                      正常推理流程
```

## 配置说明

在 `config.yaml` 中添加了 `downscale` 配置项：

```yaml
# 图像自动缩放配置（处理超大图像）
downscale:
  enabled: true  # 是否启用自动缩放（默认：启用）
  threshold: 50000  # 触发缩放的像素阈值（宽或高超过此值则自动缩放）
  quality: 95  # 缩放后JPEG质量（1-100，默认95）
```

### 参数说明

- **enabled**: 是否启用自动缩放功能
  - `true`: 启用（推荐）
  - `false`: 禁用，所有图像使用原始尺寸处理

- **threshold**: 触发缩放的像素阈值
  - 默认: 50000
  - 当图像宽度或高度超过此值时触发缩放
  - 建议根据服务器内存调整（内存充足可提高此值）

- **quality**: 缩放后图像的JPEG质量
  - 范围: 1-100
  - 默认: 95（高质量）
  - 较高值保证图像质量，但文件更大

## 目录结构

启用自动缩放后，会创建以下新目录：

```
A_sclie2inference/
├── DataWSI/                    # 原始图像目录（监听目录）
├── DataWSI_downscaled/         # 缩放后图像保存目录（新增）
│   └── xxx_downscaled.jpg      # 缩放后的图像
├── DataPatches/                # 切片目录
├── DataPatchesKeep/            # 保留的切片
├── DataPatchesTrash/           # 丢弃的切片
└── DataPatchesInference/       # 推理结果
```

## 坐标映射机制

### 问题背景

如果直接对缩小后的图像进行切片，生成的坐标是基于缩小后图像的。但用户通常需要在原始图像上标注，因此需要坐标映射。

### 解决方案

1. **记录缩放系数**：50%缩放对应系数为2.0
2. **切片时使用缩小后图像**：减少内存占用和处理时间
3. **保存坐标时自动转换**：将坐标乘以缩放系数

### 示例

假设原始图像尺寸为 `100000 x 80000` 像素：

1. **触发缩放**：宽度100000 > 50000，触发自动缩放
2. **缩放至50%**：新尺寸 `50000 x 40000`
3. **切片**：在50000x40000的图像上切片
4. **某个patch坐标**（缩小图像上）：`(1280, 640)` 到 `(1920, 1280)`
5. **保存到CSV的坐标**（映射回原图）：`(2560, 1280)` 到 `(3840, 2560)`

### CSV坐标文件示例

```csv
# Coordinates are in the original image coordinate system
# Scale factor: 2.0x
filename,x_start,y_start,x_end,y_end
patch_0.png,0,0,1280,1280
patch_1.png,1280,0,2560,1280
patch_2.png,2560,0,3840,1280
...
```

注意：CSV文件中的注释行说明了坐标系统和缩放系数。

## 使用方法

### 1. 启动服务

```bash
cd /home/ubuntu/lsn/project_new/RT-DETR-main/A_sclie2inference/DataSlice2Inference_main

# 使用配置文件启动
python service_main.py

# 或使用服务管理脚本
./service_manager.sh start
```

### 2. 放入图像

将超大图像（如 100000×80000 像素的WSI）放入监听目录：

```bash
cp your_large_image.jpg /home/ubuntu/lsn/project_new/RT-DETR-main/A_sclie2inference/DataWSI/
```

### 3. 自动处理流程

服务会自动执行以下步骤：

```
[检测到文件] your_large_image.jpg
    ↓
[步骤0] 检查图像尺寸: 100000x80000
    ↓
[判断] 宽度100000 > 50000 → 需要缩放
    ↓
[缩放] 正在缩放到 50000x40000...
    ↓
[保存] DataWSI_downscaled/your_large_image_downscaled.jpg
    ↓
[步骤1] 使用缩放后图像进行切片
    ↓
[步骤1.5] 自动分析背景参数
    ↓
[步骤2] 过滤空白patch
    ↓
[步骤3] 批量推理
    ↓
[完成] 结果保存在 DataPatchesInference/your_large_image_YYYYMMDD_HHMMSS/
```

### 4. 查看日志

日志会显示完整的处理过程：

```
======================================================================
步骤 0: 检查图像尺寸
======================================================================
⚠️ 图像尺寸 100000x80000 超过阈值 50000
正在执行自动缩放（50%）...
正在缩放图像: (100000, 80000) -> (50000, 40000) (缩放比例=0.5)
缩放完成: (100000, 80000) -> (50000, 40000) (2048.5MB -> 512.3MB)
✅ 图像已缩放: 100000x80000 -> 50000x40000
✅ 缩放后图像保存至: .../DataWSI_downscaled/xxx_downscaled.jpg
📊 坐标转换系数: 2.0x（输出坐标将自动映射到原始图像）
```

## 测试

### 测试脚本

创建一个测试图像来验证功能：

```python
# test_large_image.py
from PIL import Image
import numpy as np

# 创建一个模拟的大图像（60000x40000）
width, height = 60000, 40000
print(f"创建测试图像: {width}x{height}")

# 创建随机图像（为节省内存，分块创建）
img = Image.new('RGB', (width, height), color='white')
# 添加一些简单图案用于验证
import PIL.ImageDraw as ImageDraw
draw = ImageDraw.Draw(img)
for i in range(0, width, 5000):
    for j in range(0, height, 5000):
        draw.rectangle([i, j, i+2000, j+2000], fill=(255, 0, 0))

# 保存
output_path = "/home/ubuntu/lsn/project_new/RT-DETR-main/A_sclie2inference/DataWSI/test_large.jpg"
img.save(output_path, quality=95)
print(f"测试图像已保存: {output_path}")
```

### 验证步骤

1. **生成测试图像**：
   ```bash
   python test_large_image.py
   ```

2. **观察日志**：
   - 应该看到"步骤0: 检查图像尺寸"
   - 应该看到"图像尺寸 60000x40000 超过阈值 50000"
   - 应该看到"正在执行自动缩放"

3. **检查缩放后文件**：
   ```bash
   ls -lh /home/ubuntu/lsn/project_new/RT-DETR-main/A_sclie2inference/DataWSI_downscaled/
   ```

4. **检查坐标文件**：
   ```bash
   head -n 10 /home/ubuntu/lsn/project_new/RT-DETR-main/A_sclie2inference/DataPatches/test_large_*/patch_coordinates.csv
   ```
   应该看到注释行和缩放系数

## 常见问题

### Q1: 如何禁用自动缩放？

在 `config.yaml` 中设置：
```yaml
downscale:
  enabled: false
```

### Q2: 如何调整触发阈值？

在 `config.yaml` 中修改：
```yaml
downscale:
  threshold: 80000  # 提高到80000像素
```

### Q3: 坐标是否准确映射？

是的，坐标完全准确。缩放系数为2.0时，所有坐标都精确乘以2。您可以通过以下方式验证：

1. 在原图上某个位置(x, y)标记
2. 在缩放图上该位置应该在(x/2, y/2)
3. CSV中记录的坐标应该是(x, y)

### Q4: 是否支持其他缩放比例？

当前版本固定为50%缩放。如需其他比例，可以修改代码中的 `scale_factor=0.5` 参数。

### Q5: 缩放后图像保存在哪里？

保存在 `DataWSI_downscaled/` 目录，文件名为 `原文件名_downscaled.扩展名`。

### Q6: 缩放会影响推理精度吗？

会有轻微影响，因为：
- 图像分辨率降低，细小特征可能丢失
- 但使用LANCZOS高质量插值算法，影响很小
- 对于大多数WSI图像，50%缩放后仍保持足够的细节

建议：如果推理精度下降明显，可以提高 `threshold` 值，只对极大的图像进行缩放。

## 技术细节

### 修改的文件

1. **convert_wsi40x_to_20x.py**
   - 添加 `convert_single_image()` 函数
   - 支持单图像转换和质量控制

2. **auto_process_monitor.py**
   - 添加 `check_image_size()` 方法
   - 添加 `downscale_image()` 方法
   - 修改 `slice_image()` 支持 `scale_factor` 参数
   - 修改 `process_image()` 添加步骤0（尺寸检查和缩放）

3. **service_main.py**
   - 更新配置加载，支持 `downscale` 配置项

4. **config.yaml**
   - 添加 `downscale` 配置段

### 关键算法

```python
# 1. 检查尺寸
needs_downscale = (width > threshold or height > threshold)

# 2. 如果需要缩放
if needs_downscale:
    new_size = (width // 2, height // 2)
    scale_factor = 2.0
    
# 3. 切片时记录scale_factor
slice_image(downscaled_path, output_dir, scale_factor=2.0)

# 4. 保存坐标时映射
orig_x = patch_x * scale_factor
orig_y = patch_y * scale_factor
```

## 性能优化

### 内存使用

缩放功能显著减少内存使用：
- 原图 100000×80000 ≈ 24GB (RGB)
- 缩放后 50000×40000 ≈ 6GB (RGB)
- **节省 75% 内存**

### 处理时间

虽然增加了缩放步骤，但总体时间可能更短：
- 缩放耗时：约1-3分钟（取决于原图大小）
- 切片速度提升：4倍（图像面积减少到1/4）
- 推理速度提升：4倍（patch数量减少到1/4）

### 磁盘空间

缩放后图像会占用额外磁盘空间：
- 建议定期清理 `DataWSI_downscaled/` 目录
- 或修改代码，处理完成后自动删除缩放图像

## 总结

自动缩放功能：
✅ 自动检测超大图像
✅ 智能触发缩放处理
✅ 坐标精确映射回原图
✅ 对现有流程完全透明
✅ 显著降低内存和时间成本
✅ 可通过配置灵活控制

建议在生产环境中启用此功能，特别是处理病理切片等超大图像时。


