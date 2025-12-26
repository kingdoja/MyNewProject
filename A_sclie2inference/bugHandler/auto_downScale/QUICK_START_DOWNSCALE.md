# 自动缩放功能快速使用指南

## 功能说明

当图像的宽度或高度超过 50000 像素时，系统会自动将图像缩小到 50%，然后进行切片和推理。输出的坐标会自动映射回原始图像坐标系。

## 快速开始

### 1. 配置检查

确认 `config.yaml` 中的配置：

```yaml
downscale:
  enabled: true      # 启用自动缩放
  threshold: 50000   # 触发阈值（像素）
  quality: 95        # 缩放质量
```

### 2. 启动服务

```bash
cd /home/ubuntu/lsn/project_new/RT-DETR-main/A_sclie2inference/DataSlice2Inference_main
python service_main.py
```

### 3. 放入大图像

```bash
# 将大图像放入监听目录
cp your_large_image.jpg ../DataWSI/
```

### 4. 观察日志

服务会自动：
- ✅ 检测图像尺寸
- ✅ 如果超过阈值，自动缩放到 50%
- ✅ 使用缩放后图像进行切片
- ✅ 坐标自动映射回原图

日志示例：
```
======================================================================
步骤 0: 检查图像尺寸
======================================================================
⚠️ 图像尺寸 100000x80000 超过阈值 50000
正在执行自动缩放（50%）...
✅ 图像已缩放: 100000x80000 -> 50000x40000
✅ 缩放后图像保存至: .../DataWSI_downscaled/xxx_downscaled.jpg
📊 坐标转换系数: 2.0x（输出坐标将自动映射到原始图像）
```

### 5. 查看结果

缩放后的图像：
```bash
ls -lh ../DataWSI_downscaled/
```

坐标文件（已映射到原图）：
```bash
cat ../DataPatches/your_image_*/patch_coordinates.csv | head -10
```

## 测试功能

运行自动测试：
```bash
cd /home/ubuntu/lsn/project_new/RT-DETR-main/A_sclie2inference/DataSlice2Inference_main
python test_downscale_feature.py
```

创建测试图像：
```bash
python test_downscale_feature.py --create-large-image
```

## 常见问题

**Q: 如何禁用自动缩放？**
```yaml
downscale:
  enabled: false
```

**Q: 如何调整触发阈值？**
```yaml
downscale:
  threshold: 80000  # 提高到80000像素
```

**Q: 坐标是否准确？**  
是的，坐标会精确乘以缩放系数（2.0x），完全映射回原始图像。

**Q: 对推理精度有影响吗？**  
有轻微影响，但使用高质量的 LANCZOS 插值，影响很小。如果需要更高精度，可以提高 `threshold` 值。

## 技术细节

- **缩放算法**：PIL.Image.Resampling.LANCZOS (高质量)
- **缩放比例**：固定 50%（scale_factor = 2.0）
- **坐标转换**：`original_coord = patch_coord × scale_factor`
- **文件位置**：`DataWSI_downscaled/` 目录

## 性能提升

对于 100000×80000 的图像：
- 内存使用：从 ~24GB 降至 ~6GB (**节省 75%**)
- 处理时间：缩放耗时 1-3 分钟，但后续切片和推理速度提升 **4倍**
- 总体时间：**显著减少**

## 完整文档

详细说明请查看：[AUTO_DOWNSCALE_README.md](./AUTO_DOWNSCALE_README.md)


