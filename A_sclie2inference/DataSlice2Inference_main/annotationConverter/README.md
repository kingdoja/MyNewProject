# 标注坐标转换工具

将基于全图的标注框坐标转换为基于640x640切片的COCO格式标注文件。

## 功能说明

1. **坐标转换**：将全图坐标的标注框转换为patch内的相对坐标
2. **格式转换**：生成标准COCO格式的JSON标注文件
3. **可视化**：在patch图像上绘制标注框，便于验证转换结果

## 使用方法

### 基本用法

```bash
python convert_global_to_patch.py \
    --annotation-file /path/to/标记.json \
    --patch-dir /path/to/patch_dir \
    --output-dir /path/to/output \
    --wsi-image /path/to/wsi.jpeg
```

### 参数说明

- `--annotation-file` (必需): 输入的标注文件路径（标记.json格式）
- `--patch-dir` (必需): patch图像所在目录
- `--output-dir` (必需): 输出目录
- `--wsi-image` (可选): 全图路径（用于验证，当前版本未使用）
- `--patch-size` (可选): patch尺寸，默认640
- `--coordinates-csv` (可选): patch坐标CSV文件路径，默认在patch_dir下查找`patch_coordinates.csv`
- `--min-overlap-ratio` (可选): 标注框与patch的最小重叠比例阈值，默认0.1（10%）
- `--no-visualization` (可选): 不生成可视化图片

### 示例

```bash
python convert_global_to_patch.py \
    --annotation-file /home/ubuntu/lsn/project_new/RT-DETR-main/A_sclie2inference/test_big_conv_small/标记.json \
    --patch-dir /home/ubuntu/lsn/project_new/RT-DETR-main/A_sclie2inference/DataPatches/cj_20260106_155902 \
    --output-dir /home/ubuntu/lsn/project_new/RT-DETR-main/A_sclie2inference/DataSlice2Inference_main/annotationConverter/output \
    --wsi-image /home/ubuntu/lsn/project_new/RT-DETR-main/A_sclie2inference/DataWSI/cj.jpeg
```

## 输入格式

### 标注文件格式（标记.json）

```json
[
    {
        "box": "[21905.254, 13596.98, 21967.307, 13656.157]",
        "boxList": [21905.254, 13596.98, 21967.307, 13656.157],
        "cellType": "TC1",
        "coordinatesId": 146363,
        "cropImageName": "cj_1.jpg",
        "cropPreviewUrl": "http://...",
        "imageType": "1",
        "params": {},
        "sampleId": 36,
        "score": 0.0
    }
]
```

**说明**：
- `box` 或 `boxList`: 标注框坐标，格式为 `[x1, y1, x2, y2]`（全图坐标）
- `cellType`: 细胞类型，如 "TC1", "TC2", "AD" 等

### Patch坐标文件格式（patch_coordinates.csv）

```csv
filename,x_start,y_start,x_end,y_end
patch_0.png,0,0,640,640
patch_1.png,640,0,1280,640
...
```

**说明**：
- `filename`: patch文件名
- `x_start, y_start`: patch在全图中的左上角坐标
- `x_end, y_end`: patch在全图中的右下角坐标

## 输出格式

### COCO格式JSON（coco_format.json）

```json
{
  "info": {
    "description": "Converted from global annotations to patch-based COCO format",
    "version": "1.0",
    "year": 2024
  },
  "licenses": [],
  "categories": [
    {"id": 0, "name": "AD"},
    {"id": 1, "name": "BC"},
    ...
  ],
  "images": [
    {
      "id": 0,
      "width": 640,
      "height": 640,
      "file_name": "patch_0.png"
    }
  ],
  "annotations": [
    {
      "id": 0,
      "image_id": 0,
      "category_id": 9,
      "segmentation": [],
      "bbox": [x, y, width, height],
      "ignore": 0,
      "iscrowd": 0,
      "area": 912.82
    }
  ]
}
```

**说明**：
- `images`: 所有有标注的patch信息
- `annotations`: 所有标注框信息
- `bbox`: COCO格式，`[x, y, width, height]`（左上角坐标+宽高）

### 可视化输出

在 `output_dir/visualization/` 目录下生成可视化图片，文件名格式为 `vis_{patch_filename}`。

每个可视化图片上会显示：
- 标注框（不同类别用不同颜色）
- 类别名称和重叠比例

## 转换逻辑

1. **坐标分配**：对于每个全图标注框，找到所有与其有重叠的patch（重叠比例 >= 阈值）
2. **坐标转换**：将全图坐标转换为patch内的相对坐标
   - `patch_x = global_x - patch_x_start`
   - `patch_y = global_y - patch_y_start`
3. **坐标裁剪**：将超出patch范围的坐标裁剪到patch边界内
4. **格式转换**：将 `(x1, y1, x2, y2)` 格式转换为COCO格式 `[x, y, width, height]`

## 类别映射

支持的类别及其ID映射：

| 类别名称 | ID |
|---------|-----|
| AD | 0 |
| BC | 1 |
| EC | 2 |
| L | 3 |
| LC | 4 |
| M | 5 |
| NT | 6 |
| SM | 7 |
| SQ | 8 |
| TC1 | 9 |
| TC2 | 10 |
| TC3 | 11 |

## 注意事项

1. **重叠处理**：如果一个标注框跨越多个patch，它会被分配到所有重叠的patch中
2. **坐标裁剪**：超出patch范围的标注框会被裁剪到patch边界内
3. **最小重叠阈值**：默认只保留重叠比例 >= 10% 的标注框，可以通过 `--min-overlap-ratio` 调整
4. **文件路径**：确保patch目录下有 `patch_coordinates.csv` 文件，或通过 `--coordinates-csv` 指定路径

## 输出目录结构

```
output_dir/
├── coco_format.json          # COCO格式的标注文件
└── visualization/             # 可视化图片目录
    ├── vis_patch_0.png
    ├── vis_patch_1.png
    ...
```

## 依赖库

- numpy
- Pillow (PIL)
- tqdm

确保已安装这些依赖库。

