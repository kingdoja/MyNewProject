# Bug修复：CSV注释行解析错误

## 问题描述

当处理宽或高大于50000像素的图像时，系统会自动进行50%缩放，并在生成的坐标CSV文件中添加注释行说明坐标映射关系。但是推理脚本在读取带注释行的CSV文件时会报错。

## 错误信息

```
File ".../inferenceTool/predict_batch_torchscript.py", line 313, in load_patch_coordinates
RuntimeError: 预测脚本执行失败，返回码: 1
```

## 根本原因

### 1. CSV文件格式变化

当图像经过自动缩放后，生成的CSV文件包含注释行：

```csv
# Coordinates are in the original image coordinate system
# Scale factor: 2.0x
filename,x_start,y_start,x_end,y_end
patch_0.png,0,0,1280,1280
patch_1.png,1280,0,2560,1280
...
```

### 2. 原有代码的问题

原来的 `load_patch_coordinates` 函数使用 `csv.DictReader(f)` 直接读取文件，它会将第一行作为表头。当第一行是注释行时，会导致：

- `csv.DictReader` 将 `# Coordinates are in the original image coordinate system` 作为列名
- 第二行 `# Scale factor: 2.0x` 会被当作数据行解析
- 尝试访问 `row['filename']` 时会因为键不存在而报错

## 修复方案

### 修改的文件

`A_sclie2inference/DataSlice2Inference_main/inferenceTool/predict_batch_torchscript.py`

### 修改内容

1. **添加 `io` 模块导入**（第38行）：
   ```python
   import io
   ```

2. **修改 `load_patch_coordinates` 函数**（第301-330行）：
   ```python
   def load_patch_coordinates(csv_path: str) -> Dict[str, Tuple[int, int]]:
       """从CSV文件加载patch坐标信息
       
       CSV格式：filename, x_start, y_start, x_end, y_end
       支持跳过以 # 开头的注释行
       返回：{filename: (x_start, y_start)}
       """
       coordinates = {}
       if not os.path.exists(csv_path):
           raise FileNotFoundError(f"坐标CSV文件不存在: {csv_path}")
       
       with open(csv_path, 'r', encoding='utf-8', newline='') as f:
           # 跳过注释行（以 # 开头）
           lines = []
           for line in f:
               if not line.strip().startswith('#'):
                   lines.append(line)
           
           # 使用过滤后的行创建 DictReader
           csv_content = io.StringIO(''.join(lines))
           reader = csv.DictReader(csv_content)
           
           for row in reader:
               filename = row['filename']
               x_start = int(row['x_start'])
               y_start = int(row['y_start'])
               coordinates[filename] = (x_start, y_start)
       
       print(f"✓ 已加载 {len(coordinates)} 个patch的坐标信息")
       return coordinates
   ```

### 修复逻辑

1. 先读取所有行，过滤掉以 `#` 开头的注释行
2. 将过滤后的内容转换为 `StringIO` 对象
3. 使用 `csv.DictReader` 读取过滤后的内容

## 验证测试

```bash
cd /home/ubuntu/lsn/project_new/RT-DETR-main/A_sclie2inference/DataSlice2Inference_main
python3 -c "
import csv
import io

def load_patch_coordinates(csv_path: str):
    coordinates = {}
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        lines = []
        for line in f:
            if not line.strip().startswith('#'):
                lines.append(line)
        csv_content = io.StringIO(''.join(lines))
        reader = csv.DictReader(csv_content)
        for row in reader:
            filename = row['filename']
            x_start = int(row['x_start'])
            y_start = int(row['y_start'])
            coordinates[filename] = (x_start, y_start)
    print(f'✓ 已加载 {len(coordinates)} 个patch的坐标信息')
    return coordinates

csv_path = '../DataPatches/45-庄驷40X11_20251211_103850/patch_coordinates.csv'
coords = load_patch_coordinates(csv_path)
print(f'前5个坐标: {list(coords.items())[:5]}')
"
```

### 测试结果

```
✓ 已加载 2070 个patch的坐标信息
前5个坐标: [('patch_0.png', (0, 0)), ('patch_1.png', (1280, 0)), ('patch_2.png', (2560, 0)), ('patch_3.png', (3840, 0)), ('patch_4.png', (5120, 0))]
```

## 使用说明

修复后，系统可以正常处理大于50000像素的图像：

1. **自动缩放**：图像宽或高超过50000时，自动缩小到50%
2. **坐标映射**：切片坐标自动映射回原始图像坐标系
3. **CSV注释**：CSV文件包含注释说明坐标映射关系
4. **正常推理**：推理脚本能正确读取带注释的CSV文件

## 重新测试

```bash
# 清理之前的处理状态
rm -f /home/ubuntu/lsn/project_new/RT-DETR-main/A_sclie2inference/DataPatchesInference/processing_status.json

# 启动服务
cd /home/ubuntu/lsn/project_new/RT-DETR-main/A_sclie2inference/DataSlice2Inference_main
python service_main.py
```

## 相关文件

- `DataSlice2Inference_main/inferenceTool/predict_batch_torchscript.py` - 修复的推理脚本
- `DataSlice2Inference_main/auto_process_package/auto_process_monitor.py` - 自动缩放和切片逻辑
- `DataSlice2Inference_main/sliceTool/convert_wsi40x_to_20x.py` - 图像缩放工具
- `DataSlice2Inference_main/config.yaml` - 配置文件（`downscale.threshold: 50000`）

## 修复日期

2025-12-11

