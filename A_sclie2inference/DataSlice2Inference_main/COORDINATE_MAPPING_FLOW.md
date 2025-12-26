# 坐标转换和映射流程详解

本文档详细说明在两种情况下（图像宽/高 > 50000 和 < 50000）的坐标转换和生成流程。

---

## 一、图像尺寸 < 50000 像素的处理流程

### 1. 图像检查阶段

**代码位置**: `auto_process_monitor.py` 第327-363行

```python
needs_downscale, orig_width, orig_height = self.check_image_size(image_path)

if needs_downscale:
    # 需要缩放...
else:
    self._log(f"✅ 图像尺寸 {orig_width}x{orig_height} 无需缩放\n")
```

**关键参数**:
- `scale_factor = 1.0` (无缩放)
- `actual_image_path = image_path` (使用原始图像)

### 2. 切片阶段

**代码位置**: `auto_process_monitor.py` 第501-610行

```python
def slice_image(self, image_path: Path, output_dir: Path, scale_factor: float = 1.0):
    # 打开原始图像
    pil_img = Image.open(image_path)
    width, height = pil_img.size  # 例如: 40000 x 30000
    
    # 切片参数
    patch_size = 640  # 固定大小
    num_patches_x = width // patch_size   # 40000 // 640 = 62
    num_patches_y = height // patch_size  # 30000 // 640 = 46
    
    # 切片并记录坐标（在当前图像上的坐标）
    for y_idx in range(num_patches_y):
        for x_idx in range(num_patches_x):
            x = x_idx * patch_size  # 0, 640, 1280, 1920, ...
            y = y_idx * patch_size  # 0, 640, 1280, 1920, ...
            
            # 切片
            patch_box = (x, y, x + patch_size, y + patch_size)
            patch_pil = pil_img.crop(patch_box)
            patch_pil.save(output_dir / f'patch_{patch_count}.png')
```

### 3. 坐标保存阶段

**代码位置**: `auto_process_monitor.py` 第589-610行

```python
# 保存坐标到CSV (scale_factor = 1.0)
csv_path = output_dir / 'patch_coordinates.csv'
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    # scale_factor = 1.0 时，不添加注释行
    writer.writerow(['filename', 'x_start', 'y_start', 'x_end', 'y_end'])
    
    for i in range(patch_count):
        filename = f'patch_{i}.png'
        # 坐标映射: 乘以 scale_factor (1.0)
        orig_x_start = int(x_r[i] * 1.0)      # 0 * 1.0 = 0
        orig_y_start = int(y_r[i] * 1.0)      # 0 * 1.0 = 0
        orig_x_end = int((x_r[i] + 640) * 1.0)  # 640 * 1.0 = 640
        orig_y_end = int((y_r[i] + 640) * 1.0)  # 640 * 1.0 = 640
        writer.writerow([filename, orig_x_start, orig_y_start, orig_x_end, orig_y_end])
```

**生成的CSV文件内容** (无注释行):
```csv
filename,x_start,y_start,x_end,y_end
patch_0.png,0,0,640,640
patch_1.png,640,0,1280,640
patch_2.png,1280,0,1920,640
patch_3.png,1920,0,2560,640
...
```

### 4. 推理阶段 - 坐标加载

**代码位置**: `predict_batch_torchscript.py` 第301-330行

```python
def load_patch_coordinates(csv_path: str):
    coordinates = {}
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        # 跳过注释行（无缩放时没有注释行）
        lines = []
        for line in f:
            if not line.strip().startswith('#'):
                lines.append(line)
        
        csv_content = io.StringIO(''.join(lines))
        reader = csv.DictReader(csv_content)
        
        for row in reader:
            filename = row['filename']
            x_start = int(row['x_start'])  # 直接使用CSV中的坐标
            y_start = int(row['y_start'])  # 已经是原图坐标
            coordinates[filename] = (x_start, y_start)
    
    return coordinates  # {'patch_0.png': (0, 0), 'patch_1.png': (640, 0), ...}
```

### 5. 推理阶段 - 检测框坐标转换

**代码位置**: `predict_batch_torchscript.py` 第333-352行

```python
# 加载patch坐标
patch_offset = coordinates[patch_filename]  # 例如: (640, 0)

# 模型在patch上检测到的框 (相对于patch的坐标)
boxes_patch = model_output  # 例如: [[100, 150, 200, 250], ...]
                           # 表示: 在patch内，左上角(100,150)，右下角(200,250)

# 转换为全图坐标
def convert_to_global_coordinates(boxes, patch_offset):
    x_offset, y_offset = patch_offset  # (640, 0)
    offset_tensor = torch.tensor([x_offset, y_offset, x_offset, y_offset])
    return boxes + offset_tensor

boxes_global = convert_to_global_coordinates(boxes_patch, patch_offset)
# 结果: [[100+640, 150+0, 200+640, 250+0]] = [[740, 150, 840, 250]]
# 这就是在原始40000x30000图像上的坐标
```

### 坐标流程图 (< 50000):

```
原始图像 (40000 x 30000)
    ↓
[无需缩放]
    ↓
切片 (640x640)
    ↓
patch坐标: (0,0), (640,0), (1280,0), ...
    ↓
保存到CSV (scale_factor=1.0, 无映射)
    ↓
CSV: x_start=0, y_start=0  (原图坐标)
     x_start=640, y_start=0 (原图坐标)
    ↓
推理时读取坐标: (0,0), (640,0), ...
    ↓
检测框: patch内(100,150,200,250)
    ↓
转换为全图坐标: (100+0, 150+0, 200+0, 250+0) = (100,150,200,250)
    ↓
✅ 最终坐标: 在原始40000x30000图像上的正确位置
```

---

## 二、图像尺寸 > 50000 像素的处理流程

### 1. 图像检查和缩放阶段

**代码位置**: `auto_process_monitor.py` 第327-363行

```python
needs_downscale, orig_width, orig_height = self.check_image_size(image_path)
# 例如: needs_downscale=True, orig_width=100000, orig_height=80000

if needs_downscale:
    self._log(f"⚠️ 图像尺寸 {orig_width}x{orig_height} 超过阈值 {self.downscale_threshold}")
    
    # 执行50%缩放
    orig_w, orig_h, new_w, new_h = self.downscale_image(
        image_path,
        downscaled_path,
        scale_factor=0.5  # 缩放到50%
    )
    # orig_w=100000, orig_h=80000
    # new_w=50000, new_h=40000
    
    scale_factor = 2.0  # 坐标需要乘以2才能映射回原图
    actual_image_path = downscaled_path  # 后续处理使用缩放后的图像
```

**关键参数**:
- `scale_factor = 2.0` (需要×2映射回原图)
- `actual_image_path = downscaled_path` (使用缩放后的50000x40000图像)

### 2. 切片阶段

**代码位置**: `auto_process_monitor.py` 第501-610行

```python
def slice_image(self, image_path: Path, output_dir: Path, scale_factor: float = 2.0):
    # 打开缩放后的图像
    pil_img = Image.open(image_path)  # downscaled_path
    width, height = pil_img.size  # 50000 x 40000 (已缩放)
    
    # 切片参数
    patch_size = 640
    num_patches_x = width // patch_size   # 50000 // 640 = 78
    num_patches_y = height // patch_size  # 40000 // 640 = 62
    
    # 切片并记录坐标（在缩放后图像上的坐标）
    for y_idx in range(num_patches_y):
        for x_idx in range(num_patches_x):
            x = x_idx * patch_size  # 在50000x40000图像上: 0, 640, 1280, ...
            y = y_idx * patch_size  # 在50000x40000图像上: 0, 640, 1280, ...
            
            # 从缩放后的图像切片
            patch_box = (x, y, x + patch_size, y + patch_size)
            patch_pil = pil_img.crop(patch_box)
            patch_pil.save(output_dir / f'patch_{patch_count}.png')
```

### 3. 坐标保存阶段 (关键：自动映射)

**代码位置**: `auto_process_monitor.py` 第589-610行

```python
# 保存坐标到CSV (scale_factor = 2.0)
csv_path = output_dir / 'patch_coordinates.csv'
with open(csv_path, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    
    # 添加注释说明坐标已映射到原图
    if scale_factor != 1.0:
        writer.writerow(['# Coordinates are in the original image coordinate system'])
        writer.writerow([f'# Scale factor: {scale_factor}x'])
    
    writer.writerow(['filename', 'x_start', 'y_start', 'x_end', 'y_end'])
    
    for i in range(patch_count):
        filename = f'patch_{i}.png'
        # 🔑 关键：坐标映射回原图 (乘以 scale_factor)
        orig_x_start = int(x_r[i] * 2.0)        # 0 * 2.0 = 0
        orig_y_start = int(y_r[i] * 2.0)        # 0 * 2.0 = 0
        orig_x_end = int((x_r[i] + 640) * 2.0)  # 640 * 2.0 = 1280
        orig_y_end = int((y_r[i] + 640) * 2.0)  # 640 * 2.0 = 1280
        writer.writerow([filename, orig_x_start, orig_y_start, orig_x_end, orig_y_end])
```

**生成的CSV文件内容** (有注释行):
```csv
# Coordinates are in the original image coordinate system
# Scale factor: 2.0x
filename,x_start,y_start,x_end,y_end
patch_0.png,0,0,1280,1280
patch_1.png,1280,0,2560,1280
patch_2.png,2560,0,3840,1280
patch_3.png,3840,0,5120,1280
...
```

**注意**: 虽然patch是从50000x40000的缩放图切的640x640，但CSV中保存的坐标已经映射回原始100000x80000图像的坐标系：
- patch_0在缩放图上是(0,0)-(640,640)
- 但CSV记录的是原图坐标(0,0)-(1280,1280)

### 4. 推理阶段 - 坐标加载

**代码位置**: `predict_batch_torchscript.py` 第301-330行

```python
def load_patch_coordinates(csv_path: str):
    coordinates = {}
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        # 跳过注释行
        lines = []
        for line in f:
            if not line.strip().startswith('#'):
                lines.append(line)
        
        csv_content = io.StringIO(''.join(lines))
        reader = csv.DictReader(csv_content)
        
        for row in reader:
            filename = row['filename']
            x_start = int(row['x_start'])  # 直接读取CSV中的坐标
            y_start = int(row['y_start'])  # 已经是原图坐标（×2后的）
            coordinates[filename] = (x_start, y_start)
    
    return coordinates  # {'patch_0.png': (0, 0), 'patch_1.png': (1280, 0), ...}
    # 注意：这些坐标已经是在原始100000x80000图像上的坐标了！
```

### 5. 推理阶段 - 检测框坐标转换

**代码位置**: `predict_batch_torchscript.py` 第333-352行

```python
# 加载patch坐标 (已经是原图坐标)
patch_offset = coordinates[patch_filename]  # 例如: (1280, 0) - 原图坐标

# 模型在patch上检测到的框 (相对于patch的坐标)
# patch本身是640x640大小
boxes_patch = model_output  # 例如: [[100, 150, 200, 250], ...]
                           # 表示: 在640x640的patch内，左上角(100,150)，右下角(200,250)

# 🔑 关键：需要将patch内坐标也映射到原图
# 但是这里有个问题：patch内的检测框是在640x640的patch上的
# 而这个patch实际对应原图的1280x1280区域
# 所以需要×2映射

# 转换为全图坐标
def convert_to_global_coordinates(boxes, patch_offset):
    x_offset, y_offset = patch_offset  # (1280, 0) - 原图坐标
    offset_tensor = torch.tensor([x_offset, y_offset, x_offset, y_offset])
    return boxes + offset_tensor

boxes_global = convert_to_global_coordinates(boxes_patch, patch_offset)
# 结果: [[100+1280, 150+0, 200+1280, 250+0]] = [[1380, 150, 1480, 250]]
```

**⚠️ 发现问题**：当前实现存在坐标映射不完整的问题！

### 坐标流程图 (> 50000) - 当前实现:

```
原始图像 (100000 x 80000)
    ↓
[缩放到50%]
    ↓
缩放后图像 (50000 x 40000)
    ↓
切片 (640x640) 从缩放图切
    ↓
patch在缩放图上坐标: (0,0), (640,0), (1280,0), ...
    ↓
保存到CSV时×2映射回原图
    ↓
CSV: x_start=0, y_start=0     (原图坐标，×2)
     x_start=1280, y_start=0  (原图坐标，×2)
    ↓
推理时读取坐标: (0,0), (1280,0), ... (原图坐标)
    ↓
检测框: patch内(100,150,200,250) - 在640x640的patch上
    ↓
⚠️ 问题：检测框坐标也需要×2！
    因为patch实际对应原图1280x1280区域
    检测框(100,150)在640x640的patch上
    应该对应原图上(200,300)的位置
    ↓
当前转换: (100+1280, 150+0, 200+1280, 250+0) = (1380,150,1480,250) ❌
正确转换: ((100*2)+1280, (150*2)+0, (200*2)+1280, (250*2)+0) = (1480,300,1680,500) ✅
```

---

## 三、坐标映射问题分析

### 🔴 当前实现的问题

**问题位置**: `predict_batch_torchscript.py` 第333-352行

当图像经过缩放后：
1. ✅ CSV中的patch坐标已正确映射回原图（×2）
2. ❌ 但检测框坐标没有进行相应的缩放映射

**示例说明**:

```python
# 场景：原图100000x80000，缩放为50000x40000
# patch_0 在缩放图上是 (0,0)-(640,640)
# CSV中记录为原图坐标 (0,0)-(1280,1280)

# 模型在patch上检测到一个框：(100,150,200,250)
# 这是在640x640的patch上的坐标

# 当前实现：
boxes_global = boxes_patch + patch_offset
# = (100,150,200,250) + (0,0,0,0)
# = (100,150,200,250)  ❌ 错误！

# 正确实现应该是：
boxes_in_original_scale = boxes_patch * 2.0  # 先将patch内坐标映射到原图尺度
boxes_global = boxes_in_original_scale + patch_offset
# = (100*2, 150*2, 200*2, 250*2) + (0,0,0,0)
# = (200,300,400,500)  ✅ 正确！
```

### 🟢 解决方案

需要在 `predict_batch_torchscript.py` 中增加对检测框坐标的缩放映射。

**方案1**: 从CSV注释行读取scale_factor

```python
def load_patch_coordinates(csv_path: str) -> Tuple[Dict[str, Tuple[int, int]], float]:
    """返回坐标字典和缩放系数"""
    coordinates = {}
    scale_factor = 1.0  # 默认无缩放
    
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        lines = []
        for line in f:
            if line.strip().startswith('# Scale factor:'):
                # 提取缩放系数: "# Scale factor: 2.0x" -> 2.0
                scale_factor = float(line.split(':')[1].strip().rstrip('x'))
            elif not line.strip().startswith('#'):
                lines.append(line)
        
        csv_content = io.StringIO(''.join(lines))
        reader = csv.DictReader(csv_content)
        
        for row in reader:
            filename = row['filename']
            x_start = int(row['x_start'])
            y_start = int(row['y_start'])
            coordinates[filename] = (x_start, y_start)
    
    return coordinates, scale_factor

def convert_to_global_coordinates(
    boxes: torch.Tensor,
    patch_offset: Tuple[int, int],
    scale_factor: float = 1.0
) -> torch.Tensor:
    """将patch内的检测框坐标转换为全图坐标
    
    Args:
        boxes: 检测框坐标 (N, 4)，相对于patch
        patch_offset: patch在全图中的偏移量 (x_start, y_start) - 已经是原图坐标
        scale_factor: 图像缩放系数，用于将patch内坐标映射到原图尺度
    """
    # 先将patch内坐标按scale_factor缩放
    boxes_scaled = boxes * scale_factor
    
    # 再加上patch在原图中的偏移
    x_offset, y_offset = patch_offset
    offset_tensor = torch.tensor(
        [x_offset, y_offset, x_offset, y_offset],
        dtype=boxes_scaled.dtype,
        device=boxes_scaled.device
    )
    return boxes_scaled + offset_tensor
```

---

## 四、验证测试

### 测试场景1: 图像 < 50000

```
原始图像: 40000 x 30000
scale_factor: 1.0 (无缩放)

patch_0 CSV坐标: (0, 0)
检测框 (在patch内): (100, 150, 200, 250)

转换过程:
1. boxes_scaled = (100, 150, 200, 250) * 1.0 = (100, 150, 200, 250)
2. boxes_global = (100, 150, 200, 250) + (0, 0, 0, 0) = (100, 150, 200, 250)

✅ 结果: (100, 150, 200, 250) 在原图40000x30000上
```

### 测试场景2: 图像 > 50000

```
原始图像: 100000 x 80000
缩放后: 50000 x 40000
scale_factor: 2.0

patch_1 在缩放图上: (640, 0)-(1280, 640)
patch_1 CSV坐标: (1280, 0)  (已×2映射回原图)
检测框 (在patch内): (100, 150, 200, 250)

转换过程:
1. boxes_scaled = (100, 150, 200, 250) * 2.0 = (200, 300, 400, 500)
2. boxes_global = (200, 300, 400, 500) + (1280, 0, 1280, 0) = (1480, 300, 1680, 500)

✅ 结果: (1480, 300, 1680, 500) 在原图100000x80000上
验证: 
- patch_1左上角在原图的(1280,0)
- 检测框在patch内偏移(100,150)，缩放×2后是(200,300)
- 最终位置: (1280+200, 0+300) = (1480, 300) ✅
```

---

## 五、总结

### 当前状态

| 功能 | 尺寸<50000 | 尺寸>50000 | 状态 |
|------|-----------|-----------|------|
| 图像缩放 | ✅ 无需缩放 | ✅ 自动缩放50% | 正常 |
| 切片 | ✅ 原图切片 | ✅ 缩放图切片 | 正常 |
| CSV坐标 | ✅ 直接原图坐标 | ✅ 已×2映射回原图 | 正常 |
| CSV读取 | ✅ 正常读取 | ✅ 跳过注释行 | 已修复 |
| 检测框映射 | ✅ 直接加偏移 | ❌ 需要先×2再加偏移 | **待修复** |

### 需要修复

检测框坐标在缩放场景下的映射逻辑需要更新：

```python
# 当前（错误）:
boxes_global = boxes_patch + patch_offset

# 应该修改为（正确）:
boxes_scaled = boxes_patch * scale_factor  # 先缩放
boxes_global = boxes_scaled + patch_offset  # 再加偏移
```

### 修复文件

需要修改 `predict_batch_torchscript.py`:
1. `load_patch_coordinates()` - 增加返回scale_factor
2. `convert_to_global_coordinates()` - 增加scale_factor参数并应用缩放
3. 所有调用处 - 传入scale_factor参数

---

**创建日期**: 2025-12-11  
**相关文件**: 
- `auto_process_monitor.py` - 切片和坐标生成
- `predict_batch_torchscript.py` - 坐标加载和检测框映射
- `convert_wsi40x_to_20x.py` - 图像缩放工具

