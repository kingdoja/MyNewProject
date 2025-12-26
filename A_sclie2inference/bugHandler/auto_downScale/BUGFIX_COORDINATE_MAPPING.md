# Bug修复：检测框坐标映射到原图的缩放问题

## 问题描述

当处理宽或高大于50000像素的图像时，系统会自动进行50%缩放，但检测框坐标在映射回原图时，没有考虑缩放系数，导致坐标偏小（只有实际位置的一半）。

## 问题分析

### 场景回顾

1. **图像尺寸 < 50000**: 直接在原图上切片，坐标无需额外处理
2. **图像尺寸 > 50000**: 先缩放到50%，在缩放图上切片，需要两次坐标映射

### 坐标映射流程

#### 场景1：图像 < 50000（无问题）

```
原图 (40000 x 30000)
  ↓ [无缩放, scale_factor=1.0]
切片 (640x640)
  ↓
patch_1 坐标: (640, 0) - CSV保存
  ↓
检测框(patch内): (100, 150, 200, 250)
  ↓
全图坐标 = (100, 150, 200, 250) + (640, 0, 640, 0)
         = (740, 150, 840, 250) ✅
```

#### 场景2：图像 > 50000（有问题）

```
原图 (100000 x 80000)
  ↓ [缩放到50%, scale_factor=2.0]
缩放图 (50000 x 40000)
  ↓
切片 (640x640) - 在缩放图上切
  ↓
patch_1 在缩放图: (640, 0)-(1280, 640)
patch_1 在原图: (1280, 0)-(2560, 1280) - CSV保存(已×2)
  ↓
检测框(patch内): (100, 150, 200, 250) - 在640x640的patch上
  ↓
❌ 原来的实现：
全图坐标 = (100, 150, 200, 250) + (1280, 0, 1280, 0)
         = (1380, 150, 1480, 250) ❌ 错误！

✅ 正确的实现：
步骤1: boxes_scaled = (100, 150, 200, 250) × 2.0
                     = (200, 300, 400, 500)
步骤2: 全图坐标 = (200, 300, 400, 500) + (1280, 0, 1280, 0)
                = (1480, 300, 1680, 500) ✅ 正确！
```

### 根本原因

**关键问题**：检测框坐标是在640x640的patch上检测的，但这个patch实际对应原图1280x1280的区域。因此：

1. CSV中的patch坐标已经映射回原图（×2）✅
2. 但检测框在patch内的坐标没有映射（×2）❌

这导致检测框坐标只加了偏移，没有考虑缩放系数。

## 修复方案

### 修改文件

`A_sclie2inference/DataSlice2Inference_main/inferenceTool/predict_batch_torchscript.py`

### 修改内容

#### 1. 修改 `load_patch_coordinates` 函数（第301-348行）

**修改前**：
```python
def load_patch_coordinates(csv_path: str) -> Dict[str, Tuple[int, int]]:
    """返回坐标字典"""
    # ... 代码 ...
    return coordinates
```

**修改后**：
```python
def load_patch_coordinates(csv_path: str) -> Tuple[Dict[str, Tuple[int, int]], float]:
    """返回坐标字典和缩放系数"""
    coordinates = {}
    scale_factor = 1.0  # 默认无缩放
    
    with open(csv_path, 'r', encoding='utf-8', newline='') as f:
        lines = []
        for line in f:
            stripped = line.strip()
            # 从注释行提取scale_factor
            if stripped.startswith('# Scale factor:'):
                scale_str = stripped.split(':')[1].strip().rstrip('x')
                scale_factor = float(scale_str)
                print(f"✓ 检测到图像缩放系数: {scale_factor}x")
            elif not stripped.startswith('#'):
                lines.append(line)
        
        # ... CSV读取代码 ...
    
    return coordinates, scale_factor
```

**功能**：从CSV注释行提取缩放系数，如果没有注释则默认为1.0。

#### 2. 修改 `convert_to_global_coordinates` 函数（第351-388行）

**修改前**：
```python
def convert_to_global_coordinates(
    boxes: torch.Tensor,
    patch_offset: Tuple[int, int]
) -> torch.Tensor:
    """将patch内坐标转换为全图坐标"""
    x_offset, y_offset = patch_offset
    offset_tensor = torch.tensor([x_offset, y_offset, x_offset, y_offset])
    return boxes + offset_tensor  # ❌ 缺少scale_factor处理
```

**修改后**：
```python
def convert_to_global_coordinates(
    boxes: torch.Tensor,
    patch_offset: Tuple[int, int],
    scale_factor: float = 1.0
) -> torch.Tensor:
    """将patch内坐标转换为全图坐标
    
    说明：
        当图像经过缩放时，patch是在缩放后的图像上切的，
        但patch_offset已经映射回原图。因此需要：
        1. 先将patch内的检测框坐标按scale_factor缩放
        2. 再加上patch在原图中的偏移量
    """
    # 步骤1：将patch内坐标按scale_factor缩放到原图尺度
    boxes_scaled = boxes * scale_factor
    
    # 步骤2：加上patch在原图中的偏移量
    x_offset, y_offset = patch_offset
    offset_tensor = torch.tensor(
        [x_offset, y_offset, x_offset, y_offset],
        dtype=boxes_scaled.dtype,
        device=boxes_scaled.device
    )
    
    return boxes_scaled + offset_tensor
```

**关键改进**：增加 `scale_factor` 参数，并在加偏移前先将坐标缩放。

#### 3. 更新所有调用处（3处）

##### 3.1 main函数（第724行）
```python
# 修改前
coordinates = load_patch_coordinates(str(csv_path))

# 修改后
coordinates, scale_factor = load_patch_coordinates(str(csv_path))
```

##### 3.2 predict_single_patch函数（第618-627行，第639行）
```python
# 修改前
def predict_single_patch(...):
    # ...
    boxes_global = convert_to_global_coordinates(boxes_patch, patch_offset)

# 修改后
def predict_single_patch(..., scale_factor: float = 1.0):
    # ...
    boxes_global = convert_to_global_coordinates(boxes_patch, patch_offset, scale_factor)
```

##### 3.3 predict_batch_patches函数（第488-497行，第563行）
```python
# 修改前
def predict_batch_patches(...):
    # ...
    boxes_global = convert_to_global_coordinates(boxes_patch, patch_offset)

# 修改后
def predict_batch_patches(..., scale_factor: float = 1.0):
    # ...
    boxes_global = convert_to_global_coordinates(boxes_patch, patch_offset, scale_factor)
```

##### 3.4 main函数调用处（第812行）
```python
# 修改前
batch_results = predict_batch_patches(
    model=model,
    ...
)

# 修改后
batch_results = predict_batch_patches(
    model=model,
    ...
    scale_factor=scale_factor
)
```

## 验证测试

### 测试脚本

创建了 `test_coordinate_mapping.py` 进行全面测试。

### 测试结果

```
======================================================================
坐标映射修复验证测试
======================================================================

测试 1: 情况1: 图像尺寸 < 50000 (无缩放)
描述: 原图40000x30000，patch在(640,0)，检测框在patch内(100,150,200,250)
缩放系数: 1.0x
期望结果: [[740.0, 150.0, 840.0, 250.0]]
实际结果: [[740.0, 150.0, 840.0, 250.0]]
✅ 通过

测试 2: 情况2: 图像尺寸 > 50000 (50%缩放)
描述: 原图100000x80000，缩放后50000x40000，patch在原图(1280,0)
缩放系数: 2.0x
期望结果: [[1480.0, 300.0, 1680.0, 500.0]]
实际结果: [[1480.0, 300.0, 1680.0, 500.0]]
✅ 通过

测试 3: 情况3: 无缩放，多个检测框
✅ 通过

测试 4: 情况4: 2倍缩放，多个检测框
✅ 通过

======================================================================
🎉 所有测试通过！坐标映射修复成功。
======================================================================
```

### 手动计算验证

```
场景：原图100000x80000，缩放为50000x40000 (scale_factor=2.0)
patch_1在缩放图上: (640, 0)-(1280, 640)
patch_1在原图上: (1280, 0)-(2560, 1280)
检测框在patch内: (100, 150, 200, 250)

步骤1: 将patch内坐标×2 (映射到原图尺度)
  (100, 150, 200, 250) × 2.0 = (200, 300, 400, 500)

步骤2: 加上patch在原图的偏移 (1280, 0)
  (200, 300, 400, 500) + (1280, 0, 1280, 0)
  = (1480, 300, 1680, 500)

✅ 手动计算验证通过！
```

## 修复效果

### 修复前

| 场景 | 坐标映射 | 结果 |
|------|----------|------|
| 图像 < 50000 | ✅ 正确 | 正常工作 |
| 图像 > 50000 | ❌ 错误 | 坐标偏小50% |

### 修复后

| 场景 | 坐标映射 | 结果 |
|------|----------|------|
| 图像 < 50000 | ✅ 正确 | 正常工作 |
| 图像 > 50000 | ✅ 正确 | 正常工作 |

## 使用说明

修复后，系统可以正确处理两种情况：

### 场景1：图像尺寸 < 50000

```python
# 自动处理流程
原图40000x30000
  ↓ scale_factor=1.0
切片640x640
  ↓
CSV坐标: (640, 0) - 原图坐标
  ↓
检测框映射: boxes × 1.0 + offset
  ↓
✅ 正确的全图坐标
```

### 场景2：图像尺寸 > 50000

```python
# 自动处理流程
原图100000x80000
  ↓ 缩放50% → 50000x40000
  ↓ scale_factor=2.0
切片640x640 (在缩放图上)
  ↓
CSV坐标: (1280, 0) - 已×2映射回原图
CSV注释: # Scale factor: 2.0x
  ↓
检测框映射: boxes × 2.0 + offset
  ↓
✅ 正确的全图坐标
```

## 技术细节

### 坐标系统说明

1. **patch坐标**：patch在图像上的位置
   - < 50000: 在原图上的坐标
   - > 50000: 在缩放图上的坐标，但CSV保存时已×2映射回原图

2. **检测框坐标**：检测框在patch内的位置
   - 始终是相对于640x640的patch
   - 需要考虑patch对应原图的实际尺寸

3. **全图坐标**：检测框在原始图像上的位置
   - 公式：`boxes_global = (boxes_patch × scale_factor) + patch_offset`

### 关键公式

```python
# 通用坐标转换公式
boxes_global = (boxes_patch × scale_factor) + patch_offset

# 示例：
# scale_factor = 2.0
# patch_offset = (1280, 0)
# boxes_patch = (100, 150, 200, 250)
# 
# 计算：
# boxes_scaled = (100, 150, 200, 250) × 2.0 = (200, 300, 400, 500)
# boxes_global = (200, 300, 400, 500) + (1280, 0, 1280, 0)
#              = (1480, 300, 1680, 500)
```

## 相关文件

- `inferenceTool/predict_batch_torchscript.py` - 修复的推理脚本
- `auto_process_package/auto_process_monitor.py` - 切片和坐标生成
- `sliceTool/convert_wsi40x_to_20x.py` - 图像缩放工具
- `config.yaml` - 配置文件（`downscale.threshold: 50000`）
- `test_coordinate_mapping.py` - 测试脚本
- `COORDINATE_MAPPING_FLOW.md` - 详细流程文档

## 修复日期

2025-12-11

## 修复人员

AI Assistant (Claude)

## 测试命令

```bash
# 激活conda环境
conda activate detr

# 运行坐标映射测试
cd /home/ubuntu/lsn/project_new/RT-DETR-main/A_sclie2inference/DataSlice2Inference_main
python test_coordinate_mapping.py

# 完整服务测试
python service_main.py
```

## 总结

此次修复解决了图像自动缩放场景下检测框坐标映射不正确的问题。通过从CSV注释行提取缩放系数，并在坐标转换时应用该系数，确保了两种情况下的坐标都能正确映射回原图。

关键改进：
1. ✅ 从CSV注释提取scale_factor
2. ✅ 坐标转换时先缩放再加偏移
3. ✅ 所有测试用例通过
4. ✅ 向后兼容（scale_factor默认1.0）

