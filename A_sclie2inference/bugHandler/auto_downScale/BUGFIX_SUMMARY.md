# 坐标映射Bug修复总结

## ✅ 修复完成

**日期**: 2025-12-11  
**状态**: 已完成并通过所有测试

---

## 🔴 问题描述

当图像宽或高 > 50000像素时，系统自动缩放到50%，但检测框坐标映射回原图时**只加了偏移，没有乘以缩放系数**，导致坐标偏小50%。

### 具体表现

```
原图: 100000 x 80000
缩放: 50000 x 40000 (50%)
patch在原图: (1280, 0)
检测框(patch内): (100, 150, 200, 250)

❌ 修复前: (100, 150, 200, 250) + (1280, 0, ...) = (1380, 150, ...) 错误！
✅ 修复后: (100, 150, 200, 250) × 2.0 + (1280, 0, ...) = (1480, 300, ...) 正确！
```

---

## ✅ 修复方案

### 修改文件

`inferenceTool/predict_batch_torchscript.py`

### 核心改动

1. **提取缩放系数**: 从CSV注释行读取 `scale_factor`
2. **应用缩放**: 坐标转换公式改为 `boxes × scale_factor + offset`
3. **传递参数**: 更新所有调用处传入 `scale_factor`

### 关键代码

```python
# 修改前
def convert_to_global_coordinates(boxes, patch_offset):
    return boxes + offset_tensor  # ❌

# 修改后
def convert_to_global_coordinates(boxes, patch_offset, scale_factor=1.0):
    boxes_scaled = boxes * scale_factor  # ✅ 先缩放
    return boxes_scaled + offset_tensor   # ✅ 再加偏移
```

---

## ✅ 测试结果

```bash
🎉 所有测试通过！

测试 1: 图像 < 50000 (无缩放)     ✅ 通过
测试 2: 图像 > 50000 (50%缩放)    ✅ 通过
测试 3: 无缩放，多个检测框         ✅ 通过
测试 4: 2倍缩放，多个检测框        ✅ 通过
```

---

## ✅ 修复效果对比

| 场景 | 修复前 | 修复后 |
|------|-------|-------|
| 图像 < 50000 | ✅ 正确 | ✅ 正确 |
| 图像 > 50000 | ❌ 坐标偏小50% | ✅ 正确 |

---

## 📋 修改清单

| 函数/位置 | 修改内容 | 行号 |
|----------|---------|------|
| `load_patch_coordinates` | 返回 `(coordinates, scale_factor)` | 301-348 |
| `convert_to_global_coordinates` | 增加 `scale_factor` 参数，先缩放再加偏移 | 351-388 |
| `predict_single_patch` | 增加 `scale_factor` 参数 | 618-627 |
| `predict_batch_patches` | 增加 `scale_factor` 参数 | 488-497 |
| `main` 函数调用处 | 传入 `scale_factor` | 724, 812 |

---

## 🔧 使用方法

### 自动工作

修复后无需任何配置，系统自动处理：

```python
# 场景1: 图像 < 50000
scale_factor = 1.0  # 自动检测
坐标 = boxes × 1.0 + offset  # 正常映射

# 场景2: 图像 > 50000  
scale_factor = 2.0  # 从CSV注释自动提取
坐标 = boxes × 2.0 + offset  # 正确映射
```

### 测试验证

```bash
cd /home/ubuntu/lsn/project_new/RT-DETR-main/A_sclie2inference/DataSlice2Inference_main
conda activate detr
python test_coordinate_mapping.py
```

---

## 📄 相关文档

- `BUGFIX_COORDINATE_MAPPING.md` - 详细修复报告
- `COORDINATE_MAPPING_FLOW.md` - 坐标映射流程详解
- `test_coordinate_mapping.py` - 测试脚本

---

## ✅ 结论

**Bug已完全修复**，系统现在可以正确处理任意尺寸图像的坐标映射：

- ✅ 图像 < 50000: 直接映射
- ✅ 图像 > 50000: 先缩放×2再加偏移
- ✅ 向后兼容：scale_factor默认1.0
- ✅ 所有测试通过

