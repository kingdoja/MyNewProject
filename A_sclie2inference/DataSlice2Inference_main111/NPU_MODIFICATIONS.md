# 华为910B NPU适配修改总结

本文档列出了为支持华为昇腾910B NPU而进行的所有代码修改。

## 新增文件

1. **`utils/npu_utils.py`** - NPU工具模块
   - `is_npu_available()`: 检查NPU是否可用
   - `resolve_device()`: 统一的设备解析（优先NPU，其次GPU，最后CPU）
   - `get_device_memory_info()`: 获取设备内存信息（支持NPU/GPU）
   - `clear_device_cache()`: 清理设备缓存（支持NPU/GPU）
   - `supports_fp16()`: 检查设备是否支持FP16

2. **`NPU_DEPLOYMENT.md`** - NPU部署指南（详细文档）

3. **`NPU_MODIFICATIONS.md`** - 本文档

## 修改的文件

### 1. 推理脚本

#### `inferenceTool/predict_batch_torchscript.py`
- ✅ `resolve_device()`: 使用`npu_utils.resolve_device()`实现NPU支持
- ✅ `load_torchscript_model()`: FP16支持从`cuda`扩展到`['npu', 'cuda']`
- ✅ `prepare_image()`: FP16数据类型转换支持NPU
- ✅ Batch size自动调整：支持NPU显存检测和调整

#### `inferenceTool/predict_single_torchscript.py`
- ✅ `resolve_device()`: 使用`npu_utils.resolve_device()`实现NPU支持

#### `inferenceTool/predict_single_globalImage.py`
- ✅ `resolve_device()`: 使用`npu_utils.resolve_device()`实现NPU支持

### 2. 工具模块

#### `utils/metrics.py`
- ✅ `update_system_metrics()`: 使用NPU工具模块获取设备内存信息
- ✅ 支持NPU和GPU的内存监控

#### `utils/resources.py`
- ✅ `clear_gpu_cache()`: 重命名为支持NPU，使用`clear_device_cache()`
- ✅ `monitor_gpu_memory()`: 使用NPU工具模块，支持NPU/GPU内存监控
- ✅ `get_system_resources()`: 更新设备信息键名，兼容NPU/GPU

### 3. 配置文件

#### `config.yaml`
- ✅ `model.device`: 更新注释，说明支持`npu`设备
- ✅ `batch_inference.device`: 更新注释，说明支持`npu`设备
- ✅ `batch_inference.use_fp16`: 更新注释，说明NPU/GPU都支持FP16

#### `requirements.txt`
- ✅ 添加torch_npu安装说明和注意事项

## 关键修改点

### 1. 设备检测优先级
```python
# 修改前：auto -> cuda -> cpu
# 修改后：auto -> npu -> cuda -> cpu
```

### 2. FP16支持
```python
# 修改前：仅支持CUDA
if use_fp16 and device.type == 'cuda':

# 修改后：支持NPU和CUDA
if use_fp16 and device.type in ['npu', 'cuda']:
```

### 3. 设备内存管理
```python
# 修改前：仅支持CUDA
torch.cuda.memory_allocated()
torch.cuda.empty_cache()

# 修改后：统一接口支持NPU和CUDA
if device.type == 'npu':
    torch.npu.memory_allocated()
    torch.npu.empty_cache()
elif device.type == 'cuda':
    torch.cuda.memory_allocated()
    torch.cuda.empty_cache()
```

## 兼容性说明

### 向后兼容
- ✅ 所有修改都保持了向后兼容性
- ✅ 如果`torch_npu`未安装，代码会回退到原始逻辑（GPU/CPU）
- ✅ 配置文件中的`device: "auto"`会自动适配可用设备

### 降级支持
- 如果NPU不可用，系统会自动降级到GPU（如果可用）
- 如果GPU也不可用，系统会使用CPU
- 不会因为NPU相关代码导致系统无法运行

## 测试建议

### 1. 环境验证
```python
# 在NPU环境中运行
python -c "import torch; import torch_npu; print(torch.npu.is_available())"
```

### 2. 功能测试
```bash
# 使用NPU进行单张推理测试
python inferenceTool/predict_single_torchscript.py \
  --model models/rtdetr_torchscript_cuda.pt \
  --patch test_patch.png \
  --coordinates-csv test_coordinates.csv \
  --device npu

# 使用auto自动检测（应该优先使用NPU）
python inferenceTool/predict_batch_torchscript.py \
  --model models/rtdetr_torchscript_cuda.pt \
  --patch-dir test_patches \
  --output-dir test_output \
  --device auto
```

### 3. 性能测试
- 对比NPU和CPU的推理速度
- 测试FP16开启前后的性能差异
- 监控NPU显存使用情况

## 注意事项

1. **模型兼容性**: TorchScript模型通常是设备无关的，但建议在NPU环境下测试确认
2. **FP16精度**: 如果遇到精度问题，可以在配置中关闭FP16
3. **Batch Size**: NPU显存通常较大（32GB），可以适当增大batch_size以提升吞吐量
4. **依赖安装**: 必须按照`NPU_DEPLOYMENT.md`中的说明安装CANN和torch_npu

## 后续优化建议

1. **模型转换**: 如果性能不理想，考虑使用CANN的模型转换工具（ATC）将模型转换为OM格式
2. **算子优化**: 检查是否有NPU不支持的算子，考虑替换或优化
3. **性能调优**: 根据实际测试结果调整batch_size、FP16等参数
4. **监控增强**: 添加NPU特定的监控指标（如算力利用率等）

## 相关文档

- [NPU_DEPLOYMENT.md](./NPU_DEPLOYMENT.md) - 详细的部署指南
- [华为昇腾官方文档](https://www.hiascend.com/document) - CANN和NPU的官方文档
