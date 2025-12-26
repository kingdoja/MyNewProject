# 华为昇腾910B NPU部署指南

本文档说明如何将RT-DETR服务部署到华为昇腾910B NPU环境。

## 环境准备

### 1. 安装CANN（Compute Architecture for Neural Networks）

CANN是昇腾NPU的底层软件栈，必须首先安装。

```bash
# 从华为昇腾官方下载并安装CANN
# 下载地址: https://www.hiascend.com/software/cann
# 根据您的操作系统版本（如EulerOS, Ubuntu等）选择对应的CANN版本
```

### 2. 安装PyTorch

根据您的系统架构（通常是aarch64），安装适配的PyTorch版本：

```bash
# 对于aarch64架构（华为910B通常是这个架构）
# 下载并安装CPU版本的PyTorch（NPU会通过torch_npu扩展）
wget https://download.pytorch.org/whl/cpu/torch-2.1.0-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl
pip install torch-2.1.0-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

# 同时安装torchvision（如果需要）
pip install torchvision
```

### 3. 安装torch_npu

torch_npu是华为提供的PyTorch扩展插件，用于在PyTorch中支持NPU计算：

```bash
# 从华为昇腾官方获取torch_npu安装包
# 确保torch_npu版本与PyTorch版本匹配
# 例如，对于PyTorch 2.1.0：
pip install torch_npu-2.1.0.post4-cp39-cp39-manylinux_2_17_aarch64.whl

# 注意：torch_npu不能通过pip直接安装，必须从华为官方获取whl文件
```

### 4. 验证安装

安装完成后，验证环境是否正确：

```python
import torch
import torch_npu

# 检查NPU是否可用
print(f"NPU可用: {torch.npu.is_available()}")
print(f"NPU数量: {torch.npu.device_count()}")

# 测试NPU计算
a = torch.randn(3, 4).npu()
b = torch.randn(3, 4).npu()
c = a + b
print(f"NPU计算测试成功: {c.shape}")
```

## 代码修改说明

已完成的代码修改包括：

### 1. 新增NPU工具模块

- **文件**: `utils/npu_utils.py`
- **功能**: 统一管理NPU设备检测、设备解析、内存监控等

### 2. 修改推理脚本

以下文件已更新以支持NPU：

- `inferenceTool/predict_batch_torchscript.py` - 批量推理脚本
- `inferenceTool/predict_single_torchscript.py` - 单张推理脚本
- `inferenceTool/predict_single_globalImage.py` - 全图推理脚本

**主要修改**:
- `resolve_device()` 函数：优先检测NPU，其次GPU，最后CPU
- `load_torchscript_model()` 函数：支持NPU设备的FP16加速
- `prepare_image()` 函数：支持NPU设备的张量转换
- Batch size自动调整：支持根据NPU显存自动调整

### 3. 修改工具模块

- `utils/metrics.py` - 指标收集：支持NPU内存监控
- `utils/resources.py` - 资源管理：支持NPU缓存清理和内存监控

### 4. 配置文件更新

- `config.yaml` - 设备配置已更新，支持 `npu` 和 `auto`（自动优先使用NPU）

## 使用方法

### 1. 配置设备

在 `config.yaml` 中设置设备：

```yaml
model:
  device: "auto"  # 自动优先使用NPU，其次GPU，最后CPU
  # 或者显式指定: device: "npu" 或 device: "npu:0"

batch_inference:
  device: "auto"  # 同上
  use_fp16: true  # NPU支持FP16，可启用以提升性能
```

### 2. 运行服务

#### 方式1: 使用配置文件运行

```bash
cd /home/ubuntu/lsn/project_new/RT-DETR-main/A_sclie2inference/DataSlice2Inference_main
python service_main.py
```

#### 方式2: 命令行指定设备

```bash
# 批量推理
python inferenceTool/predict_batch_torchscript.py \
  --model models/rtdetr_torchscript_cuda.pt \
  --patch-dir /path/to/patches \
  --output-dir /path/to/output \
  --device npu \
  --threshold 0.5

# 单张推理
python inferenceTool/predict_single_torchscript.py \
  --model models/rtdetr_torchscript_cuda.pt \
  --patch /path/to/patch.png \
  --coordinates-csv /path/to/coordinates.csv \
  --device npu \
  --threshold 0.5
```

### 3. 验证NPU使用

运行服务后，查看日志确认是否使用了NPU：

```
⚡ 检测到可用 NPU，使用 NPU 进行推理
=== 加载 TorchScript 模型 ===
模型路径: models/rtdetr_torchscript_cuda.pt
设备: npu
✓ 模型加载完成
```

## 注意事项

### 1. 模型兼容性

- TorchScript模型（.pt文件）通常是设备无关的，可以在NPU上直接使用
- 如果遇到模型兼容性问题，可能需要在NPU环境下重新导出模型

### 2. FP16精度

- NPU支持FP16混合精度，可显著提升推理速度
- 如果遇到精度问题，可以在配置文件中设置 `use_fp16: false`

### 3. 批量大小调整

- 华为910B NPU通常有32GB显存，默认batch_size会根据显存自动调整
- 如果遇到显存不足，可以在代码中手动调整batch_size

### 4. 性能优化

- NPU的推理速度通常比GPU稍慢，但功耗更低
- 可以通过启用FP16和调整batch_size来优化性能

## 故障排查

### 问题1: torch_npu导入失败

**错误信息**: `ImportError: No module named 'torch_npu'`

**解决方案**:
1. 确认已安装CANN
2. 确认已安装torch_npu：`pip list | grep torch_npu`
3. 如果未安装，从华为官方获取并安装对应的torch_npu whl文件

### 问题2: NPU不可用

**错误信息**: `NPU可用: False`

**解决方案**:
1. 检查CANN是否正确安装：`npu-smi info`
2. 检查NPU驱动是否正确加载
3. 确认用户是否有访问NPU的权限

### 问题3: 模型加载失败

**错误信息**: 模型加载到NPU时出错

**解决方案**:
1. 确认模型文件路径正确
2. 尝试在NPU环境下重新导出TorchScript模型
3. 检查模型是否包含NPU不支持的算子

### 问题4: 推理速度慢

**可能原因**:
1. 未启用FP16：在配置中设置 `use_fp16: true`
2. Batch size过小：增加batch_size
3. 数据预处理开销：检查数据加载和预处理是否在CPU上执行

## 性能参考

在华为910B NPU上的典型性能：

- **单张640x640图像推理时间**: ~10-20ms（FP16）
- **批量推理吞吐量**: ~50-100 images/s（FP16, batch_size=32）
- **显存占用**: ~2-4GB（FP16, batch_size=32）

注意：实际性能取决于模型大小、输入尺寸、batch_size等因素。

## 技术支持

如遇到问题，请检查：
1. CANN版本是否匹配
2. PyTorch和torch_npu版本是否匹配
3. 模型格式是否正确
4. 日志中的错误信息

更多信息请参考：
- 华为昇腾官方文档: https://www.hiascend.com/document
- CANN开发指南: https://www.hiascend.com/document/detail/zh/CANNCommunityEdition
