# RT-DETR TorchScript 模型 CUDA 支持说明

## 🔍 问题原因

**TorchScript 不支持 CUDA 的根本原因：**

当使用 `torch.jit.trace()` 导出 TorchScript 模型时，如果**在 CPU 设备上进行 trace**，导出的模型会被"冻结"为 CPU 模型。即使后续使用 `map_location='cuda'` 加载，模型内部的某些操作（如常量、某些算子）仍然被固定在 CPU 上，导致设备不一致错误。

## ✅ 解决方案

**必须在 CUDA 设备上进行 trace**，才能导出支持 CUDA 推理的 TorchScript 模型。

## 📝 正确导出步骤

### 方法1：使用修复后的导出脚本（推荐）

```bash
cd rtdetrv2_pytorch

python tools/export_pt.py \
    --config configs/rtdetrv2/rtdetrv2_r18vd_cancer_detection.yml \
    --resume output/rtdetrv2_r18vd_cancer_detection/best.pth \
    --mode torchscript \
    --input-size 640 \
    --device cuda \  # 关键：指定在CUDA上导出
    --output exported_models/rtdetr_torchscript_cuda.pt
```

### 方法2：使用快速导出脚本

编辑 `快速导出模型.py`：

```python
# 导出模式
EXPORT_MODE = "torchscript"

# 导出设备（关键！）
EXPORT_DEVICE = None  # None=自动检测CUDA, 'cuda'=强制CUDA, 'cpu'=强制CPU
```

然后运行：

```bash
python 快速导出模型.py
```

### 方法3：手动指定设备（如果自动检测失败）

```bash
python tools/export_pt.py \
    --config configs/rtdetrv2/rtdetrv2_r18vd_cancer_detection.yml \
    --resume output/rtdetrv2_r18vd_cancer_detection/best.pth \
    --mode torchscript \
    --input-size 640 \
    --device cuda \
    --output exported_models/rtdetr_torchscript_cuda.pt
```

## 🔧 修复内容

已修复 `rtdetrv2_pytorch/tools/export_pt.py` 中的 `export_torchscript` 函数：

1. ✅ **自动检测 CUDA**：如果 CUDA 可用，自动在 CUDA 设备上导出
2. ✅ **模型移动到设备**：确保模型在 trace 前移动到目标设备
3. ✅ **输入张量在设备上**：示例输入张量在目标设备上创建
4. ✅ **验证导出结果**：导出后验证模型是否正常工作

## 📋 验证导出是否成功

导出时应该看到类似输出：

```
✅ 检测到CUDA，将在CUDA设备上导出模型（支持CUDA推理）
使用EMA模型权重
测试模型推理...
模型测试通过
正在转换为TorchScript格式（设备: cuda）...
验证TorchScript模型...
TorchScript模型验证通过
✅ TorchScript模型已保存到: exported_models/rtdetr_torchscript_cuda.pt
💡 模型已导出为支持 CUDA 推理的格式
```

## 🚀 在 Label Studio ML 后端中使用

### 1. 更新模型路径

确保 `MODEL_PATH` 环境变量指向新导出的 CUDA 模型：

```bash
export MODEL_PATH=/path/to/rtdetr_torchscript_cuda.pt
```

或在 `docker-compose.yml` 中：

```yaml
environment:
  - MODEL_PATH=/models/rtdetr_torchscript_cuda.pt
```

### 2. 确保 CUDA 可用

在 Label Studio ML 后端容器中，确保：
- CUDA 驱动已安装
- PyTorch 支持 CUDA
- 容器有 GPU 访问权限（Docker 需要 `--gpus all`）

### 3. 验证 CUDA 使用

启动后端后，应该看到：

```
使用 CUDA 设备: NVIDIA GeForce RTX 3090
✓ TorchScript模型加载成功
```

而不是：

```
检测到设备不一致（cuda 与 cpu），切换到 CPU 推理并重新加载模型
根据配置/兼容性要求，使用 CPU
```

## ⚠️ 常见问题

### Q1: 导出时提示 CUDA 不可用

**原因**：导出环境没有 CUDA 或 PyTorch 未编译 CUDA 支持

**解决**：
1. 检查 `torch.cuda.is_available()` 是否为 `True`
2. 确保在支持 CUDA 的环境中导出
3. 如果必须在 CPU 上导出，后续只能使用 CPU 推理

### Q2: 导出成功但推理时仍使用 CPU

**原因**：可能是模型加载时的设备设置问题

**解决**：
1. 检查 `model.py` 中的 `_load_model` 方法
2. 确保 `FORCE_CPU_INFERENCE` 环境变量未设置为 `true`
3. 检查容器是否有 GPU 访问权限

### Q3: 设备不一致错误

**原因**：模型在 CPU 上导出，但尝试在 CUDA 上使用

**解决**：
1. 重新在 CUDA 设备上导出模型
2. 或设置 `FORCE_CPU_INFERENCE=true` 强制使用 CPU

## 📚 相关文件

- `rtdetrv2_pytorch/tools/export_pt.py` - 导出脚本（已修复）
- `Z_export_model/快速导出模型.py` - 快速导出脚本（已更新）
- `Z_export_model/model.py` - ML 后端模型加载代码

## 🎯 总结

**关键点：**
1. ✅ TorchScript 模型必须在**目标设备上**进行 trace
2. ✅ 要在 CUDA 上推理，必须在 CUDA 上导出
3. ✅ 导出脚本已修复，会自动检测并使用 CUDA（如果可用）
4. ✅ 导出后验证模型是否正常工作

**下一步：**
1. 使用修复后的脚本重新导出模型
2. 更新 Label Studio ML 后端的模型路径
3. 重启服务并验证 CUDA 使用情况

