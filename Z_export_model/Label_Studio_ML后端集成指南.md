# Label Studio ML 后端集成指南

## 📋 模型选择建议

根据你的导出模型文件，**推荐使用 `best_full.pt`**：

### ✅ 推荐：best_full.pt (full_model 模式)
- **优点**：
  - 包含完整模型结构和权重，可直接加载
  - 使用简单：`torch.load()` 即可
  - 不需要额外的配置文件或模型定义代码
  - 适合 ML 后端快速集成
- **文件大小**：78MB
- **加载方式**：`model = torch.load('best_full.pt')`

### ⚡ 备选：best_torchscript.pt (TorchScript 模式)
- **优点**：
  - 性能更好，适合生产环境
  - 已优化的推理速度
- **缺点**：
  - 加载方式不同：`torch.jit.load()`
  - 如果模型有动态控制流可能需要额外处理
- **文件大小**：85MB
- **加载方式**：`model = torch.jit.load('best_torchscript.pt')`

### ❌ 不推荐：best_weights.pt (state_dict 模式)
- **缺点**：
  - 只包含权重，需要模型定义代码和配置文件
  - 在 ML 后端中集成较复杂
- **文件大小**：77MB

---

## 🚀 快速开始

### 步骤 1：安装 Label Studio ML 后端

```bash
pip install label-studio-ml
```

### 步骤 2：创建 ML 后端项目

```bash
label-studio-ml create my_rtdetr_backend
cd my_rtdetr_backend
```

### 步骤 3：配置模型

1. 将 `best_full.pt` 复制到 ML 后端目录，或修改代码中的模型路径
2. 将 `label_studio_ml_backend_example.py` 的内容复制到 `model.py`

```bash
# 复制模型文件（可选，如果放在其他位置需要修改路径）
cp /home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/exported_models/best_full.pt ./models/

# 复制示例代码
cp /home/ubuntu/lsn/project_new/RT-DETR-main/Z_export_model/label_studio_ml_backend_example.py ./model.py
```

3. 修改 `model.py` 中的模型路径（如果需要）：

```python
# 在 RTDETRModel.__init__ 中修改
model_path = '/path/to/your/best_full.pt'
```

### 步骤 4：安装依赖

在 `my_rtdetr_backend` 目录下创建或更新 `requirements.txt`：

```txt
torch>=1.9.0
torchvision>=0.10.0
Pillow>=8.0.0
numpy>=1.19.0
label-studio-ml>=0.0.40
```

安装依赖：

```bash
pip install -r requirements.txt
```

### 步骤 5：启动 ML 后端服务

```bash
label-studio-ml start my_rtdetr_backend
```

服务将在 `http://localhost:9090` 启动。

### 步骤 6：在 Label Studio 中连接

1. 打开 Label Studio 项目设置
2. 进入 "模型" 标签页
3. 点击 "连接模型"
4. 填写：
   - **模型名称**：RT-DETR Detection
   - **后端 URL**：`http://localhost:9090`
5. 保存

### 步骤 7：使用预测功能

1. 在数据管理页面选择任务
2. 点击 "获取预测" 按钮
3. 模型将自动标注检测结果
4. 可以在此基础上进行微调和确认

---

## 📝 代码说明

### 主要功能

1. **`load_model()`**：加载 `best_full.pt` 模型
2. **`predict()`**：对输入图像进行目标检测，返回 Label Studio 格式的预测结果
3. **`fit()`**：可选，用于使用新标注数据微调模型

### 输出格式

模型输出需要转换为 Label Studio 的标注格式：

```json
{
  "result": [
    {
      "id": "result_0",
      "type": "rectanglelabels",
      "value": {
        "x": 10.5,
        "y": 20.3,
        "width": 30.2,
        "height": 25.1,
        "rectanglelabels": ["class_0"]
      },
      "score": 0.95,
      "from_name": "label",
      "to_name": "image"
    }
  ],
  "score": 0.95
}
```

### 注意事项

1. **坐标转换**：需要将模型输出的坐标转换为 Label Studio 的百分比格式
2. **置信度阈值**：代码中默认使用 0.5，可根据需要调整
3. **类别标签**：需要根据你的实际类别修改 `rectanglelabels`
4. **图像预处理**：确保预处理方式与训练时一致

---

## 🔧 自定义配置

### 修改置信度阈值

在 `predict()` 方法中：

```python
if score < 0.5:  # 修改这里的阈值
    continue
```

### 修改输入尺寸

在 `__init__()` 方法中：

```python
self.input_size = 640  # 修改为你的模型输入尺寸
```

### 修改类别标签

在 `predict()` 方法中，将：

```python
'rectanglelabels': [f'class_{int(label)}']
```

改为你的实际类别名称，例如：

```python
# 假设你有类别映射
class_names = ['cell', 'nucleus', 'background']
'rectanglelabels': [class_names[int(label)]]
```

---

## 🐛 故障排除

### 问题 1：模型加载失败

**错误**：`FileNotFoundError: 模型文件不存在`

**解决**：
- 检查模型路径是否正确
- 使用绝对路径
- 确认文件权限

### 问题 2：CUDA 内存不足

**解决**：
- 在代码中强制使用 CPU：`self.device = torch.device('cpu')`
- 或减小 batch size

### 问题 3：输出格式不匹配

**解决**：
- 检查模型实际输出格式
- 根据实际情况调整 `predict()` 方法中的解析逻辑
- 可以先用测试脚本验证模型输出

### 问题 4：预测结果为空

**可能原因**：
- 置信度阈值设置过高
- 图像预处理方式不匹配
- 模型输出格式解析错误

**解决**：
- 降低置信度阈值
- 检查预处理方式
- 打印模型输出进行调试

---

## 📚 参考资源

- [Label Studio ML 后端文档](https://github.com/HumanSignal/label-studio-ml-backend)
- [Label Studio 官方文档](https://labelstud.io/)
- RT-DETR 模型导出说明：`导出模型说明.md`

---

## 💡 提示

1. **测试模型**：在集成到 ML 后端之前，先用测试脚本验证模型是否能正常加载和推理
2. **性能优化**：如果推理速度慢，可以考虑使用 `best_torchscript.pt` 或进行模型量化
3. **在线训练**：如果需要在线训练功能，需要实现 `fit()` 方法，并确保有足够的计算资源

---

**最后更新**：根据你的实际模型输出格式，可能需要调整代码中的解析逻辑。




