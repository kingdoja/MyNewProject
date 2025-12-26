# Auto Process Monitor 使用说明

## 目录结构概览

```
DataSlice2Inference/     监听+切片+推理根目录
├── auto_process_package/
│   ├── auto_process_monitor.py  ← 自动监控脚本
│   └── README.md  ← 使用说明
├── inferenceTool/
│   ├── predict_batch_torchscript.py  批量推理脚本
│   └── （批量推理辅助脚本）
├── models/
│   └── （TorchScript / ONNX 等权重文件）
└── sliceTool/
    └── （WSI 切片与预处理脚本）

DataWSI/                       待处理全图存放处
DataPatches/                   原始 patch 输出根目录
DataPatchesKeep/               过滤后保留 patch
DataPatchesTrash/              判定为空白的 patch
DataPatchesInference/          推理结果（JSON/可视化/统计）

```

运行 `auto_process_monitor.py` 会在 `DataSlice2Inference/` 平级目录下自动维护以下产物：

- `DataPatches/` 原始 patch
- `DataPatchesKeep/` 保留 patch
- `DataPatchesTrash/` 空白 patch
- `DataPatchesInference/` 推理结果（JSON 与可视化）

## 功能概览

1. 监听 `watch-dir`（如 `DataWSI/`）中新到的全图文件
2. 使用 `sliceTool` 自动切片、生成坐标 CSV
3. 进行背景分析与空白 patch 过滤
4. 调用 `inferenceTool/predict_batch_torchscript.py` 批量推理
5. 输出可追踪的统计信息与结果

## 环境依赖

- Python 3.8+
- 安装依赖：`pip install watchdog pillow numpy tqdm torch torchvision`
- GPU 推理需安装对应版本的 CUDA/cuDNN 和 PyTorch

## 快速开始

```bash
cd /home/ubuntu/lsn/project_new/RT-DETR-main/DataSlice2Inference/auto_process_package
python auto_process_monitor.py \
  --watch-dir /home/ubuntu/lsn/project_new/RT-DETR-main/DataWSI \
  --output-dir /home/ubuntu/lsn/project_new/RT-DETR-main/DataPatchesInference \
  --model /home/ubuntu/lsn/project_new/RT-DETR-main/DataSlice2Inference/models/rtdetr_torchscript_cuda.pt \
  --threshold 0.5 \
  --process-existing \
#   --no-visualization
```

- 将全图放入 `watch-dir` 即可触发自动处理
- `--process-existing` 会先处理目录内已有文件，可按需移除

## 常用参数

| 参数 | 说明 | 默认值 |
| --- | --- | --- |
| `--watch-dir` | 监听的全图目录 | `DataWSI` |
| `--output-dir` | 推理结果输出根目录 | `DataPatchesInference` |
| `--model` | TorchScript 模型路径 | `DataSlice2Inference/models/...pt` |
| `--patch-size` | 切片尺寸（方形） | `640` |
| `--threshold` | 置信度阈值 | `0.5` |
| `--bg-rgb`, `--tolerance`, `--bg-ratio`, `--std-thresh` | 空白过滤参数 | 见脚本注释 |
| `--disable-auto-bg` | 关闭自动背景分析 | 默认开启 |
| `--no-visualization` | 不保存带框图片，仅输出 JSON | 默认保存 |

详情可查看脚本内注释，每个步骤（切片、过滤、推理）都有日志提示。

## 日志与状态

- `DataPatchesInference/<任务名>/processing_status.json` 记录已处理文件的哈希
- 控制台实时输出各阶段进度、统计信息与错误提示

## 常见问题

1. **未生成 patch**：确认全图格式在 `IMAGE_EXTENSIONS` 列表中，或手动添加扩展名。
2. **watchdog 未安装**：执行 `pip install watchdog`。
3. **推理阶段报错**：检查模型路径是否正确、GPU 是否可用、`predict_batch_torchscript.py` 参数是否需要补充 `--coordinates-csv`。

如需自定义流程，可直接修改同目录下的 `auto_process_monitor.py`。保存后重新运行脚本即可。***

