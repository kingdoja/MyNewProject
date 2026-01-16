# RT-DETR 自动切片与推理服务

一个基于 RT-DETR 模型的自动图像切片、过滤和批量推理服务系统。该系统能够自动监听目录中的全图文件，进行智能切片、背景过滤和批量推理，并生成可视化的检测结果。

## 📋 目录

- [功能特性](#功能特性)
- [系统要求](#系统要求)
- [快速开始](#快速开始)
- [安装部署](#安装部署)
- [配置说明](#配置说明)
- [使用方法](#使用方法)
- [目录结构](#目录结构)
- [API 接口](#api-接口)
- [故障排查](#故障排查)
- [常见问题](#常见问题)
- [开发说明](#开发说明)

## ✨ 功能特性

### 核心功能

- ✅ **自动文件监听**：使用 watchdog 实时监听指定目录，自动检测新增的全图文件
- ✅ **智能图像切片**：自动将大图切分为固定尺寸的 patch（默认 640x640）
- ✅ **背景分析与过滤**：自动分析背景色，过滤空白 patch，提高推理效率
- ✅ **批量推理**：使用 TorchScript 模型进行批量目标检测推理
- ✅ **结果可视化**：自动生成带检测框的可视化图片和 JSON 结果
- ✅ **自动缩放**：支持超大图像的自动缩放处理（超过阈值自动缩小到 50%）
- ✅ **坐标映射**：自动处理缩放后的坐标映射回原始图像坐标系

### 服务特性

- ✅ **健康检查**：提供 HTTP 健康检查和指标监控接口
- ✅ **优雅关闭**：支持信号处理和优雅关闭机制
- ✅ **日志记录**：完整的日志记录系统，支持日志轮转
- ✅ **并发处理**：支持多文件并行处理，提高处理效率
- ✅ **状态追踪**：记录已处理文件的哈希值，避免重复处理
- ✅ **MinIO 集成**：支持自动上传结果到 MinIO 对象存储
- ✅ **Systemd 支持**：提供 systemd 服务配置，支持开机自启

## 🔧 系统要求

### 硬件要求

- **CPU**: 支持 x86_64 架构
- **内存**: 建议 16GB 以上（处理大图时建议 32GB+）
- **GPU**: 可选，支持 CUDA 的 NVIDIA GPU（推荐用于加速推理）
- **磁盘空间**: 建议至少 100GB 可用空间（取决于处理图像数量和大小）

### 软件要求

- **操作系统**: Linux (Ubuntu 18.04+ 推荐)
- **Python**: 3.8 或更高版本
- **CUDA**: 11.6+ (如果使用 GPU 推理)
- **cuDNN**: 8.0+ (如果使用 GPU 推理)

### Python 依赖

主要依赖包（详见 `requirements.txt`）：

- `watchdog>=3.0.0` - 文件监听
- `pillow>=10.0.0` - 图像处理
- `torch>=2.0.0` - PyTorch 框架
- `torchvision>=0.15.0` - 计算机视觉工具
- `tqdm>=4.65.0` - 进度条显示
- `pyyaml>=6.0` - YAML 配置解析
- `psutil>=5.9.0` - 系统资源监控
- `flask>=2.3.0` - Web 服务框架
- `minio>=7.2.0` - MinIO 客户端

## 🚀 快速开始

### 1. 克隆或进入项目目录

```bash
cd /home/ubuntu/lsn/project_new/RT-DETR-main/A_sclie2inference/DataSlice2Inference_main
```

### 2. 安装依赖

```bash
# 安装基础依赖
pip install -r requirements.txt

# 如果使用 GPU，安装 PyTorch CUDA 版本
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

### 3. 配置服务

编辑 `config.yaml` 文件，设置必要的路径和参数：

```yaml
paths:
  watch_dir: "/path/to/your/input/images"      # 监听目录
  output_dir: "/path/to/output"                 # 输出目录
  model_path: "/path/to/model.pt"               # 模型文件路径
```

### 4. 启动服务

#### 方式1：直接运行（测试用）

```bash
python3 service_main.py --config config.yaml
# 或直接运行（使用默认 config.yaml）
python3 service_main.py
```

#### 方式2：使用服务管理脚本（推荐）

```bash
# 安装 systemd 服务
./service_manager.sh install

# 启动服务
./service_manager.sh start

# 查看状态
./service_manager.sh status

# 查看日志
./service_manager.sh logs -f
```

### 5. 测试服务

将图像文件放入监听目录，服务会自动开始处理。

## 📦 安装部署

### 完整安装步骤

1. **准备环境**

```bash
# 激活 conda 环境（如果使用）
conda activate detr

# 或创建新的虚拟环境
python3 -m venv venv
source venv/bin/activate
```

2. **安装依赖**

```bash
cd /home/ubuntu/lsn/project_new/RT-DETR-main/A_sclie2inference/DataSlice2Inference_main
pip install -r requirements.txt
```

3. **准备模型文件**

将训练好的 TorchScript 模型文件放置在 `models/` 目录下，或更新 `config.yaml` 中的模型路径。

4. **配置服务**

编辑 `config.yaml`，根据实际环境配置路径和参数。

5. **安装为系统服务（可选）**

```bash
./service_manager.sh install
sudo systemctl enable rtdetr-processor
sudo systemctl start rtdetr-processor
```

## ⚙️ 配置说明

配置文件 `config.yaml` 支持环境变量替换，使用 `${VAR_NAME:default_value}` 格式。

### 路径配置

```yaml
paths:
  watch_dir: "${WATCH_DIR:/path/to/watch}"      # 监听目录（支持环境变量）
  output_dir: "${OUTPUT_DIR:/path/to/output}"   # 输出目录
  model_path: "${MODEL_PATH:/path/to/model.pt}" # 模型路径
  log_dir: "${LOG_DIR:/path/to/logs}"           # 日志目录
```

### 处理参数

```yaml
processing:
  patch_size: 640                    # 切片尺寸（像素）
  threshold: 0.5                     # 置信度阈值（0-1）
  file_wait_timeout: 15              # 文件写入等待超时（秒）
  max_concurrent_tasks: 2            # 最大并发任务数
  save_visualization: true            # 是否保存可视化图片
  process_existing: true              # 启动时是否处理已存在文件
```

### 图像缩放配置

```yaml
downscale:
  enabled: true                      # 是否启用自动缩放
  threshold: 50000                   # 触发缩放的像素阈值（宽或高）
  quality: 95                        # 缩放后 JPEG 质量（1-100）
```

**说明**：
- 当图像宽度或高度超过 `threshold` 时，自动缩小到 50%
- 缩小后的图像保存在 `DataWSI_downscaled/` 目录
- 输出的坐标会自动映射回原始图像坐标系（乘以 2 倍）

### 过滤参数

```yaml
filtering:
  bg_rgb: [238, 235, 235]           # 背景色 RGB（auto_bg=false 时使用）
  tolerance: 30                      # 颜色容差
  bg_ratio: 0.9                     # 背景比例阈值
  std_thresh: 10.0                  # 灰度标准差阈值
  auto_bg: true                     # 是否自动分析背景（推荐开启）
  bg_analysis_limit: 10             # 自动分析采样 patch 数量
```

### 批量推理配置

```yaml
batch_inference:
  device: "auto"                    # auto/cpu/cuda/cuda:0
  use_fp16: false                   # 是否启用 FP16（仅 GPU）
  threshold: 0.5                    # 置信度阈值
  batch_size: null                  # 批量大小（null 表示自动调整）
  pattern: "*.png"                  # 图像文件匹配模式
  save_visualization: true          # 是否保存可视化
```

### 监控配置

```yaml
monitoring:
  enabled: true
  stats_interval: 300               # 统计信息输出间隔（秒）
  health_check_port: 8081          # 健康检查端口（0 表示禁用）
```

### MinIO 配置

```yaml
minio:
  enabled: true                     # 是否启用 MinIO 上传
  endpoint: "192.168.150.157:9000" # MinIO 服务地址
  access_key: "minioadmin"          # 访问密钥
  secret_key: "minioadmin"          # 密钥
  bucket: "heathycare"              # 目标 bucket
  secure: false                     # 是否使用 HTTPS
  base_path: "source"               # 上传前缀
  upload_original_image: true       # 是否上传原始全图
  upload_json: true                 # 是否上传推理 JSON
```

### 日志配置

```yaml
logging:
  level: "INFO"                     # DEBUG/INFO/WARNING/ERROR
  console: true                     # 是否输出到控制台
  file_max_bytes: 524288000         # 单个日志文件最大大小（500MB）
  file_backup_count: 20             # 保留的备份日志文件数量
```

## 📖 使用方法

### 服务管理

#### 使用服务管理脚本

```bash
./service_manager.sh <command>
```

可用命令：

- `install` - 安装 systemd 服务
- `start` - 启动服务
- `stop` - 停止服务
- `restart` - 重启服务
- `status` - 查看服务状态
- `logs` - 查看日志（最近 100 行）
- `logs -f` - 实时查看日志
- `start-direct` - 直接启动（不使用 systemd）

#### 使用 systemd 命令

```bash
# 启动服务
sudo systemctl start rtdetr-processor

# 停止服务
sudo systemctl stop rtdetr-processor

# 重启服务
sudo systemctl restart rtdetr-processor

# 查看状态
sudo systemctl status rtdetr-processor

# 查看日志
sudo journalctl -u rtdetr-processor -n 100
sudo journalctl -u rtdetr-processor -f  # 实时日志

# 开机自启
sudo systemctl enable rtdetr-processor

# 禁用开机自启
sudo systemctl disable rtdetr-processor
```

### 独立运行自动处理

如果需要独立运行自动处理模块（不启动 Flask 服务）：

```bash
cd auto_process_package
python run_auto_process.py --config ../config.yaml
```

### 批量推理

直接使用批量推理脚本：

```bash
python inferenceTool/predict_batch_torchscript.py \
    --patch-dir /path/to/patches \
    --output-dir /path/to/output \
    --model /path/to/model.pt \
    --threshold 0.5 \
    --batch-size 16
```

### 单张图像推理

```bash
# 单张全图推理
python inferenceTool/predict_single_globalImage.py \
    --image /path/to/image.jpg \
    --model /path/to/model.pt \
    --output /path/to/output

# 单张 TorchScript 推理
python inferenceTool/predict_single_torchscript.py \
    --image /path/to/image.jpg \
    --model /path/to/model.pt \
    --output /path/to/output
```

### 标注坐标转换

将全图标注转换为 patch 级别的 COCO 格式：

```bash
python annotationConverter/convert_global_to_patch.py \
    --annotation-file /path/to/标记.json \
    --patch-dir /path/to/patch_dir \
    --output-dir /path/to/output \
    --wsi-image /path/to/wsi.jpeg
```

## 📁 目录结构

```
A_sclie2inference/
├── DataSlice2Inference_main/                # 主服务与自动处理代码
│   ├── auto_process_package/                # 监听/切片/过滤/预测核心逻辑
│   │   ├── auto_process_monitor.py             # watchdog 监听 + 切片/过滤/批量预测调度
│   │   ├── run_auto_process.py                 # 独立运行入口
│   │   └── README.md                           # 子模块说明
│   ├── inferenceTool/                       # 推理脚本
│   │   ├── predict_batch_torchscript.py        # 批量 TorchScript 预测
│   │   ├── predict_single_globalImage.py       # 单张全图预测
│   │   └── predict_single_torchscript.py        # 单张 TorchScript 预测
│   ├── utils/                               # 通用工具
│   │   ├── __init__.py                         # 包初始化
│   │   ├── clear_processed_record.py           # 清除 processing_status 记录
│   │   ├── config.py                           # 配置加载与解析
│   │   ├── logger.py                           # 日志封装
│   │   ├── metrics.py                          # 指标收集
│   │   ├── monitor.py                          # 健康检查/Flask 服务
│   │   ├── notification.py                     # 通知相关
│   │   ├── progress.py                         # 进度显示工具
│   │   ├── queue.py                            # 队列封装
│   │   ├── resources.py                        # 资源检测
│   │   ├── retry.py                            # 重试装饰器
│   │   ├── security.py                         # 安全校验
│   │   ├── shutdown.py                         # 优雅关闭工具
│   │   ├── task_queue.py                       # 任务队列
│   │   ├── validation.py                       # 校验与哈希工具
│   │   └── minio_helper.py                     # MinIO 上传工具
│   ├── models/                              # 模型文件目录
│   │   └── rtdetr_torchscript_cuda.pt          # TorchScript 模型文件
│   ├── logs/                                # 运行日志目录
│   │   ├── auto_process_*.log                  # 自动处理日志
│   │   └── rtdetr_service_*.log                # 服务日志
│   ├── global_visualizer/                   # 全局可视化（全图绘制标注）
│   │   └── visualize_global_annotations.py     # 在全图上绘制检测结果
│   ├── sliceTool/                           # 切片工具
│   │   ├── 1step1_pre_process_jpeg.py          # JPEG 预处理
│   │   ├── 2analyze_background.py              # 背景分析
│   │   ├── 3filter_patch.py                    # Patch 过滤
│   │   └── convert_wsi40x_to_20x.py            # 40X 转 20X 辅助
│   ├── annotationConverter/                 # 标注转换工具
│   │   ├── convert_global_to_patch.py          # 全图标注转 patch 标注
│   │   └── README.md                           # 转换工具说明
│   ├── systemd/
│   │   └── rtdetr-processor.service            # systemd 服务模板
│   ├── service_main.py                      # 主入口（Flask 健康检查 + 监听器）
│   ├── service_manager.sh                   # 管理脚本（安装/启动/停止/日志）
│   ├── config.yaml                          # 配置文件
│   ├── requirements.txt                     # Python 依赖
│   ├── README.md                            # 本文件
│   └── SERVICE_README.md                    # 服务使用说明
├── DataWSI/                                 # 输入全图目录（监听目录）
├── DataWSI_downscaled/                      # 自动缩放后的图像目录
├── DataPatchesInference/                    # 推理输出/可视化/状态记录
│   ├── processing_status.json               # 已处理文件哈希记录
│   └── <运行生成的输出目录>                  # 批量预测结果与可视化
└── DataSliceTools_disused/                  # 旧的/备用切片工具脚本
```

## 🌐 API 接口

如果启用了健康检查（`config.yaml` 中 `monitoring.health_check_port > 0`），可以通过 HTTP 接口检查服务状态。

### 健康检查

```bash
curl http://localhost:8081/health
```

响应示例：
```json
{
  "status": "healthy",
  "uptime_seconds": 3600,
  "timestamp": "2024-01-01T12:00:00"
}
```

### 获取指标

```bash
curl http://localhost:8081/metrics
```

响应示例：
```json
{
  "files": {
    "total_processed": 10,
    "successful": 9,
    "failed": 1
  },
  "patches": {
    "created": 1000,
    "kept": 800
  },
  "detections": {
    "total": 150
  }
}
```

### 获取统计信息

```bash
curl http://localhost:8081/stats
```

响应示例：
```json
{
  "uptime_seconds": 3600,
  "files": {
    "total": 10,
    "successful": 9,
    "failed": 1
  }
}
```

## 🔍 故障排查

### 服务无法启动

1. **检查配置文件**

```bash
python service_main.py --config config.yaml
```

2. **检查日志**

```bash
./service_manager.sh logs
# 或
sudo journalctl -u rtdetr-processor -n 100
```

3. **检查路径和权限**

```bash
# 确保监听目录存在且有读权限
ls -la /path/to/watch_dir

# 确保输出目录存在且有写权限
mkdir -p /path/to/output_dir
chmod 755 /path/to/output_dir

# 确保模型文件存在
ls -la /path/to/model.pt
```

### 服务频繁重启

1. **查看详细日志**

```bash
sudo journalctl -u rtdetr-processor -n 200 --no-pager
```

2. **检查资源限制**

```bash
# 检查内存使用
free -h

# 检查磁盘空间
df -h

# 检查 GPU 使用（如果使用 GPU）
nvidia-smi
```

3. **检查模型文件**

```bash
# 确保模型文件完整
file /path/to/model.pt
```

### 文件处理失败

1. **检查文件格式**

确保文件格式在支持列表中：`.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`

2. **检查文件完整性**

确保文件已完全写入，避免处理正在传输的文件。

3. **查看处理日志**

```bash
tail -f logs/auto_processor_*.log
```

### GPU 相关问题

1. **检查 CUDA 可用性**

```python
python -c "import torch; print(torch.cuda.is_available())"
```

2. **检查 GPU 内存**

```bash
nvidia-smi
```

3. **调整批量大小**

如果 GPU 内存不足，在 `config.yaml` 中减小 `batch_inference.batch_size`。

### MinIO 上传失败

1. **检查网络连接**

```bash
ping <minio_endpoint>
```

2. **检查凭证**

确保 `config.yaml` 中的 `minio.access_key` 和 `minio.secret_key` 正确。

3. **检查 bucket 权限**

确保 MinIO 用户有对应 bucket 的读写权限。

## ❓ 常见问题

### Q1: 如何处理超大图像？

A: 启用自动缩放功能（`downscale.enabled: true`），系统会自动将超过阈值的图像缩小到 50%，并自动处理坐标映射。

### Q2: 如何避免重复处理同一文件？

A: 系统使用文件哈希值记录已处理文件，记录保存在 `{output_dir}/processing_status.json`。如需重新处理，可删除该文件或使用 `utils/clear_processed_record.py` 清除记录。

### Q3: 如何调整并发处理数量？

A: 在 `config.yaml` 中修改 `processing.max_concurrent_tasks` 参数（建议根据 CPU/GPU 资源调整）。

### Q4: 如何禁用可视化输出？

A: 在 `config.yaml` 中设置 `processing.save_visualization: false`。

### Q5: 如何查看实时处理进度？

A: 使用 `./service_manager.sh logs -f` 查看实时日志，或访问健康检查接口 `http://localhost:8081/metrics`。

### Q6: 支持哪些图像格式？

A: 当前支持 JPEG、PNG、TIFF 格式。如需支持其他格式，可在 `auto_process_monitor.py` 中的 `IMAGE_EXTENSIONS` 列表中添加。

### Q7: 如何自定义背景过滤参数？

A: 在 `config.yaml` 的 `filtering` 部分调整参数，或启用 `auto_bg: true` 让系统自动分析背景。

### Q8: 服务占用内存过高怎么办？

A: 可以减小 `processing.max_concurrent_tasks`，或减小 `batch_inference.batch_size`。

## 🛠️ 开发说明

### 代码结构

- **service_main.py**: 主服务入口，负责启动 Flask 健康检查服务和文件监听
- **auto_process_package/**: 核心处理逻辑，包括文件监听、切片、过滤和推理调度
- **utils/**: 通用工具模块，包括日志、监控、配置等
- **inferenceTool/**: 推理脚本集合
- **sliceTool/**: 图像切片和预处理工具

### 扩展开发

1. **添加新的图像格式支持**

修改 `auto_process_package/auto_process_monitor.py` 中的 `IMAGE_EXTENSIONS` 列表。

2. **自定义过滤逻辑**

修改 `sliceTool/3filter_patch.py` 中的过滤函数。

3. **添加新的监控指标**

在 `utils/metrics.py` 中添加新的指标字段，并在处理逻辑中更新。

### 测试

```bash
# 测试配置加载
python -c "from utils.config import load_config; print(load_config('config.yaml'))"

# 测试模型加载
python -c "import torch; model = torch.jit.load('models/your_model.pt'); print('OK')"
```

## 📝 日志位置

- **Systemd 日志**: `sudo journalctl -u rtdetr-processor`
- **应用日志**: `logs/rtdetr_service_YYYYMMDD.log`
- **处理日志**: `logs/auto_process_YYYYMMDD.log`
- **处理状态**: `{output_dir}/processing_status.json`

## 📄 相关文档

- [SERVICE_README.md](./SERVICE_README.md) - 服务使用详细说明
- [auto_process_package/README.md](./auto_process_package/README.md) - 自动处理模块说明
- [annotationConverter/README.md](./annotationConverter/README.md) - 标注转换工具说明

## 📞 支持与反馈

如有问题或建议，请：

1. 查看日志文件获取详细错误信息
2. 检查配置文件格式和路径设置
3. 确认依赖是否完整安装
4. 参考故障排查章节

## 📜 许可证

请参考项目根目录的 LICENSE 文件。

---

**最后更新**: 2024年
