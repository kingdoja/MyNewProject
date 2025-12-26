# RT-DETR 自动处理服务使用指南

这是一个完整的服务包装，用于自动监听文件、切片和推理。

## 📋 功能特性

- ✅ 自动监听指定目录，检测新增图片
- ✅ 自动切片、过滤和批量预测
- ✅ 健康检查接口
- ✅ 优雅关闭
- ✅ 日志记录
- ✅ 监控和统计
- ✅ Systemd 服务支持

## 🚀 快速开始

### 1. 安装依赖

```bash
cd /home/ubuntu/lsn/project_new/RT-DETR-main/A_sclie2inference/DataSlice2Inference_main
pip install -r requirements.txt
```

### 2. 配置服务

编辑 `config.yaml` 文件，设置：
- 监听目录 (`paths.watch_dir`)
- 输出目录 (`paths.output_dir`)
- 模型路径 (`paths.model_path`)
- 其他处理参数

### 3. 启动服务

#### 方式1：直接运行（测试用）

```bash
python service_main.py --config config.yaml
```

#### 方式2：使用服务管理脚本

```bash
# 安装 systemd 服务
./service_manager.sh install

# 启动服务
./service_manager.sh start

# 查看状态
./service_manager.sh status

# 查看日志
./service_manager.sh logs

# 实时查看日志
./service_manager.sh logs -f
```

#### 方式3：使用 systemd 命令

```bash
# 安装服务（首次使用）
sudo cp systemd/rtdetr-processor.service /etc/systemd/system/
sudo sed -i "s|@PROJECT_ROOT@|$(pwd)|g" /etc/systemd/system/rtdetr-processor.service
sudo sed -i "s|@PYTHON_PATH@|$(which python3)|g" /etc/systemd/system/rtdetr-processor.service
sudo systemctl daemon-reload
sudo systemctl enable rtdetr-processor

# 启动服务
sudo systemctl start rtdetr-processor

# 查看状态
sudo systemctl status rtdetr-processor

# 查看日志
sudo journalctl -u rtdetr-processor -f
```

## 🔧 服务管理

### 使用服务管理脚本

```bash
./service_manager.sh <command>
```

可用命令：
- `install` - 安装 systemd 服务
- `start` - 启动服务
- `stop` - 停止服务
- `restart` - 重启服务
- `status` - 查看服务状态
- `logs` - 查看日志（最近100行）
- `logs -f` - 实时查看日志
- `start-direct` - 直接启动（不使用systemd）

### 使用 systemd 命令

```bash
# 启动
sudo systemctl start rtdetr-processor

# 停止
sudo systemctl stop rtdetr-processor

# 重启
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

## 🏥 健康检查

如果启用了健康检查（`config.yaml` 中 `monitoring.health_check_port > 0`），可以通过 HTTP 接口检查服务状态：

```bash
# 健康检查
curl http://localhost:8081/health

# 获取指标
curl http://localhost:8081/metrics

# 获取统计信息
curl http://localhost:8081/stats
```

## 📁 目录结构

```
DataSlice2Inference_main/
├── service_main.py          # 主服务入口
├── service_manager.sh       # 服务管理脚本
├── config.yaml              # 配置文件
├── systemd/
│   └── rtdetr-processor.service  # systemd 服务文件
├── auto_process_package/
│   ├── auto_process_monitor.py   # 核心处理逻辑
│   └── run_auto_process.py       # 配置加载器
├── utils/                    # 工具模块
└── logs/                     # 日志目录
```

## ⚙️ 配置说明

主要配置项在 `config.yaml` 中：

### 路径配置
```yaml
paths:
  watch_dir: "/path/to/watch"      # 监听目录
  output_dir: "/path/to/output"    # 输出目录
  model_path: "/path/to/model.pt"   # 模型路径
  log_dir: "/path/to/logs"         # 日志目录
```

### 处理参数
```yaml
processing:
  patch_size: 640                  # 切片大小
  threshold: 0.5                   # 置信度阈值
  file_wait_timeout: 30            # 文件等待超时（秒）
  save_visualization: true         # 是否保存可视化
  process_existing: true           # 启动时处理已存在文件
```

### 监控配置
```yaml
monitoring:
  enabled: true
  health_check_port: 8081          # 健康检查端口（0表示禁用）
  stats_interval: 300               # 统计输出间隔（秒）
```

## 🐛 故障排查

### 服务无法启动

1. 检查配置文件：
```bash
python service_main.py --config config.yaml
```

2. 检查日志：
```bash
./service_manager.sh logs
# 或
sudo journalctl -u rtdetr-processor -n 100
```

3. 检查路径和权限：
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

1. 查看详细日志：
```bash
sudo journalctl -u rtdetr-processor -n 200 --no-pager
```

2. 检查资源限制：
```bash
# 检查内存使用
free -h

# 检查磁盘空间
df -h
```

3. 检查模型文件：
```bash
# 确保模型文件完整
file /path/to/model.pt
```

### 文件处理失败

1. 检查文件格式是否支持
2. 检查文件是否完整写入
3. 查看处理日志：
```bash
tail -f logs/auto_processor_*.log
```

## 📝 日志位置

- Systemd 日志：`sudo journalctl -u rtdetr-processor`
- 应用日志：`logs/auto_processor_YYYYMMDD.log`
- 处理状态：`{output_dir}/processing_status.json`

## 🔄 更新服务

1. 停止服务：
```bash
./service_manager.sh stop
```

2. 更新代码

3. 重启服务：
```bash
./service_manager.sh restart
```

## 📞 支持

如有问题，请检查：
1. 日志文件
2. 配置文件格式
3. 路径和权限
4. 依赖是否完整安装

---

**最后更新**: 2024年

