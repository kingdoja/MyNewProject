
# 方式1: 直接运行（测试）
conda activate detr
cd /home/ubuntu/lsn/project_new/RT-DETR-main/A_sclie2inference/DataSlice2Inference_main 
python3 service_main.py --config config.yaml
python3 service_main.py  # 可以直接运行


# 方式2: 使用服务管理脚本（推荐）
./service_manager.sh install
./service_manager.sh start
./service_manager.sh status
./service_manager.sh logs -f
./service_manager.sh stop



# 目录结构（A_sclie2inference，细化到文件）
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
│   │   └── predict_single_torchscript.py       # 单张 TorchScript 预测
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
│   │   └── validation.py                       # 校验与哈希工具
│   ├── models/                              # 模型文件放这里（如 rtdetr_torchscript_cuda.pt）
│   │   └── rtdetr_torchscript_cuda.pt          # 当前部署模型
│   ├── logs/                                # 运行日志目录
│   │   ├── auto_process_*.log                  # 自动处理日志
│   │   └── rtdetr_service_*.log                # 服务日志
│   ├── global_visualizer/                   # 全局可视化（全图绘制标注）
│   │   └── visualize_global_annotations.py     # 在全图上绘制检测结果
│   ├── sliceTool/                           # 切片工具（当前使用）
│   │   ├── 1step1_pre_process_jpeg.py          # JPEG 预处理
│   │   ├── 2analyze_background.py              # 背景分析
│   │   ├── 3filter_patch.py                    # Patch 过滤
│   │   └── convert_wsi40x_to_20x.py            # 40X 转 20X 辅助
│   ├── systemd/
│   │   └── rtdetr-processor.service            # systemd 服务模板
│   ├── service_main.py                      # 主入口（Flask 健康检查 + 监听器）
│   ├── service_manager.sh                   # 管理脚本（安装/启动/停止/日志）
│   ├── config.yaml                          # 配置文件
│   ├── requirements.txt                     # Python 依赖
│   ├── README.md                            # 本文件
│   ├── SERVICE_README.md                    # 服务使用说明
│   ├── IMPLEMENTATION_SUMMARY.md            # 实施摘要
│   ├── PRODUCTION_IMPROVEMENTS.md           # 生产改进记录
│   └── QUICK_START_IMPROVEMENTS.md          # 快速起步改进
├── DataWSI/                                 # 输入全图目录（监听目录）
├── DataWSI_copy/                            # 备份/副本目录（可选）
├── DataPatchesInference/                    # 推理输出/可视化/状态记录
│   ├── processing_status.json               # 已处理文件哈希记录
│   └── <运行生成的输出目录>                  # 批量预测结果与可视化
├── DataSliceTools_disused/                  # 旧的/备用切片工具脚本
│   ├── 1step1_pre_process_jpeg.py              # JPEG 预处理
│   ├── 2analyze_background.py                  # 背景分析
│   ├── 3filter_patch.py                        # Patch 过滤
│   ├── step1_pre_process_ndpi.py               # NDPI 预处理
│   ├── step1_pre_process_svs.py                # SVS 预处理
│   └── step1_pre_process_tif.py                # TIF 预处理
└── DataSlice2Inference_main.tar.gz          # 打包备份（如有）
```



