RT-DETRv2 PyTorch 项目说明
===========================

本目录是 RT-DETRv2 的 PyTorch 实现，包含从数据配置、训练、推理到训练过程分析和可视化的一整套工具。下面包含使用 `configs/rtdetrv2` 进行训练以及使用 `training_analysis` 做评估和可视化的完整命令流程，和目录结构。



# 从安装到训练再到评估流程
--------------------------------

下面给出一个尽量精简但可直接跑通的使用流程，更多细节可自行阅读各子目录的 `README` 或脚本注释。

### 1. 安装环境

```bash
cd rtdetrv2_pytorch
pip install -r requirements.txt
```

建议使用 conda/venv 独立环境，Python ≥ 3.8，PyTorch 与 CUDA 版本按本机 GPU 选择。

### 2. 准备数据与配置

- 在 `configs/dataset/` 中选择数据集配置并修改路径/类别等字段（如 `data_root`、`img_folder`、`ann_file`）。  
- 在 `configs/rtdetrv2/` 中选择对应模型配置，确保其中引用的数据集配置与你刚才修改的一致。

### 3. 启动训练

推荐使用 `tools/train.py` 统一入口：

```bash
cd rtdetrv2_pytorch
python tools/train.py \
  -c configs/rtdetrv2/<your_config>.yml \
  -d cuda:0 \
  --use-amp \
  --seed 42 \
  --output-dir outputs/<exp_name> \
  --summary-dir outputs/<exp_name>/tb
```

如需快速体验，可以直接运行根目录的：

```bash
cd rtdetrv2_pytorch
python train.py
```

其内部默认使用一个痰液检测配置与 `cuda:0` 设备，你可以在 `train.py` 中调整默认参数。

### 4. 在验证/测试集上导出预测

- 使用你自己的推理脚本，或参考项目中的 `inference.py` / `training_analysis/scripts/visualize_validation.py`，在验证/测试集上批量推理；  
- 将预测结果以 COCO detection JSON 格式保存到 `training_analysis/output/<exp_name>/val_detections.json` 或 `test_detections.json`。

典型可视化+导出预测的方式（示例）：

```bash
cd rtdetrv2_pytorch
python -m training_analysis.visualize_validation \
  --config configs/rtdetrv2/<your_config>.yml \
  --model outputs/<exp_name>/best.pth \
  --device cuda:0 \
  --images-dir /path/to/val/images \
  --output-dir training_analysis/output/<exp_name>_val_vis
```

### 5. 训练后评估与可视化

当你拥有：

- GT 标注（如 `instances_val.json`）；  
- 预测结果 JSON（如 `training_analysis/output/<exp_name>/val_detections.json`），  

可以使用 `training_analysis` 中的脚本做进一步分析：

- `accuracy.py`：计算 per-class/overall 准确率等分类指标；  
- `confusion_matrix.py`：生成混淆矩阵图；  
- `plot_training_curves.py`：从 `--summary-dir` 日志中绘制 loss/mAP/学习率曲线；  
- `visualEval_plot/` 下各脚本：绘制 PR 曲线、多实验对比、loss–mAP 关系等。

一句话概括：**配置与环境 → `tools/train.py` 训练 → 导出 COCO JSON 预测 → 用 `training_analysis`/`visualEval_plot` 做全面评估与可视化即可。**




# 目录结构说明
-----------

下面以树形结构列出 `rtdetrv2_pytorch` 项目中各个文件/目录及其作用（仅列出与使用密切相关的源码与脚本，常规缓存/中间文件不列出）。

```text
rtdetrv2_pytorch/                      # RT-DETRv2 PyTorch 实现根目录
├── README.md                          # 本说明文档
├── README1.md                         # 旧版/补充说明（可忽略或参考）
├── requirements.txt                   # Python 依赖列表
├── Dockerfile                         # 构建 Docker 训练/推理环境
├── docker-compose.yml                 # 使用 docker-compose 启动环境的配置
├── train.py                           # 封装的训练入口（内部调用 tools/train.py，内置默认配置）
├── inference.py                       # 通用推理脚本（可在数据集上批量推理）
├── simple_inference.py                # 简化版推理脚本（适合快速测试）
├── predict_single_image.py            # 单张图片预测与可视化脚本
├── visualize_results.py               # 对预测结果进行可视化的工具脚本
├── visualize_test.py                  # 针对测试集结果的可视化脚本
├── training_analysis/scripts/visualize_validation.py # 针对验证集结果的可视化脚本
├── check_coco_info.py                 # 检查 COCO 标注文件（类别、图像数等）的小工具
├── check_gpu.py                       # 检查 GPU/显存是否可用的脚本
├── check_validation.py                # 检查验证集配置与路径的小脚本
├── generate_class_tables.py           # 根据数据集/配置生成类别表格（用于报告或论文）
├── configs/                           # 所有 YAML 配置文件
│   ├── runtime.yml                    # 通用运行时配置（日志/分布式/设备等）
│   ├── dataset/                       # 各类数据集配置
│   │   ├── coco_detection.yml         # COCO 检测数据集配置示例
│   │   ├── voc_detection.yml          # VOC 检测数据集配置示例
│   │   ├── cancer_detection.yml       # 痰液细胞检测数据集配置（版本 1）
│   │   ├── cancer_detection1.yml      # 痰液细胞检测数据集配置（版本 2）
│   │   ├── cancer_detection2.yml      # 痰液细胞检测数据集配置（版本 3）
│   ├── rtdetr/                        # 原始 RT-DETR 模型配置
│   │   ├── readme.md                  # RT-DETR 配置说明
│   │   ├── rtdetr_r18vd_6x_coco.yml   # RT-DETR-R18 在 COCO 上 6x 训练配置
│   │   ├── rtdetr_r34vd_6x_coco.yml   # RT-DETR-R34 在 COCO 上 6x 训练配置
│   │   ├── rtdetr_r50vd_6x_coco.yml   # RT-DETR-R50 在 COCO 上 6x 训练配置
│   │   ├── rtdetr_r50vd_m_6x_coco.yml # RT-DETR-R50-M 变体在 COCO 上 6x 训练配置
│   │   ├── rtdetr_r101vd_6x_coco.yml  # RT-DETR-R101 在 COCO 上 6x 训练配置
│   │   └── include/                   # RT-DETR 子配置（被上面各 yml 引用）
│   │       ├── dataloader.yml         # RT-DETR 通用数据加载设置
│   │       ├── optimizer.yml          # RT-DETR 通用优化器设置
│   │       └── rtdetr_r50vd.yml       # RT-DETR-R50 模型结构子配置
│   └── rtdetrv2/                      # RT-DETRv2 主配置（本项目核心）
│       ├── rtdetrv2_r18vd_120e_coco.yml        # RT-DETRv2-R18 COCO 120e 训练
│       ├── rtdetrv2_r34vd_120e_coco.yml        # RT-DETRv2-R34 COCO 120e 训练
│       ├── rtdetrv2_r50vd_6x_coco.yml          # RT-DETRv2-R50 COCO 6x 训练
│       ├── rtdetrv2_r50vd_m_7x_coco.yml        # RT-DETRv2-R50-M COCO 7x 训练
│       ├── rtdetrv2_r50vd_dsp_1x_coco.yml      # RT-DETRv2-R50 DSP 1x 训练
│       ├── rtdetrv2_r50vd_m_dsp_3x_coco.yml    # RT-DETRv2-R50-M DSP 3x 训练
│       ├── rtdetrv2_r18vd_dsp_3x_coco.yml      # RT-DETRv2-R18 DSP 3x 训练
│       ├── rtdetrv2_r18vd_sp1_120e_coco.yml    # RT-DETRv2-R18 不同 schedule/sp 实验 1
│       ├── rtdetrv2_r18vd_sp2_120e_coco.yml    # RT-DETRv2-R18 不同 schedule/sp 实验 2
│       ├── rtdetrv2_r18vd_sp3_120e_coco.yml    # RT-DETRv2-R18 不同 schedule/sp 实验 3
│       ├── rtdetrv2_r18vd_120e_voc.yml         # RT-DETRv2-R18 VOC 120e 训练
│       ├── rtdetrv2_r34vd_dsp_1x_coco.yml      # RT-DETRv2-R34 DSP 1x 训练
│       ├── rtdetrv2_r101vd_6x_coco.yml         # RT-DETRv2-R101 COCO 6x 训练
│       ├── rtdetrv2_hgnetv2_l_6x_coco.yml      # RT-DETRv2-HGNetV2-L COCO 6x 训练
│       ├── rtdetrv2_hgnetv2_h_6x_coco.yml      # RT-DETRv2-HGNetV2-H COCO 6x 训练
│       ├── rtdetrv2_hgnetv2_x_6x_coco.yml      # RT-DETRv2-HGNetV2-X COCO 6x 训练
│       ├── rtdetrv2_r18vd_cancer_detection.yml       # RT-DETRv2-R18 痰液检测配置（版本 1）
│       ├── rtdetrv2_r18vd_cancer_detection1.yml      # RT-DETRv2-R18 痰液检测配置（版本 2）
│       ├── rtdetrv2_r50vd_cancer_detection.yml       # RT-DETRv2-R50 痰液检测配置（版本 1）
│       ├── rtdetrv2_r50vd_cancer_detection1.yml      # RT-DETRv2-R50 痰液检测配置（版本 2）
│       ├── rtdetrv2_r50vd_cancer_detection1_optimized.yml # RT-DETRv2-R50 痰液检测优化版配置
│       ├── rtdetrv2_r50vd_cancer_detection2.yml      # RT-DETRv2-R50 痰液检测配置（版本 3）
│       ├── rtdetrv2_r18vd_sputum_aug.yml             # RT-DETRv2-R18 痰液数据增强配置
│       ├── rtdetrv2_r50vd_sputum_aug.yml             # RT-DETRv2-R50 痰液数据增强配置
│       ├── include/                          # RT-DETRv2 子配置（被上面各 yml 引用）
│       │   ├── dataloader.yml               # RT-DETRv2 通用数据加载设置
│       │   ├── optimizer.yml                # RT-DETRv2 通用优化器设置
│       │   └── rtdetrv2_r50vd.yml           # RT-DETRv2-R50 模型结构子配置
├── src/                              # 项目核心源码（模型/数据/训练框架等）
│   ├── __init__.py                   # 使 src 成为 Python 包
│   ├── core/                         # 核心配置与工作区管理
│   │   ├── __init__.py
│   │   ├── _config.py                # 低层配置定义
│   │   ├── workspace.py              # 实验 workspace 管理
│   │   ├── yaml_config.py            # YAMLConfig 实现，负责解析 yml 并构建组件
│   │   └── yaml_utils.py             # YAML 解析与命令行覆盖工具
│   ├── data/                         # 数据集与数据增强
│   │   ├── __init__.py
│   │   ├── _misc.py                  # 数据加载相关的辅助函数
│   │   ├── dataloader.py             # 通用 DataLoader 构建逻辑
│   │   ├── dataset/                  # 各具体数据集实现
│   │   │   ├── __init__.py
│   │   │   ├── _dataset.py           # 抽象数据集基类
│   │   │   ├── cifar_dataset.py      # CIFAR 数据集示例
│   │   │   ├── coco_dataset.py       # COCO 检测数据集
│   │   │   ├── coco_eval.py          # COCO 评估封装
│   │   │   ├── coco_utils.py         # COCO 数据集工具函数
│   │   │   ├── voc_detection.py      # VOC 检测数据集
│   │   │   └── voc_eval.py           # VOC 评估封装
│   │   └── transforms/               # 数据增强与图像变换
│   │       ├── __init__.py
│   │       ├── _transforms.py        # 基本图像变换定义
│   │       ├── container.py          # 变换容器/组合
│   │       ├── functional.py         # 具体图像操作函数
│   │       ├── mosaic.py             # Mosaic 等增强策略
│   │       └── presets.py            # 预设增强 pipeline
│   ├── misc/                         # 杂项工具
│   │   ├── __init__.py
│   │   ├── box_ops.py                # 边界框操作（IoU 等）
│   │   ├── dist_utils.py             # 分布式训练工具
│   │   ├── lazy_loader.py            # 延迟加载工具
│   │   ├── logger.py                 # 日志与打印工具
│   │   ├── profiler_utils.py         # 性能分析工具
│   │   └── visualizer.py             # 中间结果可视化辅助
│   ├── nn/                           # 神经网络模块（模型骨干/头等）
│   │   ├── __init__.py
│   │   ├── arch/                     # 高层网络结构（分类/检测等）
│   │   │   ├── __init__.py
│   │   │   ├── classification.py     # 分类网络结构
│   │   │   └── yolo.py               # YOLO 类结构示例
│   │   ├── backbone/                 # 各类 backbone
│   │   │   ├── __init__.py
│   │   │   ├── common.py             # backbone 通用模块
│   │   │   ├── csp_darknet.py        # CSP-Darknet 实现
│   │   │   ├── csp_resnet.py         # CSP-ResNet 实现
│   │   │   ├── hgnetv2.py            # HGNetV2 实现
│   │   │   ├── presnet.py            # Paddle 风格 ResNet 适配
│   │   │   ├── test_resnet.py        # ResNet 测试脚本
│   │   │   ├── timm_model.py         # 基于 timm 的 backbone 封装
│   │   │   ├── torchvision_model.py  # 基于 torchvision 的 backbone 封装
│   │   │   └── utils.py              # backbone 相关工具函数
│   │   ├── criterion/                # 损失与训练目标
│   │   │   ├── __init__.py
│   │   │   └── det_criterion.py      # 检测任务损失函数封装
│   │   └── postprocessor/            # 后处理模块
│   │       ├── __init__.py
│   │       ├── box_revert.py         # 预测框坐标反变换
│   │       ├── detr_postprocessor.py # DETR 风格后处理
│   │       └── nms_postprocessor.py  # NMS 后处理
│   ├── optim/                        # 优化器与训练调度
│   │   ├── __init__.py
│   │   ├── amp.py                    # 自动混合精度相关工具
│   │   ├── ema.py                    # EMA（指数滑动平均）工具
│   │   ├── optim.py                  # 优化器创建与封装
│   │   └── warmup.py                 # 学习率 warmup 等调度策略
│   ├── solver/                       # 训练/验证 Solver 框架
│   │   ├── __init__.py               # 定义 TASKS 等入口
│   │   ├── _solver.py                # 通用 Solver 抽象类
│   │   ├── clas_engine.py            # 分类任务训练/验证 engine
│   │   ├── clas_solver.py            # 分类任务 solver 封装
│   │   ├── det_engine.py             # 检测任务训练/验证 engine
│   │   └── det_solver.py             # 检测任务 solver 封装（RT-DETRv2 主要入口）
│   └── zoo/                          # 模型“动物园”，含 RT-DETR/RT-DETRv2 具体实现
│       ├── __init__.py
│       └── rtdetr/                   # RT-DETR/RT-DETRv2 相关模块
│           ├── __init__.py
│           ├── box_ops.py            # RT-DETR 专用 box 操作
│           ├── conver_params.py      # 参数转换工具
│           ├── denoising.py          # 去噪训练相关模块
│           ├── hybrid_encoder.py     # 混合编码器实现
│           ├── matcher.py            # 匹配算法（如匈牙利匹配）
│           ├── rtdetr_criterion.py   # RT-DETR 损失定义
│           ├── rtdetr_decoder.py     # RT-DETR 解码器
│           ├── rtdetr_postprocessor.py # RT-DETR 后处理器
│           ├── rtdetr.py             # RT-DETR 主模型
│           ├── rtdetrv2_criterion.py # RT-DETRv2 损失定义
│           ├── rtdetrv2_decoder.py   # RT-DETRv2 解码器
│           └── utils.py              # RT-DETR/RT-DETRv2 通用工具
├── tools/                            # 训练/导出等命令行工具
│   ├── README.md                     # 工具脚本使用说明
│   ├── train.py                      # 通用训练/验证入口（推荐使用的 CLI）
│   ├── export_pt.py                  # 导出 PyTorch 原生权重（.pt/.pth）
│   ├── export_onnx.py                # 导出 ONNX 模型
│   ├── export_trt.py                 # 基于 ONNX/Torch 导出 TensorRT 引擎
│   ├── onnx2trt.sh                   # 使用 trtexec 将 ONNX 转换为 TRT 的脚本
│   └── run_profile.py                # 性能 profile（吞吐/延迟）工具
├── training_analysis/                # 训练结果分析与可视化
│   ├── accuracy.py                   # 计算分类相关指标（per-class/mean accuracy 等）
│   ├── confusion_matrix.py           # 生成并绘制分类混淆矩阵
│   ├── extract_best_results.py       # 从多次实验结果中提取最佳指标
│   ├── plot_training_curves.py       # 绘制 loss/mAP/学习率 等训练曲线
│   ├── scripts/visualize_validation.py # 批量可视化验证集推理结果
│   ├── output/                       # 保存各类评估/可视化输出
│   │   ├── best_results_*.txt        # 不同实验汇总的最佳指标
│   │   ├── validation_visualization_*/ # 多组验证集预测 JSON（val/test）
│   │   │   ├── val_detections.json   # 验证集检测结果（COCO 格式）
│   │   │   └── test_detections.json  # 测试集检测结果（COCO 格式）
│   └── 消融实验/                       # 消融实验目录（下沉一层）
│       ├── ablation_study/           # 消融实验脚本（基础）
│       │   ├── README.md
│       │   └── extract_ablation_data.py
│       └── ablation_study_virtual/   # 消融实验相关代码与结果（虚拟实验）
│           ├── __init__.py
│           ├── data/                 # 消融实验数据（csv/json）
│           ├── extract_ablation_data.py
│           ├── generate_ablation_report.py
│           ├── generate_ablation_table.py
│           ├── visualize_ablation.py
│           ├── run_ablation_study.py # 一键运行虚拟消融实验
│           ├── report/               # 输出的 markdown 报告
│           └── tables/               # 输出的 TeX/markdown 表格
├── optimization_tools/               # 训练问题诊断与快速优化
│   ├── diagnose_dataset.py           # 检查数据集质量（类别分布/标注异常等）
│   ├── monitor_training.py           # 监控训练过程（loss/mAP 等）
│   ├── OPTIMIZATION_GUIDE.md         # 深度优化指南
│   ├── QUICK_FIX.md                  # 常见问题快速修复方案
│   └── README.md                     # 优化工具使用说明
├── references/                       # 参考实现与部署示例
│   └── deploy/                       # 多种推理/部署后端示例
│       ├── readme.md                 # 部署示例说明
│       ├── rtdetrv2_onnxruntime.py   # 基于 ONNX Runtime 部署示例
│       ├── rtdetrv2_openvino.py      # 基于 OpenVINO 部署示例
│       ├── rtdetrv2_tensorrt.py      # 基于 TensorRT 部署示例
│       └── rtdetrv2_torch.py         # 纯 PyTorch 推理部署示例
├── visualEval_plot/                  # 额外的训练指标可视化脚本
│   ├── accuracy.py                   # 准确率相关的可视化/统计
│   ├── analyze_sputum_dataset.py     # 痰液数据集分布/标签分析
│   ├── evalPlot.py                   # 综合评估结果绘图
│   ├── lossPlot.py                   # 训练损失曲线绘制
│   ├── mAP_Loss.py                   # mAP 与 loss 关联分析
│   └── precision_recall_metrics.py   # 精度-召回等指标绘制
└── references（同上）/其它中间/缓存目录 # 日常使用中生成的中间文件请根据需要自行管理
```
