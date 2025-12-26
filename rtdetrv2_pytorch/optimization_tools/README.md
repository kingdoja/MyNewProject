# 模型优化工具集

本文件夹包含用于诊断和优化RT-DETR模型训练的工具。

## 文件说明

- **diagnose_dataset.py** - 数据集诊断工具，用于检查数据质量、类别分布、目标大小等
- **monitor_training.py** - 训练监控工具，用于实时分析训练过程中的loss和指标变化
- **OPTIMIZATION_GUIDE.md** - 详细的优化指南文档

## 使用方法

### 1. 数据集诊断

在项目根目录（rtdetrv2_pytorch）下运行：

```bash
python optimization_tools/diagnose_dataset.py \
    --ann-file /home/ubuntu/lsn/project_new/RT-DETR-main/DATA/SputumCell/split_dataset_aug/train/annotations/instances_train.json \
    --img-folder /home/ubuntu/lsn/project_new/RT-DETR-main/DATA/SputumCell/split_dataset_aug/train/images \
    --output-dir ./diagnosis_output
```

### 2. 训练监控

在训练过程中，定期运行监控脚本：

```bash
python optimization_tools/monitor_training.py \
    --log-file ./logs/training.log \
    --output-dir ./training_monitor
```

## 注意事项

- 所有脚本需要在项目根目录（rtdetrv2_pytorch）下运行
- 输出目录路径是相对于运行脚本时的当前工作目录
- 详细的使用说明请参考 OPTIMIZATION_GUIDE.md

