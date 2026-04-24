# 训练脚本说明

本文件夹包含用于基于预训练模型进行微调训练的脚本和说明文档。

## 文件说明

- **训练方案说明.md**: 详细的训练方案对比和实施步骤说明
- **train_with_pretrained.sh**: 基于预训练模型微调的训练脚本
- **README.md**: 本说明文件

## 快速开始

### 使用自动训练脚本（推荐）

```bash
cd /home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch
./training_scripts/train_with_pretrained.sh
```

脚本会自动：
- 使用 `configs/rtdetrv2/rtdetrv2_r50vd_cancer_detection1_incremental_ft.yml`
- 读取其中的 `output_dir`
- 自动追加当天日期（`YYYYMMDD`）后启动训练  
  例如：`./output/xxx_20260323`

### 可选环境变量

```bash
cd /home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch
PRETRAINED_MODEL=/path/to/best.pth DEVICE=cuda:1 ./training_scripts/train_with_pretrained.sh
```

## 注意事项

- 脚本会自动检测预训练模型和配置文件是否存在
- 脚本会自动切换到项目根目录执行
- 训练日志和模型会保存在“配置 output_dir + 当天日期”的目录

## 更多信息

详细说明请参考 `训练方案说明.md` 文件。
