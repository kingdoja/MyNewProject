#!/bin/bash

# 痰液细胞检测优化训练脚本
# 针对小目标检测进行了多项优化

cd /home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch

# 设置环境变量
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# 检查GPU状态
echo "检查GPU状态..."
nvidia-smi

# 清理GPU缓存
python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None"

# 训练参数配置
CONFIG_FILE="/home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_sputum_cell_optimized.yml"
# PRETRAINED_MODEL="./premodel/rtdetr-l.pt"  # 请确保预训练模型存在
OUTPUT_DIR="./output/sputum_cell_optimized_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="./logs/sputum_cell_optimized_$(date +%Y%m%d_%H%M%S)"

# 创建输出目录
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

echo "开始痰液细胞检测训练..."
echo "配置文件: $CONFIG_FILE"
echo "预训练模型: $PRETRAINED_MODEL"
echo "输出目录: $OUTPUT_DIR"
echo "日志目录: $LOG_DIR"
echo "=========================================="

# 使用优化后的训练脚本
# --tuning $PRETRAINED_MODEL \
python train.py \
  -c $CONFIG_FILE \
  --use-amp \
  --seed 42 \
  --device cuda:0 \
  --output-dir $OUTPUT_DIR \
  --summary-dir $LOG_DIR \
  --small-object-mode \
  --num-queries 500 \
  --batch-size 4 \
  --learning-rate 0.0002 \
  --epochs 150 \
  --focal-loss-alpha 0.75 \
  --focal-loss-gamma 2.0 \
  -u "RTDETRCriterionv2.weight_dict.loss_vfl=8.0" \
  -u "RTDETRCriterionv2.weight_dict.loss_bbox=3.0" \
  -u "RTDETRCriterionv2.weight_dict.loss_giou=2.0" \
  -u "RTDETRCriterionv2.matcher.weight_dict.cost_class=3.0" \
  -u "RTDETRCriterionv2.matcher.weight_dict.cost_bbox=8.0" \
  -u "RTDETRCriterionv2.matcher.weight_dict.cost_giou=3.0"

# 检查训练结果
if [ $? -eq 0 ]; then
    echo "=========================================="
    echo "训练成功完成!"
    echo "模型保存在: $OUTPUT_DIR"
    echo "日志保存在: $LOG_DIR"
    echo "=========================================="
    
    # 显示最终模型信息
    echo "最终模型文件:"
    ls -la $OUTPUT_DIR/*.pt 2>/dev/null || echo "未找到模型文件"
    
    # 显示训练日志
    echo "训练日志:"
    ls -la $LOG_DIR/ 2>/dev/null || echo "未找到日志文件"
else
    echo "=========================================="
    echo "训练失败!"
    echo "请检查错误信息并重试"
    echo "=========================================="
    exit 1
fi
