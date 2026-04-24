#!/bin/bash

# RT-DETR 模型导出快速脚本
# 使用说明：修改下面的路径参数，然后运行此脚本

echo "======================================"
echo "RT-DETR 模型导出工具"
echo "======================================"
echo ""

# ===== 配置区域 - 请根据你的实际情况修改 =====

# 选择版本：rtdetr_pytorch 或 rtdetrv2_pytorch
VERSION="rtdetrv2_pytorch"

# 配置文件路径（相对于VERSION目录）
CONFIG_FILE="configs/rtdetrv2/rtdetrv2_r18vd.yml"

# 训练好的模型checkpoint路径
CHECKPOINT="premodel/best.pth"

# 输出目录
OUTPUT_DIR="exported_models"

# 导出模式：state_dict / full_model / torchscript / deploy
# 推荐使用 state_dict（体积小，灵活）
EXPORT_MODE="state_dict"

# 输入图像尺寸（仅用于torchscript模式）
INPUT_SIZE=640

# ============================================

# 进入版本目录
cd $VERSION

# 创建输出目录
mkdir -p $OUTPUT_DIR

# 获取checkpoint文件名（不含扩展名）
BASENAME=$(basename "$CHECKPOINT" .pth)

# 根据导出模式设置输出文件名
if [ "$EXPORT_MODE" = "state_dict" ]; then
    OUTPUT_FILE="${OUTPUT_DIR}/${BASENAME}_weights.pt"
elif [ "$EXPORT_MODE" = "full_model" ]; then
    OUTPUT_FILE="${OUTPUT_DIR}/${BASENAME}_full.pt"
elif [ "$EXPORT_MODE" = "torchscript" ]; then
    OUTPUT_FILE="${OUTPUT_DIR}/${BASENAME}_torchscript.pt"
elif [ "$EXPORT_MODE" = "deploy" ]; then
    OUTPUT_FILE="${OUTPUT_DIR}/${BASENAME}_deploy.pt"
fi

echo "配置信息："
echo "  版本: $VERSION"
echo "  配置文件: $CONFIG_FILE"
echo "  Checkpoint: $CHECKPOINT"
echo "  导出模式: $EXPORT_MODE"
echo "  输出文件: $OUTPUT_FILE"
echo ""
echo "开始导出..."
echo ""

# 执行导出命令
python tools/export_pt.py \
    --config "$CONFIG_FILE" \
    --resume "$CHECKPOINT" \
    --mode "$EXPORT_MODE" \
    --output "$OUTPUT_FILE" \
    --input-size $INPUT_SIZE

# 检查导出是否成功
if [ $? -eq 0 ]; then
    echo ""
    echo "======================================"
    echo "✅ 导出成功!"
    echo "======================================"
    echo "输出文件位置: $VERSION/$OUTPUT_FILE"
    echo ""
    
    # 显示文件信息
    if [ -f "$OUTPUT_FILE" ]; then
        FILE_SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
        echo "文件大小: $FILE_SIZE"
    fi
else
    echo ""
    echo "======================================"
    echo "❌ 导出失败，请检查错误信息"
    echo "======================================"
    exit 1
fi

