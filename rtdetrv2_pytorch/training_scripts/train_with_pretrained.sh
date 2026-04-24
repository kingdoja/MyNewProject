#!/bin/bash
# 基于预训练模型的自动化增量微调脚本
# 功能：
# 1) 使用增量微调配置 rtdetrv2_r50vd_cancer_detection1_incremental_ft.yml
# 2) 自动将配置中的 output_dir 追加当天日期（YYYYMMDD）
# 3) 生成临时配置并启动训练

set -euo pipefail

# 获取脚本所在目录的父目录（项目根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 设置工作目录为项目根目录
cd "$PROJECT_ROOT"

# 预训练模型路径（可通过环境变量 PRETRAINED_MODEL 覆盖）
PRETRAINED_MODEL="${PRETRAINED_MODEL:-$PROJECT_ROOT/output/rtdetrv2_r50vd_cancer_detection_split_dataset_0105/best.pth}"

# 配置文件（固定使用增量微调配置，可通过环境变量 CONFIG 覆盖）
CONFIG="${CONFIG:-configs/rtdetrv2/rtdetrv2_r50vd_cancer_detection1_incremental_ft.yml}"
CONFIG_ABS="$PROJECT_ROOT/$CONFIG"

# 训练设备（可通过环境变量 DEVICE 覆盖）
DEVICE="${DEVICE:-cuda:0}"

# 日期后缀格式
DATE_SUFFIX="$(date +%Y%m%d)"

# 临时配置文件路径
TMP_CONFIG="$(mktemp "$PROJECT_ROOT/configs/rtdetrv2/tmp.incremental_ft.${DATE_SUFFIX}.XXXXXX.yml")"

cleanup() {
    if [ -f "$TMP_CONFIG" ]; then
        rm -f "$TMP_CONFIG"
    fi
}
trap cleanup EXIT

# 检查配置文件是否存在
if [ ! -f "$CONFIG_ABS" ]; then
    echo "错误: 配置文件不存在: $CONFIG_ABS"
    exit 1
fi

# 检查预训练模型是否存在
if [ ! -f "$PRETRAINED_MODEL" ]; then
    echo "错误: 预训练模型不存在: $PRETRAINED_MODEL"
    exit 1
fi

# 读取并重写 output_dir（追加日期）
python3 - "$CONFIG_ABS" "$TMP_CONFIG" "$DATE_SUFFIX" <<'PY'
import re
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
date_suffix = sys.argv[3]

content = src.read_text(encoding="utf-8")
pattern = re.compile(r"^(output_dir:\s*)(\S+)\s*$", re.MULTILINE)
match = pattern.search(content)
if not match:
    raise SystemExit(f"未在配置中找到 output_dir: {src}")

origin_dir = match.group(2).strip().strip("'\"")
new_dir = f"{origin_dir}_{date_suffix}"
new_line = f"{match.group(1)}{new_dir}"
updated = pattern.sub(new_line, content, count=1)
dst.write_text(updated, encoding="utf-8")

print(new_dir)
PY

FINAL_OUTPUT_DIR="$(python3 - "$TMP_CONFIG" <<'PY'
import re
import sys
from pathlib import Path

cfg = Path(sys.argv[1]).read_text(encoding="utf-8")
m = re.search(r"^output_dir:\s*(\S+)\s*$", cfg, re.MULTILINE)
if not m:
    raise SystemExit("无法从临时配置读取 output_dir")
print(m.group(1).strip().strip("'\""))
PY
)"

echo "=========================================="
echo "开始自动化增量微调训练"
echo "=========================================="
echo "原始配置文件: $CONFIG"
echo "临时配置文件: ${TMP_CONFIG#$PROJECT_ROOT/}"
echo "output_dir  : $FINAL_OUTPUT_DIR"
echo "预训练模型: $PRETRAINED_MODEL"
echo "训练设备  : $DEVICE"
echo "=========================================="

# 运行训练（使用 tuning 模式）
python3 tools/train.py \
    -c "$TMP_CONFIG" \
    -t "$PRETRAINED_MODEL" \
    --use-amp \
    --seed=42 \
    -d "$DEVICE"

echo "=========================================="
echo "训练完成"
echo "输出目录: $FINAL_OUTPUT_DIR"
echo "=========================================="
