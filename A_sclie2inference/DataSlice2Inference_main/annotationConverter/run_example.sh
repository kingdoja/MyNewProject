#!/bin/bash
# 标注坐标转换工具使用示例

# 设置路径
ANNOTATION_FILE="/home/ubuntu/lsn/project_new/RT-DETR-main/A_sclie2inference/test_big_conv_small/标记.json" # 全图标注文件
PATCH_DIR="/home/ubuntu/lsn/project_new/RT-DETR-main/A_sclie2inference/DataPatches/cj_20260106_155902" # 切片目录
OUTPUT_DIR="/home/ubuntu/lsn/project_new/RT-DETR-main/A_sclie2inference/DataSlice2Inference_main/annotationConverter/output" # 输出目录
WSI_IMAGE="/home/ubuntu/lsn/project_new/RT-DETR-main/A_sclie2inference/DataWSI/cj.jpeg" # 全图图像

# 运行转换脚本
python convert_global_to_patch.py \
    --annotation-file "$ANNOTATION_FILE" \
    --patch-dir "$PATCH_DIR" \
    --output-dir "$OUTPUT_DIR" \
    --wsi-image "$WSI_IMAGE"

echo ""
echo "转换完成！输出文件："
echo "  - COCO格式JSON: $OUTPUT_DIR/coco_format.json"
echo "  - 可视化图片: $OUTPUT_DIR/visualization/"

