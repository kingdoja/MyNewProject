#!/usr/bin/env python3
"""
RT-DETR v2 单图推理脚本

功能：
1. 加载指定的配置与 checkpoint，构建 deploy 模型
2. 对单张图片执行前向推理，输出检测框、类别与置信度
3. 自动保存可视化图片与 JSON 结果，保证坐标/尺寸计算准确

使用示例：
python predict_single_image.py \
    --config /home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/configs/rtdetrv2/rtdetrv2_r18vd_cancer_detection.yml \
    --checkpoint /home/ubuntu/lsn/project_new/RT-DETR-main/rtdetrv2_pytorch/output/rtdetrv2_r18vd_cancer_detection/best.pth \
    --image /home/ubuntu/lsn/project_new/RT-DETR-main/ZZ/11.png \
    --device auto \
    --threshold 0.5 \
    --output-image /home/ubuntu/lsn/project_new/RT-DETR-main/ZZ/pred_11.png \
    --output-json /home/ubuntu/lsn/project_new/RT-DETR-main/ZZ/annotations.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

import torch

# 确保可以导入 inference.py 中的公共函数
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from inference import load_model, predict_image  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description="RT-DETR v2 单图推理工具")
    parser.add_argument("--config", required=True, help="训练时使用的配置文件绝对路径")
    parser.add_argument("--checkpoint", required=True, help="训练好的 checkpoint (.pth)")
    parser.add_argument("--image", required=True, help="待预测图片路径")
    parser.add_argument(
        "--output-image",
        default=None,
        help="可视化结果保存路径（默认同目录下自动命名）",
    )
    parser.add_argument(
        "--output-json",
        default=None,
        help="JSON 结果保存路径（默认不保存）",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="推理设备：auto / cpu / cuda / cuda:0 等，auto 优先使用可用的 GPU",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="置信度阈值，低于该值的预测会被过滤",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印更多调试信息",
    )
    return parser.parse_args()


def resolve_device(device_str: str) -> torch.device:
    if device_str.lower() == "auto":
        if torch.cuda.is_available():
            print("⚡ 检测到可用 GPU，使用 CUDA 进行推理")
            return torch.device("cuda")
        print("⚠️ 未检测到 GPU，退回 CPU 推理")
        return torch.device("cpu")
    return torch.device(device_str)


def ensure_parent_dir(path: str):
    if path is None:
        return
    parent = Path(path).expanduser().resolve().parent
    parent.mkdir(parents=True, exist_ok=True)


def main():
    args = parse_args()
    device = resolve_device(args.device)

    # 路径合法性检查
    for value, desc in [
        (args.config, "配置文件"),
        (args.checkpoint, "Checkpoint"),
        (args.image, "输入图片"),
    ]:
        if not os.path.exists(value):
            raise FileNotFoundError(f"{desc} 不存在: {value}")

    # 预创建输出目录
    ensure_parent_dir(args.output_image)
    ensure_parent_dir(args.output_json)

    # 加载模型
    model = load_model(args.config, args.checkpoint, device=device)
    if model is None:
        raise RuntimeError("模型加载失败，请检查配置与 checkpoint 是否匹配")

    # 执行推理
    result = predict_image(
        model=model,
        image_path=args.image,
        output_path=args.output_image,
        device=device,
        verbose=args.verbose,
        threshold=args.threshold,
    )

    if result is None:
        raise RuntimeError("推理失败，未得到有效输出")

    # 如果未显式指定输出图片路径，predict_image 会自动命名
    saved_image_path = args.output_image
    if saved_image_path is None:
        saved_image_path = os.path.join(
            os.path.dirname(args.image),
            f"prediction_result_{os.path.basename(args.image)}",
        )

    print("\n=== 推理结果 ===")
    print(f"图片: {result['image_name']} ({result['image_size']['width']}x{result['image_size']['height']})")
    print(f"检测目标数: {result['detection_count']}")
    for idx, det in enumerate(result["detections"], 1):
        bbox = det["bbox"]
        print(
            f"{idx:02d}. 类别: {det['class_name']} "
            f"(ID={det['class_id']}), 置信度: {det['score']:.3f}, "
            f"BBox: [{bbox[0]:.1f}, {bbox[1]:.1f}, {bbox[2]:.1f}, {bbox[3]:.1f}]"
        )

    print(f"\n📸 可视化结果: {saved_image_path}")

    # 保存 JSON
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"📝 JSON 结果: {args.output_json}")

    print("✅ 单图推理完成")


if __name__ == "__main__":
    main()




