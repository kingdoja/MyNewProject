#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试自动缩放功能

该脚本创建一个测试用的大图像，验证自动缩放功能是否正常工作。
"""

import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import random

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent.resolve()
# 添加 DataSlice2Inference_main 目录到路径，以便导入相关模块
DATA_SLICE_MAIN_DIR = PROJECT_ROOT.parent / "DataSlice2Inference_main"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(DATA_SLICE_MAIN_DIR))
sys.path.insert(0, str(DATA_SLICE_MAIN_DIR / "auto_process_package"))


def create_test_image(width: int, height: int, output_path: Path) -> None:
    """
    创建测试用的大图像
    
    Args:
        width: 图像宽度
        height: 图像高度
        output_path: 输出路径
    """
    print(f"正在创建测试图像: {width}x{height} 像素")
    print(f"预计内存使用: {width * height * 3 / (1024**3):.2f} GB")
    
    # 创建白色背景图像
    img = Image.new('RGB', (width, height), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    
    # 添加网格线（帮助验证坐标）
    grid_step = 10000
    print("添加网格线...")
    for x in range(0, width, grid_step):
        draw.line([(x, 0), (x, height)], fill=(200, 200, 200), width=10)
    for y in range(0, height, grid_step):
        draw.line([(0, y), (width, y)], fill=(200, 200, 200), width=10)
    
    # 添加一些彩色方块（模拟组织区域）
    print("添加测试图案...")
    num_blocks = min(50, (width // 5000) * (height // 5000))
    for i in range(num_blocks):
        x = random.randint(0, width - 5000)
        y = random.randint(0, height - 5000)
        w = random.randint(2000, 5000)
        h = random.randint(2000, 5000)
        color = (
            random.randint(100, 255),
            random.randint(100, 255),
            random.randint(100, 255)
        )
        draw.rectangle([x, y, x + w, y + h], fill=color, outline=(0, 0, 0), width=20)
    
    # 添加标记文本（在四个角）
    print("添加坐标标记...")
    try:
        # 尝试使用系统字体，如果失败则跳过文字
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 500)
    except:
        font = None
    
    if font:
        # 左上角
        draw.text((100, 100), f"(0, 0)", fill=(255, 0, 0), font=font)
        # 右上角
        draw.text((width - 3000, 100), f"({width}, 0)", fill=(255, 0, 0), font=font)
        # 左下角
        draw.text((100, height - 700), f"(0, {height})", fill=(255, 0, 0), font=font)
        # 右下角
        draw.text((width - 3500, height - 700), f"({width}, {height})", fill=(255, 0, 0), font=font)
    
    # 保存图像
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"正在保存图像到: {output_path}")
    img.save(output_path, 'JPEG', quality=95, optimize=True)
    
    # 显示文件大小
    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"✅ 测试图像已创建: {output_path}")
    print(f"✅ 文件大小: {file_size_mb:.1f} MB")


def test_image_size_check():
    """测试图像尺寸检查功能"""
    print("\n" + "="*70)
    print("测试1: 图像尺寸检查")
    print("="*70)
    
    from auto_process_package.auto_process_monitor import ImageProcessHandler
    from pathlib import Path
    
    # 创建临时handler
    handler = ImageProcessHandler(
        watch_dir=Path("/tmp"),
        output_base_dir=Path("/tmp"),
        model_path="dummy.pt",
        auto_downscale=True,
        downscale_threshold=50000
    )
    
    # 测试用例
    test_cases = [
        ("小图像", 10000, 8000, False),
        ("中等图像", 40000, 30000, False),
        ("大图像（宽超限）", 60000, 40000, True),
        ("大图像（高超限）", 40000, 60000, True),
        ("超大图像", 100000, 80000, True),
    ]
    
    print("\n图像尺寸检查测试：")
    print(f"阈值设置: {handler.downscale_threshold} 像素\n")
    
    all_passed = True
    for name, width, height, expected_downscale in test_cases:
        # 模拟检查（不实际创建图像）
        needs_downscale = (width > handler.downscale_threshold or 
                          height > handler.downscale_threshold)
        
        status = "✅ 通过" if needs_downscale == expected_downscale else "❌ 失败"
        action = "需要缩放" if needs_downscale else "无需缩放"
        print(f"{status} | {name:15s} | {width:6d}x{height:6d} | {action}")
        
        if needs_downscale != expected_downscale:
            all_passed = False
    
    print(f"\n{'='*70}")
    if all_passed:
        print("✅ 所有测试用例通过！")
    else:
        print("❌ 部分测试用例失败！")
    print(f"{'='*70}\n")
    
    return all_passed


def test_coordinate_mapping():
    """测试坐标映射功能"""
    print("\n" + "="*70)
    print("测试2: 坐标映射计算")
    print("="*70)
    
    # 测试用例：(patch坐标, scale_factor, 期望的原图坐标)
    test_cases = [
        # scale_factor=1.0（无缩放）
        ((0, 0, 640, 640), 1.0, (0, 0, 640, 640)),
        ((1280, 640, 1920, 1280), 1.0, (1280, 640, 1920, 1280)),
        
        # scale_factor=2.0（50%缩放）
        ((0, 0, 640, 640), 2.0, (0, 0, 1280, 1280)),
        ((1280, 640, 1920, 1280), 2.0, (2560, 1280, 3840, 2560)),
        ((320, 320, 960, 960), 2.0, (640, 640, 1920, 1920)),
    ]
    
    print("\n坐标映射测试：\n")
    print(f"{'Patch坐标':25s} | {'缩放系数':8s} | {'原图坐标':25s} | {'状态':6s}")
    print("-" * 70)
    
    all_passed = True
    for patch_coords, scale_factor, expected_orig in test_cases:
        x1, y1, x2, y2 = patch_coords
        orig_x1 = int(x1 * scale_factor)
        orig_y1 = int(y1 * scale_factor)
        orig_x2 = int(x2 * scale_factor)
        orig_y2 = int(y2 * scale_factor)
        
        calculated = (orig_x1, orig_y1, orig_x2, orig_y2)
        passed = calculated == expected_orig
        status = "✅ 通过" if passed else "❌ 失败"
        
        print(f"{str(patch_coords):25s} | {scale_factor:8.1f} | {str(calculated):25s} | {status}")
        
        if not passed:
            print(f"  期望: {expected_orig}")
            all_passed = False
    
    print(f"\n{'='*70}")
    if all_passed:
        print("✅ 所有坐标映射测试通过！")
    else:
        print("❌ 部分坐标映射测试失败！")
    print(f"{'='*70}\n")
    
    return all_passed


def test_downscale_function():
    """测试缩放函数"""
    print("\n" + "="*70)
    print("测试3: 图像缩放函数")
    print("="*70)
    
    import sys
    from pathlib import Path
    
    # 导入缩放函数
    slice_tool_dir = DATA_SLICE_MAIN_DIR / "sliceTool"
    if str(slice_tool_dir) not in sys.path:
        sys.path.insert(0, str(slice_tool_dir))
    
    from convert_wsi40x_to_20x import convert_single_image
    
    # 创建一个小测试图像
    test_dir = PROJECT_ROOT / "test_tmp"
    test_dir.mkdir(exist_ok=True)
    
    src_path = test_dir / "test_src.jpg"
    dst_path = test_dir / "test_dst.jpg"
    
    print("\n创建测试图像 (5000x4000)...")
    test_img = Image.new('RGB', (5000, 4000), color=(128, 128, 255))
    draw = ImageDraw.Draw(test_img)
    # 添加一些图案用于验证
    draw.rectangle([1000, 1000, 2000, 2000], fill=(255, 0, 0))
    test_img.save(src_path, 'JPEG', quality=95)
    print(f"✅ 测试图像已创建: {src_path}")
    
    # 测试缩放
    print("\n执行50%缩放...")
    try:
        orig_w, orig_h, new_w, new_h = convert_single_image(
            src_path=src_path,
            dst_path=dst_path,
            quality=95,
            scale_factor=0.5
        )
        
        print(f"✅ 缩放成功:")
        print(f"   原始尺寸: {orig_w}x{orig_h}")
        print(f"   新尺寸: {new_w}x{new_h}")
        
        # 验证尺寸
        expected_w, expected_h = 2500, 2000
        if new_w == expected_w and new_h == expected_h:
            print(f"✅ 尺寸验证通过")
            result = True
        else:
            print(f"❌ 尺寸验证失败: 期望 {expected_w}x{expected_h}, 实际 {new_w}x{new_h}")
            result = False
        
        # 验证文件存在
        if dst_path.exists():
            file_size = dst_path.stat().st_size / 1024
            print(f"✅ 输出文件已创建: {dst_path} ({file_size:.1f} KB)")
        else:
            print(f"❌ 输出文件不存在: {dst_path}")
            result = False
        
    except Exception as e:
        print(f"❌ 缩放失败: {e}")
        import traceback
        traceback.print_exc()
        result = False
    finally:
        # 清理测试文件
        print("\n清理测试文件...")
        if src_path.exists():
            src_path.unlink()
        if dst_path.exists():
            dst_path.unlink()
        if test_dir.exists():
            test_dir.rmdir()
        print("✅ 清理完成")
    
    print(f"\n{'='*70}")
    if result:
        print("✅ 图像缩放功能测试通过！")
    else:
        print("❌ 图像缩放功能测试失败！")
    print(f"{'='*70}\n")
    
    return result


def main():
    """主测试函数"""
    print("\n" + "="*70)
    print("RT-DETR 自动缩放功能测试")
    print("="*70)
    
    # 运行所有测试
    results = []
    
    # 测试1: 图像尺寸检查
    results.append(("图像尺寸检查", test_image_size_check()))
    
    # 测试2: 坐标映射
    results.append(("坐标映射计算", test_coordinate_mapping()))
    
    # 测试3: 缩放函数
    results.append(("图像缩放函数", test_downscale_function()))
    
    # 汇总结果
    print("\n" + "="*70)
    print("测试结果汇总")
    print("="*70)
    
    for test_name, passed in results:
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"{status} | {test_name}")
    
    all_passed = all(result for _, result in results)
    
    print("="*70)
    if all_passed:
        print("🎉 所有测试通过！自动缩放功能正常工作。")
    else:
        print("⚠️ 部分测试失败，请检查代码。")
    print("="*70)
    
    # 提供下一步建议
    if all_passed:
        print("\n下一步：创建真实测试图像")
        print("-" * 70)
        print("运行以下命令创建一个大图像进行完整测试：")
        print()
        print("  python test_downscale_feature.py --create-large-image")
        print()
        print("然后将图像放入监听目录：")
        print(f"  cp test_large_image_*.jpg {PROJECT_ROOT.parent}/DataWSI/")
        print()
        print("启动服务观察处理过程：")
        print(f"  cd {DATA_SLICE_MAIN_DIR}")
        print("  python service_main.py")
        print()
    
    return 0 if all_passed else 1


def create_large_test_image():
    """创建大型测试图像的入口"""
    print("\n" + "="*70)
    print("创建大型测试图像")
    print("="*70)
    
    # 询问用户
    print("\n选择要创建的测试图像尺寸：")
    print("1. 60000 x 40000 (轻度超限，约 6.9 GB 内存)")
    print("2. 80000 x 60000 (中度超限，约 13.8 GB 内存)")
    print("3. 100000 x 80000 (重度超限，约 23.0 GB 内存)")
    print("4. 自定义尺寸")
    
    choice = input("\n请选择 (1-4, 或按 Enter 使用选项1): ").strip() or "1"
    
    size_options = {
        "1": (60000, 40000),
        "2": (80000, 60000),
        "3": (100000, 80000),
    }
    
    if choice in size_options:
        width, height = size_options[choice]
    elif choice == "4":
        try:
            width = int(input("请输入宽度 (像素): "))
            height = int(input("请输入高度 (像素): "))
        except ValueError:
            print("❌ 输入无效，使用默认值 60000x40000")
            width, height = 60000, 40000
    else:
        print("❌ 选择无效，使用默认值 60000x40000")
        width, height = 60000, 40000
    
    output_path = PROJECT_ROOT / f"test_large_image_{width}x{height}.jpg"
    
    # 警告
    estimated_memory_gb = width * height * 3 / (1024**3)
    print(f"\n⚠️  警告: 创建 {width}x{height} 图像大约需要 {estimated_memory_gb:.1f} GB 内存")
    confirm = input("是否继续? (y/N): ").strip().lower()
    
    if confirm != 'y':
        print("已取消")
        return
    
    try:
        create_test_image(width, height, output_path)
        print(f"\n✅ 成功！现在可以将图像复制到监听目录进行测试:")
        print(f"   cp {output_path} {PROJECT_ROOT.parent}/DataWSI/")
    except Exception as e:
        print(f"\n❌ 创建失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="测试自动缩放功能")
    parser.add_argument(
        "--create-large-image",
        action="store_true",
        help="创建大型测试图像"
    )
    
    args = parser.parse_args()
    
    if args.create_large_image:
        create_large_test_image()
    else:
        sys.exit(main())


