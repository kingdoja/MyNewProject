#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试坐标映射修复

验证两种情况下的坐标转换是否正确：
1. 图像尺寸 < 50000：scale_factor = 1.0
2. 图像尺寸 > 50000：scale_factor = 2.0
"""

import sys
from pathlib import Path
import torch

# 添加项目路径
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT / "inferenceTool"))

from predict_batch_torchscript import convert_to_global_coordinates


def test_coordinate_mapping():
    """测试坐标映射逻辑"""
    print("\n" + "="*70)
    print("坐标映射修复验证测试")
    print("="*70)
    
    # 测试用例
    test_cases = [
        {
            "name": "情况1: 图像尺寸 < 50000 (无缩放)",
            "scale_factor": 1.0,
            "patch_offset": (640, 0),
            "boxes_patch": torch.tensor([[100.0, 150.0, 200.0, 250.0]]),
            "expected": torch.tensor([[740.0, 150.0, 840.0, 250.0]]),
            "description": "原图40000x30000，patch在(640,0)，检测框在patch内(100,150,200,250)"
        },
        {
            "name": "情况2: 图像尺寸 > 50000 (50%缩放)",
            "scale_factor": 2.0,
            "patch_offset": (1280, 0),
            "boxes_patch": torch.tensor([[100.0, 150.0, 200.0, 250.0]]),
            "expected": torch.tensor([[1480.0, 300.0, 1680.0, 500.0]]),
            "description": "原图100000x80000，缩放后50000x40000，patch在原图(1280,0)，检测框在patch内(100,150,200,250)"
        },
        {
            "name": "情况3: 无缩放，多个检测框",
            "scale_factor": 1.0,
            "patch_offset": (0, 0),
            "boxes_patch": torch.tensor([
                [50.0, 60.0, 150.0, 160.0],
                [200.0, 210.0, 300.0, 310.0]
            ]),
            "expected": torch.tensor([
                [50.0, 60.0, 150.0, 160.0],
                [200.0, 210.0, 300.0, 310.0]
            ]),
            "description": "patch在原图(0,0)，两个检测框"
        },
        {
            "name": "情况4: 2倍缩放，多个检测框",
            "scale_factor": 2.0,
            "patch_offset": (2560, 1280),
            "boxes_patch": torch.tensor([
                [50.0, 60.0, 150.0, 160.0],
                [200.0, 210.0, 300.0, 310.0]
            ]),
            "expected": torch.tensor([
                [2660.0, 1400.0, 2860.0, 1600.0],
                [2960.0, 1700.0, 3160.0, 1900.0]
            ]),
            "description": "patch在原图(2560,1280)，检测框需要先×2再加偏移"
        }
    ]
    
    all_passed = True
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n测试 {i}: {test_case['name']}")
        print(f"描述: {test_case['description']}")
        print(f"缩放系数: {test_case['scale_factor']}x")
        print(f"Patch偏移: {test_case['patch_offset']}")
        print(f"Patch内检测框: {test_case['boxes_patch'].tolist()}")
        
        # 执行转换
        result = convert_to_global_coordinates(
            test_case['boxes_patch'],
            test_case['patch_offset'],
            test_case['scale_factor']
        )
        
        print(f"期望结果: {test_case['expected'].tolist()}")
        print(f"实际结果: {result.tolist()}")
        
        # 验证结果
        if torch.allclose(result, test_case['expected'], atol=1e-5):
            print("✅ 通过")
        else:
            print("❌ 失败")
            print(f"差异: {(result - test_case['expected']).tolist()}")
            all_passed = False
    
    print("\n" + "="*70)
    if all_passed:
        print("🎉 所有测试通过！坐标映射修复成功。")
    else:
        print("❌ 部分测试失败，请检查代码。")
    print("="*70)
    
    return all_passed


def test_manual_calculation():
    """手动计算验证"""
    print("\n" + "="*70)
    print("手动计算验证")
    print("="*70)
    
    print("\n场景：原图100000x80000，缩放为50000x40000 (scale_factor=2.0)")
    print("patch_1在缩放图上: (640, 0)-(1280, 640)")
    print("patch_1在原图上: (1280, 0)-(2560, 1280)")
    print("\n检测框在patch内: (100, 150, 200, 250)")
    
    # 步骤1：将patch内坐标缩放到原图尺度
    print("\n步骤1: 将patch内坐标×2 (映射到原图尺度)")
    print("  (100, 150, 200, 250) × 2.0 = (200, 300, 400, 500)")
    
    # 步骤2：加上patch在原图中的偏移
    print("\n步骤2: 加上patch在原图的偏移 (1280, 0)")
    print("  (200, 300, 400, 500) + (1280, 0, 1280, 0)")
    print("  = (1480, 300, 1680, 500)")
    
    # 验证
    boxes_patch = torch.tensor([[100.0, 150.0, 200.0, 250.0]])
    patch_offset = (1280, 0)
    scale_factor = 2.0
    
    result = convert_to_global_coordinates(boxes_patch, patch_offset, scale_factor)
    
    expected = torch.tensor([[1480.0, 300.0, 1680.0, 500.0]])
    
    print(f"\n使用函数计算结果: {result.tolist()}")
    print(f"期望结果: {expected.tolist()}")
    
    if torch.allclose(result, expected, atol=1e-5):
        print("\n✅ 手动计算验证通过！")
        return True
    else:
        print("\n❌ 手动计算验证失败！")
        return False


def main():
    """主测试函数"""
    test1_passed = test_coordinate_mapping()
    test2_passed = test_manual_calculation()
    
    print("\n" + "="*70)
    print("测试结果汇总")
    print("="*70)
    print(f"坐标映射测试: {'✅ 通过' if test1_passed else '❌ 失败'}")
    print(f"手动计算验证: {'✅ 通过' if test2_passed else '❌ 失败'}")
    print("="*70)
    
    if test1_passed and test2_passed:
        print("\n🎉 所有测试通过！")
        print("\n修复已完成，系统可以正确处理两种情况：")
        print("  ✅ 图像尺寸 < 50000：坐标正确映射")
        print("  ✅ 图像尺寸 > 50000：坐标正确映射（先×scale_factor再加偏移）")
        return 0
    else:
        print("\n❌ 测试失败，请检查代码。")
        return 1


if __name__ == "__main__":
    sys.exit(main())

