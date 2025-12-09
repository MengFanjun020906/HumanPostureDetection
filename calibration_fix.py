#!/usr/bin/env python3
"""
标定参数修复工具
用于诊断和修复当前标定参数的问题：
1. valid_roi全零的问题
2. depth_range_m不合理的问题
"""

import numpy as np
import cv2
import yaml
import os
import argparse
from datetime import datetime


def load_calibration_data(yaml_file):
    """加载标定参数"""
    with open(yaml_file, 'r', encoding='utf-8') as f:
        # 使用load而不是safe_load来处理Python特定类型
        data = yaml.load(f, Loader=yaml.Loader)
    return data


def save_calibration_data(data, output_file):
    """保存标定参数"""
    with open(output_file, 'w', encoding='utf-8') as f:
        yaml.dump(data, f, default_flow_style=False)


def fix_valid_roi(data, image_size):
    """修复valid_roi全零的问题"""
    print("\n=== 修复valid_roi问题 ===")
    
    # 检查当前valid_roi
    valid_roi_left = data['rectification']['valid_roi_left']
    valid_roi_right = data['rectification']['valid_roi_right']
    
    print(f"当前valid_roi_left: {valid_roi_left}")
    print(f"当前valid_roi_right: {valid_roi_right}")
    
    # 判断是否全零
    if (np.array(valid_roi_left) == 0).all() or (np.array(valid_roi_right) == 0).all():
        print("发现全零valid_roi，需要重新计算")
        
        # 重新计算立体校正参数
        mtx_left = np.array(data['camera_matrix_left'])
        dist_left = np.array(data['distortion_coeffs_left'])
        mtx_right = np.array(data['camera_matrix_right'])
        dist_right = np.array(data['distortion_coeffs_right'])
        R = np.array(data['rotation_matrix'])
        T = np.array(data['translation_vector'])
        
        # 使用不同的alpha值重新计算
        alpha_values = [1.0, 0.9, 0.8, 0.5]
        best_valid_roi = None
        best_alpha = None
        
        for alpha in alpha_values:
            print(f"尝试alpha={alpha}...")
            R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
                mtx_left, dist_left, mtx_right, dist_right,
                image_size, R, T,
                alpha=alpha,  # 尝试不同的alpha值
                flags=cv2.CALIB_ZERO_DISPARITY
            )
            
            # 检查是否有有效的ROI
            if (np.array(validPixROI1) != 0).any() and (np.array(validPixROI2) != 0).any():
                print(f"✓ alpha={alpha} 成功生成有效ROI")
                print(f"  validPixROI1: {validPixROI1}")
                print(f"  validPixROI2: {validPixROI2}")
                best_valid_roi = (validPixROI1, validPixROI2)
                best_alpha = alpha
                break
        
        if best_valid_roi:
            # 更新数据
            data['rectification']['valid_roi_left'] = best_valid_roi[0]
            data['rectification']['valid_roi_right'] = best_valid_roi[1]
            data['rectification']['alpha'] = best_alpha
            print(f"✓ 使用alpha={best_alpha}修复了valid_roi")
            return True
        else:
            print("✗ 无法修复valid_roi，建议重新标定")
            return False
    else:
        print("✓ valid_roi已经有效，无需修复")
        return True


def fix_depth_range(data, actual_baseline_m, actual_fx):
    """修复depth_range_m不合理的问题"""
    print("\n=== 修复depth_range_m问题 ===")
    
    # 获取当前深度范围
    current_range = data.get('depth_range_m', [0, 0])
    print(f"当前depth_range_m: {current_range}")
    
    # 计算合理的深度范围
    # 基于实际使用场景（<10m）和视差范围（5-500像素）
    fx = actual_fx if actual_fx else np.mean([data['camera_matrix_left'][0][0], data['camera_matrix_right'][0][0]])
    baseline = actual_baseline_m if actual_baseline_m else data['baseline_m']
    
    # 合理的视差范围应该基于实际使用场景
    # 对于10m x 30m的场景，计算合适的视差范围
    max_distance = 30.0  # 最大距离30米
    min_distance = 0.5   # 最小距离0.5米
    
    # 计算对应的视差范围
    max_disparity = int(baseline * fx / min_distance)
    min_disparity = int(baseline * fx / max_distance)
    
    # 确保视差范围合理
    max_disparity = min(max_disparity, 1000)  # 限制最大视差
    min_disparity = max(min_disparity, 1)     # 确保最小视差至少为1
    
    # 计算新的深度范围
    new_min_depth = baseline * fx / max_disparity
    new_max_depth = baseline * fx / min_disparity
    
    print(f"\n基于实际场景计算新的深度范围：")
    print(f"  基线长度: {baseline:.4f} 米")
    print(f"  平均焦距: {fx:.1f} 像素")
    print(f"  视差范围: {min_disparity}-{max_disparity} 像素")
    print(f"  深度范围: {new_min_depth:.2f}-{new_max_depth:.2f} 米")
    
    # 更新数据
    data['depth_range_m'] = [float(new_min_depth), float(new_max_depth)]
    data['estimated_disparity_range'] = [min_disparity, max_disparity]
    print(f"✓ 修复了depth_range_m")
    return True


def diagnose_calibration(data, image_size=None):
    """诊断标定参数问题"""
    print("=== 标定参数诊断 ===")
    
    issues = []
    
    # 检查valid_roi
    valid_roi_left = data['rectification']['valid_roi_left']
    valid_roi_right = data['rectification']['valid_roi_right']
    
    if (np.array(valid_roi_left) == 0).all() or (np.array(valid_roi_right) == 0).all():
        issues.append("✗ valid_roi全零，立体校正可能失败")
    else:
        print(f"✓ valid_roi_left: {valid_roi_left}")
        print(f"✓ valid_roi_right: {valid_roi_right}")
    
    # 检查深度范围
    depth_range = data.get('depth_range_m', [0, 0])
    if depth_range[0] < 0.1 or depth_range[1] > 100.0:
        issues.append(f"✗ depth_range_m不合理: {depth_range} 米")
    else:
        print(f"✓ depth_range_m: {depth_range} 米")
    
    # 检查基线长度
    baseline = data['baseline_m']
    if baseline < 0.01 or baseline > 10.0:
        issues.append(f"✗ 基线长度不合理: {baseline:.4f} 米")
    else:
        print(f"✓ baseline_m: {baseline:.4f} 米")
    
    # 检查重投影误差
    rms = data['reprojection_error']['rms']
    if rms > 1.0:
        issues.append(f"✗ 重投影误差过大: {rms:.4f} 像素")
    else:
        print(f"✓ rms_error: {rms:.4f} 像素")
    
    print("\n=== 诊断结果 ===")
    if issues:
        for issue in issues:
            print(issue)
        return False
    else:
        print("✓ 所有参数都在合理范围内")
        return True


def main():
    parser = argparse.ArgumentParser(description='标定参数修复工具')
    parser.add_argument('--input', required=True, help='输入的标定YAML文件路径')
    parser.add_argument('--output', default='', help='输出的修复后YAML文件路径')
    parser.add_argument('--image_width', type=int, default=3840, help='图像宽度')
    parser.add_argument('--image_height', type=int, default=2160, help='图像高度')
    parser.add_argument('--actual_baseline', type=float, default=0, help='实际基线长度（米）')
    parser.add_argument('--actual_fx', type=float, default=0, help='实际焦距（像素）')
    args = parser.parse_args()
    
    # 加载标定数据
    print(f"加载标定数据: {args.input}")
    data = load_calibration_data(args.input)
    
    # 诊断问题
    image_size = (args.image_width, args.image_height)
    diagnose_calibration(data, image_size)
    
    # 修复问题
    print("\n=== 开始修复 ===")
    
    # 修复valid_roi
    valid_roi_fixed = fix_valid_roi(data, image_size)
    
    # 修复depth_range
    depth_fixed = fix_depth_range(data, args.actual_baseline, args.actual_fx)
    
    # 再次诊断
    print("\n=== 修复后诊断 ===")
    diagnose_calibration(data, image_size)
    
    # 保存修复后的数据
    output_file = args.output if args.output else args.input.replace('.yaml', '_fixed.yaml')
    save_calibration_data(data, output_file)
    print(f"\n✓ 修复后的标定数据已保存到: {output_file}")
    
    if valid_roi_fixed or depth_fixed:
        print("\n=== 修复建议 ===")
        if not valid_roi_fixed:
            print("✗ 无法修复valid_roi，建议：")
            print("  1. 重新进行标定，确保使用近距离数据")
            print("  2. 确保标定板在图像中有足够大的尺寸")
            print("  3. 使用正确的棋盘格尺寸")
        
        print("\n✓ 建议重新标定的注意事项：")
        print("  1. 使用近距离标定数据（0.5-10米）")
        print("  2. 覆盖实际使用场景的整个范围")
        print("  3. 确保标定板在图像的各个位置都有分布")
        print("  4. 使用足够数量的标定图像（至少15对）")
        print("  5. 确保图像清晰，没有运动模糊")
    

if __name__ == "__main__":
    main()
