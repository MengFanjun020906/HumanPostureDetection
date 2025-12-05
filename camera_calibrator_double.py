#!/usr/bin/env python3
"""
双目摄像头标定工具 - 完全修复畸变系数问题
================================================
特点:
- 彻底解决 OpenCV 畸变系数格式问题
- 智能处理单目标定结果集成
- 增强错误诊断和恢复机制
"""

import numpy as np
import cv2
import glob
import os
import re
import yaml
from pathlib import Path
import argparse
import json
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont


def put_text_cn(img, text, org, color=(0, 255, 0), font_size=28, font_path=None):
    """在OpenCV图像上绘制中文文本。"""
    if font_path is None:
        # Windows 常见中文字体
        font_paths = [
            "C:/Windows/Fonts/msyh.ttc",  # 微软雅黑
            "C:/Windows/Fonts/simsun.ttc",  # 宋体
            "C:/Windows/Fonts/msyhbd.ttc",  # 微软雅黑 Bold
            "C:/Windows/Fonts/simhei.ttf"   # 黑体
        ]
        for path in font_paths:
            if os.path.exists(path):
                font_path = path
                break
    
    # 转为PIL图像进行中文绘制
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            font = ImageFont.load_default()
        
        # PIL 使用RGB颜色
        rgb = (int(color[2]), int(color[1]), int(color[0]))
        draw.text((int(org[0]), int(org[1])), str(text), font=font, fill=rgb)
        return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"中文绘制失败: {e}, 退化为英文")
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_size/32.0, color, 2, cv2.LINE_AA)
        return img


def parse_args():
    parser = argparse.ArgumentParser(description='双目摄像头标定工具')
    parser.add_argument('--left', default='left', help='左相机图像目录')
    parser.add_argument('--right', default='right', help='右相机图像目录')
    parser.add_argument('--size', default='11x8', help='棋盘格内角点尺寸 (宽x高)')
    parser.add_argument('--square', type=float, default=0.025, help='棋盘格方格大小(米)')
    parser.add_argument('--alpha', type=float, default=0.8, help='立体校正alpha参数 (0.0-1.0)')
    parser.add_argument('--output', default='calibration_results_double', help='输出目录')
    parser.add_argument('--test', action='store_true', help='标定后立即测试校正效果')
    parser.add_argument('--single_calib_left', default='', help='左相机单目标定结果文件路径')
    parser.add_argument('--single_calib_right', default='', help='右相机单目标定结果文件路径')
    parser.add_argument('--fix_intrinsic', action='store_true', help='强制固定内参（当单目标定质量高时）')
    return parser.parse_args()


def load_single_calibration(calib_path):
    """加载单目标定结果 - 修复所有格式问题"""
    if not calib_path or not os.path.exists(calib_path):
        print(f"  ❌ 单目标定文件不存在: {calib_path}")
        return None
    
    print(f"\n加载单目标定结果: {calib_path}")
    
    try:
        # 尝试加载JSON
        if calib_path.endswith('.json'):
            with open(calib_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 提取相机矩阵
            camera_matrix = None
            for key in ['camera_matrix', 'intrinsic', 'K', 'mtx', 'cameraMatrix']:
                if key in data:
                    camera_matrix = np.array(data[key], dtype=np.float64)
                    break
            
            # 提取畸变系数
            dist_coeffs = None
            for key in ['distortion_coefficients', 'dist', 'distCoeffs', 'D', 'distortion']:
                if key in data:
                    dist_coeffs = np.array(data[key], dtype=np.float64)
                    break
            
            # 获取图像尺寸
            image_size = None
            if 'image_size' in data:
                size = data['image_size']
                if isinstance(size, dict):
                    image_size = (size.get('width', 0), size.get('height', 0))
                elif isinstance(size, list) or isinstance(size, tuple):
                    image_size = (size[0], size[1])
            
            # 获取重投影误差
            reprojection_error = None
            for key in ['reprojection_error', 'error', 'rms_error']:
                if key in data:
                    if isinstance(data[key], dict):
                        reprojection_error = data[key].get('mean', data[key].get('overall_mean', None))
                    else:
                        reprojection_error = data[key]
                    break
            
            if camera_matrix is None:
                print("  ❌ 未找到有效的相机矩阵")
                return None
            
            print(f"  ✅ 成功加载JSON格式标定结果")
            print(f"    相机矩阵:\n{camera_matrix}")
            if dist_coeffs is not None:
                print(f"    原始畸变系数形状: {dist_coeffs.shape}")
                print(f"    原始畸变系数值: {dist_coeffs.ravel()[:8]}...")
            if image_size:
                print(f"    图像尺寸: {image_size[0]}x{image_size[1]}")
            if reprojection_error:
                print(f"    重投影误差: {reprojection_error:.4f} 像素")
            
            return {
                'camera_matrix': camera_matrix,
                'dist_coeffs': dist_coeffs,
                'image_size': image_size,
                'reprojection_error': reprojection_error
            }
    
    except Exception as e:
        print(f"  ⚠️ 加载JSON失败: {e}")
    
    try:
        # 尝试加载YAML
        if calib_path.endswith('.yaml') or calib_path.endswith('.yml'):
            with open(calib_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            
            camera_matrix = None
            dist_coeffs = None
            
            # 处理不同的可能结构
            if 'camera_matrix_left' in data:
                camera_matrix = np.array(data['camera_matrix_left'])
            elif 'camera_matrix' in data:
                camera_matrix = np.array(data['camera_matrix'])
            elif 'intrinsics' in data:
                camera_matrix = np.array(data['intrinsics'])
            
            if 'distortion_coeffs_left' in data:
                dist_coeffs = np.array(data['distortion_coeffs_left'])
            elif 'distortion_coefficients' in data:
                dist_coeffs = np.array(data['distortion_coefficients'])
            elif 'dist_coeffs' in data:
                dist_coeffs = np.array(data['dist_coeffs'])
            
            if camera_matrix is not None and camera_matrix.size > 0:
                print(f"  ✅ 成功加载YAML格式标定结果")
                return {
                    'camera_matrix': camera_matrix,
                    'dist_coeffs': dist_coeffs
                }
    except Exception as e:
        print(f"  ⚠️ 加载YAML失败: {e}")
    
    try:
        # 尝试加载XML
        if calib_path.endswith('.xml'):
            fs = cv2.FileStorage(calib_path, cv2.FILE_STORAGE_READ)
            if fs.isOpened():
                camera_matrix = fs.getNode("camera_matrix").mat()
                if camera_matrix is None or camera_matrix.size == 0:
                    camera_matrix = fs.getNode("intrinsic").mat()
                if camera_matrix is None or camera_matrix.size == 0:
                    camera_matrix = fs.getNode("K").mat()
                
                dist_coeffs = fs.getNode("distortion_coefficients").mat()
                if dist_coeffs is None or dist_coeffs.size == 0:
                    dist_coeffs = fs.getNode("dist").mat()
                if dist_coeffs is None or dist_coeffs.size == 0:
                    dist_coeffs = fs.getNode("D").mat()
                
                fs.release()
                
                if camera_matrix is not None and camera_matrix.size > 0:
                    print(f"  ✅ 成功加载XML格式标定结果")
                    return {
                        'camera_matrix': camera_matrix,
                        'dist_coeffs': dist_coeffs
                    }
    except Exception as e:
        print(f"  ⚠️ 加载XML失败: {e}")
    
    try:
        # 尝试加载NPZ
        if calib_path.endswith('.npz'):
            data = np.load(calib_path)
            camera_matrix = None
            dist_coeffs = None
            
            if 'camera_matrix' in data:
                camera_matrix = data['camera_matrix']
            elif 'K' in data:
                camera_matrix = data['K']
            
            if 'distortion_coefficients' in data:
                dist_coeffs = data['distortion_coefficients']
            elif 'D' in data:
                dist_coeffs = data['D']
            
            if camera_matrix is not None:
                print(f"  ✅ 成功加载NPZ格式标定结果")
                return {
                    'camera_matrix': camera_matrix,
'dist_coeffs': dist_coeffs
                }
    except Exception as e:
        print(f"  ⚠️ 加载NPZ失败: {e}")
    
    print(f"  ❌ 未找到有效的单目标定结果，将使用独立标定")
    return None


def pair_images(left_dir, right_dir):
    """基于文件名序号智能配对图像"""
    left_files = glob.glob(os.path.join(left_dir, '*.jpg')) + glob.glob(os.path.join(left_dir, '*.png'))
    right_files = glob.glob(os.path.join(right_dir, '*.jpg')) + glob.glob(os.path.join(right_dir, '*.png'))
    
    if not left_files or not right_files:
        raise ValueError(f"未找到图像! 检查目录: left='{left_dir}', right='{right_dir}'")
    
    print(f"\n找到图像: 左={len(left_files)}, 右={len(right_files)}")
    
    # 提取序号 (支持多种命名格式)
    def extract_index(filename):
        basename = os.path.basename(filename)
        # 尝试匹配数字序号
        match = re.search(r'(\d+)[^\d]*$', os.path.splitext(basename)[0])
        if match:
            return int(match.group(1))
        # 尝试匹配时间戳
        match = re.search(r'(\d{8}_\d{6})', basename)
        if match:
            return int(''.join(filter(str.isdigit, match.group(1))))
        return -1
    
    left_pairs = [(extract_index(f), f) for f in left_files]
    right_pairs = [(extract_index(f), f) for f in right_files]
    
    # 过滤无效序号
    left_pairs = [(idx, f) for idx, f in left_pairs if idx != -1]
    right_pairs = [(idx, f) for idx, f in right_pairs if idx != -1]
    
    if not left_pairs or not right_pairs:
        raise ValueError("无法从文件名提取有效序号! 请重命名图像为 left_001.jpg, right_001.jpg 格式")
    
    # 按序号排序
    left_pairs.sort(key=lambda x: x[0])
    right_pairs.sort(key=lambda x: x[0])
    
    # 创建字典
    left_dict = {idx: f for idx, f in left_pairs}
    right_dict = {idx: f for idx, f in right_pairs}
    
    # 找共同序号
    common_indices = sorted(set(left_dict.keys()) & set(right_dict.keys()))
    paired_images = [(left_dict[i], right_dict[i]) for i in common_indices]
    
    print(f"成功配对 {len(paired_images)} 对图像")
    for i, (l, r) in enumerate(paired_images[:5]):
        print(f"  示例对 {i+1}: {os.path.basename(l)} ↔ {os.path.basename(r)}")
    if len(paired_images) > 5:
        print(f"  ... 及 {len(paired_images)-5} 对")
    
    return paired_images


def check_chessboard_consistency(corners_left, corners_right, chessboard_size):
    """
    检查左右相机检测到的棋盘格是否满足一致性
    
    参数:
        corners_left: 左相机检测到的角点，形状为(N, 1, 2)
        corners_right: 右相机检测到的角点，形状为(N, 1, 2)
        chessboard_size: 棋盘格尺寸 (width, height)
    
    返回:
        consistent: 是否一致
        consistency_info: 一致性详细信息
    """
    consistency_info = {
        "角点数量": True,
        "角点排列": True,
        "棋盘格形状": True,
        "扭曲程度": True,
        "详细信息": []
    }
    
    # 首先处理角点数组的维度，确保它们是(N, 2)的形状
    try:
        corners_left_2d = corners_left.reshape(-1, 2)
        corners_right_2d = corners_right.reshape(-1, 2)
    except Exception as e:
        consistency_info["详细信息"].append(f"角点数组维度处理错误: {e}")
        # 如果无法处理维度，返回不一致
        return False, consistency_info
    
    # 1. 检查角点数量
    if len(corners_left_2d) != len(corners_right_2d):
        consistency_info["角点数量"] = False
        consistency_info["详细信息"].append(f"左右相机检测到的角点数量不一致 (左: {len(corners_left_2d)}, 右: {len(corners_right_2d)})")
    
    # 2. 检查棋盘格形状
    expected_points = chessboard_size[0] * chessboard_size[1]
    if len(corners_left_2d) != expected_points:
        consistency_info["棋盘格形状"] = False
        consistency_info["详细信息"].append(f"左相机检测到的角点数量({len(corners_left_2d)})与预期({expected_points})不符")
    
    if len(corners_right_2d) != expected_points:
        consistency_info["棋盘格形状"] = False
        consistency_info["详细信息"].append(f"右相机检测到的角点数量({len(corners_right_2d)})与预期({expected_points})不符")
    
    # 3. 检查角点排列 (计算行和列的平均间隔，检查是否一致)
    if len(corners_left_2d) >= 4 and len(corners_right_2d) >= 4:
        try:
            # 计算左相机角点的行间隔
            left_corners = corners_left_2d.reshape(chessboard_size[1], chessboard_size[0], 2)
            left_row_distances = []
            for i in range(chessboard_size[1]-1):
                for j in range(chessboard_size[0]):
                    pt1 = left_corners[i, j]
                    pt2 = left_corners[i+1, j]
                    distance = np.linalg.norm(pt1 - pt2)
                    left_row_distances.append(distance)
            left_row_mean = np.mean(left_row_distances) if left_row_distances else 0
            left_row_std = np.std(left_row_distances) if left_row_distances else 0
            
            # 计算左相机角点的列间隔
            left_col_distances = []
            for i in range(chessboard_size[1]):
                for j in range(chessboard_size[0]-1):
                    pt1 = left_corners[i, j]
                    pt2 = left_corners[i, j+1]
                    distance = np.linalg.norm(pt1 - pt2)
                    left_col_distances.append(distance)
            left_col_mean = np.mean(left_col_distances) if left_col_distances else 0
            left_col_std = np.std(left_col_distances) if left_col_distances else 0
            
            # 计算右相机角点的行间隔
            right_corners = corners_right_2d.reshape(chessboard_size[1], chessboard_size[0], 2)
            right_row_distances = []
            for i in range(chessboard_size[1]-1):
                for j in range(chessboard_size[0]):
                    pt1 = right_corners[i, j]
                    pt2 = right_corners[i+1, j]
                    distance = np.linalg.norm(pt1 - pt2)
                    right_row_distances.append(distance)
            right_row_mean = np.mean(right_row_distances) if right_row_distances else 0
            right_row_std = np.std(right_row_distances) if right_row_distances else 0
            
            # 计算右相机角点的列间隔
            right_col_distances = []
            for i in range(chessboard_size[1]):
                for j in range(chessboard_size[0]-1):
                    pt1 = right_corners[i, j]
                    pt2 = right_corners[i, j+1]
                    distance = np.linalg.norm(pt1 - pt2)
                    right_col_distances.append(distance)
            right_col_mean = np.mean(right_col_distances) if right_col_distances else 0
            right_col_std = np.std(right_col_distances) if right_col_distances else 0
            
            # 检查行间隔一致性
            if left_row_mean > 0 and left_row_std > left_row_mean * 0.3:
                consistency_info["角点排列"] = False
                consistency_info["详细信息"].append("左相机棋盘格行间隔不一致，可能存在扭曲")
            
            if right_row_mean > 0 and right_row_std > right_row_mean * 0.3:
                consistency_info["角点排列"] = False
                consistency_info["详细信息"].append("右相机棋盘格行间隔不一致，可能存在扭曲")
            
            # 检查列间隔一致性
            if left_col_mean > 0 and left_col_std > left_col_mean * 0.3:
                consistency_info["角点排列"] = False
                consistency_info["详细信息"].append("左相机棋盘格列间隔不一致，可能存在扭曲")
            
            if right_col_mean > 0 and right_col_std > right_col_mean * 0.3:
                consistency_info["角点排列"] = False
                consistency_info["详细信息"].append("右相机棋盘格列间隔不一致，可能存在扭曲")
            
            # 检查左右相机之间的间隔一致性
            if left_row_mean > 0 and right_row_mean > 0:
                row_ratio = abs(left_row_mean - right_row_mean) / max(left_row_mean, right_row_mean)
                if row_ratio > 0.1:
                    consistency_info["角点排列"] = False
                    consistency_info["详细信息"].append("左右相机棋盘格行间隔差异较大，可能存在尺度不一致")
            
            if left_col_mean > 0 and right_col_mean > 0:
                col_ratio = abs(left_col_mean - right_col_mean) / max(left_col_mean, right_col_mean)
                if col_ratio > 0.1:
                    consistency_info["角点排列"] = False
                    consistency_info["详细信息"].append("左右相机棋盘格列间隔差异较大，可能存在尺度不一致")
            
            # 4. 检查扭曲程度 (计算棋盘格的透视变换)
            # 计算左相机的扭曲程度
            try:
                # 取四个角的点
                left_corners_4 = np.array([
                    left_corners[0, 0],
                    left_corners[0, -1],
                    left_corners[-1, -1],
                    left_corners[-1, 0]
                ], dtype=np.float32)
                
                # 计算矩形区域
                scale_factor = max(left_row_mean, left_col_mean) if max(left_row_mean, left_col_mean) > 0 else 1.0
                expected_rect = np.array([
                    [0, 0],
                    [chessboard_size[0]-1, 0],
                    [chessboard_size[0]-1, chessboard_size[1]-1],
                    [0, chessboard_size[1]-1]
                ], dtype=np.float32) * scale_factor
                
                # 计算透视变换矩阵
                M_left = cv2.getPerspectiveTransform(left_corners_4, expected_rect)
                
                # 计算透视变换后的点与原始点的误差
                left_transformed = cv2.perspectiveTransform(left_corners_4.reshape(1, -1, 2), M_left)
                left_transform_error = np.mean(np.abs(left_transformed - expected_rect)) / scale_factor
                
                if left_transform_error > 0.2:
                    consistency_info["扭曲程度"] = False
                    consistency_info["详细信息"].append("左相机棋盘格扭曲程度较大，可能影响标定精度")
            except Exception as e:
                consistency_info["详细信息"].append(f"计算左相机扭曲程度时出错: {e}")
            
            # 计算右相机的扭曲程度
            try:
                # 取四个角的点
                right_corners_4 = np.array([
                    right_corners[0, 0],
                    right_corners[0, -1],
                    right_corners[-1, -1],
                    right_corners[-1, 0]
                ], dtype=np.float32)
                
                # 计算矩形区域
                scale_factor_right = max(right_row_mean, right_col_mean) if max(right_row_mean, right_col_mean) > 0 else 1.0
                expected_rect_right = np.array([
                    [0, 0],
                    [chessboard_size[0]-1, 0],
                    [chessboard_size[0]-1, chessboard_size[1]-1],
                    [0, chessboard_size[1]-1]
                ], dtype=np.float32) * scale_factor_right
                
                # 计算透视变换矩阵
                M_right = cv2.getPerspectiveTransform(right_corners_4, expected_rect_right)
                
                # 计算透视变换后的点与原始点的误差
                right_transformed = cv2.perspectiveTransform(right_corners_4.reshape(1, -1, 2), M_right)
                right_transform_error = np.mean(np.abs(right_transformed - expected_rect_right)) / scale_factor_right
                
                if right_transform_error > 0.2:
                    consistency_info["扭曲程度"] = False
                    consistency_info["详细信息"].append("右相机棋盘格扭曲程度较大，可能影响标定精度")
            except Exception as e:
                consistency_info["详细信息"].append(f"计算右相机扭曲程度时出错: {e}")
        except Exception as e:
            consistency_info["详细信息"].append(f"角点排列或扭曲程度检查错误: {e}")
            # 如果出现错误，将角点排列和扭曲程度标记为不一致
            consistency_info["角点排列"] = False
            consistency_info["扭曲程度"] = False
    else:
        # 如果角点数量不足4个，无法进行排列和扭曲检查
        consistency_info["角点排列"] = False
        consistency_info["扭曲程度"] = False
        consistency_info["详细信息"].append(f"角点数量不足，无法进行排列和扭曲检查 (左: {len(corners_left_2d)}, 右: {len(corners_right_2d)})")
    
    # 综合判断是否一致
    consistent = (consistency_info["角点数量"] and 
                 consistency_info["角点排列"] and 
                 consistency_info["棋盘格形状"] and 
                 consistency_info["扭曲程度"])
    
    return consistent, consistency_info


def compute_epipolar_error(objpoints, imgpoints_left, imgpoints_right, 
                          mtx_left, dist_left, mtx_right, dist_right,
                          R1, R2, P1, P2):
    """
    计算校正后的平均极线误差 (像素)
    """
    total_error = 0.0
    total_points = 0
    
    for i in range(len(objpoints)):
        # 校正左图点
        pts_left = imgpoints_left[i]
        pts_left_rect = cv2.undistortPoints(pts_left, mtx_left, dist_left, R=R1, P=P1)
        
        # 校正右图点
        pts_right = imgpoints_right[i]
        pts_right_rect = cv2.undistortPoints(pts_right, mtx_right, dist_right, R=R2, P=P2)
        
        # 计算y坐标差 (极线误差)
        for j in range(len(pts_left_rect)):
            pt_l = pts_left_rect[j, 0]
            pt_r = pts_right_rect[j, 0]
            error = abs(pt_l[1] - pt_r[1])  # y坐标差
            total_error += error
            total_points += 1
    
    mean_error = total_error / total_points if total_points > 0 else float('inf')
    return mean_error, total_points


def safe_calibrate_camera(object_points, image_points, image_size, camera_matrix=None, dist_coeffs=None, flags=0):
    """
    安全的单目标定函数，处理畸变系数格式问题
    """
    # 如果提供了畸变系数，但格式不正确，我们忽略它并让OpenCV创建
    if dist_coeffs is not None and dist_coeffs.size > 0:
        # 检查畸变系数是否符合flags要求
        max_coeffs = 14  # OpenCV 4.x 最大支持14个畸变系数
        
        # 标准化畸变系数
        dist_coeffs = np.array(dist_coeffs, dtype=np.float64).ravel()
        
        if len(dist_coeffs) > max_coeffs:
            print(f"  ⚠️ 畸变系数数量 {len(dist_coeffs)} 超过最大值 {max_coeffs}，截断")
            dist_coeffs = dist_coeffs[:max_coeffs]
        
        # 重塑为1xN
        dist_coeffs = dist_coeffs.reshape(1, -1)
        print(f"  标准化后畸变系数形状: {dist_coeffs.shape}")
    else:
        # 不提供畸变系数，让OpenCV创建
        dist_coeffs = None
    
    # 如果提供了相机矩阵，确保它是3x3
    if camera_matrix is not None:
        camera_matrix = np.array(camera_matrix, dtype=np.float64)
        if camera_matrix.shape != (3, 3):
            print(f"  ⚠️ 相机矩阵形状 {camera_matrix.shape} 不正确，需要3x3，重置为None")
            camera_matrix = None
        else:
            flags |= cv2.CALIB_USE_INTRINSIC_GUESS
    
    # 根据flags调整畸变系数数量
    if flags & cv2.CALIB_RATIONAL_MODEL:
        max_dist_size = 8
    elif flags & cv2.CALIB_THIN_PRISM_MODEL:
        max_dist_size = 12
    else:
        max_dist_size = 5  # 标准模型
    
    # 如果我们提供了畸变系数，但数量超过当前flags支持，截断
    if dist_coeffs is not None and dist_coeffs.shape[1] > max_dist_size:
        print(f"  ⚠️ 截断畸变系数从 {dist_coeffs.shape[1]} 到 {max_dist_size} 以匹配标定标志")
        dist_coeffs = dist_coeffs[:, :max_dist_size].copy()
    
    # 安全调用calibrateCamera
    try:
        print("  尝试使用提供的初始值进行标定...")
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            object_points, 
            image_points, 
            image_size,
            camera_matrix,
            dist_coeffs,
            flags=flags,
            criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        )
        print(f"  ✅ 标定成功! RMS误差: {ret:.4f}")
        return ret, mtx, dist, rvecs, tvecs
    except cv2.error as e:
        print(f"  ❌ 标定失败: {e}")
        
        # 尝试简化模型
        print("  尝试简化畸变模型...")
        simple_flags = flags & ~(cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_THIN_PRISM_MODEL)
        
        try:
            ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                object_points, 
                image_points, 
                image_size,
                camera_matrix,
                None,  # 不提供畸变系数
                flags=simple_flags,
                criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
            )
            print(f"  ✅ 简化模型标定成功! RMS误差: {ret:.4f}")
            return ret, mtx, dist, rvecs, tvecs
        except cv2.error as e2:
            print(f"  ❌ 简化模型也失败: {e2}")
            
            # 最后尝试：完全不使用初始值
            print("  尝试完全不使用初始值...")
            try:
                ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                    object_points, 
                    image_points, 
                    image_size,
                    None,  # 不提供相机矩阵
                    None,  # 不提供畸变系数
                    flags=simple_flags,
                    criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
                )
                print(f"  ✅ 无初始值标定成功! RMS误差: {ret:.4f}")
                return ret, mtx, dist, rvecs, tvecs
            except cv2.error as e3:
                print(f"  ❌ 所有尝试都失败: {e3}")
                raise


def stereo_calibration(args):
    """主标定函数"""
    print("="*60)
    print("双目摄像头标定工具 - 畸变系数完全修复版")
    print("="*60)
    
    # 解析棋盘格尺寸
    try:
        chessboard_size = tuple(map(int, args.size.split('x')))
        assert len(chessboard_size) == 2
    except Exception as e:
        raise ValueError(f"无效的棋盘格尺寸! 格式应为 '9x6', 错误: {e}")
    
    print(f"配置:")
    print(f"  棋盘格: {chessboard_size[0]}x{chessboard_size[1]} 内角点")
    print(f"  方格尺寸: {args.square*1000:.1f} mm")
    print(f"  立体校正 alpha: {args.alpha:.2f} (0=裁剪最大, 1=保留全部)")
    print(f"  输出目录: '{args.output}'")
    print(f"  强制固定内参: {'是' if args.fix_intrinsic else '否'}")
    
    # 加载单目标定结果
    single_calib_left = None
    single_calib_right = None
    
    if args.single_calib_left.strip():
        single_calib_left = load_single_calibration(args.single_calib_left.strip())
    
    if args.single_calib_right.strip():
        single_calib_right = load_single_calibration(args.single_calib_right.strip())
    
    # 创建输出目录
    os.makedirs(args.output, exist_ok=True)
    
    # 配对图像
    paired_images = pair_images(args.left, args.right)
    if len(paired_images) < 10:
        print(f"⚠️ 警告: 仅 {len(paired_images)} 对图像，建议至少15对以获得更好精度")
    
    # 准备对象点
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= args.square  # 转换为物理单位(米)
    
    # 存储点
    objpoints = []  # 3D点
    imgpoints_left = []  # 左相机2D点
    imgpoints_right = []  # 右相机2D点
    
    # 处理每对图像
    print("\n" + "-"*50)
    print("检测棋盘格角点...")
    print("-"*50)
    
    valid_pairs = 0
    for idx, (left_path, right_path) in enumerate(paired_images):
        # 读取图像
        img_left = cv2.imread(left_path)
        img_right = cv2.imread(right_path)
        
        if img_left is None or img_right is None:
            print(f"  跳过对 {idx+1}: 无法读取图像 ({left_path}, {right_path})")
            continue
        
        h, w = img_left.shape[:2]
        
        # 转灰度
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        
        # 查找角点
        ret_left, corners_left = cv2.findChessboardCorners(
            gray_left, chessboard_size, 
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | 
                  cv2.CALIB_CB_FAST_CHECK | 
                  cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        ret_right, corners_right = cv2.findChessboardCorners(
            gray_right, chessboard_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | 
                  cv2.CALIB_CB_FAST_CHECK | 
                  cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        # 亚像素精化
        if ret_left and ret_right:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_left_refined = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
            corners_right_refined = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
            
            # 检查棋盘格一致性
            consistent, consistency_info = check_chessboard_consistency(
                corners_left_refined, corners_right_refined, chessboard_size
            )
            
            objpoints.append(objp)
            imgpoints_left.append(corners_left_refined)
            imgpoints_right.append(corners_right_refined)
            valid_pairs += 1
            
            print(f"  对 {idx+1}: 成功检测角点 (累计: {valid_pairs})")
            
            # 输出一致性检查结果
            if consistent:
                print(f"    ✅ 棋盘格一致性: 满足要求")
            else:
                print(f"    ❌ 棋盘格一致性: 不满足要求")
                for info in consistency_info["详细信息"]:
                    print(f"    - {info}")
        else:
            print(f"  对 {idx+1}: 角点检测失败")
    
    if valid_pairs < 5:
        print(f"❌ 错误: 仅 {valid_pairs} 对有效图像，需要至少5对")
        return None
    
    print(f"\n✅ 成功检测 {valid_pairs} 对图像的角点")
    image_size = (w, h)
    
    # 单目优化标志
    calib_flags = cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_THIN_PRISM_MODEL
    
    # 左相机标定
    print("\n" + "-"*50)
    print("左相机标定...")
    print("-"*50)
    
    if single_calib_left:
        print("  使用单目标定结果作为初始值")
        camera_matrix_init = single_calib_left['camera_matrix']
        dist_coeffs_init = single_calib_left.get('dist_coeffs', None)
        
        # 验证初始值
        if camera_matrix_init.shape != (3, 3):
            print(f"  ⚠️ 左相机矩阵形状 {camera_matrix_init.shape} 不正确，重置为None")
            camera_matrix_init = None
    else:
        camera_matrix_init = None
        dist_coeffs_init = None
    
    # 安全标定
    ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = safe_calibrate_camera(
        objpoints, imgpoints_left, image_size,
        camera_matrix_init, dist_coeffs_init,
        flags=calib_flags
    )
    print(f"  最终畸变系数形状: {dist_left.shape}")
    print(f"  最终畸变系数值: {dist_left.ravel()[:8]}...")
    
    # 右相机标定
    print("\n" + "-"*50)
    print("右相机标定...")
    print("-"*50)
    
    if single_calib_right:
        print("  使用单目标定结果作为初始值")
        camera_matrix_init = single_calib_right['camera_matrix']
        dist_coeffs_init = single_calib_right.get('dist_coeffs', None)
        
        # 验证初始值
        if camera_matrix_init.shape != (3, 3):
            print(f"  ⚠️ 右相机矩阵形状 {camera_matrix_init.shape} 不正确，重置为None")
            camera_matrix_init = None
    else:
        camera_matrix_init = None
        dist_coeffs_init = None
    
    ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = safe_calibrate_camera(
        objpoints, imgpoints_right, image_size,
        camera_matrix_init, dist_coeffs_init,
        flags=calib_flags
    )
    print(f"  最终畸变系数形状: {dist_right.shape}")
    print(f"  最终畸变系数值: {dist_right.ravel()[:8]}...")
    
    # 立体标定策略
    print("\n" + "-"*50)
    print("立体标定策略选择...")
    print("-"*50)
    
    stereo_flags = cv2.CALIB_USE_INTRINSIC_GUESS
    
    # 基于单目标定质量决定策略
    use_quality_check = False
    if single_calib_left and single_calib_right:
        left_error = single_calib_left.get('reprojection_error', None)
        right_error = single_calib_right.get('reprojection_error', None)
        
        if left_error and right_error:
            use_quality_check = True
            print(f"  单目标定质量: 左={left_error:.4f} 像素, 右={right_error:.4f} 像素")
            
            if args.fix_intrinsic or (left_error < 0.3 and right_error < 0.3):
                stereo_flags |= cv2.CALIB_FIX_INTRINSIC
                print("  ✅ 使用高质量单目标定，固定内参")
            else:
                print("  ⚠️ 单目标定质量一般，允许优化内参")
        else:
            print("  ℹ️ 无法获取单目标定质量，仅使用初始值")
    
    if not use_quality_check:
        if args.fix_intrinsic:
            stereo_flags |= cv2.CALIB_FIX_INTRINSIC
            print("  ✅ 强制固定内参 (命令行参数指定)")
        else:
            print("  ℹ️ 没有单目标定或质量未知，仅使用初始值")
    
    # 添加模型标志
    stereo_flags |= cv2.CALIB_RATIONAL_MODEL
    
    print(f"  使用立体标定标志: {stereo_flags}")
    
    # 立体标定 - 安全调用
    print("\n" + "-"*50)
    print("立体标定...")
    print("-"*50)
    
    try:
        ret, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
            objpoints, 
            imgpoints_left, 
            imgpoints_right,
            mtx_left, 
            dist_left, 
            mtx_right, 
            dist_right,
            image_size, 
            flags=stereo_flags,
            criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        )
        print(f"  ✅ 立体标定成功! RMS误差: {ret:.4f} 像素")
    except cv2.error as e:
        print(f"  ❌ 立体标定失败: {e}")
        print("  尝试降级策略...")
        
        # 尝试降级策略
        fallback_flags = cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_USE_INTRINSIC_GUESS
        
        try:
            ret, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
                objpoints, imgpoints_left, imgpoints_right,
                mtx_left, dist_left, mtx_right, dist_right,
                image_size, flags=fallback_flags,
                criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
            )
            print(f"  ✅ 降级标定成功! RMS误差: {ret:.4f}")
        except cv2.error as e2:
            print(f"  ❌ 降级标定也失败: {e2}")
            print("  尝试最简模型...")
            
            try:
                # 最简模型
                simple_flags = cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_USE_INTRINSIC_GUESS
                simple_flags &= ~cv2.CALIB_RATIONAL_MODEL
                
                ret, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
                    objpoints, imgpoints_left, imgpoints_right,
                    mtx_left, dist_left[:, :5],  # 仅用前5个系数
                    mtx_right, dist_right[:, :5],  # 仅用前5个系数
                    image_size, flags=simple_flags,
                    criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
                )
                print(f"  ✅ 最简模型标定成功! RMS误差: {ret:.4f}")
            except cv2.error as e3:
                print(f"  ❌ 所有策略都失败: {e3}")
                return None
    
    # 诊断：检查立体标定结果
    baseline = np.linalg.norm(T)
    print(f"\n  诊断信息:")
    print(f"    基线长度: {baseline:.4f} 米")
    if baseline < 0.01:
        print(f"    ❌ 基线过短! 可能左右图像未正确配对或相机位置异常")
    elif baseline > 1.0:
        print(f"    ⚠️ 基线过长! 可能图像配对错误")
    else:
        print(f"    ✓ 基线长度合理")
    
    # 检查旋转矩阵
    # 添加重投影计算
    if len(objpoints) > 0:
        # 计算左相机的重投影点
        reprojected_left, _ = cv2.projectPoints(
            objpoints[0], rvecs_left[0], tvecs_left[0], mtx_left, dist_left
        )
        # 计算右相机的重投影点
        reprojected_right, _ = cv2.projectPoints(
            objpoints[0], rvecs_right[0], tvecs_right[0], mtx_right, dist_right
        )
    
    # 计算立体三维重投影误差
    stereo_3d_errors = []
    axis_errors_x = []  # 新增：X轴误差数组
    axis_errors_y = []  # 新增：Y轴误差数组
    axis_errors_z = []  # 新增：Z轴误差数组
    
    for j in range(len(objpoints[0])):
        # 左图像误差
        detected_left = imgpoints_left[0][j, 0]
        reproj_left = reprojected_left[j, 0]
        error_left = np.linalg.norm(detected_left - reproj_left)
        
        # 右图像误差
        detected_right = imgpoints_right[0][j, 0]
        reproj_right = reprojected_right[j, 0]
        error_right = np.linalg.norm(detected_right - reproj_right)
        
        # 立体三维误差（取左右误差的平均值）
        stereo_error = (error_left + error_right) / 2
        stereo_3d_errors.append(stereo_error)
        
        # 计算各轴误差
        error_left_x = abs(detected_left[0] - reproj_left[0])
        error_left_y = abs(detected_left[1] - reproj_left[1])
        error_right_x = abs(detected_right[0] - reproj_right[0])
        error_right_y = abs(detected_right[1] - reproj_right[1])
        
        # 平均X/Y轴误差
        error_x = (error_left_x + error_right_x) / 2
        error_y = (error_left_y + error_right_y) / 2
        
        # 计算视差和深度误差
        disparity_detected = detected_right[0] - detected_left[0]  # 检测视差
        disparity_reproj = reproj_right[0] - reproj_left[0]        # 重投影视差
        error_z = abs(disparity_detected - disparity_reproj)       # 深度误差（视差误差）
        
        # 收集轴误差数据用于统计
        axis_errors_x.append(error_x)
        axis_errors_y.append(error_y)
        axis_errors_z.append(error_z)
        
        # 显示前10个点的详细误差
        # if j < 10:
        #     print(f"  点 {j+1}:")
        #     print(f"    左检测位置: ({detected_left[0]:.1f}, {detected_left[1]:.1f})")
        #     print(f"    左重投影位置: ({reproj_left[0]:.1f}, {reproj_left[1]:.1f})")
        #     print(f"    左误差: {error_left:.4f} 像素")
        #     print(f"    左X轴误差: {error_left_x:.4f} 像素")
        #     print(f"    左Y轴误差: {error_left_y:.4f} 像素")
            
        #     print(f"    右检测位置: ({detected_right[0]:.1f}, {detected_right[1]:.1f})")
        #     print(f"    右重投影位置: ({reproj_right[0]:.1f}, {reproj_right[1]:.1f})")
        #     print(f"    右误差: {error_right:.4f} 像素")
        #     print(f"    右X轴误差: {error_right_x:.4f} 像素")
        #     print(f"    右Y轴误差: {error_right_y:.4f} 像素")
            
        #     print(f"    立体三维误差: {stereo_error:.4f} 像素")
        #     print(f"    平均X轴误差: {error_x:.4f} 像素")
        #     print(f"    平均Y轴误差: {error_y:.4f} 像素")
        #     print(f"    深度(Z轴)误差: {error_z:.4f} 像素")
            
        #     # 视差信息
        #     print(f"    检测视差: {disparity_detected:.1f} 像素")
        #     print(f"    重投影视差: {disparity_reproj:.1f} 像素")
        #     print(f"    视差误差: {error_z:.4f} 像素")
            
        #     # 误差方向分析
        #     print(f"    误差方向分析:")
        #     if error_x > error_y and error_x > error_z:
        #         print(f"      → 主要误差在X轴（水平方向）")
        #     elif error_y > error_x and error_y > error_z:
        #         print(f"      → 主要误差在Y轴（垂直方向）")
        #     elif error_z > error_x and error_z > error_y:
        #         print(f"      → 主要误差在Z轴（深度方向）")
        #     else:
        #         print(f"      → 误差分布相对均匀")
            
        #     print(f"    {'-'*60}")
    
    # 统计信息
    stereo_errors = np.array(stereo_3d_errors)
    axis_errors_x = np.array(axis_errors_x)  # 转换为numpy数组
    axis_errors_y = np.array(axis_errors_y)  # 转换为numpy数组
    axis_errors_z = np.array(axis_errors_z)  # 转换为numpy数组
    
    # print(f"\n【立体三维重投影误差统计】")
    # print(f"  平均误差: {np.mean(stereo_errors):.4f} 像素")
    # print(f"  标准差: {np.std(stereo_errors):.4f} 像素")
    # print(f"  最小误差: {np.min(stereo_errors):.4f} 像素")
    # print(f"  最大误差: {np.max(stereo_errors):.4f} 像素")
    
    # 轴误差统计
    # print(f"\n【轴误差分析】")
    # print(f"  X轴（水平）误差: {np.mean(axis_errors_x):.4f} ± {np.std(axis_errors_x):.4f} 像素")
    # print(f"  Y轴（垂直）误差: {np.mean(axis_errors_y):.4f} ± {np.std(axis_errors_y):.4f} 像素")
    # print(f"  Z轴（深度）误差: {np.mean(axis_errors_z):.4f} ± {np.std(axis_errors_z):.4f} 像素")
    
    # 误差贡献度分析
    total_error = np.mean(stereo_errors)
    x_contribution = np.mean(axis_errors_x) / total_error * 100
    y_contribution = np.mean(axis_errors_y) / total_error * 100
    z_contribution = np.mean(axis_errors_z) / total_error * 100
    
    # print(f"\n【误差贡献度】")
    # print(f"  X轴贡献度: {x_contribution:.1f}%")
    # print(f"  Y轴贡献度: {y_contribution:.1f}%")
    # print(f"  Z轴贡献度: {z_contribution:.1f}%")
    
    # 诊断建议
    # print(f"\n【诊断建议】")
    # if np.mean(axis_errors_z) > np.mean(axis_errors_x) and np.mean(axis_errors_z) > np.mean(axis_errors_y):
    #     print(f"  ⚠️ 主要问题在深度方向（Z轴）")
    #     print(f"    可能原因: 基线长度估计不准确，外参T向量有问题")
    # elif np.mean(axis_errors_x) > np.mean(axis_errors_y):
    #     print(f"  ⚠️ 主要问题在水平方向（X轴）")
    #     print(f"    可能原因: 旋转矩阵R不准确，极线几何约束失效")
    # else:
    #     print(f"  ⚠️ 主要问题在垂直方向（Y轴）")
    #     print(f"    可能原因: 相机未水平对齐，图像配对有问题")
    
    # 检查是否有异常误差
    if np.max(stereo_errors) > 100:
        print(f"\n⚠️ 警告: 检测到异常大的重投影误差!")
        print(f"    最大误差: {np.max(stereo_errors):.1f} 像素")
        print(f"    可能原因: 右相机标定参数严重错误")
        print(f"    建议: 重新检查右相机的单目标定结果")

    rotation_angle = np.linalg.norm(cv2.Rodrigues(R)[0]) * 180 / np.pi
    print(f"    旋转角度: {rotation_angle:.2f} 度")
    if rotation_angle > 45:
        print(f"    ⚠️ 旋转角度较大，可能左右相机未对齐")
    
    # 立体校正
    print("\n" + "-"*50)
    print("立体校正优化...")
    print("-"*50)
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        mtx_left, dist_left, mtx_right, dist_right,
        image_size, R, T,
        alpha=args.alpha,  # 关键参数!
        flags=cv2.CALIB_ZERO_DISPARITY
    )
    
    # 计算极线误差
    epi_error, total_points = compute_epipolar_error(
        objpoints, imgpoints_left, imgpoints_right,
        mtx_left, dist_left, mtx_right, dist_right,
        R1, R2, P1, P2
    )
    print(f"  平均极线误差: {epi_error:.4f} 像素 (基于 {total_points} 个点)")
    
    # 物理参数验证
    fx_avg = (mtx_left[0,0] + mtx_right[0,0]) / 2
    fy_avg = (mtx_left[1,1] + mtx_right[1,1]) / 2
    min_depth = baseline * fx_avg / 200  # 最大视差200像素
    max_depth = baseline * fx_avg / 5    # 最小视差5像素
    
    print("\n" + "="*50)
    print("标定质量评估")
    print("="*50)
    print(f"【几何精度】")
    print(f"  左相机 RMS 误差: {ret_left:.4f} 像素 {'✅' if ret_left < 0.5 else '⚠️' if ret_left < 1.0 else '❌'}")
    print(f"  右相机 RMS 误差: {ret_right:.4f} 像素 {'✅' if ret_right < 0.5 else '⚠️' if ret_right < 1.0 else '❌'}")
    print(f"  立体 RMS 误差: {ret:.4f} 像素 {'✅' if ret < 0.5 else '⚠️' if ret < 1.0 else '❌'}")
    print(f"  极线误差: {epi_error:.4f} 像素 {'✅ 优秀' if epi_error < 0.5 else '⚠️ 良好' if epi_error < 1.0 else '❌ 需改进'}")
    
    print(f"\n【物理参数】")
    print(f"  基线长度: {baseline:.4f} 米 {'✅ 合理' if 0.05 < baseline < 0.3 else '⚠️ 验证'}")
    print(f"  左焦距: {mtx_left[0,0]:.1f} 像素, 右焦距: {mtx_right[0,0]:.1f} 像素")
    print(f"  有效深度范围: {min_depth:.2f}m - {max_depth:.2f}m {'✅ 适用' if max_depth > 2.5 else '⚠️ 有限适用'}")

    # 详细误差分析 - 每个点的误差分布
    print("\n" + "="*60)
    print("每个点的重投影误差分析")
    print("="*60)
    
    # 分析第一对图像作为示例
    if len(objpoints) > 0:
        print("\n【第一对图像的详细误差分析】")
        
        # 左相机每个点的误差
        img_points_left_projected, _ = cv2.projectPoints(
            objpoints[0], rvecs_left[0], tvecs_left[0], mtx_left, dist_left
        )
        
        # 计算每个点的误差
        point_errors_left = []
        for j in range(len(objpoints[0])):
            detected_point = imgpoints_left[0][j, 0]  # 检测到的角点
            projected_point = img_points_left_projected[j, 0]  # 重投影的点
            error = np.linalg.norm(detected_point - projected_point)
            point_errors_left.append(error)
        
        # 右相机每个点的误差
        img_points_right_projected, _ = cv2.projectPoints(
            objpoints[0], rvecs_right[0], tvecs_right[0], mtx_right, dist_right
        )
        
        point_errors_right = []
        for j in range(len(objpoints[0])):
            detected_point = imgpoints_right[0][j, 0]
            projected_point = img_points_right_projected[j, 0]
            error = np.linalg.norm(detected_point - projected_point)
            point_errors_right.append(error)
        
        # 统计信息
        errors_left = np.array(point_errors_left)
        errors_right = np.array(point_errors_right)
        
        print(f"左相机:")
        print(f"  平均误差: {np.mean(errors_left):.4f} 像素")
        print(f"  标准差: {np.std(errors_left):.4f} 像素")
        print(f"  最小误差: {np.min(errors_left):.4f} 像素")
        print(f"  最大误差: {np.max(errors_left):.4f} 像素")
        print(f"  误差范围: {np.max(errors_left) - np.min(errors_left):.4f} 像素")
        
        print(f"右相机:")
        print(f"  平均误差: {np.mean(errors_right):.4f} 像素")
        print(f"  标准差: {np.std(errors_right):.4f} 像素")
        print(f"  最小误差: {np.min(errors_right):.4f} 像素")
        print(f"  最大误差: {np.max(errors_right):.4f} 像素")
        print(f"  误差范围: {np.max(errors_right) - np.min(errors_right):.4f} 像素")
        
        # 分析误差分布
        print(f"\n【误差分布分析】")
        print(f"左相机误差分布:")
        print(f"  < 0.1 像素: {np.sum(errors_left < 0.1)} 个点")
        print(f"  0.1-0.5 像素: {np.sum((errors_left >= 0.1) & (errors_left < 0.5))} 个点")
        print(f"  0.5-1.0 像素: {np.sum((errors_left >= 0.5) & (errors_left < 1.0))} 个点")
        print(f"  ≥ 1.0 像素: {np.sum(errors_left >= 1.0)} 个点")
        
        # 显示每个点的具体误差（前10个点）
        print(f"\n【前10个点的详细误差】")
        for j in range(min(10, len(objpoints[0]))):
            detected_left = imgpoints_left[0][j, 0]
            projected_left = img_points_left_projected[j, 0]
            error_left = point_errors_left[j]
            
            detected_right = imgpoints_right[0][j, 0]
            projected_right = img_points_right_projected[j, 0]
            error_right = point_errors_right[j]
            
            print(f"  点 {j+1}: 左误差={error_left:.4f} 像素, 右误差={error_right:.4f} 像素")
            print(f"        左检测位置: ({detected_left[0]:.1f}, {detected_left[1]:.1f})")
            print(f"        左重投影位置: ({projected_left[0]:.1f}, {projected_left[1]:.1f})")
            print(f"        右检测位置: ({detected_right[0]:.1f}, {detected_right[1]:.1f})")
            print(f"        右重投影位置: ({projected_right[0]:.1f}, {projected_right[1]:.1f})")

    # 详细误差分析 - 每个图像对的误差
    print("\n" + "-"*50)
    print("详细误差分析 (每个图像对)")
    print("-"*50)
    for i in range(len(objpoints)):
        # 计算每个图像对的平均误差
        img_points_left_projected, _ = cv2.projectPoints(
            objpoints[i], rvecs_left[i], tvecs_left[i], mtx_left, dist_left
        )
        error_left = cv2.norm(imgpoints_left[i], img_points_left_projected, cv2.NORM_L2) / len(objpoints[i])
        
        img_points_right_projected, _ = cv2.projectPoints(
            objpoints[i], rvecs_right[i], tvecs_right[i], mtx_right, dist_right
        )
        error_right = cv2.norm(imgpoints_right[i], img_points_right_projected, cv2.NORM_L2) / len(objpoints[i])
        
        print(f"  图像对 {i+1}: 左相机误差={error_left:.4f} 像素, 右相机误差={error_right:.4f} 像素")
        # 立体三维重投影误差分析
    print("\n" + "="*60)
    print("立体三维重投影误差分析")
    print("="*60)
    
    # 分析第一对图像作为示例
    if len(objpoints) > 0:
        print("\n【第一对图像的立体三维重投影误差】")
        
        # 使用三角测量重建三维点
        imgpoints_left_norm = cv2.undistortPoints(imgpoints_left[0], mtx_left, dist_left, P=P1)
        imgpoints_right_norm = cv2.undistortPoints(imgpoints_right[0], mtx_right, dist_right, P=P2)
        
        # 三角测量重建三维点
        # 使用线性三角测量方法，通过左右相机的投影矩阵和对应点重建三维点
        # P1, P2: 左右相机的投影矩阵 [3x4]，包含内参和外参信息
        # imgpoints_left_norm, imgpoints_right_norm: 左右图像中归一化的对应点坐标 [Nx1x2]
        # 返回值points_4d: 齐次坐标下的三维点 [4xN]，需要转换为3D坐标
        points_4d = cv2.triangulatePoints(P1, P2, imgpoints_left_norm, imgpoints_right_norm)
        points_3d = points_4d[:3] / points_4d[3]  # 齐次坐标转换为3D坐标
        
        # 将重建的三维点重投影到左右图像
        reprojected_left, _ = cv2.projectPoints(points_3d.T, np.zeros(3), np.zeros(3), mtx_left, dist_left)
        reprojected_right, _ = cv2.projectPoints(points_3d.T, np.zeros(3), np.zeros(3), mtx_right, dist_right)
        
        # 计算立体三维重投影误差
        stereo_3d_errors = []
        axis_errors_x = []  # 新增：X轴误差数组
        axis_errors_y = []  # 新增：Y轴误差数组
        axis_errors_z = []  # 新增：Z轴误差数组
        
        for j in range(len(objpoints[0])):
            # 左图像误差
            detected_left = imgpoints_left[0][j, 0]
            reproj_left = reprojected_left[j, 0]
            error_left = np.linalg.norm(detected_left - reproj_left)
            
            # 右图像误差
            detected_right = imgpoints_right[0][j, 0]
            reproj_right = reprojected_right[j, 0]
            error_right = np.linalg.norm(detected_right - reproj_right)
            
            # 立体三维误差（取左右误差的平均值）
            stereo_error = (error_left + error_right) / 2
            stereo_3d_errors.append(stereo_error)
            
            # 计算各轴误差
            # X轴误差计算：左右相机X坐标误差的平均值
            # - error_left_x: 左相机检测点与重投影点的X坐标绝对误差
            # - error_right_x: 右相机检测点与重投影点的X坐标绝对误差  
            # - error_x: 左右相机X轴误差的平均值，反映水平方向的标定精度
            error_left_x = abs(detected_left[0] - reproj_left[0])
            error_left_y = abs(detected_left[1] - reproj_left[1])
            error_right_x = abs(detected_right[0] - reproj_right[0])
            # Y轴误差计算：左右相机Y坐标误差的平均值
            # - error_left_y: 左相机检测点与重投影点的Y坐标绝对误差
            # - error_right_y: 右相机检测点与重投影点的Y坐标绝对误差
            # - error_y: 左右相机Y轴误差的平均值，反映垂直方向的标定精度
            error_right_y = abs(detected_right[1] - reproj_right[1])
            
            # 平均X/Y轴误差
            error_x = (error_left_x + error_right_x) / 2
            error_y = (error_left_y + error_right_y) / 2
            
            # 计算视差和深度误差
            # - disparity_detected: 检测视差 = 右相机检测点X坐标 - 左相机检测点X坐标
            # - disparity_reproj: 重投影视差 = 右相机重投影点X坐标 - 左相机重投影点X坐标  
            # - error_z: 视差误差 = |检测视差 - 重投影视差|，反映深度方向的标定精度
            # 注意：Z轴误差不是真正的三维深度误差，而是通过视差差异来估计深度方向的误差
            disparity_detected = detected_right[0] - detected_left[0]  # 检测视差
            disparity_reproj = reproj_right[0] - reproj_left[0]        # 重投影视差
            error_z = abs(disparity_detected - disparity_reproj)       # 深度误差（视差误差）
            
            # 显示前10个点的详细误差
            if j < 10:
                print(f"  点 {j+1}:")
                print(f"    左检测位置: ({detected_left[0]:.1f}, {detected_left[1]:.1f})")
                print(f"    左重投影位置: ({reproj_left[0]:.1f}, {reproj_left[1]:.1f})")
                print(f"    左误差: {error_left:.4f} 像素")
                print(f"    左X轴误差: {error_left_x:.4f} 像素")
                print(f"    左Y轴误差: {error_left_y:.4f} 像素")
                
                print(f"    右检测位置: ({detected_right[0]:.1f}, {detected_right[1]:.1f})")
                print(f"    右重投影位置: ({reproj_right[0]:.1f}, {reproj_right[1]:.1f})")
                print(f"    右误差: {error_right:.4f} 像素")
                print(f"    右X轴误差: {error_right_x:.4f} 像素")
                print(f"    右Y轴误差: {error_right_y:.4f} 像素")
                
                print(f"    立体三维误差: {stereo_error:.4f} 像素")
                print(f"    平均X轴误差: {error_x:.4f} 像素")
                print(f"    平均Y轴误差: {error_y:.4f} 像素")
                print(f"    深度(Z轴)误差: {error_z:.4f} 像素")
                
                # 视差信息
                print(f"    检测视差: {disparity_detected:.1f} 像素")
                print(f"    重投影视差: {disparity_reproj:.1f} 像素")
                print(f"    视差误差: {error_z:.4f} 像素")
                
                # 误差方向分析
                print(f"    误差方向分析:")
                if error_x > error_y and error_x > error_z:
                    print(f"      → 主要误差在X轴（水平方向）")
                elif error_y > error_x and error_y > error_z:
                    print(f"      → 主要误差在Y轴（垂直方向）")
                elif error_z > error_x and error_z > error_y:
                    print(f"      → 主要误差在Z轴（深度方向）")
                else:
                    print(f"      → 误差分布相对均匀")
                
                print(f"    {'-'*60}")

        # 统计信息
        stereo_errors = np.array(stereo_3d_errors)
        print(f"\n【立体三维重投影误差统计】")
        print(f"  平均误差: {np.mean(stereo_errors):.4f} 像素")
        print(f"  标准差: {np.std(stereo_errors):.4f} 像素")
        print(f"  最小误差: {np.min(stereo_errors):.4f} 像素")
        print(f"  最大误差: {np.max(stereo_errors):.4f} 像素")
        print(f"  误差范围: {np.max(stereo_errors) - np.min(stereo_errors):.4f} 像素")
        
        # 误差分布分析
        print(f"\n【误差分布】")
        print(f"  < 0.1 像素: {np.sum(stereo_errors < 0.1)} 个点")
        print(f"  0.1-0.5 像素: {np.sum((stereo_errors >= 0.1) & (stereo_errors < 0.5))} 个点")
        print(f"  0.5-1.0 像素: {np.sum((stereo_errors >= 0.5) & (stereo_errors < 1.0))} 个点")
        print(f"  ≥ 1.0 像素: {np.sum(stereo_errors >= 1.0)} 个点")
        
        # 计算立体RMS误差（与cv2.stereoCalibrate的ret值对比）
        stereo_rms = np.sqrt(np.mean(stereo_errors**2))
        print(f"\n【立体RMS误差对比】")
        print(f"  cv2.stereoCalibrate返回的RMS: {ret:.4f} 像素")
        print(f"  手动计算的立体RMS: {stereo_rms:.4f} 像素")
        print(f"  差异: {abs(ret - stereo_rms):.4f} 像素")

    # 保存结果
    output_xml = os.path.join(args.output, 'stereo_calibration.xml')
    output_yaml = os.path.join(args.output, 'stereo_calibration.yaml')
    
    # 保存XML (OpenCV标准格式)
    fs = cv2.FileStorage(output_xml, cv2.FILE_STORAGE_WRITE)
    fs.write("cameraMatrix1", mtx_left)
    fs.write("distCoeffs1", dist_left)
    fs.write("cameraMatrix2", mtx_right)
    fs.write("distCoeffs2", dist_right)
    fs.write("R", R)
    fs.write("T", T)
    fs.write("E", E)
    fs.write("F", F)
    fs.write("R1", R1)
    fs.write("R2", R2)
    fs.write("P1", P1)
    fs.write("P2", P2)
    fs.write("Q", Q)
    fs.write("image_width", w)
    fs.write("image_height", h)
    fs.write("rms_error", ret)
    fs.write("epipolar_error", epi_error)
    fs.write("baseline", baseline)
    fs.write("validPixROI1", np.array(validPixROI1))
    fs.write("validPixROI2", np.array(validPixROI2))
    fs.release()
    
    # 保存YAML (人类可读)
    calibration_data = {
        'calibration_date': str(datetime.now()),
        'image_size': {'width': w, 'height': h},
        'chessboard': {
            'size': [chessboard_size[0], chessboard_size[1]],
            'square_size_m': args.square
        },
        'reprojection_error': {
            'rms': float(ret),
            'left_camera': float(ret_left),
            'right_camera': float(ret_right),
            'epipolar': float(epi_error)
        },
        'baseline_m': float(baseline),
        'camera_matrix_left': mtx_left.tolist(),
        'distortion_coeffs_left': dist_left.ravel().tolist(),
        'camera_matrix_right': mtx_right.tolist(),
        'distortion_coeffs_right': dist_right.ravel().tolist(),
        'rotation_matrix': R.tolist(),
        'translation_vector': T.ravel().tolist(),
        'rectification': {
            'alpha': args.alpha,
            'valid_roi_left': validPixROI1,
            'valid_roi_right': validPixROI2
        },
        'depth_range_m': [float(min_depth), float(max_depth)],
        'calibration_flags': {
            'stereo_flags': int(stereo_flags),
            'used_single_calibration': {
                'left': bool(single_calib_left),
                'right': bool(single_calib_right)
            }
        }
    }
    
    with open(output_yaml, 'w', encoding='utf-8') as f:
        yaml.dump(calibration_data, f, default_flow_style=False)
    
    print(f"\n✅ 标定结果已保存:")
    print(f"  - OpenCV标准格式: {output_xml}")
    print(f"  - 人类可读格式: {output_yaml}")
    
    # 可视化校正效果
    if paired_images:
        visualize_rectification(paired_images[0], mtx_left, dist_left, mtx_right, dist_right, 
                               R1, R2, P1, P2, args.output, validPixROI1, validPixROI2)
    
    print(f"\n{'='*60}")
    if epi_error < 0.5 and ret < 0.5:
        print("🎉 标定成功! 结果质量优秀")
    elif epi_error < 1.0 and ret < 1.0:
        print("👍 标定成功! 结果质量良好")
    else:
        print("⚠️ 标定完成，但质量不足! 建议增加更多图像")
    
    return {
        'mtx_left': mtx_left,
        'dist_left': dist_left,
        'mtx_right': mtx_right,
        'dist_right': dist_right,
        'R': R,
        'T': T,
        'R1': R1,
        'R2': R2,
        'P1': P1,
        'P2': P2,
        'Q': Q,
        'epi_error': epi_error,
        'baseline': baseline
    }


def visualize_rectification(first_pair, mtx_left, dist_left, mtx_right, dist_right,
                          R1, R2, P1, P2, output_dir, roi1, roi2):
    """可视化立体校正效果"""
    print("\n" + "-"*50)
    print("生成校正效果可视化...")
    print("-"*50)
    
    left_path, right_path = first_pair
    img_left = cv2.imread(left_path)
    img_right = cv2.imread(right_path)
    
    if img_left is None or img_right is None:
        print("  警告: 无法读取测试图像，跳过可视化")
        return
    
    h, w = img_left.shape[:2]
    
    # 计算校正映射
    map1_left, map2_left = cv2.initUndistortRectifyMap(
        mtx_left, dist_left, R1, P1, (w, h), cv2.CV_16SC2)
    map1_right, map2_right = cv2.initUndistortRectifyMap(
        mtx_right, dist_right, R2, P2, (w, h), cv2.CV_16SC2)
    
    # 应用校正
    img_left_rect = cv2.remap(img_left, map1_left, map2_left, cv2.INTER_LANCZOS4)
    img_right_rect = cv2.remap(img_right, map1_right, map2_right, cv2.INTER_LANCZOS4)
    
    # 绘制水平线
    line_img_left = img_left_rect.copy()
    line_img_right = img_right_rect.copy()
    for y in range(50, h, 50):
        cv2.line(line_img_left, (0, y), (w, y), (0, 255, 0), 1)
        cv2.line(line_img_right, (0, y), (w, y), (0, 255, 0), 1)
    
    # 标记有效区域
    if roi1[2] > 0 and roi1[3] > 0:
        cv2.rectangle(line_img_left, (roi1[0], roi1[1]), (roi1[0]+roi1[2], roi1[1]+roi1[3]), (0, 0, 255), 2)
    if roi2[2] > 0 and roi2[3] > 0:
        cv2.rectangle(line_img_right, (roi2[0], roi2[1]), (roi2[0]+roi2[2], roi2[1]+roi2[3]), (0, 0, 255), 2)
    
    # 拼接结果
    top_row = np.hstack((img_left, img_right))
    bottom_row = np.hstack((line_img_left, line_img_right))
    result = np.vstack((top_row, bottom_row))
    
    # 添加标注
    result = put_text_cn(result, "原始图像", (50, 20), (255, 255, 255), 28)
    result = put_text_cn(result, "校正后图像 + 水平线", (w + 50, 20 + h), (255, 255, 255), 28)
    
    # 保存
    output_path = os.path.join(output_dir, 'rectification_visualization.jpg')
    cv2.imwrite(output_path, result)
    print(f"  ✅ 可视化已保存到: {output_path}")


if __name__ == "__main__":
    args = parse_args()
    
    # 打印实际使用的参数
    print("\n" + "="*60)
    print("使用的参数:")
    print(f"  左图像目录: {args.left}")
    print(f"  右图像目录: {args.right}")
    print(f"  棋盘格尺寸: {args.size}")
    print(f"  方格大小: {args.square} 米")
    print(f"  单标左: {args.single_calib_left or '无'}")
    print(f"  单标右: {args.single_calib_right or '无'}")
    print("="*60 + "\n")
    
    # 运行标定
    calib_data = stereo_calibration(args)
    
    if calib_data is None:
        print("❌ 标定失败，程序退出")
        exit(1)
    
    print("\n" + "="*60)
    print("标定流程完成!")
    print("="*60)
'''
D:/anaconda3/envs/retinaface_env/python.exe camera_calibrator_double.py --left left --right right --square 0.1 --size 11x8 --single_calib_left ""E:\Investigation\姿态绕杆检测\Code\calibration_results\calibration_data_left_1126\calibration_report.json"" --single_calib_right "E:\Investigation\姿态绕杆检测\Code\calibration_results\calibration_report.json"
'''