#!/usr/bin/env python3
"""
双目摄像头标定工具 - 专业优化版
================================================
特点:
- 鲁棒图像配对 (基于文件名序号)
- 深度优化的立体校正 (alpha参数控制)
- 全面的质量验证 (极线误差/物理参数)
- 生产级结果保存 (XML+YAML)
- 篮球绕杆场景专属验证

使用说明:
1. 准备同步采集的左右相机图像 (命名如 left_001.jpg, right_001.jpg)
2. 调整 chessboard_size 和 square_size 匹配你的标定板
3. 运行: python stereo_calibrator.py
4. 按任意键继续每张图像的检测
"""

import numpy as np
import cv2
import glob
import os
import re
import yaml
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont


def put_text_cn(img, text, org, color=(0, 255, 0), font_size=28, font_path=None):
    """在OpenCV图像上绘制中文文本。
    参数:
        img: OpenCV图像(BGR)
        text: 要绘制的中文字符串
        org: 左上角位置(x, y)
        color: 文本颜色(BGR)
        font_size: 字号
        font_path: 字体路径，默认使用 Windows 微软雅黑
    返回:
        绘制后的图像
    """
    if font_path is None:
        # Windows 常见中文字体
        font_path = "C:/Windows/Fonts/msyh.ttc"
        if not os.path.exists(font_path):
            # 兜底：尝试宋体
            alt = "C:/Windows/Fonts/simsun.ttc"
            if os.path.exists(alt):
                font_path = alt
            else:
                # 无中文字体时，退化为英文绘制（仍然返回原图）
                cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, font_size/32.0, color, 2, cv2.LINE_AA)
                return img

    # 转为PIL图像进行中文绘制
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception:
        font = ImageFont.load_default()
    # PIL 使用RGB颜色
    rgb = (int(color[2]), int(color[1]), int(color[0]))
    draw.text((int(org[0]), int(org[1])), str(text), font=font, fill=rgb)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def parse_args():
    parser = argparse.ArgumentParser(description='双目摄像头标定工具')
    parser.add_argument('--left', default='left', help='左相机图像目录')
    parser.add_argument('--right', default='right', help='右相机图像目录')
    parser.add_argument('--size', default='9x6', help='棋盘格内角点尺寸 (宽x高)')
    parser.add_argument('--square', type=float, default=0.025, help='棋盘格方格大小(米)')
    parser.add_argument('--alpha', type=float, default=0.8, help='立体校正alpha参数 (0.0-1.0)')
    parser.add_argument('--output', default='calibration_results_double', help='输出目录')
    parser.add_argument('--test', action='store_true', help='标定后立即测试校正效果')
    return parser.parse_args()

def pair_images(left_dir, right_dir):
    """基于文件名序号智能配对图像"""
    left_files = glob.glob(os.path.join(left_dir, '*.jpg')) + glob.glob(os.path.join(left_dir, '*.png'))
    right_files = glob.glob(os.path.join(right_dir, '*.jpg')) + glob.glob(os.path.join(right_dir, '*.png'))
    
    if not left_files or not right_files:
        raise ValueError(f"未找到图像! 检查目录: left='{left_dir}', right='{right_dir}'")
    
    print(f"找到图像: 左={len(left_files)}, 右={len(right_files)}")
    
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

def compute_epipolar_error(objpoints, imgpoints_left, imgpoints_right, 
                          mtx_left, dist_left, mtx_right, dist_right,
                          R1, R2, P1, P2):
    """
    计算校正后的平均极线误差 (像素)
    
    极线误差计算原理：
    1. 立体校正后，左右图像中的对应点应该位于同一水平线上（y坐标相同）
    2. 对每个标定图像对：
       - 将左图角点投影到校正后的坐标系
       - 将右图对应角点投影到校正后的坐标系
       - 计算对应点y坐标的差值：error = |y_left - y_right|
    3. 对所有对应点的y坐标差求平均，得到平均极线误差
    
    误差越小越好：
    - < 0.5像素：优秀，极线对齐完美
    - 0.5-1.0像素：良好，可用于大多数应用
    - > 1.0像素：需改进，可能影响立体匹配精度
    """
    total_error = 0.0
    total_points = 0
    
    for i in range(len(objpoints)):
        # 校正左图点：将原始图像坐标转换为校正后的坐标
        pts_left = imgpoints_left[i]
        pts_left_rect = cv2.undistortPoints(pts_left, mtx_left, dist_left, R=R1, P=P1)
        
        # 校正右图点：将原始图像坐标转换为校正后的坐标
        pts_right = imgpoints_right[i]
        pts_right_rect = cv2.undistortPoints(pts_right, mtx_right, dist_right, R=R2, P=P2)
        
        # 计算y坐标差 (极线误差)
        # 理想情况下，校正后对应点的y坐标应该完全相同
        for j in range(len(pts_left_rect)):
            pt_l = pts_left_rect[j, 0]
            pt_r = pts_right_rect[j, 0]
            error = abs(pt_l[1] - pt_r[1])  # y坐标差
            total_error += error
            total_points += 1
    
    mean_error = total_error / total_points if total_points > 0 else float('inf')
    return mean_error, total_points

def stereo_calibration(args):
    """主标定函数"""
    print("="*60)
    print("双目摄像头标定工具 - 专业优化版")
    print("="*60)
    
    # 解析棋盘格尺寸
    try:
        chessboard_size = tuple(map(int, args.size.split('x')))
        assert len(chessboard_size) == 2
    except:
        raise ValueError("无效的棋盘格尺寸! 格式应为 '9x6'")
    
    print(f"配置:")
    print(f"  棋盘格: {chessboard_size[0]}x{chessboard_size[1]} 内角点")
    print(f"  方格尺寸: {args.square*1000:.1f} mm")
    print(f"  立体校正 alpha: {args.alpha:.2f} (0=裁剪最大, 1=保留全部)")
    print(f"  输出目录: '{args.output}'")
    
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
        
        # 可视化
        display_left = img_left.copy()
        display_right = img_right.copy()
        cv2.drawChessboardCorners(display_left, chessboard_size, corners_left, ret_left)
        cv2.drawChessboardCorners(display_right, chessboard_size, corners_right, ret_right)
        
        # 显示
        combined = np.hstack((display_left, display_right))
        combined = put_text_cn(combined, f"图像对: {idx+1}/{len(paired_images)}", (20, 30), (0, 255, 0), 28)
        
        if ret_left and ret_right:
            combined = put_text_cn(combined, "状态: 角点检测成功", (20, 70), (0, 255, 0), 26)
        else:
            status = []
            if not ret_left: status.append("左图失败")
            if not ret_right: status.append("右图失败")
            combined = put_text_cn(combined, f"状态: {' & '.join(status)}", (20, 70), (0, 0, 255), 26)
        
        cv2.imshow('角点检测 - 按任意键继续', combined)
        key = cv2.waitKey(0) & 0xFF
        if key == 27:  # ESC
            print("用户中断标定过程")
            cv2.destroyAllWindows()
            return None
        
        # 亚像素精化
        if ret_left and ret_right:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_left_refined = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
            corners_right_refined = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
            
            objpoints.append(objp)
            imgpoints_left.append(corners_left_refined)
            imgpoints_right.append(corners_right_refined)
            valid_pairs += 1
            
            print(f"  对 {idx+1}: 成功检测角点 (累计: {valid_pairs})")
        else:
            print(f"  对 {idx+1}: 角点检测失败")
    
    cv2.destroyAllWindows()
    
    if valid_pairs < 5:
        print(f"❌ 错误: 仅 {valid_pairs} 对有效图像，需要至少5对")
        return None
    
    print(f"\n✅ 成功检测 {valid_pairs} 对图像的角点")
    
    # 单目优化标志
    calib_flags = cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_THIN_PRISM_MODEL
    
    # 左相机标定
    # 单目RMS误差计算原理：
    # 1. 使用标定得到的相机内参(K)和畸变系数，将3D标定板角点投影回2D图像
    # 2. 计算投影点与检测到的角点之间的欧氏距离
    # 3. 对所有点求均方根(RMS)：RMS = sqrt(mean((x_proj - x_detected)^2 + (y_proj - y_detected)^2))
    # 误差越小越好：<0.5像素=优秀, 0.5-1.0像素=良好, >1.0像素=需改进
    print("\n" + "-"*50)
    print("左相机标定...")
    print("-"*50)
    ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
        objpoints, imgpoints_left, (w, h), None, None, flags=calib_flags)
    print(f"  重投影误差 (RMS): {ret_left:.4f} 像素")
    
    # 右相机标定
    print("\n" + "-"*50)
    print("右相机标定...")
    print("-"*50)
    ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
        objpoints, imgpoints_right, (w, h), None, None, flags=calib_flags)
    print(f"  重投影误差 (RMS): {ret_right:.4f} 像素")
    
    # 立体标定
    # 立体RMS误差计算原理：
    # 1. 同时优化左右相机的内参、外参(R, T)和畸变系数
    # 2. 将3D标定板角点分别投影到左右图像
    # 3. 计算左右图像投影点与检测角点的误差，并考虑立体几何约束
    # 4. 对所有点求均方根，得到立体RMS误差
    # 误差越小越好：<0.5像素=优秀, 0.5-1.0像素=良好, >1.0像素=需改进
    print("\n" + "-"*50)
    print("立体标定...")
    print("-"*50)
    stereo_flags = cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_USE_INTRINSIC_GUESS
    ret, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        mtx_left, dist_left, mtx_right, dist_right,
        (w, h), flags=stereo_flags,
        criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
    )
    print(f"  立体重投影误差 (RMS): {ret:.4f} 像素")
    
    # 立体校正
    print("\n" + "-"*50)
    print("立体校正优化...")
    print("-"*50)
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        mtx_left, dist_left, mtx_right, dist_right,
        (w, h), R, T,
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
    baseline = np.linalg.norm(T)
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
    print(f"  立体 RMS 误差: {ret:.4f} 像素(0.5以内可接受) {'✅' if ret < 0.5 else '⚠️' if ret < 1.0 else '❌'}")
    print(f"  极线误差: {epi_error:.4f} 像素(1以内可接受) {'✅ 优秀' if epi_error < 0.5 else '⚠️ 良好' if epi_error < 1.0 else '❌ 需改进'}")
    
    print(f"\n【物理参数】")
    print(f"  基线长度: {baseline:.4f} 米 {'✅ 合理' if 0.05 < baseline < 0.3 else '⚠️ 验证'}")
    print(f"  左焦距: {mtx_left[0,0]:.1f} 像素, 右焦距: {mtx_right[0,0]:.1f} 像素 (差异: {abs(mtx_left[0,0]-mtx_right[0,0])/fx_avg:.1%})")
    print(f"  有效深度范围: {min_depth:.2f}m - {max_depth:.2f}m")
    print(f"  篮球绕杆适用性: {'✅ 适用' if (1.0 < min_depth < 2.0 and max_depth > 3.0) else '⚠️ 部分适用' if max_depth > 2.5 else '❌ 不适用'}")
    
    print(f"\n【有效区域】")
    print(f"  左相机有效区域: {validPixROI1}")
    print(f"  右相机有效区域: {validPixROI2}")
    
    # 保存结果
    output_xml = os.path.join(args.output, 'stereo_calibration.xml')
    output_yaml = os.path.join(args.output, 'stereo_calibration.yaml')
    output_vis = os.path.join(args.output, 'rectification_visualization.jpg')
    
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
            'right_camera': float(ret_right)
        },
        'epipolar_error': float(epi_error),
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
        'depth_range_m': [float(min_depth), float(max_depth)]
    }
    
    with open(output_yaml, 'w') as f:
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
        print("🎉 标定成功! 结果质量优秀，适用于篮球绕杆场景")
    elif epi_error < 1.0 and ret < 1.0:
        print("👍 标定成功! 结果质量良好，可用于篮球绕杆，但边缘精度略低")
    else:
        print("⚠️ 标定完成，但质量不足! 建议:")
        print("   - 增加更多图像 (特别是边缘区域)")
        print("   - 检查标定板是否平整")
        print(f"   - 调整alpha参数 (当前:{args.alpha})")
    
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
    result = put_text_cn(result, "有效区域", (50, h-30), (0, 0, 255), 24)
    
    # 保存
    output_path = os.path.join(output_dir, 'rectification_visualization.jpg')
    cv2.imwrite(output_path, result)
    print(f"  ✅ 可视化已保存到: {output_path}")
    
    # 显示
    cv2.imshow('立体校正效果 (按任意键关闭)', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def test_rectification(calib_data, test_left, test_right, output_dir):
    """测试校正效果"""
    print("\n" + "="*60)
    print("立体校正效果测试")
    print("="*60)
    
    # 加载测试图像
    img_left = cv2.imread(test_left)
    img_right = cv2.imread(test_right)
    
    if img_left is None or img_right is None:
        print(f"❌ 无法读取测试图像: {test_left}, {test_right}")
        return
    
    h, w = img_left.shape[:2]
    
    # 计算校正映射
    map1_left, map2_left = cv2.initUndistortRectifyMap(
        calib_data['mtx_left'], calib_data['dist_left'], 
        calib_data['R1'], calib_data['P1'], (w, h), cv2.CV_16SC2)
    map1_right, map2_right = cv2.initUndistortRectifyMap(
        calib_data['mtx_right'], calib_data['dist_right'], 
        calib_data['R2'], calib_data['P2'], (w, h), cv2.CV_16SC2)
    
    # 应用校正
    img_left_rect = cv2.remap(img_left, map1_left, map2_left, cv2.INTER_LANCZOS4)
    img_right_rect = cv2.remap(img_right, map1_right, map2_right, cv2.INTER_LANCZOS4)
    
    # 显示
    combined = np.hstack((img_left_rect, img_right_rect))
    cv2.putText(combined, "左相机校正后", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(combined, "右相机校正后", (w + 50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow('校正效果测试 - 按任意键保存', combined)
    cv2.waitKey(0)
    
    # 保存
    output_path = os.path.join(output_dir, 'test_rectification.jpg')
    cv2.imwrite(output_path, combined)
    print(f"✅ 测试结果已保存到: {output_path}")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    args = parse_args()
    
    # 运行标定
    calib_data = stereo_calibration(args)
    
    if calib_data is None:
        exit(1)
    
    # 测试校正
    if args.test and hasattr(args, 'test_left') and hasattr(args, 'test_right'):
        test_rectification(calib_data, args.test_left, args.test_right, args.output)
    
    print("\n" + "="*60)
    print("标定流程完成!")
    print("="*60)
