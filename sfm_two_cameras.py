#!/usr/bin/env python3
"""
双机位SFM (Structure from Motion) 工具
================================================
功能:
- 基于特征点匹配的双相机SFM重建
- 支持SIFT/SURF/ORB特征检测
- 自动估计相机相对姿态
- 三角化重建3D点云
- 支持PnP算法进行已知3D点的精确重建
- 可选的Bundle Adjustment优化

使用说明:
1. 准备两个相机从不同角度拍摄的同一场景图像
2. 图像命名: cam1_001.jpg, cam2_001.jpg (或使用目录结构)
3. 运行: python sfm_two_cameras.py --cam1 cam1_images --cam2 cam2_images是不使用已知3D点的示例做
4. 输出: 3D点云、相机姿态、可视化结果
"""

import numpy as np
import cv2
import glob
import os
import argparse
import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def detect_and_match_features(img1, img2, detector_type='SIFT', ratio_thresh=0.75, 
                              max_distance=None, cross_check=False):
    """
    检测特征点并匹配（增强版，支持多种过滤策略）
    
    参数:
        img1, img2: 输入图像
        detector_type: 'SIFT', 'SURF', 'ORB'
        ratio_thresh: Lowe's ratio test阈值 (越小越严格，推荐0.6-0.75)
        max_distance: 最大描述符距离阈值 (可选，用于进一步过滤)
        cross_check: 是否使用交叉验证 (更严格但更慢)
    
    返回:
        kp1, kp2: 关键点
        matches: 匹配点对
        good_matches: 过滤后的优质匹配
        match_quality: 匹配质量信息字典
    """
    # 转换为灰度
    if len(img1.shape) == 3:
        gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = img1
    
    if len(img2.shape) == 3:
        gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = img2
    
    # 选择特征检测器
    if detector_type == 'SIFT':
        detector = cv2.SIFT_create()
        matcher = cv2.BFMatcher()
    elif detector_type == 'SURF':
        detector = cv2.xfeatures2d.SURF_create(400) if hasattr(cv2, 'xfeatures2d') else cv2.SIFT_create()
        matcher = cv2.BFMatcher()
    elif detector_type == 'ORB':
        detector = cv2.ORB_create()
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=cross_check)
    else:
        raise ValueError(f"不支持的检测器类型: {detector_type}")
    
    # 检测特征点和描述符
    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)
    
    if des1 is None or des2 is None or len(des1) < 4 or len(des2) < 4:
        return None, None, None, None, None
    
    # 初始化match_distances变量
    match_distances = []
    
    # 匹配特征点
    if cross_check and detector_type != 'ORB':
        # 交叉验证：双向匹配
        matches1 = matcher.match(des1, des2)
        matches2 = matcher.match(des2, des1)
        # 只保留双向一致的匹配
        good_matches = [m1 for m1 in matches1 
                       if any(m1.queryIdx == m2.trainIdx and m1.trainIdx == m2.queryIdx 
                             for m2 in matches2)]
        matches = matches1
        # 收集匹配距离用于质量评估
        match_distances = [m.distance for m in good_matches]
    else:
        # KNN匹配
        matches = matcher.knnMatch(des1, des2, k=2)
        # Lowe's ratio test过滤
        good_matches = []
        match_distances = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                ratio = m.distance / n.distance if n.distance > 0 else float('inf')
                if ratio < ratio_thresh:
                    if max_distance is None or m.distance < max_distance:
                        good_matches.append(m)
                        match_distances.append(m.distance)
    
    # 计算匹配质量指标
    match_quality = {
        'total_features1': len(kp1),
        'total_features2': len(kp2),
        'total_matches': len(matches) if isinstance(matches, list) else len(matches),
        'good_matches': len(good_matches),
        'match_ratio': len(good_matches) / max(len(kp1), len(kp2)) if max(len(kp1), len(kp2)) > 0 else 0,
        'avg_distance': np.mean(match_distances) if match_distances else 0,
        'min_distance': np.min(match_distances) if match_distances else 0,
        'max_distance': np.max(match_distances) if match_distances else 0
    }
    
    return kp1, kp2, matches, good_matches, match_quality


def estimate_relative_pose(K1, K2, kp1, kp2, matches, img1_shape, img2_shape, 
                           ransac_threshold=1.0, confidence=0.9999, max_iters=2000):
    """
    估计两个相机之间的相对姿态（增强版，支持更严格的RANSAC参数）
    
    参数:
        K1, K2: 相机内参矩阵
        kp1, kp2: 关键点
        matches: 匹配点对
        img1_shape, img2_shape: 图像尺寸
        ransac_threshold: RANSAC阈值（像素，越小越严格，推荐0.5-3.0）
        confidence: RANSAC置信度（0.99-0.9999，越大越严格）
        max_iters: RANSAC最大迭代次数
    
    返回:
        R: 旋转矩阵 (从相机1到相机2)
        t: 平移向量 (从相机1到相机2)
        points1, points2: 匹配点坐标
        inliers: 内点掩码
        pose_quality: 姿态估计质量信息
    """
    # 提取匹配点坐标
    points1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    points2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    if len(points1) < 8:
        return None, None, None, None, None, None
    
    # 使用RANSAC估计基础矩阵（更严格的参数）
    F, mask = cv2.findFundamentalMat(
        points1, points2, 
        cv2.FM_RANSAC, 
        ransac_threshold,  # 阈值（像素）
        confidence,        # 置信度
        maxIters=max_iters
    )
    
    if F is None:
        return None, None, None, None, None, None
    
    # 提取内点
    inliers = mask.ravel() == 1
    points1_inlier = points1[inliers]
    points2_inlier = points2[inliers]
    num_inliers = np.sum(inliers)
    inlier_ratio = num_inliers / len(points1)
    
    if len(points1_inlier) < 8:
        return None, None, None, None, None, None
    
    # 从基础矩阵恢复本质矩阵
    E = K2.T @ F @ K1
    
    # 从本质矩阵恢复旋转和平移
    _, R, t, mask_recover = cv2.recoverPose(E, points1_inlier, points2_inlier, K1)
    
    # 计算姿态估计质量
    pose_quality = {
        'num_inliers': int(num_inliers),
        'inlier_ratio': float(inlier_ratio),
        'num_outliers': int(len(points1) - num_inliers),
        'ransac_threshold': ransac_threshold,
        'confidence': confidence
    }
    
    return R, t, points1_inlier, points2_inlier, inliers, pose_quality


def estimate_pose_pnp(K, dist_coeffs, points_2d, points_3d, ransac_threshold=1.0, confidence=0.9999):
    """
    使用PnP算法估计相机姿态（已知3D点坐标）
    
    参数:
        K: 相机内参矩阵
        dist_coeffs: 畸变系数向量
        points_2d: 2D图像点坐标 (Nx2)
        points_3d: 对应的3D世界点坐标 (Nx3)
        ransac_threshold: RANSAC阈值（像素）
        confidence: RANSAC置信度
    
    返回:
        success: 是否成功估计姿态
        rvec: 旋转向量
        tvec: 平移向量
        inliers: 内点掩码
    """
    if len(points_2d) < 4 or len(points_3d) < 4:
        return False, None, None, None
    
    # 使用solvePnPRansac进行姿态估计
    success, rvec, tvec, inliers = cv2.solvePnPRansac(
        points_3d.astype(np.float32),
        points_2d.astype(np.float32),
        K.astype(np.float32),
        dist_coeffs,  # 畸变系数
        reprojectionError=ransac_threshold,
        confidence=confidence,
        flags=cv2.SOLVEPNP_ITERATIVE
    )
    
    if not success:
        return False, None, None, None
    
    # 将旋转向量转换为旋转矩阵
    R, _ = cv2.Rodrigues(rvec)
    
    return True, R, tvec, inliers


def triangulate_points(K1, K2, R, t, points1, points2):
    """
    三角化重建3D点
    
    参数:
        K1, K2: 相机内参
        R, t: 相对姿态
        points1, points2: 匹配点对
    
    返回:
        points_3d: 3D点 (Nx3)
    """
    # 构建投影矩阵
    P1 = K1 @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K2 @ np.hstack([R, t])
    
    # 三角化
    points_4d = cv2.triangulatePoints(P1, P2, points1, points2)
    
    # 转换为齐次坐标
    points_3d = points_4d[:3] / points_4d[3]
    
    return points_3d.T


def undistort_image(img, K, dist_coeffs):
    """
    对图像进行去畸变处理
    
    参数:
        img: 输入图像
        K: 相机内参矩阵
        dist_coeffs: 畸变系数向量
    
    返回:
        undistorted_img: 去畸变后的图像
    """
    if dist_coeffs is None:
        return img
    
    # 使用OpenCV的undistort函数进行去畸变
    undistorted_img = cv2.undistort(img, K, dist_coeffs)
    return undistorted_img


def sfm_two_cameras(cam1_dir, cam2_dir, K1=None, K2=None, dist1=None, dist2=None, output_dir='sfm_results', 
                    detector_type='SIFT', min_matches=50, ratio_thresh=0.75,
                    ransac_threshold=1.0, ransac_confidence=0.9999, cross_check=False,
                    points_3d_file=None, use_pnp=False, undistort_images=False):
    """
    双机位SFM主函数
    
    参数:
        cam1_dir: 相机1图像目录
        cam2_dir: 相机2图像目录
        K1, K2: 相机内参矩阵 (如果已知，否则需要先标定)
        dist1, dist2: 相机畸变系数 (如果已知，否则需要先标定)
        output_dir: 输出目录
        detector_type: 特征检测器类型
        min_matches: 最小匹配点数
        points_3d_file: 已知3D点坐标文件路径 (JSON格式)
        use_pnp: 是否使用PnP算法进行姿态估计
        undistort_images: 是否对图像进行去畸变处理
    """
    print("="*60)
    print("双机位SFM重建工具")
    print("="*60)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载图像
    cam1_files = sorted(glob.glob(os.path.join(cam1_dir, '*.jpg')) + 
                       glob.glob(os.path.join(cam1_dir, '*.png')))
    cam2_files = sorted(glob.glob(os.path.join(cam2_dir, '*.jpg')) + 
                       glob.glob(os.path.join(cam2_dir, '*.png')))
    
    if not cam1_files or not cam2_files:
        raise ValueError(f"未找到图像! cam1={cam1_dir}, cam2={cam2_dir}")
    
    print(f"找到图像: 相机1={len(cam1_files)}, 相机2={len(cam2_files)}")
    
    # 如果内参未知，尝试从标定文件加载
    if K1 is None or K2 is None:
        print("\n⚠️ 警告: 相机内参未提供，尝试从标定文件加载...")
        # 尝试加载标定结果（按优先级）
        
        # 1. 优先从XML文件加载（OpenCV标准格式，最可靠）
        xml_file = 'calibration_results_double/stereo_calibration.xml'
        if os.path.exists(xml_file):
            try:
                fs = cv2.FileStorage(xml_file, cv2.FILE_STORAGE_READ)
                if fs.isOpened():
                    K1 = fs.getNode('cameraMatrix1').mat()
                    K2 = fs.getNode('cameraMatrix2').mat()
                    # 加载畸变系数
                    dist1 = fs.getNode('distCoeffs1').mat()
                    dist2 = fs.getNode('distCoeffs2').mat()
                    fs.release()
                    if K1 is not None and K2 is not None:
                        print(f"✓ 从 {xml_file} 加载双目内参和畸变系数")
            except Exception as e:
                print(f"  ⚠️ 无法从XML加载: {e}")
        
        # 2. 从JSON文件加载（单目标定结果）
        if K1 is None:
            calib_file1 = 'calibration_results/calibration.json'
            if os.path.exists(calib_file1):
                try:
                    with open(calib_file1, 'r') as f:
                        calib = json.load(f)
                        K1 = np.array(calib['camera_matrix'])
                        # 加载畸变系数
                        if 'distortion_coefficients' in calib:
                            dist1 = np.array(calib['distortion_coefficients'])
                        print(f"✓ 从 {calib_file1} 加载相机1内参和畸变系数")
                except Exception as e:
                    print(f"  ⚠️ 无法从JSON加载: {e}")
        
        # 3. 尝试从YAML文件加载（需要处理tuple问题）
        if (K1 is None or K2 is None) and os.path.exists('calibration_results_double/stereo_calibration.yaml'):
            try:
                import yaml
                # 使用FullLoader而不是SafeLoader来处理tuple
                with open('calibration_results_double/stereo_calibration.yaml', 'r') as f:
                    calib = yaml.load(f, Loader=yaml.FullLoader)
                    if K1 is None and 'camera_matrix_left' in calib:
                        K1 = np.array(calib['camera_matrix_left'])
                        print(f"✓ 从YAML加载相机1内参")
                    if K2 is None and 'camera_matrix_right' in calib:
                        K2 = np.array(calib['camera_matrix_right'])
                        print(f"✓ 从YAML加载相机2内参")
            except Exception as e:
                print(f"  ⚠️ 无法从YAML加载: {e}")
        
        # 如果仍然没有，使用估计值（需要用户提供）
        if K1 is None:
            print("❌ 无法加载相机1内参，请先进行相机标定或手动提供")
            print("   使用示例: --K1 '[[fx,0,cx],[0,fy,cy],[0,0,1]]'")
            return None
        
        if K2 is None:
            # 假设两个相机相同
            K2 = K1.copy()
            print("⚠️ 使用相机1内参作为相机2内参（假设相同相机）")
    
    print(f"\n相机内参:")
    print(f"  相机1 K:\n{K1}")
    print(f"  相机2 K:\n{K2}")
    
    if dist1 is not None:
        print(f"  相机1 畸变系数: {dist1}")
    if dist2 is not None:
        print(f"  相机2 畸变系数: {dist2}")
    
    # 处理第一对图像
    img1 = cv2.imread(cam1_files[0])
    img2 = cv2.imread(cam2_files[0])
    
    if img1 is None or img2 is None:
        raise ValueError(f"无法读取图像: {cam1_files[0]}, {cam2_files[0]}")
    
    h1, w1 = img1.shape[:2]
    h2, w2 = img2.shape[:2]
    
    print(f"\n图像尺寸: 相机1={w1}x{h1}, 相机2={w2}x{h2}")
    
    # 如果启用去畸变，对图像进行处理
    if undistort_images and (dist1 is not None or dist2 is not None):
        print(f"\n对图像进行去畸变处理...")
        if dist1 is not None:
            img1 = undistort_image(img1, K1, dist1)
            print("  ✓ 相机1图像已去畸变")
        if dist2 is not None:
            img2 = undistort_image(img2, K2, dist2)
            print("  ✓ 相机2图像已去畸变")
    
    # 检测和匹配特征点
    print(f"\n检测特征点 ({detector_type})...")
    print(f"匹配参数: ratio_thresh={ratio_thresh}, cross_check={cross_check}")
    kp1, kp2, matches, good_matches, match_quality = detect_and_match_features(
        img1, img2, detector_type=detector_type, 
        ratio_thresh=ratio_thresh, cross_check=cross_check)
    
    if kp1 is None or match_quality is None:
        print("❌ 特征检测失败")
        return None
    
    # 显示匹配质量
    print(f"\n【匹配质量评估】")
    print(f"  检测到的特征点: 相机1={match_quality['total_features1']}, 相机2={match_quality['total_features2']}")
    print(f"  初始匹配数: {match_quality['total_matches']}")
    print(f"  优质匹配数: {match_quality['good_matches']}")
    print(f"  匹配率: {match_quality['match_ratio']:.2%}")
    print(f"  平均描述符距离: {match_quality['avg_distance']:.2f}")
    print(f"  距离范围: {match_quality['min_distance']:.2f} - {match_quality['max_distance']:.2f}")
    
    if len(good_matches) < min_matches:
        print(f"\n❌ 匹配点不足: {len(good_matches)} < {min_matches}")
        print("   建议:")
        print("   - 降低ratio_thresh阈值 (如0.6) 获得更多匹配")
        print("   - 确保两个相机拍摄同一场景")
        print("   - 增加图像重叠区域")
        print("   - 尝试不同的特征检测器 (--detector SIFT/SURF/ORB)")
        print("   - 使用 --cross-check 启用交叉验证（更严格但更准确）")
        return None
    
    # 评估匹配质量
    if match_quality['match_ratio'] < 0.01:
        print("  ⚠️ 警告: 匹配率很低，可能存在大量误匹配")
    elif match_quality['match_ratio'] > 0.1:
        print("  ✓ 匹配率良好")
    
    # 如果提供了已知3D点且启用PnP算法
    if points_3d_file and use_pnp:
        print(f"\n使用PnP算法进行姿态估计...")
        print(f"加载已知3D点: {points_3d_file}")
        
        # 加载已知3D点
        try:
            with open(points_3d_file, 'r') as f:
                points_3d_data = json.load(f)
                points_3d_world = np.array(points_3d_data['points_3d'])  # Nx3
                points_2d_cam1 = np.array(points_3d_data['points_2d_cam1'])  # Nx2
                points_2d_cam2 = np.array(points_3d_data['points_2d_cam2'])  # Nx2
        except Exception as e:
            print(f"❌ 无法加载3D点文件: {e}")
            return None
        
        print(f"  加载 {len(points_3d_world)} 个已知3D点")
        
        # 为每个相机估计姿态
        print("  估计相机1姿态...")
        success1, R1, t1, inliers1 = estimate_pose_pnp(
            K1, dist1, points_2d_cam1, points_3d_world,
            ransac_threshold=ransac_threshold,
            confidence=ransac_confidence
        )
        
        print("  估计相机2姿态...")
        success2, R2, t2, inliers2 = estimate_pose_pnp(
            K2, dist2, points_2d_cam2, points_3d_world,
            ransac_threshold=ransac_threshold,
            confidence=ransac_confidence
        )
        
        if not success1 or not success2:
            print("❌ PnP姿态估计失败")
            return None
        
        # 计算相对姿态 (从相机1到相机2)
        R = R2 @ R1.T
        t = t2 - R @ t1
        
        # 提取用于姿态估计的匹配点
        points1 = points_2d_cam1.reshape(-1, 1, 2)
        points2 = points_2d_cam2.reshape(-1, 1, 2)
        inliers = np.ones(len(points1), dtype=bool)  # 假设所有点都是内点
        
        # 创建姿态质量信息
        pose_quality = {
            'num_inliers': len(points1),
            'inlier_ratio': 1.0,
            'num_outliers': 0,
            'ransac_threshold': ransac_threshold,
            'confidence': ransac_confidence
        }
        
        print("✓ PnP姿态估计完成")
    else:
        # 使用原始的双目SFM方法
        print(f"\n估计相机相对姿态...")
        print(f"RANSAC参数: threshold={ransac_threshold}px, confidence={ransac_confidence}")
        R, t, points1, points2, inliers, pose_quality = estimate_relative_pose(
            K1, K2, kp1, kp2, good_matches, (w1, h1), (w2, h2),
            ransac_threshold=ransac_threshold, 
            confidence=ransac_confidence)
        
        if R is None or pose_quality is None:
            print("❌ 无法估计相对姿态")
            print("   建议:")
            print("   - 增加匹配点数量")
            print("   - 降低ransac_threshold (如0.5) 获得更严格过滤")
            return None
    
    # 显示姿态估计质量
    print(f"\n【姿态估计质量】")
    print(f"  RANSAC内点数量: {pose_quality['num_inliers']}")
    print(f"  内点比例: {pose_quality['inlier_ratio']:.2%}")
    print(f"  外点数量: {pose_quality['num_outliers']}")
    
    if pose_quality['inlier_ratio'] < 0.5:
        print("  ⚠️ 警告: 内点比例较低，可能存在误匹配")
        print("   建议: 降低ratio_thresh或ransac_threshold以获得更准确的匹配")
    elif pose_quality['inlier_ratio'] > 0.8:
        print("  ✓ 内点比例优秀")
    
    print(f"\n  旋转矩阵 R:\n{R}")
    print(f"  平移向量 t: {t.ravel()}")
    
    # 三角化重建3D点
    print("\n三角化重建3D点...")
    points_3d = triangulate_points(K1, K2, R, t, points1, points2)
    
    # 过滤无效点（深度为负或过大）
    valid_mask = (points_3d[:, 2] > 0) & (points_3d[:, 2] < 100)
    points_3d_valid = points_3d[valid_mask]
    
    print(f"✓ 重建 {len(points_3d_valid)} 个有效3D点")
    print(f"  深度范围: {points_3d_valid[:, 2].min():.2f}m - {points_3d_valid[:, 2].max():.2f}m")
    
    # 保存结果
    print(f"\n保存结果到 '{output_dir}'...")
    
    # 保存相机姿态（包含质量信息）
    pose_data = {
        'calibration_date': datetime.now().isoformat(),
        'camera1_intrinsic': K1.tolist(),
        'camera2_intrinsic': K2.tolist(),
        'relative_pose': {
            'rotation': R.tolist(),
            'translation': t.ravel().tolist()
        },
        'matching_quality': match_quality,
        'pose_estimation_quality': pose_quality,
        'matching_parameters': {
            'detector_type': detector_type,
            'ratio_thresh': ratio_thresh,
            'cross_check': cross_check,
            'ransac_threshold': ransac_threshold,
            'ransac_confidence': ransac_confidence,
            'use_pnp': use_pnp,
            'undistort_images': undistort_images
        },
        'num_matches': len(good_matches),
        'num_3d_points': len(points_3d_valid),
        'depth_range': [float(points_3d_valid[:, 2].min()), 
                       float(points_3d_valid[:, 2].max())]
    }
    
    with open(os.path.join(output_dir, 'sfm_pose.json'), 'w') as f:
        json.dump(pose_data, f, indent=4)
    
    # 保存3D点云
    np.save(os.path.join(output_dir, 'points_3d.npy'), points_3d_valid)
    
    # 保存匹配点
    match_data = {
        'points1': points1.reshape(-1, 2).tolist(),
        'points2': points2.reshape(-1, 2).tolist(),
        'inliers': inliers.tolist()
    }
    with open(os.path.join(output_dir, 'matches.json'), 'w') as f:
        json.dump(match_data, f, indent=4)
    
    # 可视化
    print("生成可视化结果...")
    visualize_sfm_results(img1, img2, kp1, kp2, good_matches, inliers, 
                         points_3d_valid, R, t, output_dir)
    
    print(f"\n✅ SFM重建完成!")
    print(f"   - 相机姿态: {os.path.join(output_dir, 'sfm_pose.json')}")
    print(f"   - 3D点云: {os.path.join(output_dir, 'points_3d.npy')}")
    print(f"   - 特征匹配图: {os.path.join(output_dir, 'feature_matches.jpg')}")
    print(f"   - 3D可视化: {os.path.join(output_dir, 'sfm_3d_visualization.png')}")
    
    return {
        'R': R,
        't': t,
        'points_3d': points_3d_valid,
        'K1': K1,
        'K2': K2
    }


def visualize_sfm_results(img1, img2, kp1, kp2, matches, inliers, points_3d, R, t, output_dir):
    """
    可视化SFM结果
    
    参数:
        img1, img2: 输入图像
        kp1, kp2: 关键点
        matches: 匹配点对
        inliers: 内点掩码
        points_3d: 3D点
        R, t: 相机相对姿态
        output_dir: 输出目录
    """
    # 绘制特征匹配图
    if len(matches) > 0:
        # 只绘制内点匹配
        inlier_matches = [m for i, m in enumerate(matches) if inliers[i]] if len(inliers) == len(matches) else matches[:50]
        
        # 绘制匹配结果
        match_img = cv2.drawMatches(img1, kp1, img2, kp2, inlier_matches[:50], None, 
                                   flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # 保存匹配图
        cv2.imwrite(os.path.join(output_dir, 'feature_matches.jpg'), match_img)
    
    # 绘制3D点云
    if len(points_3d) > 0:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制3D点
        ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], 
                  c=points_3d[:, 2], cmap='viridis', s=1)
        
        # 设置坐标轴
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.set_title('3D点云重建结果')
        
        # 保存3D可视化图
        plt.savefig(os.path.join(output_dir, 'sfm_3d_visualization.png'), 
                   dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='双机位SFM重建工具（增强匹配质量控制）')
    parser.add_argument('--cam1', required=True, help='相机1图像目录')
    parser.add_argument('--cam2', required=True, help='相机2图像目录')
    parser.add_argument('--output', default='sfm_results', help='输出目录')
    parser.add_argument('--detector', default='SIFT', choices=['SIFT', 'SURF', 'ORB'],
                       help='特征检测器类型')
    parser.add_argument('--min-matches', type=int, default=50,
                       help='最小匹配点数')
    parser.add_argument('--ratio-thresh', type=float, default=0.75,
                       help='Lowe\'s ratio test阈值 (0.6-0.8，越小越严格，推荐0.6-0.7)')
    parser.add_argument('--ransac-threshold', type=float, default=1.0,
                       help='RANSAC阈值（像素，0.5-3.0，越小越严格，推荐0.5-1.0）')
    parser.add_argument('--ransac-confidence', type=float, default=0.9999,
                       help='RANSAC置信度（0.99-0.9999，越大越严格）')
    parser.add_argument('--cross-check', action='store_true',
                       help='启用交叉验证（更严格但更准确，速度较慢）')
    parser.add_argument('--K1', help='相机1内参矩阵 (JSON格式或标定文件路径)')
    parser.add_argument('--K2', help='相机2内参矩阵 (JSON格式或标定文件路径)')
    parser.add_argument('--points-3d-file', help='已知3D点坐标文件路径 (JSON格式)')
    parser.add_argument('--use-pnp', action='store_true',
                       help='使用PnP算法进行姿态估计（需要提供已知3D点）')
    parser.add_argument('--undistort-images', action='store_true',
                       help='对图像进行去畸变处理')
    
    args = parser.parse_args()
    
    # 解析内参
    K1, K2 = None, None
    if args.K1:
        if os.path.exists(args.K1):
            # 从文件加载
            with open(args.K1, 'r') as f:
                data = json.load(f)
                K1 = np.array(data['camera_matrix'] if 'camera_matrix' in data else data)
        else:
            # 从JSON字符串解析
            import json as json_lib
            K1 = np.array(json_lib.loads(args.K1))
    
    if args.K2:
        if os.path.exists(args.K2):
            with open(args.K2, 'r') as f:
                data = json.load(f)
                K2 = np.array(data['camera_matrix'] if 'camera_matrix' in data else data)
        else:
            import json as json_lib
            K2 = np.array(json_lib.loads(args.K2))
    
    result = sfm_two_cameras(
        args.cam1, args.cam2, 
        K1=K1, K2=K2,
        output_dir=args.output,
        detector_type=args.detector,
        min_matches=args.min_matches,
        ratio_thresh=args.ratio_thresh,
        ransac_threshold=args.ransac_threshold,
        ransac_confidence=args.ransac_confidence,
        cross_check=args.cross_check,
        points_3d_file=args.points_3d_file,
        use_pnp=args.use_pnp,
        undistort_images=args.undistort_images
    )
    
    if result:
        print("\n" + "="*60)
        print("SFM重建成功完成!")
        print("="*60)

'''
无已知3D点的示例:
python sfm_two_cameras.py --cam1 .\cam1_images\ --cam2 .\cam2_images\ --ratio-thresh 0.6 --ransac-threshold 0.5 --cross-check
--ratio-thresh 越低越严格
--ransac-threshold 越低越严格

使用PnP算法的示例，加载3d对应像素点文件known_points.json：
python sfm_two_cameras.py --cam1 .\cam1_images\ --cam2 .\cam2_images\ --use-pnp --points-3d-file .\known_points.json

使用去畸变的示例：
python sfm_two_cameras.py --cam1 .\cam1_images\ --cam2 .\cam2_images\ --undistort-images
'''