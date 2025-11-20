#!/usr/bin/env python3
"""
相机标定程序 - 支持普通USB相机
================================================
使用说明：
1. 准备一个棋盘格标定板（推荐9x6内角点）
2. 运行程序采集图像: python camera_calibrator_single.py --capture
3. 放置标定板在不同位置/角度（覆盖整个画面）
4. 按空格键保存图像，按ESC结束采集
5. 运行标定: python camera_calibrator_single.py --calibrate

标定结果输出：
标定完成后，程序会在 'calibration_results' 目录下生成以下文件：

【标定数据文件】（3种格式）
1. camera_calibration.xml
   - OpenCV标准格式，可直接用 cv2.FileStorage 读取
   - 包含：相机内参矩阵(K)、畸变系数、重投影误差、图像尺寸等

2. calibration.json
   - JSON格式，易于阅读和程序解析
   - 包含完整的标定信息：标定日期、图像尺寸、棋盘格参数、
     重投影误差(RMS/平均值)、相机内参矩阵、畸变系数、使用的图像列表等

3. calibration.txt
   - 简单文本格式，适用于嵌入式系统或快速查看
   - 包含：标定日期、图像尺寸、重投影误差、相机内参矩阵、畸变系数

【可视化结果文件】（保存在 visualizations/ 子目录）
1. distortion_correction.jpg
   - 畸变校正效果对比图（左右对比：原始图像 vs 校正后图像）
   - 用于直观验证标定效果和畸变校正能力

2. 3d_point_cloud.png
   - 3D点云分布可视化图
   - 展示不同标定图像中棋盘格在相机坐标系中的3D位置和姿态分布

【关键参数说明】
- camera_matrix (K): 3x3相机内参矩阵，包含焦距(fx, fy)和主点坐标(cx, cy)
  [fx  0  cx]
  [ 0 fy  cy]
  [ 0  0   1]
  
- distortion_coefficients: 畸变系数数组 (k1, k2, p1, p2, k3, s1, s2, s3, s4)
  - k1, k2, k3: 径向畸变系数
  - p1, p2: 切向畸变系数
  - s1, s2, s3, s4: 薄棱镜畸变系数（如果使用）
  
- reprojection_error: 重投影误差（像素单位）
  - RMS误差: 所有点的重投影误差的均方根值
  - 平均误差: 所有点的平均重投影误差
  - 评估标准(rms): <0.5像素=优秀, <1.0像素=良好, >1.0像素=需要重新标定

【文件结构】
calibration_results/
├── camera_calibration.xml      # OpenCV标准格式
├── calibration.json            # JSON格式（易读）
├── calibration.txt             # 文本格式（简单）
└── visualizations/
    ├── distortion_correction.jpg  # 畸变校正对比图
    └── 3d_point_cloud.png         # 3D点云可视化
"""

import cv2
import numpy as np
import os
import argparse
import glob
import json
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ======================
# 配置参数 (可自行调整)
# ======================
class Config:
    # 棋盘格参数 (根据你的标定板修改!)
    CHESSBOARD_SIZE = (9, 6)  # 内角点数量 (宽, 高)
    SQUARE_SIZE = 0.025       # 棋盘格单格尺寸(米) - 25mm
    
    # 采集设置
    CAPTURE_DIR = "calibration_images_sxh"  # 保存图像的目录 (统一使用此目录)
    MIN_IMAGES = 10                    # 最小图像数量
    MAX_IMAGES = 30                    # 最大图像数量
    
    # 质量控制阈值
    MIN_BLUR_VAR = 100.0      # 拉普拉斯方差阈值 (防模糊)
    MIN_CONTRAST = 30.0       # 最小对比度
    MAX_SATURATION = 220.0    # 最大饱和度 (防过曝)
    
    # 标定参数
    CALIB_FLAGS = cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_THIN_PRISM_MODEL | cv2.CALIB_FIX_PRINCIPAL_POINT
    
    # 相机设置 (手机用户注意!)
    CAMERA_ID = 0             # 0=内置摄像头, 1=外接USB, 手机用DroidCam时通常为1
    RESOLUTION = (1920, 1080)  # 设置分辨率 (手机建议720p)
    FPS = 30                  # 帧率

# ======================
# 辅助函数：验证定位精度
# ======================
def verify_localization_accuracy(mtx, dist, working_distance=1.0, board_size_mm=25.0):
    """
    验证1米距离下的实际定位精度
    
    参数:
    mtx, dist: 相机标定参数
    working_distance: 工作距离(米)
    board_size_mm: 棋盘格单格物理尺寸(mm)
    
    返回:
    实际定位精度(mm)
    """
    print(f"\n验证在{working_distance}米工作距离下的定位精度...")
    
    # 1. 创建一个虚拟标定板 (使用与实际标定相同的尺寸)
    objp = np.zeros((Config.CHESSBOARD_SIZE[0] * Config.CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:Config.CHESSBOARD_SIZE[0], 0:Config.CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= board_size_mm / 1000  # 转为米
    
    # 2. 模拟相机在工作距离的位姿
    rvec = np.array([0, 0, 0], dtype=np.float64)  # 无旋转
    tvec = np.array([0, 0, working_distance], dtype=np.float64)
    
    # 3. 生成理想图像点
    imgpoints, _ = cv2.projectPoints(objp, rvec, tvec, mtx, dist)
    
    # 4. 添加0.5像素误差模拟实际检测误差
    np.random.seed(42)  # 固定随机种子以保证结果可重复
    noisy_imgpoints = imgpoints + np.random.normal(0, 0.5, imgpoints.shape)
    
    # 5. 反向计算3D点（使用PnP算法模拟定位）
    success, rvec_calc, tvec_calc = cv2.solvePnP(objp, noisy_imgpoints, mtx, dist)
    
    if not success:
        print("  警告: PnP求解失败，无法计算定位精度")
        return 0
    
    # 6. 计算实际定位误差
    total_error_mm = 0
    
    # 对每个点计算误差
    for i in range(len(objp)):
        # 将3D点投影到图像平面
        projected_point, _ = cv2.projectPoints(objp[i:i+1], rvec_calc, tvec_calc, mtx, dist)
        
        # 计算重投影误差
        reprojection_error = np.linalg.norm(imgpoints[i] - projected_point[0])
        
        # 通过重投影误差反推3D空间误差
        # 这是一个简化的估算，实际误差会因点的位置而异
        # 使用小角度近似: 误差(米) ≈ 重投影误差(像素) * 工作距离(米) / 焦距(像素)
        focal_length = (mtx[0, 0] + mtx[1, 1]) / 2  # 平均焦距
        error_3d_m = reprojection_error * working_distance / focal_length
        total_error_mm += error_3d_m * 1000  # 转为毫米
    
    avg_error_mm = total_error_mm / len(objp)
    
    print(f"  在{working_distance}米距离下，0.5像素检测误差对应平均定位精度: {avg_error_mm:.3f} mm")
    return avg_error_mm

# ======================
# 辅助函数：保存标定结果 (更新版本)
# ======================
def save_calibration_results(mtx, dist, rvecs, tvecs, rms_error, mean_error, image_size, image_files):
    """保存标定结果到多种格式"""
    print("\n保存标定结果...")
    
    # 创建输出目录
    output_dir = "calibration_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. OpenCV 标准XML/YAML格式
    fs = cv2.FileStorage(os.path.join(output_dir, "camera_calibration.xml"), cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", mtx)
    fs.write("distortion_coefficients", dist)
    fs.write("rms_error", rms_error)
    fs.write("mean_error", mean_error)
    fs.write("image_width", image_size[0])
    fs.write("image_height", image_size[1])
    fs.write("used_images", " ".join([os.path.basename(f) for f in image_files]))
    fs.release()
    
    # 2. JSON格式 (易读)
    calibration_data = {
        "calibration_date": datetime.now().isoformat(),
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "chessboard": {
            "size": [Config.CHESSBOARD_SIZE[0], Config.CHESSBOARD_SIZE[1]],
            "square_size_m": Config.SQUARE_SIZE
        },
        "reprojection_error": {
            "rms": float(rms_error),
            "mean": float(mean_error)
        },
        "camera_matrix": mtx.tolist(),
        "distortion_coefficients": dist.ravel().tolist(),
        "used_images_count": len(image_files),
        "used_images": [os.path.basename(f) for f in image_files]
    }
    
    with open(os.path.join(output_dir, "calibration.json"), 'w') as f:
        json.dump(calibration_data, f, indent=4)
    
    # 3. 简单文本格式 (用于嵌入式系统)
    with open(os.path.join(output_dir, "calibration.txt"), 'w') as f:
        f.write(f"Camera Calibration Results\n")
        f.write(f"Generated: {datetime.now()}\n")
        f.write(f"Image Size: {image_size[0]}x{image_size[1]}\n")
        f.write(f"Reprojection Error (RMS): {rms_error:.4f} pixels\n")
        f.write(f"\nCamera Matrix (K):\n")
        for row in mtx:
            f.write(f"  {row[0]:.6f} {row[1]:.6f} {row[2]:.6f}\n")
        f.write("\nDistortion Coefficients (k1,k2,p1,p2,k3,s1,s2,s3,s4):\n")
        f.write("  " + " ".join([f"{d:.6f}" for d in dist.ravel()]))
    
    print(f"结果已保存到 '{output_dir}':")
    print(f"  - camera_calibration.xml (OpenCV标准格式)")
    print(f"  - calibration.json (易读格式)")
    print(f"  - calibration.txt (简单文本)")
    
    return output_dir

# ======================
# 辅助函数：可视化标定结果 (更新版本)
# ======================
def visualize_calibration_results(mtx, dist, image_files, output_dir):
    """可视化标定结果 - 优化版本"""
    print("\n生成可视化结果...")
    
    # 确保可视化目录存在
    vis_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(vis_dir, exist_ok=True)
    
    # 1. 畸变校正效果对比 (只处理第一张图像以提高速度)
    print("  生成畸变校正对比图...")
    sample_img = cv2.imread(image_files[0])
    if sample_img is None:
        print(f"警告: 无法读取样本图像 {image_files[0]}，跳过畸变校正可视化")
        return
    
    # 获取图像尺寸
    h, w = sample_img.shape[:2]
    # 计算最优新相机矩阵
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    
    # 校正图像
    dst = cv2.undistort(sample_img, mtx, dist, None, newcameramtx)
    
    # 裁剪图像
    x, y, w_roi, h_roi = roi
    dst_cropped = dst[y:y+h_roi, x:x+w_roi]
    
    # 创建对比图
    comparison = np.zeros((h, w*2, 3), dtype=np.uint8)
    comparison[:, :w] = sample_img
    comparison[:, w:] = cv2.resize(dst_cropped, (w, h))
    
    # 添加标注
    cv2.putText(comparison, "Original Image", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(comparison, "Distortion Correction", (w+50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # 保存对比图
    cv2.imwrite(os.path.join(vis_dir, "distortion_correction.jpg"), comparison)
    
    # 2. 3D点云可视化 (优化版本 - 只使用前5张图像以提高速度)
    print("  生成3D点云可视化...")
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 准备对象点 (3D点)
    objp = np.zeros((Config.CHESSBOARD_SIZE[0] * Config.CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:Config.CHESSBOARD_SIZE[0], 0:Config.CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= Config.SQUARE_SIZE  # 转换为物理单位(米)
    
    # 限制处理的图像数量以提高速度 (最多5张)
    max_visualization_images = min(5, len(image_files))
    
    # 为不同图像使用不同颜色
    colors = plt.cm.jet(np.linspace(0, 1, max_visualization_images))
    
    # 绘制棋盘格 (优化版本)
    successful_processing = 0
    for i, fname in enumerate(image_files[:max_visualization_images]):
        print(f"    处理图像 {i+1}/{max_visualization_images}: {os.path.basename(fname)}")
        
        # 对于高分辨率图像，使用降采样策略
        img = cv2.imread(fname)
        if img is None:
            continue
            
        original_h, original_w = img.shape[:2]
        
        # 根据图像大小选择合适的降采样比例 (与标定过程保持一致)
        if max(original_h, original_w) > 4000:
            target_size = 1000
        elif max(original_h, original_w) > 2000:
            target_size = 1500
        else:
            target_size = 2000
            
        if max(original_h, original_w) > target_size:
            scale_factor = target_size / max(original_h, original_w)
            new_w = int(original_w * scale_factor)
            new_h = int(original_h * scale_factor)
            img_scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            img_scaled = img
            scale_factor = 1.0
            
        gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)
        
        # 检测棋盘格角点 (带超时机制)
        import threading
        
        def find_corners_thread(gray, pattern_size, flags, result_dict):
            try:
                result_dict['result'] = cv2.findChessboardCorners(
                    gray, pattern_size, 
                    flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
                )
                result_dict['success'] = True
            except Exception as e:
                result_dict['error'] = str(e)
                result_dict['success'] = False
                
        result_dict = {}
        thread = threading.Thread(
            target=find_corners_thread, 
            args=(gray, Config.CHESSBOARD_SIZE, 
                  cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE, 
                  result_dict)
        )
        thread.start()
        
        # 设置合理的超时时间
        timeout = 5.0
        thread.join(timeout=timeout)
        
        # 检查线程是否完成
        if thread.is_alive():
            print(f"    跳过 {os.path.basename(fname)}: 角点检测超时(>{timeout}秒)")
            continue
            
        if not result_dict.get('success', False):
            error_msg = result_dict.get('error', '未知错误')
            print(f"    跳过 {os.path.basename(fname)}: 角点检测出错 - {error_msg}")
            continue
            
        found, corners = result_dict['result']
        
        if not found:
            continue
            
        # 亚像素级精化（在降采样图像上进行）
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # 如果图像被缩放了，需要将角点坐标还原到原始图像尺寸
        if scale_factor != 1.0:
            corners_original = corners_sub.copy()
            corners_original[:, :, 0] = corners_sub[:, :, 0] / scale_factor
            corners_original[:, :, 1] = corners_sub[:, :, 1] / scale_factor
        else:
            corners_original = corners_sub
            
        # 计算外参
        try:
            _, rvec, tvec = cv2.solvePnP(objp, corners_original, mtx, dist)
        except Exception as e:
            print(f"    跳过 {os.path.basename(fname)}: 姿态解算失败 - {str(e)}")
            continue
        
        # 将3D点转换到相机坐标系
        R, _ = cv2.Rodrigues(rvec)
        camera_points = (R @ objp.T).T + tvec.T
        
        # 绘制
        ax.scatter(camera_points[:, 0], camera_points[:, 1], camera_points[:, 2], 
                  c=[colors[i]], s=10, alpha=0.6, label=f'Image {i+1}')
        successful_processing += 1
    
    if successful_processing == 0:
        print("  警告: 所有图像都未能成功处理，跳过3D点云可视化")
        plt.close()
        return
        
    # 绘制相机位置 (原点)
    ax.scatter(0, 0, 0, c='red', s=200, marker='o', label='Camera Position')
    
    # 设置坐标轴
    ax.set_xlabel('X (m)', fontsize=12)
    ax.set_ylabel('Y (m)', fontsize=12)
    ax.set_zlabel('Z (m)', fontsize=12)
    ax.set_title('3D Point Cloud from Calibration Images', fontsize=14)
    
    # 添加图例（但限制数量避免重叠）
    if successful_processing <= 5:
        ax.legend(loc='best', fontsize=8)
    
    # 保存3D可视化
    plt.savefig(os.path.join(vis_dir, "3d_point_cloud.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"可视化结果已保存到 '{vis_dir}':")
    print(f"  - distortion_correction.jpg (畸变校正对比)")
    print(f"  - 3d_point_cloud.png (3D点云分布)")

# ======================
# 第一部分：图像采集
# ======================
def capture_calibration_images():
    """采集标定图像 - 智能质量控制版"""
    print("="*50)
    print("相机标定图像采集工具")
    print("="*50)
    print(f"配置: {Config.CHESSBOARD_SIZE[0]}x{Config.CHESSBOARD_SIZE[1]} 棋盘格, 单格{Config.SQUARE_SIZE*1000:.1f}mm")
    print(f"保存目录: '{Config.CAPTURE_DIR}'")
    print(f"按 [空格] 保存图像, [ESC] 结束采集, [D] 删除上一张")
    print("-"*50)
    
    # 创建保存目录
    os.makedirs(Config.CAPTURE_DIR, exist_ok=True)
    
    # 初始化相机
    cap = cv2.VideoCapture(Config.CAMERA_ID)
    if not cap.isOpened():
        print(f"错误: 无法打开相机ID {Config.CAMERA_ID}")
        print("尝试选项:")
        print("- 普通电脑: CAMERA_ID=0 (内置) 或 1 (外接USB)")
        print("- 手机DroidCam: 通常为1-3, 试不同ID")
        print("- 用手机浏览器访问 http://[手机IP]:4747 作为网络摄像头")
        return False
# 设置分辨率 (手机摄像头可能不支持高分辨率)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.RESOLUTION[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.RESOLUTION[1])
    cap.set(cv2.CAP_PROP_FPS, Config.FPS)
    
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"相机分辨率: {actual_width}x{actual_width}")
    
    saved_images = []
    last_valid_image = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("警告: 无法获取相机画面，尝试重连...")
            cap.release()
            cap = cv2.VideoCapture(Config.CAMERA_ID)
            continue
        
        # 复制原始帧用于保存
        display_frame = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 智能质量评估
        blur_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        contrast = gray.std()
        saturation = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1].mean()
        
        # 检测棋盘格
        found, corners = cv2.findChessboardCorners(
            gray, Config.CHESSBOARD_SIZE,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        # 可视化质量指标
        status_color = (0, 255, 0) if found else (0, 0, 255)
        cv2.putText(display_frame, f"图像: {len(saved_images)}/{Config.MAX_IMAGES}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"清晰度: {blur_var:.1f} (>{Config.MIN_BLUR_VAR})", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display_frame, f"对比度: {contrast:.1f} (>{Config.MIN_CONTRAST})", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display_frame, f"饱和度: {saturation:.1f} (<{Config.MAX_SATURATION})", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # 绘制棋盘格角点 (如果找到)
        if found:
            # 亚像素级精化
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(display_frame, Config.CHESSBOARD_SIZE, corners, found)
            
            # 标记质量状态
            quality_ok = (
                blur_var > Config.MIN_BLUR_VAR and
                contrast > Config.MIN_CONTRAST and
                saturation < Config.MAX_SATURATION
            )
            
            status_text = "优质" if quality_ok else "低质"
            status_color = (0, 255, 0) if quality_ok else (0, 165, 255)
            cv2.putText(display_frame, f"状态: {status_text}", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        else:
            cv2.putText(display_frame, "未检测到棋盘格", (10, 150), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # 显示图像覆盖区域提示
        h, w = frame.shape[:2]
        cv2.rectangle(display_frame, (0, 0), (w//4, h//4), (255, 255, 0), 2)  # 左上
        cv2.rectangle(display_frame, (3*w//4, 0), (w, h//4), (255, 255, 0), 2)  # 右上
        cv2.rectangle(display_frame, (0, 3*h//4), (w//4, h), (255, 255, 0), 2)  # 左下
        cv2.rectangle(display_frame, (3*w//4, 3*h//4), (w, h), (255, 255, 0), 2)  # 右下
        cv2.putText(display_frame, "确保标定板覆盖这些区域!", (w//2-150, h-20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.imshow("标定图像采集 - 按[空格]保存, [ESC]退出, [D]删除上一张", display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == 27 or key == ord('q'):  # ESC 或 Q
            break
        elif key == 32 and found:  # 空格键 (且检测到棋盘格)
            # 质量检查
            quality_ok = (
                blur_var > Config.MIN_BLUR_VAR and
                contrast > Config.MIN_CONTRAST and
                saturation < Config.MAX_SATURATION
            )
            
            if not quality_ok:
                print(f"警告: 图像质量不足! (清晰度={blur_var:.1f}, 对比度={contrast:.1f}, 饱和度={saturation:.1f})")
                print(f"      要求: 清晰度>{Config.MIN_BLUR_VAR}, 对比度>{Config.MIN_CONTRAST}, 饱和度<{Config.MAX_SATURATION}")
                print("      按[空格]强制保存, 或调整后重试")
                continue
            
            if len(saved_images) >= Config.MAX_IMAGES:
                print(f"达到最大图像数量({Config.MAX_IMAGES}), 无法保存更多")
                continue
            
            # 生成唯一文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = os.path.join(Config.CAPTURE_DIR, f"calib_{timestamp}.jpg")
            
            # 保存原始图像 (非显示图像)
            cv2.imwrite(filename, frame)
            saved_images.append(filename)
            last_valid_image = frame.copy()
            
            print(f"已保存: {filename} (共{len(saved_images)}/{Config.MAX_IMAGES})")
            print(f"  质量: 清晰度={blur_var:.1f}, 对比度={contrast:.1f}, 饱和度={saturation:.1f}")
            
        elif key == ord('d') and saved_images:  # D键删除上一张
            last_file = saved_images.pop()
            os.remove(last_file)
            print(f"已删除: {last_file} (剩余{len(saved_images)})")
    
    cap.release()
    cv2.destroyAllWindows()
    
    # 验证是否满足最小图像数量
    if len(saved_images) < Config.MIN_IMAGES:
        print(f"错误: 仅采集到{len(saved_images)}张图像，需要至少{Config.MIN_IMAGES}张")
        print("建议: 重新采集，确保标定板覆盖不同位置和角度")
        return False
    
    print(f"\n成功采集 {len(saved_images)} 张标定图像到 '{Config.CAPTURE_DIR}'")
    print("下一步: 运行标定计算: python camera_calibrator.py --calibrate")
    return True

# ======================
# 第二部分：标定计算 (更新版本)
# ======================
def perform_calibration():
    """执行相机标定计算"""
    print("="*50)
    print("相机标定计算")
    print("="*50)
    
    # 检查图像是否存在 (同时查找.jpg和.JPG文件)
    image_files_jpg = sorted(glob.glob(os.path.join(Config.CAPTURE_DIR, "*.jpg")))
    image_files_JPG = sorted(glob.glob(os.path.join(Config.CAPTURE_DIR, "*.JPG")))
    image_files = image_files_jpg + image_files_JPG
    image_files = sorted(image_files)  # 重新排序
    
    if not image_files:
        print(f"错误: 在 '{Config.CAPTURE_DIR}' 中未找到图像")
        print("请先运行: python camera_calibrator.py --capture")
        return False
    
    print(f"找到 {len(image_files)} 张标定图像")
    if len(image_files) < Config.MIN_IMAGES:
        print(f"警告: 仅 {len(image_files)} 张图像，建议至少 {Config.MIN_IMAGES} 张以获得更好精度")
    
    # 准备对象点 (3D点)
    # 在棋盘格坐标系中，Z=0
    objp = np.zeros((Config.CHESSBOARD_SIZE[0] * Config.CHESSBOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:Config.CHESSBOARD_SIZE[0], 0:Config.CHESSBOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= Config.SQUARE_SIZE  # 转换为物理单位(米)
    
    objpoints = []  # 3D点
    imgpoints = []  # 2D点
    valid_images = []  # 有效图像列表
    
    # 处理所有图像
    print("检测棋盘格角点...")
    for i, fname in enumerate(image_files):
        print(f"正在处理图像 {i+1}/{len(image_files)}: {os.path.basename(fname)}")
        img = cv2.imread(fname)
        if img is None:
            print(f"警告: 无法读取图像 {fname}, 跳过")
            continue
        
        # 对于高分辨率图像(如6000*4000)，使用更激进的降采样策略
        original_h, original_w = img.shape[:2]
        # 根据图像大小选择合适的降采样比例
        if max(original_h, original_w) > 4000:
            # 对于超大图像(>4000像素)，降采样到1000像素
            target_size = 1000
        elif max(original_h, original_w) > 2000:
            # 对于大图像(>2000像素)，降采样到1500像素
            target_size = 1500
        else:
            # 对于较小图像，降采样到2000像素
            target_size = 2000
            
        if max(original_h, original_w) > target_size:
            scale_factor = target_size / max(original_h, original_w)
            new_w = int(original_w * scale_factor)
            new_h = int(original_h * scale_factor)
            # 使用INTER_AREA插值获得更好的缩放效果
            img_scaled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            print(f"  图像尺寸从 {original_w}x{original_h} 降采样到 {new_w}x{new_h} (比例: {scale_factor:.2f})")
        else:
            img_scaled = img
            scale_factor = 1.0
            
        gray = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        
        # 寻找棋盘格角点（优化的参数和超时机制）
        import threading
        
        # 定义一个包装函数用于在线程中执行findChessboardCorners
        def find_corners_thread(gray, pattern_size, flags, result_dict):
            try:
                # 使用优化的参数组合来提高检测速度
                result_dict['result'] = cv2.findChessboardCorners(
                    gray, pattern_size, 
                    flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
                )
                result_dict['success'] = True
            except Exception as e:
                result_dict['error'] = str(e)
                result_dict['success'] = False
                
        # 使用字典来存储线程执行结果
        result_dict = {}
        
        # 创建并启动线程
        thread = threading.Thread(
            target=find_corners_thread, 
            args=(gray, Config.CHESSBOARD_SIZE, 
                  cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE, 
                  result_dict)
        )
        thread.start()
        
        # 根据图像大小调整超时时间
        if max(original_h, original_w) > 4000:
            timeout = 15.0  # 超大图像最多等待15秒
        elif max(original_h, original_w) > 2000:
            timeout = 10.0  # 大图像最多等待10秒
        else:
            timeout = 5.0   # 小图像最多等待5秒
            
        # 等待检测完成
        thread.join(timeout=timeout)
        
        # 检查线程是否完成
        if thread.is_alive():
            print(f"  跳过 {os.path.basename(fname)}: 角点检测超时(>{timeout}秒)")
            continue
            
        if not result_dict.get('success', False):
            error_msg = result_dict.get('error', '未知错误')
            print(f"  跳过 {os.path.basename(fname)}: 角点检测出错 - {error_msg}")
            continue
            
        found, corners = result_dict['result']
        
        if not found:
            print(f"  跳过 {os.path.basename(fname)}: 未检测到棋盘格")
            continue
        
        # 如果图像被缩放了，需要将角点坐标还原到原始图像尺寸
        if scale_factor != 1.0:
            corners_original = corners.copy()
            corners_original[:, :, 0] = corners[:, :, 0] / scale_factor
            corners_original[:, :, 1] = corners[:, :, 1] / scale_factor
        else:
            corners_original = corners
            
        # 亚像素级精化（在降采样图像上进行以提高速度，然后还原坐标）
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # 同样需要将亚像素角点坐标还原到原始图像尺寸
        if scale_factor != 1.0:
            corners_sub_original = corners_sub.copy()
            corners_sub_original[:, :, 0] = corners_sub[:, :, 0] / scale_factor
            corners_sub_original[:, :, 1] = corners_sub[:, :, 1] / scale_factor
        else:
            corners_sub_original = corners_sub
        
        # 保存点（使用原始图像坐标）
        objpoints.append(objp)
        imgpoints.append(corners_sub_original)
        valid_images.append(fname)
        
        print(f"  成功处理图像 {os.path.basename(fname)} (缩放比例: {scale_factor:.2f})")
        
        # 可选: 显示检测结果（使用降采样后的图像以提高显示速度）
        if i % 5 == 0:  # 每5张显示一次
            img_disp = img_scaled.copy()
            cv2.drawChessboardCorners(img_disp, Config.CHESSBOARD_SIZE, corners_sub, found)
            cv2.putText(img_disp, f"处理中: {i+1}/{len(image_files)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # 调整显示窗口大小以便更好地查看
            display_scale = min(1.0, 600.0 / max(img_disp.shape[:2]))
            if display_scale < 1.0:
                img_display = cv2.resize(img_disp, None, fx=display_scale, fy=display_scale, interpolation=cv2.INTER_AREA)
            else:
                img_display = img_disp
            cv2.imshow("角点检测", img_display)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC键退出
                cv2.destroyAllWindows()
                break
    
    cv2.destroyAllWindows()
    
    if len(objpoints) < Config.MIN_IMAGES:
        print(f"错误: 仅 {len(objpoints)} 张有效图像，需要至少 {Config.MIN_IMAGES} 张")
        return False
    
    print(f"\n成功检测 {len(objpoints)} 张图像的角点")
    
    # 执行相机标定
    print("计算相机参数...")
    start_time = cv2.getTickCount()
    
    try:
        # 获取图像尺寸
        h, w = cv2.imread(valid_images[0]).shape[:2]
        
        # 执行标定
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, (w, h),
            None, None,
            flags=Config.CALIB_FLAGS,
            criteria=(cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
        )
        
        # 计算处理时间
        elapsed = (cv2.getTickCount() - start_time) / cv2.getTickFrequency()
        
        # 计算重投影误差 (更准确的评估)
        total_error = 0
        total_points = 0
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) ** 2
            total_error += error
            total_points += len(objpoints[i])
        
        mean_error = np.sqrt(total_error / total_points)
        
        print(f"\n标定完成! (耗时: {elapsed:.2f}秒)")
        print(f"重投影误差 (RMS): {ret:.4f} 像素")
        print(f"平均重投影误差: {mean_error:.4f} 像素")
        print(f"内参矩阵 (K):")
        print(mtx)
        print(f"畸变系数 (k1,k2,p1,p2,k3,s1,s2,s3,s4):")
        print(dist.ravel())
        
        # 评估标定质量
        quality = "优秀" if ret < 0.3 else "良好" if ret < 0.5 else "一般" if ret < 1.0 else "较差"
        print(f"\n标定质量: {quality}")
        
        if ret > 1.0:
            print("警告: 重投影误差较大(>1.0像素)，标定结果可能不准确!")
            print("建议: 重新采集更多图像，尤其覆盖图像边缘区域")
        
        # 保存标定结果
        output_dir = save_calibration_results(mtx, dist, rvecs, tvecs, ret, mean_error, (w, h), valid_images)
        
        # 验证定位精度
        print("\n" + "="*50)
        print("定位精度验证")
        print("="*50)
        # 在不同工作距离下验证定位精度
        for distance in [0.5, 1.0, 1.5, 2.0]:
            verify_localization_accuracy(mtx, dist, working_distance=distance, board_size_mm=Config.SQUARE_SIZE*1000)
        
        # 可视化验证
        visualize_calibration_results(mtx, dist, valid_images, output_dir)
        
        return True
    except Exception as e:
        print(f"标定过程出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

# ======================
# 程序入口
# ======================
def main():
    parser = argparse.ArgumentParser(description='相机标定工具')
    parser.add_argument('--capture', action='store_true', help='采集标定图像')
    parser.add_argument('--calibrate', action='store_true', help='执行标定计算')
    parser.add_argument('--all', action='store_true', help='采集+标定 (全自动)')
    parser.add_argument('--image-dir', type=str, default='calibration_images', 
                       help='标定照片所在的文件夹路径 (默认: calibration_images)')
    args = parser.parse_args()
    args.calibrate = True
    # 允许通过命令行覆盖配置
    # if args.image_dir:
    #     Config.CAPTURE_DIR = args.image_dir
    #     print(f"使用自定义图像目录: {Config.CAPTURE_DIR}")
    
    if not any([args.capture, args.calibrate, args.all]):
        print("用法: python camera_calibrator.py [选项]")
        print("选项:")
        print("  --capture    采集标定图像")
        print("  --calibrate  执行标定计算")
        print("  --all        全自动流程 (先采集后标定)")
        print("\n示例:")
        print("  1. 采集图像: python camera_calibrator_single.py --capture")
        print("  2. 执行标定: python camera_calibrator_single.py --calibrate")
        print("  3. 全自动:   python camera_calibrator_single.py --all")
        return

    # 全自动流程
    if args.all:
        print("运行全自动标定流程...")
        if capture_calibration_images():
            perform_calibration()
        return
    
    # 仅采集
    if args.capture:
        capture_calibration_images()
        return
    
    # 仅标定
    if args.calibrate:
        perform_calibration()
        return

if __name__ == "__main__":
    # 检查OpenCV版本
    print(f"OpenCV 版本: {cv2.__version__}")
    if cv2.__version__ < '4.5':
        print("警告: 检测到旧版OpenCV (需要4.5+)，某些功能可能受限")
        print("建议升级: pip install --upgrade opencv-python")
    
    main()