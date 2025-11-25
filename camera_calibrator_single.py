import cv2
import numpy as np
import glob
import os
import argparse
import json
import matplotlib.pyplot as plt
from datetime import datetime

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='相机标定与畸变校正工具')
    parser.add_argument('--image_folder', type=str, default='./calibration_images',
                        help='标定图像文件夹路径')
    parser.add_argument('--output_dir', type=str, default='./calibration_results',
                        help='结果输出目录')
    parser.add_argument('--chessboard_width', type=int, default=9,
                        help='棋盘格内角点宽度 (例如9表示10x7棋盘格的宽度方向有9个内角点)')
    parser.add_argument('--chessboard_height', type=int, default=6,
                        help='棋盘格内角点高度')
    parser.add_argument('--square_size', type=float, default=18.1,
                        help='棋盘格方格大小(mm)')
    parser.add_argument('--camera_index', type=int, default=0,
                        help='摄像头索引 (0通常是内置摄像头)')
    parser.add_argument('--display_size', type=int, nargs=2, default=[800, 600],
                        help='显示窗口大小 [宽 高]')
    parser.add_argument('--save_corrected', action='store_true',
                        help='保存校正后的实时图像')
    parser.add_argument('--validate_calibration', action='store_true',
                        help='执行标定验证并生成报告')
    return parser.parse_args()

def create_output_directories(output_dir):
    """创建输出目录结构"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'detected_corners'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'corrected_images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'validation'), exist_ok=True)
    return output_dir

def detect_chessboard_corners(image_path, chessboard_size, criteria, output_dir=None, display_size=(800, 600)):
    """
    检测单张图像中的棋盘格角点
    
    Args:
        image_path: 图像路径
        chessboard_size: (width, height) 棋盘格内角点数量
        criteria: 亚像素精确化标准
        output_dir: 保存检测结果的目录
        display_size: 显示窗口大小
        
    Returns:
        success: 是否成功检测到角点
        corners: 检测到的角点
        image: 带有标记的图像
    """
    try:
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            print(f"警告: 无法加载图像 {image_path}")
            return False, None, None
            
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 寻找棋盘格角点
        ret, corners = cv2.findChessboardCorners(
            gray, 
            chessboard_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + 
            cv2.CALIB_CB_FAST_CHECK + 
            cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        # 如果找到角点，进行亚像素精确化
        if ret:
            corners_refined = cv2.cornerSubPix(
                gray, 
                corners, 
                (11, 11), 
                (-1, -1), 
                criteria
            )
            
            # 绘制检测结果
            img_display = img.copy()
            cv2.drawChessboardCorners(img_display, chessboard_size, corners_refined, ret)
            
            # 添加文字信息
            cv2.putText(img_display, f"Found: {chessboard_size[0]}x{chessboard_size[1]} corners", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(img_display, f"Image: {os.path.basename(image_path)}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 保存检测结果
            if output_dir:
                output_path = os.path.join(output_dir, 'detected_corners', 
                                         f"detected_{os.path.basename(image_path)}")
                cv2.imwrite(output_path, img_display)
                print(f"✓ 保存角点检测结果到: {output_path}")
            
            # 显示结果
            cv2.namedWindow('Chessboard Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Chessboard Detection', display_size[0], display_size[1])
            cv2.imshow('Chessboard Detection', img_display)
            cv2.waitKey(300)  # 短暂停留以便观察
            
            return True, corners_refined, img_display
        
        else:
            print(f"✗ 未在图像中找到 {chessboard_size[0]}x{chessboard_size[1]} 棋盘格角点: {image_path}")
            # 显示未找到的图像
            img_display = img.copy()
            cv2.putText(img_display, "Chessboard Not Found", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.putText(img_display, f"Expected: {chessboard_size[0]}x{chessboard_size[1]} corners", 
                       (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(img_display, f"Image: {os.path.basename(image_path)}", 
                       (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.namedWindow('Chessboard Detection', cv2.WINDOW_NORMAL)
            cv2.resizeWindow('Chessboard Detection', display_size[0], display_size[1])
            cv2.imshow('Chessboard Detection', img_display)
            cv2.waitKey(200)
            
            return False, None, img_display
            
    except Exception as e:
        print(f"处理图像 {image_path} 时出错: {str(e)}")
        return False, None, None

def calibrate_camera(objpoints, imgpoints, image_size, square_size_mm, chessboard_size):
    """执行相机标定并返回所有参数和统计信息"""
    try:
        # 首先进行基础标定
        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, 
            imgpoints, 
            image_size[::-1],  # 注意：OpenCV需要 (width, height)
            None, 
            None,
            flags=cv2.CALIB_RATIONAL_MODEL
        )
        
        # 计算重投影误差
        mean_error = 0
        errors = []
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            errors.append(error)
            mean_error += error
        
        mean_error /= len(objpoints)
        
        # 计算每张图像的误差统计
        error_stats = {
            'mean_error': mean_error,
            'max_error': max(errors),
            'min_error': min(errors),
            'std_error': np.std(errors),
            'per_image_errors': errors
        }
        
        # 计算最优相机矩阵
        alpha = 0.5  # 0=无黑边, 1=保留所有像素
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(
            mtx, 
            dist, 
            image_size[::-1], 
            alpha, 
            image_size[::-1]
        )
        
        # 创建标定报告
        report = {
            'calibration_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'chessboard_size': {
                'width': chessboard_size[0],
                'height': chessboard_size[1],
                'square_size_mm': square_size_mm
            },
            'image_count': len(objpoints),
            'image_size': {
                'width': image_size[1],
                'height': image_size[0]
            },
            'reprojection_error': {
                'overall_mean': mean_error,
                'max': max(errors),
                'min': min(errors),
                'std': np.std(errors)
            },
            'camera_matrix': mtx.tolist(),
            'distortion_coefficients': dist.flatten().tolist(),
            'optimal_camera_matrix': newcameramtx.tolist(),
            'roi': [int(x) for x in roi]
        }
        
        return {
            'success': True,
            'ret': ret,
            'mtx': mtx,
            'dist': dist,
            'rvecs': rvecs,
            'tvecs': tvecs,
            'newcameramtx': newcameramtx,
            'roi': roi,
            'error_stats': error_stats,
            'report': report
        }
        
    except Exception as e:
        print(f"标定过程中出错: {str(e)}")
        return {
            'success': False,
            'error': str(e)
        }

def validate_calibration(calib_result, objpoints, imgpoints, output_dir, image_paths):
    """验证标定结果并生成可视化报告"""
    if not calib_result['success']:
        print("标定失败，无法验证")
        return
        
    mtx = calib_result['mtx']
    dist = calib_result['dist']
    newcameramtx = calib_result['newcameramtx']
    roi = calib_result['roi']
    errors = calib_result['error_stats']['per_image_errors']
    
    # 创建验证结果目录
    val_dir = os.path.join(output_dir, 'validation')
    
    # 1. 重投影误差可视化
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(len(errors)), errors, 'b-o', linewidth=2)
    plt.axhline(y=1.0, color='r', linestyle='--', label='阈值 (1.0像素)')
    plt.xlabel('图像索引')
    plt.ylabel('重投影误差 (像素)')
    plt.title('每张图像的重投影误差')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.hist(errors, bins=10, alpha=0.7, color='blue', edgecolor='black')
    plt.axvline(x=np.mean(errors), color='r', linestyle='--', label=f'平均误差: {np.mean(errors):.3f}')
    plt.xlabel('重投影误差 (像素)')
    plt.ylabel('图像数量')
    plt.title('重投影误差分布')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(val_dir, 'reprojection_errors.png'))
    plt.close()
    
    # 2. 选择3张代表性图像进行可视化
    # 选择: 误差最小、平均、最大的图像
    sorted_indices = np.argsort(errors)
    indices_to_show = [
        sorted_indices[0],  # 最小误差
        sorted_indices[len(sorted_indices)//2],  # 中等误差
        sorted_indices[-1]  # 最大误差
    ]
    
    plt.figure(figsize=(15, 10))
    for i, idx in enumerate(indices_to_show, 1):
        img = cv2.imread(image_paths[idx])
        if img is None:
            continue
            
        # 重投影角点
        projected_points, _ = cv2.projectPoints(
            objpoints[idx], 
            calib_result['rvecs'][idx], 
            calib_result['tvecs'][idx], 
            mtx, 
            dist
        )
        projected_points = projected_points.reshape(-1, 2)
        
        # 绘制原始角点和重投影角点
        img_vis = img.copy()
        for j, (orig_pt, proj_pt) in enumerate(zip(imgpoints[idx], projected_points)):
            orig_pt = tuple(orig_pt[0].astype(int))
            proj_pt = tuple(proj_pt.astype(int))
            
            # 原始角点 (绿色)
            cv2.circle(img_vis, orig_pt, 5, (0, 255, 0), -1)
            # 重投影角点 (红色)
            cv2.circle(img_vis, proj_pt, 3, (0, 0, 255), -1)
            # 连接线 (蓝色)
            cv2.line(img_vis, orig_pt, proj_pt, (255, 0, 0), 1)
            
            # 每5个点标记一个数字
            if j % 5 == 0:
                cv2.putText(img_vis, str(j), (orig_pt[0]+10, orig_pt[1]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 添加标题
        title = f"Image {idx}: Reprojection Error = {errors[idx]:.3f} pixels"
        cv2.putText(img_vis, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # 保存验证图像
        output_path = os.path.join(val_dir, f"validation_img_{idx}_error_{errors[idx]:.3f}.jpg")
        cv2.imwrite(output_path, img_vis)
        
        # 显示在matplotlib中
        img_rgb = cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB)
        plt.subplot(2, 2, i)
        plt.imshow(img_rgb)
        plt.title(title)
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(val_dir, 'validation_visualization.png'))
    plt.close()
    
    # 3. 畸变校正效果对比
    sample_idx = sorted_indices[len(sorted_indices)//2]  # 选择中等误差的图像
    sample_img = cv2.imread(image_paths[sample_idx])
    
    if sample_img is not None:
        # 畸变校正
        h, w = sample_img.shape[:2]
        mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
        dst = cv2.remap(sample_img, mapx, mapy, cv2.INTER_LINEAR)
        
        # 裁剪
        x, y, w_roi, h_roi = roi
        if w_roi > 0 and h_roi > 0:
            dst_cropped = dst[y:y+h_roi, x:x+w_roi]
        else:
            dst_cropped = dst.copy()
        
        # 创建对比图像
        plt.figure(figsize=(15, 8))
        
        # 原始图像
        plt.subplot(1, 3, 1)
        plt.imshow(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
        plt.title(f'原始图像 (误差: {errors[sample_idx]:.3f}像素)')
        plt.axis('off')
        
        # 校正后完整图像
        plt.subplot(1, 3, 2)
        plt.imshow(cv2.cvtColor(dst, cv2.COLOR_BGR2RGB))
        plt.title('校正后 (完整)')
        plt.axis('off')
        
        # 校正后裁剪图像
        plt.subplot(1, 3, 3)
        plt.imshow(cv2.cvtColor(dst_cropped, cv2.COLOR_BGR2RGB))
        plt.title('校正后 (裁剪)')
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(val_dir, 'undistortion_comparison.png'))
        plt.close()
        
        # 保存校正图像
        cv2.imwrite(os.path.join(val_dir, 'undistorted_full.jpg'), dst)
        cv2.imwrite(os.path.join(val_dir, 'undistorted_cropped.jpg'), dst_cropped)
    
    print(f"✓ 标定验证报告已保存到: {val_dir}")

def save_calibration_results(calib_result, output_dir, image_paths=None):
    """保存标定结果到多种格式"""
    if not calib_result['success']:
        print("标定失败，无法保存结果")
        return
        
    # 1. 保存为NPZ格式 (推荐用于程序加载)
    np.savez(
        os.path.join(output_dir, 'calibration_data.npz'),
        camera_matrix=calib_result['mtx'],
        distortion_coefficients=calib_result['dist'],
        rotation_vectors=np.array(calib_result['rvecs'], dtype=object),
        translation_vectors=np.array(calib_result['tvecs'], dtype=object),
        optimal_camera_matrix=calib_result['newcameramtx'],
        roi=np.array(calib_result['roi']),
        reprojection_errors=np.array(calib_result['error_stats']['per_image_errors'])
    )
    
    # 2. 保存为JSON格式 (人类可读)
    with open(os.path.join(output_dir, 'calibration_report.json'), 'w') as f:
        json.dump(calib_result['report'], f, indent=4)
    
    # 3. 保存为YAML格式 (OpenCV兼容)
    fs = cv2.FileStorage(os.path.join(output_dir, 'calibration.yaml'), cv2.FILE_STORAGE_WRITE)
    fs.write("camera_matrix", calib_result['mtx'])
    fs.write("distortion_coefficients", calib_result['dist'].reshape(1, -1))
    fs.write("optimal_camera_matrix", calib_result['newcameramtx'])
    fs.write("roi", np.array(calib_result['roi']))
    fs.write("reprojection_error", calib_result['error_stats']['mean_error'])
    fs.release()
    
    # 4. 保存为文本格式
    with open(os.path.join(output_dir, 'calibration_summary.txt'), 'w') as f:
        f.write("="*50 + "\n")
        f.write("相机标定结果摘要\n")
        f.write("="*50 + "\n\n")
        
        f.write(f"标定日期: {calib_result['report']['calibration_date']}\n")
        f.write(f"使用图像数量: {calib_result['report']['image_count']}\n")
        f.write(f"图像分辨率: {calib_result['report']['image_size']['width']}x{calib_result['report']['image_size']['height']}\n")
        f.write(f"棋盘格规格: {calib_result['report']['chessboard_size']['width']}x{calib_result['report']['chessboard_size']['height']} 内角点\n")
        f.write(f"方格尺寸: {calib_result['report']['chessboard_size']['square_size_mm']} mm\n\n")
        
        f.write("重投影误差统计:\n")
        f.write(f"  平均误差: {calib_result['error_stats']['mean_error']:.4f} 像素\n")
        f.write(f"  最大误差: {calib_result['error_stats']['max_error']:.4f} 像素\n")
        f.write(f"  最小误差: {calib_result['error_stats']['min_error']:.4f} 像素\n")
        f.write(f"  标准差: {calib_result['error_stats']['std_error']:.4f} 像素\n\n")
        
        f.write("相机内参矩阵:\n")
        f.write(str(calib_result['mtx']) + "\n\n")
        
        f.write("畸变系数 (k1,k2,p1,p2,k3,k4,k5,k6):\n")
        f.write(str(calib_result['dist'].flatten()) + "\n\n")
        
        f.write("最优相机矩阵 (用于校正):\n")
        f.write(str(calib_result['newcameramtx']) + "\n\n")
        
        f.write("ROI (x,y,width,height):\n")
        f.write(str(calib_result['roi']) + "\n")
    
    # 5. 保存角点检测图像列表
    if image_paths:
        with open(os.path.join(output_dir, 'detected_images.txt'), 'w') as f:
            for i, path in enumerate(image_paths):
                f.write(f"Image {i+1}: {path}\n")
    
    print(f"✓ 标定结果已保存到: {output_dir}")
    print(f"  - NPZ格式: calibration_data.npz (推荐用于程序加载)")
    print(f"  - JSON格式: calibration_report.json (人类可读报告)")
    print(f"  - YAML格式: calibration.yaml (OpenCV兼容)")
    print(f"  - 文本摘要: calibration_summary.txt")

def live_undistortion_demo(camera_index, mtx, dist, newcameramtx, roi, output_dir, save_corrected=False):
    """实时畸变校正演示"""
    print("\n" + "="*50)
    print("启动实时畸变校正演示")
    print("="*50)
    print("控制说明:")
    print("  'q' - 退出程序")
    print("  's' - 保存当前校正图像")
    print("  'r' - 重置ROI裁剪")
    print("  '1' - 显示原始图像")
    print("  '2' - 显示校正后完整图像")
    print("  '3' - 显示校正后裁剪图像")
    print("  '4' - 显示对比视图")
    
    # 尝试打开摄像头
    camera = cv2.VideoCapture(camera_index)
    
    # 尝试备用摄像头索引
    if not camera.isOpened():
        print(f"错误: 无法打开摄像头 (索引 {camera_index})")
        # 尝试其他常见索引
        for idx in [1, 2, -1]:
            print(f"尝试备用摄像头索引: {idx}")
            camera = cv2.VideoCapture(idx)
            if camera.isOpened():
                print(f"成功打开摄像头 (索引 {idx})")
                break
    
    if not camera.isOpened():
        print("错误: 无法打开任何摄像头")
        print("建议: 检查摄像头连接或使用 --camera_index 参数指定正确的索引")
        return
    
    # 设置摄像头分辨率
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # 获取实际分辨率
    frame_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"摄像头分辨率: {frame_width}x{frame_height}")
    
    # 预计算校正映射
    mapx, mapy = cv2.initUndistortRectifyMap(
        mtx, 
        dist, 
        None, 
        newcameramtx, 
        (frame_width, frame_height), 
        5
    )
    
    display_mode = 4  # 4=对比视图
    frame_count = 0
    saved_count = 0
    
    while True:
        ret, frame = camera.read()
        if not ret:
            print("错误: 无法从摄像头获取帧")
            break
            
        frame_count += 1
        
        # 应用畸变校正
        dst = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        
        # 裁剪图像
        x, y, w_roi, h_roi = roi
        if w_roi > 0 and h_roi > 0 and display_mode in [3, 4]:
            dst_cropped = dst[y:y+h_roi, x:x+w_roi]
        else:
            dst_cropped = dst.copy()
        
        # 根据显示模式显示不同的视图
        if display_mode == 1:  # 原始图像
            display_frame = frame.copy()
            cv2.putText(display_frame, "原始图像 (有畸变)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        elif display_mode == 2:  # 校正后完整图像
            display_frame = dst.copy()
            cv2.putText(display_frame, "校正后 (完整图像)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # 绘制ROI区域
            cv2.rectangle(display_frame, (x, y), (x+w_roi, y+h_roi), (0, 255, 255), 2)
            
        elif display_mode == 3:  # 校正后裁剪图像
            display_frame = dst_cropped.copy()
            cv2.putText(display_frame, f"校正后 (裁剪 {w_roi}x{h_roi})", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        elif display_mode == 4:  # 对比视图
            # 创建对比视图
            display_frame = np.zeros((frame_height, frame_width*2, 3), dtype=np.uint8)
            
            # 左侧: 原始图像
            display_frame[:, :frame_width] = frame
            cv2.putText(display_frame[:, :frame_width], "原始图像 (有畸变)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # 右侧: 校正后裁剪图像
            right_width = min(frame_width, w_roi)
            right_height = min(frame_height, h_roi)
            display_frame[:right_height, frame_width:frame_width+right_width] = dst_cropped[:right_height, :right_width]
            cv2.putText(display_frame[:, frame_width:], f"校正后 (裁剪 {w_roi}x{h_roi})", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # 添加状态信息
        status_text = [
            f"帧: {frame_count}",
            f"模式: {display_mode} (按1-4切换)",
            f"按 's' 保存, 'q' 退出"
        ]
        
        for i, text in enumerate(status_text):
            cv2.putText(display_frame, text, (10, frame_height - 20 - i*25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        
        # 显示
        cv2.namedWindow('Camera Undistortion', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Camera Undistortion', 1400, 800)
        cv2.imshow('Camera Undistortion', display_frame)
        
        # 处理按键
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # 退出
            break
        elif key == ord('s'):  # 保存
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            if display_mode == 1:
                filename = f"original_{timestamp}.jpg"
                cv2.imwrite(os.path.join(output_dir, 'corrected_images', filename), frame)
            elif display_mode == 2:
                filename = f"undistorted_full_{timestamp}.jpg"
                cv2.imwrite(os.path.join(output_dir, 'corrected_images', filename), dst)
            elif display_mode == 3:
                filename = f"undistorted_cropped_{timestamp}.jpg"
                cv2.imwrite(os.path.join(output_dir, 'corrected_images', filename), dst_cropped)
            elif display_mode == 4:
                filename = f"comparison_{timestamp}.jpg"
                cv2.imwrite(os.path.join(output_dir, 'corrected_images', filename), display_frame)
            
            print(f"✓ 保存图像: {filename}")
            saved_count += 1
        elif key == ord('r'):  # 重置ROI
            print("ROI重置为原始值:", roi)
        elif key in [ord('1'), ord('2'), ord('3'), ord('4')]:  # 切换显示模式
            display_mode = int(chr(key))
            print(f"切换到显示模式 {display_mode}")
        elif key == ord('p'):  # 暂停
            print("暂停 - 按任意键继续")
            cv2.waitKey(0)
    
    # 清理
    camera.release()
    cv2.destroyAllWindows()
    
    print("\n" + "="*50)
    print("实时畸变校正演示结束")
    print(f"总共处理帧数: {frame_count}")
    print(f"保存图像数量: {saved_count}")
    print("="*50)

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 创建输出目录
    output_dir = create_output_directories(args.output_dir)
    print(f"✓ 输出目录: {output_dir}")
    
    # 显示参数
    print("\n" + "="*50)
    print("标定参数:")
    print(f"  棋盘格尺寸: {args.chessboard_width}x{args.chessboard_height} 内角点")
    print(f"  方格大小: {args.square_size} mm")
    print(f"  图像文件夹: {args.image_folder}")
    print(f"  摄像头索引: {args.camera_index}")
    print("="*50)
    
    # 检查图像文件夹
    if not os.path.exists(args.image_folder):
        print(f"错误: 图像文件夹不存在 - {args.image_folder}")
        print("建议: 创建文件夹并将标定图像放入其中，或使用 --image_folder 参数指定正确的路径")
        return
    
    # 设置寻找亚像素角点的参数
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # 准备棋盘格世界坐标
    chessboard_size = (args.chessboard_width, args.chessboard_height)
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp = objp * args.square_size  # 转换为毫米
    
    # 存储角点
    objpoints = []  # 3D点
    imgpoints = []  # 2D点
    valid_image_paths = []  # 有效图像路径
    detected_count = 0
    
    # 加载图像
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    images = []
    for ext in image_extensions:
        images.extend(glob.glob(os.path.join(args.image_folder, ext)))
    
    print(f"\n找到 {len(images)} 张候选图像")
    
    if len(images) == 0:
        print(f"错误: 在 {args.image_folder} 中未找到图像")
        print("支持的格式: JPG, PNG, BMP")
        return
    
    # 按文件名排序
    images.sort()
    
    # 处理每张图像
    print("\n" + "="*50)
    print("开始角点检测...")
    print("="*50)
    
    for i, fname in enumerate(images):
        print(f"\n处理图像 [{i+1}/{len(images)}]: {os.path.basename(fname)}")
        success, corners, _ = detect_chessboard_corners(
            fname, 
            chessboard_size, 
            criteria, 
            output_dir,
            tuple(args.display_size)
        )
        
        if success:
            detected_count += 1
            objpoints.append(objp.copy())
            imgpoints.append(corners)
            valid_image_paths.append(fname)
            print(f"✓ 检测成功! 累计: {detected_count}/{len(images)}")
        else:
            print(f"✗ 检测失败")
    
    cv2.destroyAllWindows()
    
    # 检查结果
    print("\n" + "="*50)
    print(f"角点检测完成: {detected_count}/{len(images)} 张图像成功")
    print("="*50)
    
    if detected_count < 10:
        print(f"警告: 仅 {detected_count} 张图像成功检测到角点")
        print("建议: 至少需要10-15张良好分布的图像以获得准确标定")
        if detected_count < 3:
            print("错误: 无法标定 - 需要至少3张图像")
            return
    
    # 确保我们有图像尺寸
    if valid_image_paths:
        sample_img = cv2.imread(valid_image_paths[0])
        image_size = sample_img.shape[:2]  # (高度, 宽度)
    else:
        print("错误: 无有效图像用于标定")
        return
    
    # 相机标定
    print("\n" + "="*50)
    print("开始相机标定...")
    print("="*50)
    
    calib_result = calibrate_camera(objpoints, imgpoints, image_size, args.square_size, chessboard_size)
    
    if not calib_result['success']:
        print(f"错误: 标定失败 - {calib_result['error']}")
        return
    
    print("\n" + "="*50)
    print("标定成功!")
    print(f"平均重投影误差: {calib_result['error_stats']['mean_error']:.4f} 像素")
    print(f"最大重投影误差: {calib_result['error_stats']['max_error']:.4f} 像素")
    print("="*50)
    
    # 保存结果
    save_calibration_results(calib_result, output_dir, valid_image_paths)
    
    # 验证标定 (如果请求)
    if args.validate_calibration:
        print("\n" + "="*50)
        print("生成标定验证报告...")
        print("="*50)
        validate_calibration(calib_result, objpoints, imgpoints, output_dir, valid_image_paths)
    
    # 实时畸变校正演示 (如果请求)
    print("\n" + "="*50)
    choice = input("是否启动实时畸变校正演示? (y/n): ").strip().lower()
    print("="*50)
    
    if choice.startswith('y'):
        live_undistortion_demo(
            args.camera_index,
            calib_result['mtx'],
            calib_result['dist'],
            calib_result['newcameramtx'],
            calib_result['roi'],
            output_dir,
            args.save_corrected
        )
    
    print("\n" + "="*50)
    print("程序完成!")
    print(f"所有结果已保存到: {output_dir}")
    print("="*50)

if __name__ == "__main__":
    main()

'''
D:/anaconda3/envs/retinaface_env/python.exe camera_calibrator_single.py --chessboard_width 11 --chessboard_height 8 --square_size 100.0 --image_folder left
'''