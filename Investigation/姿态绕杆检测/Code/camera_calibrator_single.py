# ======================
# 第一部分：图像采集
# ======================
def capture_calibration_images(image_dir=None):
    """采集标定图像 - 智能质量控制版"""
    # 如果提供了image_dir参数，则使用它，否则使用Config中的默认值
    capture_dir = image_dir if image_dir is not None else Config.CAPTURE_DIR
    
    print("="*50)
    print("相机标定图像采集工具")
    print("="*50)
    print(f"配置: {Config.CHESSBOARD_SIZE[0]}x{Config.CHESSBOARD_SIZE[1]} 棋盘格, 单格{Config.SQUARE_SIZE*1000:.1f}mm")
    print(f"保存目录: '{capture_dir}'")
    print(f"按 [空格] 保存图像, [ESC] 结束采集, [D] 删除上一张")
    print("-"*50)
    
    # 创建保存目录
    os.makedirs(capture_dir, exist_ok=True)
    
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
    
    print(f"相机分辨率: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    
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
            filename = os.path.join(capture_dir, f"calib_{timestamp}.jpg")
            
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
    
    print(f"\n成功采集 {len(saved_images)} 张标定图像到 '{capture_dir}'")
    print("下一步: 运行标定计算: python camera_calibrator.py --calibrate")
    return True


# ======================
# 第二部分：标定计算
# ======================
def perform_calibration(image_dir=None):
    """执行相机标定计算"""
    # 如果提供了image_dir参数，则使用它，否则使用Config中的默认值
    capture_dir = image_dir if image_dir is not None else Config.CAPTURE_DIR
    
    print("="*50)
    print("相机标定计算")
    print("="*50)
    
    # 检查图像是否存在
    image_files = sorted(glob.glob(os.path.join(capture_dir, "*.jpg")))
    if not image_files:
        print(f"错误: 在 '{capture_dir}' 中未找到图像")
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
        img = cv2.imread(fname)
        if img is None:
            print(f"警告: 无法读取图像 {fname}, 跳过")
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # 寻找棋盘格角点
        found, corners = cv2.findChessboardCorners(
            gray, Config.CHESSBOARD_SIZE,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_FAST_CHECK | cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        
        if not found:
            print(f"  跳过 {os.path.basename(fname)}: 未检测到棋盘格")
            continue
        
        # 亚像素级精化
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # 保存点
        objpoints.append(objp)
        imgpoints.append(corners_sub)
        valid_images.append(fname)
        
        # 可选: 显示检测结果
        if i % 5 == 0:  # 每5张显示一次
            img_disp = img.copy()
            cv2.drawChessboardCorners(img_disp, Config.CHESSBOARD_SIZE, corners_sub, found)
            cv2.putText(img_disp, f"处理中: {i+1}/{len(image_files)}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imshow("角点检测", img_disp)
            cv2.waitKey(1)
    
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
        for i in range(len(objpoints)):
            imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
            error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
            total_error += error
        mean_error = total_error / len(objpoints)
        
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
        save_calibration_results(mtx, dist, rvecs, tvecs, ret, mean_error, (w, h), valid_images)
        
        # 可视化验证
        visualize_calibration_results(mtx, dist, valid_images)
        
        return True
        
    except Exception as e:
        print(f"标定错误: {e}")
        print("常见原因:")
        print("- 标定板在图像中太小 (<100像素宽)")
        print("- 图像数量不足 (<10张)")
        print("- 所有图像中棋盘格姿态相似")
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
    
    if not any([args.capture, args.calibrate, args.all]):
        print("用法: python camera_calibrator.py [选项]")
        print("选项:")
        print("  --capture    采集标定图像")
        print("  --calibrate  执行标定计算")
        print("  --all        全自动流程 (先采集后标定)")
        print("  --image-dir  标定照片所在的文件夹路径 (默认: calibration_images)")
        print("\n示例:")
        print("  1. 采集图像: python camera_calibrator.py --capture")
        print("  2. 执行标定: python camera_calibrator.py --calibrate")
        print("  3. 全自动:   python camera_calibrator.py --all")
        print("  4. 指定目录: python camera_calibrator.py --calibrate --image-dir my_images")
        return
    
    # 全自动流程
    if args.all:
        print("运行全自动标定流程...")
        if capture_calibration_images(args.image_dir):
            perform_calibration(args.image_dir)
        return
    
    # 仅采集
    if args.capture:
        capture_calibration_images(args.image_dir)
        return
    
    # 仅标定
    if args.calibrate:
        perform_calibration(args.image_dir)
        return