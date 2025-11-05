import numpy as np
import cv2
import glob
import yaml

def stereo_calibration():
    # 棋盘格尺寸 (内角点数量)
    chessboard_size = (9, 6)  # 根据你的棋盘格调整
    square_size = 0.025  # 棋盘格方格大小，单位：米
    
    # 准备对象点
    objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2)
    objp *= square_size
    
    # 存储对象点和图像点
    objpoints = []  # 3D点
    imgpoints_left = []  # 左相机2D点
    imgpoints_right = []  # 右相机2D点
    
    # 读取图像
    left_images = sorted(glob.glob('left/*.jpg'))  # 左相机图像路径
    right_images = sorted(glob.glob('right/*.jpg'))  # 右相机图像路径
    
    print(f"找到左相机图像: {len(left_images)} 张")
    print(f"找到右相机图像: {len(right_images)} 张")
    
    # 检测角点
    for i, (left_img_path, right_img_path) in enumerate(zip(left_images, right_images)):
        img_left = cv2.imread(left_img_path)
        img_right = cv2.imread(right_img_path)
        
        gray_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
        gray_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
        
        # 查找棋盘格角点
        ret_left, corners_left = cv2.findChessboardCorners(gray_left, chessboard_size, None)
        ret_right, corners_right = cv2.findChessboardCorners(gray_right, chessboard_size, None)
        
        if ret_left and ret_right:
            print(f"图像对 {i+1}: 成功检测到角点")
            
            # 亚像素级角点检测
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_left_refined = cv2.cornerSubPix(gray_left, corners_left, (11, 11), (-1, -1), criteria)
            corners_right_refined = cv2.cornerSubPix(gray_right, corners_right, (11, 11), (-1, -1), criteria)
            
            objpoints.append(objp)
            imgpoints_left.append(corners_left_refined)
            imgpoints_right.append(corners_right_refined)
            
            # 可视化角点
            cv2.drawChessboardCorners(img_left, chessboard_size, corners_left_refined, ret_left)
            cv2.drawChessboardCorners(img_right, chessboard_size, corners_right_refined, ret_right)
            
            # 显示图像
            img_combined = np.hstack((img_left, img_right))
            cv2.imshow('Detected Corners', img_combined)
            cv2.waitKey(500)
        else:
            print(f"图像对 {i+1}: 未检测到角点")
    
    cv2.destroyAllWindows()
    
    if len(objpoints) < 5:
        print("有效图像对数量不足，至少需要5对图像")
        return
    
    print(f"使用 {len(objpoints)} 对图像进行标定...")
    
    # 单相机标定
    print("进行左相机标定...")
    ret_left, mtx_left, dist_left, rvecs_left, tvecs_left = cv2.calibrateCamera(
        objpoints, imgpoints_left, gray_left.shape[::-1], None, None)
    
    print("进行右相机标定...")
    ret_right, mtx_right, dist_right, rvecs_right, tvecs_right = cv2.calibrateCamera(
        objpoints, imgpoints_right, gray_right.shape[::-1], None, None)
    
    # 立体标定
    print("进行立体标定...")
    flags = cv2.CALIB_FIX_INTRINSIC
    ret, mtx_left, dist_left, mtx_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpoints_left, imgpoints_right,
        mtx_left, dist_left, mtx_right, dist_right,
        gray_left.shape[::-1], flags=flags)
    
    # 立体校正
    print("计算立体校正参数...")
    R1, R2, P1, P2, Q, validPixROI1, validPixROI2 = cv2.stereoRectify(
        mtx_left, dist_left, mtx_right, dist_right,
        gray_left.shape[::-1], R, T)
    
    # 保存标定结果
    calibration_data = {
        'camera_matrix_left': mtx_left.tolist(),
        'distortion_coeffs_left': dist_left.tolist(),
        'camera_matrix_right': mtx_right.tolist(),
        'distortion_coeffs_right': dist_right.tolist(),
        'rotation_matrix': R.tolist(),
        'translation_vector': T.tolist(),
        'essential_matrix': E.tolist(),
        'fundamental_matrix': F.tolist(),
        'rectification_transform_left': R1.tolist(),
        'rectification_transform_right': R2.tolist(),
        'projection_matrix_left': P1.tolist(),
        'projection_matrix_right': P2.tolist(),
        'disparity_to_depth_matrix': Q.tolist()
    }
    
    with open('stereo_calibration.yaml', 'w') as f:
        yaml.dump(calibration_data, f)
    
    print("标定完成！结果已保存到 stereo_calibration.yaml")
    
    # 打印结果
    print("\n=== 标定结果 ===")
    print(f"左相机内参:\n{mtx_left}")
    print(f"左相机畸变系数: {dist_left.ravel()}")
    print(f"右相机内参:\n{mtx_right}")
    print(f"右相机畸变系数: {dist_right.ravel()}")
    print(f"旋转矩阵 R:\n{R}")
    print(f"平移向量 T: {T.ravel()}")
    print(f"重投影误差: {ret}")
    
    return calibration_data

def test_rectification():
    """测试立体校正效果"""
    # 加载标定数据
    with open('stereo_calibration.yaml', 'r') as f:
        data = yaml.safe_load(f)
    
    mtx_left = np.array(data['camera_matrix_left'])
    dist_left = np.array(data['distortion_coeffs_left'])
    mtx_right = np.array(data['camera_matrix_right'])
    dist_right = np.array(data['distortion_coeffs_right'])
    R1 = np.array(data['rectification_transform_left'])
    R2 = np.array(data['rectification_transform_right'])
    P1 = np.array(data['projection_matrix_left'])
    P2 = np.array(data['projection_matrix_right'])
    
    # 计算校正映射
    map1_left, map2_left = cv2.initUndistortRectifyMap(
        mtx_left, dist_left, R1, P1, (640, 480), cv2.CV_16SC2)
    map1_right, map2_right = cv2.initUndistortRectifyMap(
        mtx_right, dist_right, R2, P2, (640, 480), cv2.CV_16SC2)
    
    # 测试图像
    img_left = cv2.imread('left/test.jpg')
    img_right = cv2.imread('right/test.jpg')
    
    # 应用校正
    img_left_rect = cv2.remap(img_left, map1_left, map2_left, cv2.INTER_LINEAR)
    img_right_rect = cv2.remap(img_right, map1_right, map2_right, cv2.INTER_LINEAR)
    
    # 绘制水平线以检查校正效果
    for i in range(0, img_left_rect.shape[0], 50):
        cv2.line(img_left_rect, (0, i), (img_left_rect.shape[1], i), (0, 255, 0), 1)
        cv2.line(img_right_rect, (0, i), (img_right_rect.shape[1], i), (0, 255, 0), 1)
    
    # 显示结果
    result = np.hstack((img_left_rect, img_right_rect))
    cv2.imshow('Rectified Images', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 运行立体标定
    calibration_data = stereo_calibration()
    
    # 测试校正效果
    # test_rectification()
