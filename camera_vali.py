import cv2
import numpy as np
import os
import yaml

# 配置参数
CALIBRATOR_YAML = r"E:\Investigation\姿态绕杆检测\Code\calibration_results_double\stereo_calibration.yaml"  # 标定结果文件路径
LEFT_IMAGE_DIR = r"E:\Investigation\姿态绕杆检测\Code\left"  # 左相机图像目录
RIGHT_IMAGE_DIR = r"E:\Investigation\姿态绕杆检测\Code\right"  # 右相机图像目录
OUTPUT_DIR = "validation_results"  # 验证结果输出目录


# 确保输出目录存在
os.makedirs("validation_results", exist_ok=True)

# 检查左右图像目录是否存在
if not os.path.exists(LEFT_IMAGE_DIR):
    print(f"左图像目录不存在: {LEFT_IMAGE_DIR}")
    print("请创建该目录并添加标定图像")
    exit(1)

if not os.path.exists(RIGHT_IMAGE_DIR):
    print(f"右图像目录不存在: {RIGHT_IMAGE_DIR}")
    print("请创建该目录并添加标定图像")
    exit(1)

# 从YAML文件加载标定参数
def load_calibration_params(yaml_path):
    """从YAML文件加载立体相机标定参数"""
    import re
    
    # 首先读取文件内容，替换掉Python特定的类型标签
    with open(yaml_path, 'r', encoding='utf-8') as f:
        yaml_content = f.read()
    
    # 替换Python tuple标签为普通列表
    yaml_content = re.sub(r'tag:yaml.org,2002:python/tuple', r'', yaml_content)
    yaml_content = re.sub(r'!!python/tuple', r'', yaml_content)
    
    # 使用yaml.load并添加安全的构造函数
    def tuple_constructor(loader, node):
        return tuple(loader.construct_sequence(node))
    
    yaml.add_constructor('tag:yaml.org,2002:python/tuple', tuple_constructor, Loader=yaml.SafeLoader)
    yaml.add_constructor('!!python/tuple', tuple_constructor, Loader=yaml.SafeLoader)
    
    # 加载数据
    data = yaml.safe_load(yaml_content)
    
    # 左相机参数
    camera_matrix_left = np.array(data["camera_matrix_left"]).reshape(3, 3)
    distortion_coeffs_left = np.array(data["distortion_coeffs_left"]).reshape(-1, 1)
    
    # 右相机参数
    camera_matrix_right = np.array(data["camera_matrix_right"]).reshape(3, 3)
    distortion_coeffs_right = np.array(data["distortion_coeffs_right"]).reshape(-1, 1)
    
    # 外参
    rotation_matrix = np.array(data["rotation_matrix"]).reshape(3, 3)
    translation_vector = np.array(data["translation_vector"]).reshape(3, 1)
    
    # 图像尺寸
    image_size = (int(data["image_size"]["width"]), int(data["image_size"]["height"]))
    
    # 棋盘格参数
    chessboard_size = (int(data["chessboard"]["size"][0]), int(data["chessboard"]["size"][1]))
    square_size = data["chessboard"]["square_size_m"]
    
    return {
        "camera_matrix_left": camera_matrix_left,
        "distortion_coeffs_left": distortion_coeffs_left,
        "camera_matrix_right": camera_matrix_right,
        "distortion_coeffs_right": distortion_coeffs_right,
        "rotation_matrix": rotation_matrix,
        "translation_vector": translation_vector,
        "image_size": image_size,
        "chessboard_size": chessboard_size,
        "square_size": square_size
    }

# 1. 重投影误差可视化
def visualize_reprojection_error(params, output_dir):
    """可视化重投影误差"""
    print("正在进行重投影误差可视化...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取参数
    camera_matrix = params["camera_matrix_left"]
    distortion_coeffs = params["distortion_coeffs_left"]
    chessboard_size = params["chessboard_size"]
    square_size = params["square_size"]
    
    try:
        # 读取一张左相机标定图像
        left_images = [f for f in os.listdir(LEFT_IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not left_images:
            print("未找到左相机标定图像")
            return
        
        # 选择第一张图像进行测试
        img_path = os.path.join(LEFT_IMAGE_DIR, left_images[0])
        print(f"正在读取图像: {img_path}")
        
        # 使用OpenCV的imdecode方法来处理中文路径
        with open(img_path, 'rb') as f:
            img_data = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(img_data, cv2.IMREAD_COLOR)
        
        if img is None:
            print(f"无法读取图像: {img_path}")
            return
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 检测棋盘格角点
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        ret, corners = cv2.findChessboardCorners(gray, chessboard_size, None)
        
        # 生成3D点（世界坐标系，Z=0）
        objp = np.zeros((chessboard_size[0] * chessboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:chessboard_size[0], 0:chessboard_size[1]].T.reshape(-1, 2) * square_size
    except Exception as e:
        print(f"读取图像时发生错误: {e}")
        return
    
    if ret:
        # 亚像素级角点精化
        corners_refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        
        # 使用棋盘格的前几张图像的平均外参作为近似
        # 注意：这里使用的是近似的外参，实际应用中应该使用每张图像对应的rvec和tvec
        # 简单处理：假设世界坐标系原点在棋盘格左上角
        rvec = np.zeros((3, 1), np.float32)
        tvec = np.zeros((3, 1), np.float32)
        
        # 计算实际的外参（使用solvePnP）
        _, rvec, tvec = cv2.solvePnP(objp, corners_refined, camera_matrix, distortion_coeffs)
        
        # 使用标定参数重投影
        projected_points, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, distortion_coeffs)
        
        # 绘制检测点（绿色）和重投影点（红色）
        img_copy = img.copy()
        for i in range(len(corners_refined)):
            # 检测点
            cv2.circle(img_copy, tuple(corners_refined[i][0].astype(int)), 5, (0, 255, 0), -1)
            # 重投影点
            cv2.circle(img_copy, tuple(projected_points[i][0].astype(int)), 3, (0, 0, 255), 2)
            # 连接检测点和重投影点
            cv2.line(img_copy, tuple(corners_refined[i][0].astype(int)), 
                     tuple(projected_points[i][0].astype(int)), (0, 255, 255), 1)
        
        # 计算重投影误差
        error = []
        for i in range(len(corners_refined)):
            error.append(np.linalg.norm(corners_refined[i][0] - projected_points[i][0]))
        
        mean_error = np.mean(error)
        max_error = np.max(error)
        min_error = np.min(error)
        
        # 在图像上显示误差信息
        cv2.putText(img_copy, f"Mean Error: {mean_error:.4f} px", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img_copy, f"Max Error: {max_error:.4f} px", (10, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(img_copy, f"Min Error: {min_error:.4f} px", (10, 110), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # 保存结果
        output_path = os.path.join("validation_results", "reprojection_error.jpg")
        cv2.imwrite(output_path, img_copy)
        print(f"重投影误差可视化结果已保存至: {output_path}")
        print(f"平均重投影误差: {mean_error:.4f} px")
        print(f"最大重投影误差: {max_error:.4f} px")
        print(f"最小重投影误差: {min_error:.4f} px")
    else:
        print("未检测到棋盘格角点")

# 2. 极线校正效果验证
def validate_stereo_rectification(params, output_dir):
    """验证极线校正效果"""
    print("\n正在进行极线校正效果验证...")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取参数
    camera_matrix_left = params["camera_matrix_left"]
    distortion_coeffs_left = params["distortion_coeffs_left"]
    camera_matrix_right = params["camera_matrix_right"]
    distortion_coeffs_right = params["distortion_coeffs_right"]
    rotation_matrix = params["rotation_matrix"]
    translation_vector = params["translation_vector"]
    image_size = params["image_size"]
    
    try:
        # 检查左右图像目录是否存在
        if not os.path.exists(LEFT_IMAGE_DIR):
            print(f"左图像目录不存在: {LEFT_IMAGE_DIR}")
            return
        if not os.path.exists(RIGHT_IMAGE_DIR):
            print(f"右图像目录不存在: {RIGHT_IMAGE_DIR}")
            return
        
        # 读取一对左右图像
        left_images = [f for f in os.listdir(LEFT_IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        right_images = [f for f in os.listdir(RIGHT_IMAGE_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not left_images:
            print(f"左图像目录中未找到图像: {LEFT_IMAGE_DIR}")
            return
        if not right_images:
            print(f"右图像目录中未找到图像: {RIGHT_IMAGE_DIR}")
            return
        
        # 选择第一张图像对
        img_left_name = left_images[0]
        img_right_name = right_images[0]
        
        left_img_path = os.path.join(LEFT_IMAGE_DIR, img_left_name)
        right_img_path = os.path.join(RIGHT_IMAGE_DIR, img_right_name)
        
        print(f"正在读取左右图像: {left_img_path}, {right_img_path}")
        
        # 使用OpenCV的imdecode方法来处理中文路径
        with open(left_img_path, 'rb') as f:
            left_img_data = np.frombuffer(f.read(), dtype=np.uint8)
            img_left = cv2.imdecode(left_img_data, cv2.IMREAD_COLOR)
        
        with open(right_img_path, 'rb') as f:
            right_img_data = np.frombuffer(f.read(), dtype=np.uint8)
            img_right = cv2.imdecode(right_img_data, cv2.IMREAD_COLOR)
        
        if img_left is None:
            print(f"无法读取左图像: {left_img_path}")
            return
        if img_right is None:
            print(f"无法读取右图像: {right_img_path}")
            return
    except Exception as e:
        print(f"读取图像时发生错误: {e}")
        return
    
    # 生成校正映射
    alpha = 0.8  # 保留所有像素
    R1, R2, P1, P2, Q, valid_roi1, valid_roi2 = cv2.stereoRectify(
        camera_matrix_left, distortion_coeffs_left,
        camera_matrix_right, distortion_coeffs_right,
        image_size,
        rotation_matrix,
        translation_vector,
        alpha=alpha
    )
    
    # 初始化校正映射
    map1_l, map2_l = cv2.initUndistortRectifyMap(
        camera_matrix_left, distortion_coeffs_left, R1, P1, image_size, cv2.CV_32F
    )
    map1_r, map2_r = cv2.initUndistortRectifyMap(
        camera_matrix_right, distortion_coeffs_right, R2, P2, image_size, cv2.CV_32F
    )
    
    # 校正图像
    img_l_rect = cv2.remap(img_left, map1_l, map2_l, cv2.INTER_LINEAR)
    img_r_rect = cv2.remap(img_right, map1_r, map2_r, cv2.INTER_LINEAR)
    
    # 在校正后的图像上绘制水平线
    step = 50  # 水平线间距
    for y in range(0, image_size[1], step):
        cv2.line(img_l_rect, (0, y), (image_size[0], y), (0, 255, 0), 1)
        cv2.line(img_r_rect, (0, y), (image_size[0], y), (0, 255, 0), 1)
    
    # 显示极线校正效果
    cv2.putText(img_l_rect, "Rectified Left Image", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img_r_rect, "Rectified Right Image", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 保存校正后的图像
    output_left_path = os.path.join("validation_results", "rectified_left.jpg")
    output_right_path = os.path.join("validation_results", "rectified_right.jpg")
    cv2.imwrite(output_left_path, img_l_rect)
    cv2.imwrite(output_right_path, img_r_rect)
    
    # 创建并排显示的图像
    combined_img = np.hstack((img_l_rect, img_r_rect))
    output_combined_path = os.path.join("validation_results", "rectified_combined.jpg")
    cv2.imwrite(output_combined_path, combined_img)
    
    print(f"极线校正结果已保存至:")
    print(f"  - 左图像: {output_left_path}")
    print(f"  - 右图像: {output_right_path}")
    print(f"  - 合并图像: {output_combined_path}")
    print("\n验证说明：")
    print("1. 校正后的左右图像中的同名点应位于同一水平线上")
    print("2. 绿色水平线用于辅助检查极线是否水平对齐")

# 主函数
if __name__ == "__main__":
    print("开始相机标定验证...")
    
    # 加载标定参数
    params = load_calibration_params(CALIBRATOR_YAML)
    print("标定参数加载完成")
    
    # 1. 重投影误差可视化
    visualize_reprojection_error(params, OUTPUT_DIR)
    
    # 2. 极线校正效果验证
    validate_stereo_rectification(params, OUTPUT_DIR)
    
    print("\n相机标定验证完成！")
    print("验证结果保存在 'validation_results' 目录中")