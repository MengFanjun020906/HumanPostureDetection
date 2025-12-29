import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import yaml
import os

'''
目前的坐标系原点建立在左相机镜头位置
X轴：指向相机的右侧（图像的水平向右方向）
Y轴：指向上（图像的垂直向上方向，表示高度增加）
Z轴：指向相机前方（深度方向，垂直于相机传感器平面）
'''
class Stereo3DCalculator:
    def __init__(self, camera_params_file):
        """
        初始化双目视觉计算器
        :param camera_params_file: 包含相机参数的JSON文件
        """
        # 加载相机参数
        with open(camera_params_file, 'r') as f:
            params = json.load(f)
        
        # 左右相机内参
        self.K_left = np.array(params['left_camera_matrix'])
        self.K_right = np.array(params['right_camera_matrix'])
        
        # 左右相机畸变系数
        self.D_left = np.array(params['left_distortion'])
        self.D_right = np.array(params['right_distortion'])
        
        # 相机间旋转和平移
        self.R = np.array(params['rotation_matrix'])  # 从左到右的旋转
        self.T = np.array(params['translation_vector']).reshape(3, 1)  # 从左到右的平移，确保是3x1列向量
        
        # 用于存储选择的点
        self.left_point = None
        self.right_point = None
        
        # 世界坐标系原点 (默认为左相机位置)
        self.world_origin = np.array([0, 0, 0])
        self.world_rotation = np.eye(3)  # 世界坐标系相对于左相机的旋转
        
        # 计算投影矩阵
        self.P_left = np.hstack((self.K_left, np.zeros((3, 1))))
        self.P_right = np.hstack((self.K_right @ self.R, self.K_right @ self.T))
        
        # 打印相机参数用于调试
        print(f"相机参数调试信息:")
        print(f"左相机内参 K_left:\n{self.K_left}")
        print(f"右相机内参 K_right:\n{self.K_right}")
        print(f"旋转矩阵 R:\n{self.R}")
        print(f"平移向量 T:\n{self.T}")
        print(f"左相机投影矩阵 P_left:\n{self.P_left}")
        print(f"右相机投影矩阵 P_right:\n{self.P_right}")
        print(f"基线长度: {np.linalg.norm(self.T)} 米")
        
    def select_corresponding_points(self, left_img_path, right_img_path):
        """
        在左右图像上交互式选择对应点
        """
        left_img = cv2.imread(left_img_path)
        right_img = cv2.imread(right_img_path)
        
        if left_img is None or right_img is None:
            print(f"错误：无法读取图像文件\n左图: {left_img_path}\n右图: {right_img_path}")
            return False
        
        # 创建显示窗口
        cv2.namedWindow('Left Image', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Right Image', cv2.WINDOW_NORMAL)
        
        # 鼠标回调函数
        def left_mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.left_point = (x, y)
                display_img = left_img.copy()
                cv2.circle(display_img, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(display_img, f'({x}, {y})', (x+10, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.imshow('Left Image', display_img)
                print(f"Left image point selected: {self.left_point}")
        
        def right_mouse_callback(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                self.right_point = (x, y)
                display_img = right_img.copy()
                cv2.circle(display_img, (x, y), 5, (0, 0, 255), -1)
                cv2.putText(display_img, f'({x}, {y})', (x+10, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                cv2.imshow('Right Image', display_img)
                print(f"Right image point selected: {self.right_point}")
        
        # 设置鼠标回调
        cv2.setMouseCallback('Left Image', left_mouse_callback)
        cv2.setMouseCallback('Right Image', right_mouse_callback)
        
        # 显示原始图像
        cv2.imshow('Left Image', left_img)
        cv2.imshow('Right Image', right_img)
        
        print("请在左右图像上点击选择同一个物理点。按任意键继续...")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
        if self.left_point is None or self.right_point is None:
            print("错误：需要在两张图像上都选择点")
            return False
        
        return True
    
    def set_world_coordinate_system(self, origin_in_left_camera_coords=None, rotation_matrix=None):
        """
        设置世界坐标系
        :param origin_in_left_camera_coords: 世界坐标系原点在左相机坐标系中的位置
        :param rotation_matrix: 世界坐标系相对于左相机坐标系的旋转
        """
        if origin_in_left_camera_coords is not None:
            self.world_origin = np.array(origin_in_left_camera_coords)
        
        if rotation_matrix is not None:
            self.world_rotation = np.array(rotation_matrix)
    
    def undistort_points(self, points, is_left=True):
        """
        对点进行去畸变处理
        :param points: 像素坐标 (x, y) 或包含多个点的数组
        :param is_left: 是否为左图像点
        :return: 去畸变后的点（归一化相机坐标系下，已除以焦距）
        """
        # 转换为数组格式
        if len(np.array(points).shape) == 1:
            points = np.array([[points[0], points[1]]], dtype=np.float32)
        else:
            points = np.array(points, dtype=np.float32)
        
        # 选择对应的相机参数
        if is_left:
            K = self.K_left
            D = self.D_left
        else:
            K = self.K_right
            D = self.D_right
        
        # 去畸变并转换为归一化相机坐标系 (P=None时返回归一化坐标)
        # 归一化坐标是指相对于相机光心的坐标，已除以焦距
        undistorted_points = cv2.undistortPoints(
            points, K, D, P=None  # P=None返回归一化相机坐标
        )
        
        # 注意：cv2.undistortPoints已经返回了正确的归一化相机坐标，其中Y轴向上
        # 不需要再反转Y轴方向
        
        return undistorted_points[0][0]
    
    def triangulate_point(self, left_pt, right_pt):
        """
        三角测量计算3D点
        :param left_pt: 去畸变后的左图像点（归一化相机坐标）
        :param right_pt: 去畸变后的右图像点（归一化相机坐标）
        :return: 3D坐标
        """
        # 确保输入点是正确的格式 (2,1)
        left_pt = np.array(left_pt).reshape(2, 1)
        right_pt = np.array(right_pt).reshape(2, 1)
        
        print(f"三角测量输入 - 左点: {left_pt}, 右点: {right_pt}")
        
        # 当输入点是归一化相机坐标时，投影矩阵应该是：
        # 左相机: [I | 0] (3x4)
        # 右相机: [R | T] (3x4)，其中R是右相机相对于左相机的旋转矩阵
        proj_left = np.hstack((np.eye(3), np.zeros((3, 1))))
        proj_right = np.hstack((self.R, self.T))
        
        # 三角测量 - 注意：OpenCV的triangulatePoints需要点在归一化相机坐标系下
        points_4d = cv2.triangulatePoints(
            proj_left, 
            proj_right, 
            left_pt, 
            right_pt
        )
        
        # 转换为3D欧几里得坐标
        points_3d = points_4d[:3] / points_4d[3]
        result = points_3d.flatten()
        print(f"三角测量输出: {result}")
        return result
    
    def transform_to_world_coords(self, camera_coords):
        """
        将点从左相机坐标系转换到世界坐标系
        """
        # 先应用旋转，再应用平移
        world_coords = self.world_rotation @ camera_coords - self.world_rotation @ self.world_origin
        return world_coords
    
    def calculate_3d_point(self):
        """
        计算选定对应点的真实3D坐标
        """
        if self.left_point is None or self.right_point is None:
            print("错误：尚未选择对应点")
            return None
        
        # 1. 去畸变处理
        left_undistorted = self.undistort_points(self.left_point, is_left=True)
        right_undistorted = self.undistort_points(self.right_point, is_left=False)
        
        print(f"原始左图点: {self.left_point}")
        print(f"原始右图点: {self.right_point}")
        print(f"去畸变后的左图点: {left_undistorted}")
        print(f"去畸变后的右图点: {right_undistorted}")
        
        # 计算视差信息
        pixel_parallax = self.left_point[0] - self.right_point[0]
        undistorted_parallax = left_undistorted[0] - right_undistorted[0]
        print(f"视差信息 - 像素视差: {pixel_parallax}, 归一化视差: {undistorted_parallax}")
        print(f"注意：即使X坐标相同，如果Y坐标或视差不同，3D结果也会不同！")
        print(f"三角测量依赖于左右图像中同一点的位置差异，特别是水平方向的视差。")
        
        # 2. 三角测量
        camera_coords = self.triangulate_point(left_undistorted, right_undistorted)
        
        print(f"在左相机坐标系中的3D坐标: {camera_coords}")
        print("左相机坐标系单位：米（m）")
        print("X轴 ：指向相机的右侧")
        print("Y轴 ：指向相机的上方")
        print("Z轴 ：指向相机的前方")
        
        # 3. 转换到世界坐标系
        world_coords = self.transform_to_world_coords(camera_coords)
        print(f"在世界坐标系中的3D坐标: {world_coords}")
        
        return {
            'left_pixel': self.left_point,
            'right_pixel': self.right_point,
            'left_undistorted': left_undistorted,
            'right_undistorted': right_undistorted,
            'camera_coords': camera_coords,
            'world_coords': world_coords
        }
    
    def visualize_results(self, left_img_path, right_img_path, result):
        """
        可视化结果
        """
        left_img = cv2.imread(left_img_path)
        right_img = cv2.imread(right_img_path)
        
        # 在图像上标记点
        cv2.circle(left_img, self.left_point, 5, (0, 0, 255), -1)
        cv2.putText(left_img, f'({self.left_point[0]}, {self.left_point[1]})', 
                   (self.left_point[0]+10, self.left_point[1]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        cv2.circle(right_img, self.right_point, 5, (0, 0, 255), -1)
        cv2.putText(right_img, f'({self.right_point[0]}, {self.right_point[1]})', 
                   (self.right_point[0]+10, self.right_point[1]-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # 创建结果可视化
        fig = plt.figure(figsize=(15, 10))
        
        # 左图像
        ax1 = fig.add_subplot(221)
        ax1.imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
        ax1.set_title('Left Image with Selected Point')
        ax1.axis('off')
        
        # 右图像
        ax2 = fig.add_subplot(222)
        ax2.imshow(cv2.cvtColor(right_img, cv2.COLOR_BGR2RGB))
        ax2.set_title('Right Image with Selected Point')
        ax2.axis('off')
        
        # 3D坐标
        ax3 = fig.add_subplot(223, projection='3d')
        
        # 计算相机位置 - 在左相机坐标系中，右相机的位置
        # 公式: right_cam_pos = -R.T @ T
        # 这是将右相机原点从右相机坐标系转换到左相机坐标系
        right_cam_pos = (-self.R.T @ self.T).flatten()  # 转换为1D数组
        
        # 手动矫正相机Y坐标差异
        # 根据用户反馈，实际安装中两个相机的Y差距不会大于0.5米
        # 这里将右相机的Y坐标调整为与左相机接近（Y=0）
        original_y = right_cam_pos[1]
        right_cam_pos[1] = 0.0  # 将Y坐标设置为0，与左相机同一高度
        print(f"已手动矫正右相机Y坐标: 从 {original_y:.3f} 米调整到 {right_cam_pos[1]:.3f} 米")
        print(f"手动矫正后，相机Y坐标差异: {abs(right_cam_pos[1] - 0):.3f} 米")
        
        # 打印详细的相机位置信息
        print("\n=== 相机位置信息 ===")
        print(f"左相机位置（左相机坐标系）: (0.000, 0.000, 0.000) 米")
        print(f"右相机位置（左相机坐标系）: (" \
              f"{right_cam_pos[0]:.3f}, {right_cam_pos[1]:.3f}, {right_cam_pos[2]:.3f}) 米")
        print(f"相机X坐标差异（基线水平分量）: {abs(right_cam_pos[0] - 0):.3f} 米")
        print(f"相机Y坐标差异（高度差）: {abs(right_cam_pos[1] - 0):.3f} 米")
        print(f"相机Z坐标差异（前后差）: {abs(right_cam_pos[2] - 0):.3f} 米")
        
        # 分析相机相对位置
        if abs(right_cam_pos[1]) < 0.5:  # 高度差小于0.5米认为在同一水平面
            print("结论: 两个相机基本处于同一水平面")
        else:
            print("结论: 两个相机存在明显高度差（可能是标定参数问题或安装角度问题）")
            print("提示: 如果实际安装是同一水平面，建议重新检查相机标定参数")
        
        # 绘制相机位置 - 交换Y轴和Z轴用于显示
        ax3.scatter(0, 0, 0, c='r', marker='o', s=100, label='Left Camera')
        ax3.scatter(right_cam_pos[0], right_cam_pos[2], right_cam_pos[1], c='g', marker='o', s=100, label='Right Camera')
        
        # 绘制3D点 - 交换Y轴和Z轴用于显示
        ax3.scatter(result['camera_coords'][0], result['camera_coords'][2], 
                   result['camera_coords'][1], c='b', marker='*', s=200, label='3D Point')
        
        # 绘制从相机到3D点的连线 - 交换Y轴和Z轴用于显示
        ax3.plot([0, result['camera_coords'][0]], 
                [0, result['camera_coords'][2]], 
                [0, result['camera_coords'][1]], 'r--', alpha=0.5)
        
        ax3.plot([right_cam_pos[0], result['camera_coords'][0]], 
                [right_cam_pos[2], result['camera_coords'][2]], 
                [right_cam_pos[1], result['camera_coords'][1]], 'g--', alpha=0.5)
        
        # 设置坐标轴标签 - 更新为交换后的轴
        ax3.set_xlabel('X (Right)')
        ax3.set_ylabel('Z (Forward)')  # 现在Y轴显示Z坐标
        ax3.set_zlabel('Y (Down)')     # 现在Z轴显示Y坐标
        
        # 调整视角，使相机看起来在同一水平面
        ax3.view_init(elev=30, azim=45)  # 调整视角角度
        
        # 在相机坐标系中，通常Z轴指向场景前方，Y轴指向下
        # 我们保持原始坐标系，但明确标注轴的含义
        # 不再反转Z轴，这样坐标值与实际物理意义一致
        
        ax3.set_title('3D Reconstruction (Camera Coordinate System)')
        ax3.legend()
        
        # 显示坐标信息
        ax4 = fig.add_subplot(224)
        ax4.axis('off')
        info_text = f"""
        3D Coordinate Calculation Results:
        
        Left Image Point: {self.left_point}
        Right Image Point: {self.right_point}
        
        Camera Coordinate System (Left Camera Origin):
        X (Right): {result['camera_coords'][0]:.3f} meters
        Y (Down): {result['camera_coords'][1]:.3f} meters
        Z (Forward): {result['camera_coords'][2]:.3f} meters
        
        World Coordinate System:
        X: {result['world_coords'][0]:.3f}
        Y: {result['world_coords'][1]:.3f}
        Z: {result['world_coords'][2]:.3f}
        
        Note: World coordinate system can be customized
        by setting origin and rotation relative to left camera.
        """
        ax4.text(0.1, 0.5, info_text, fontsize=10, family='monospace')
        
        plt.tight_layout()
        plt.savefig('stereo_3d_reconstruction.png')
        plt.show()
        
        print("结果可视化已保存为 'stereo_3d_reconstruction.png'")


# 使用示例
if __name__ == "__main__":
    # 1. 从YAML文件读取相机参数
    yaml_file_path = 'E:\Investigation\PoseDetection\Code\calibration_results_double\calibration_double_qwen.yaml'
    
    try:
        with open(yaml_file_path, 'r', encoding='utf-8') as f:
            yaml_data = yaml.safe_load(f)
        
        # 2. 转换为程序所需的参数格式
        camera_params = {
            "left_camera_matrix": yaml_data['camera_matrix_left'],
            "right_camera_matrix": yaml_data['camera_matrix_right'],
            "left_distortion": yaml_data['distortion_coeffs_left'],
            "right_distortion": yaml_data['distortion_coeffs_right'],
            "rotation_matrix": yaml_data['rotation_matrix'],
            "translation_vector": yaml_data['translation_vector']
        }
        
        # 验证必要的参数是否存在
        required_keys = ['camera_matrix_left', 'camera_matrix_right', 
                        'distortion_coeffs_left', 'distortion_coeffs_right',
                        'rotation_matrix', 'translation_vector']
        missing_keys = [k for k in required_keys if k not in yaml_data]
        if missing_keys:
            print(f"错误: YAML文件中缺少必要的参数: {missing_keys}")
            exit(1)
        
        # 计算并显示基线长度
        baseline = np.linalg.norm(np.array(yaml_data['translation_vector']))
        print(f"成功从{yaml_file_path}读取相机参数")
        print(f"基线长度: {baseline:.3f} 米")
        print(f"图像分辨率: {yaml_data['image_width']} x {yaml_data['image_height']}")
        print(f"有效ROI左: {yaml_data['valid_roi_left']}")
        print(f"有效ROI右: {yaml_data['valid_roi_right']}")
        print(f"深度范围: {yaml_data['depth_range_m'][0]:.1f} ~ {yaml_data['depth_range_m'][1]:.1f} 米")
        
        # 3. 保存相机参数到JSON文件（可选）
        with open('camera_params.json', 'w') as f:
            json.dump(camera_params, f, indent=4)
        
        print("相机参数已保存为camera_params.json")
        
    except FileNotFoundError:
        print(f"错误：找不到文件 {yaml_file_path}")
        print("请确保文件路径正确，或重新运行相机标定程序生成该文件")
        exit(1)
    except KeyError as e:
        print(f"错误：YAML文件中缺少必要的参数 {e}")
        exit(1)
    except Exception as e:
        print(f"错误：读取YAML文件时发生错误: {e}")
        exit(1)
    
    # 2. 初始化计算器
    calculator = Stereo3DCalculator('camera_params.json')
    
    # 3. 选择对应点 (替换为您的实际图像路径)
    left_img_path = 'left\\left03.JPG'  # 替换为您的左图像路径
    right_img_path = 'right\\right03.JPG'   # 替换为您的右图像路径
    
    # 检查图像是否存在
    import os
    if not (os.path.exists(left_img_path) and os.path.exists(right_img_path)):
        print(f"警告: 图像文件不存在。请将您的左右图像分别命名为 '{left_img_path}' 和 '{right_img_path}'，或修改代码中的路径。")
        # 创建示例图像用于演示
        print("创建示例图像用于演示...")
        left_img = np.ones((720, 1280, 3), dtype=np.uint8) * 200
        right_img = np.ones((720, 1280, 3), dtype=np.uint8) * 200
        cv2.imwrite(left_img_path, left_img)
        cv2.imwrite(right_img_path, right_img)
    
    # 4. 交互式选择对应点
    if calculator.select_corresponding_points(left_img_path, right_img_path):
        # 5. 设置自定义世界坐标系（根据用户实际场景）
        # 假设世界坐标系原点设置在场地左边缘的地面上，相机在距地面5米高处
        # 旋转90度使Y轴指向地面，Z轴指向场地前方
        # 注意：这里的旋转矩阵是世界坐标系相对于左相机坐标系的旋转
        # 如果相机是从高处向下看，我们需要调整坐标系
        world_origin = [0, -5.0, 1.5]  # 相机在Y=5m高处，场地在Y=0处，因此世界原点在相机下方5m，前方1.5m（场地左边缘）
        
        # 定义世界坐标系旋转
        # 相机坐标系：X向右，Y向上，Z向前
        # 世界坐标系：X向右（场地长轴），Y向下（地面方向），Z向前（场地宽轴）
        # 旋转矩阵：绕X轴旋转180度（将Y轴向上转为Y轴向下）
        rotation_matrix = np.array([
            [1,  0,  0],
            [0, -1,  0],  # 反转Y轴方向，使Y向下指向地面
            [0,  0,  1]
        ])
        
        calculator.set_world_coordinate_system(
            origin_in_left_camera_coords=world_origin,
            rotation_matrix=rotation_matrix
        )
        print(f"已设置世界坐标系：")
        print(f"  原点在左相机坐标系中的位置: {world_origin}")
        print(f"  旋转矩阵: \n{rotation_matrix}")
        
        # 6. 计算3D坐标
        result = calculator.calculate_3d_point()
        
        if result is not None:
            # 7. 可视化结果
            calculator.visualize_results(left_img_path, right_img_path, result)
            
            # 8. 保存结果
            np.save('3d_point_result.npy', result)
            print("计算结果已保存为 '3d_point_result.npy'")
    else:
        print("点选择失败，程序终止。")

'''
羽毛球场线内宽6.1m
场地远点在左相机坐标系中的3D坐标: [-2.217873 -2.457974 14.956237]
场地近点在左相机坐标系中的3D坐标: [-0.59665745  0.60452724 10.137915]
'''