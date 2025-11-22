import cv2
import numpy as np
import matplotlib.pyplot as plt
import json

'''
目前的坐标系原点建立在左相机镜头位置
X轴：指向相机的右侧（图像的水平向右方向）
Y轴：指向下（图像的垂直向下方向，表示高度降低）
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
        self.T = np.array(params['translation_vector'])  # 从左到右的平移
        
        # 计算投影矩阵
        self.P_left = np.hstack((self.K_left, np.zeros((3, 1))))
        self.P_right = self.K_right @ np.hstack((self.R, self.T.reshape(3, 1)))
        
        # 用于存储选择的点
        self.left_point = None
        self.right_point = None
        
        # 世界坐标系原点 (默认为左相机位置)
        self.world_origin = np.array([0, 0, 0])
        self.world_rotation = np.eye(3)  # 世界坐标系相对于左相机的旋转
        
    def select_corresponding_points(self, left_img_path, right_img_path):
        """
        在左右图像上交互式选择对应点
        """
        left_img = cv2.imread(left_img_path)
        right_img = cv2.imread(right_img_path)
        
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
        
        # 显示图像
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
    
    def undistort_points(self, point, camera_matrix, dist_coeffs):
        """
        对点进行去畸变处理
        """
        points = np.array([[[point[0], point[1]]]], dtype=np.float32)
        undistorted_points = cv2.undistortPoints(
            points, 
            camera_matrix, 
            dist_coeffs,
            P=camera_matrix
        )
        return undistorted_points[0][0]
    
    def triangulate_point(self, left_pt, right_pt):
        """
        三角测量计算3D点
        """
        # 确保输入点是正确的格式 (2,1) 或 (2,)
        if len(left_pt.shape) == 1:
            left_pt = left_pt.reshape(2, 1)
        if len(right_pt.shape) == 1:
            right_pt = right_pt.reshape(2, 1)
            
        # 三角测量
        points_4d = cv2.triangulatePoints(
            self.P_left, 
            self.P_right, 
            left_pt, 
            right_pt
        )
        
        # 转换为3D欧几里得坐标
        points_3d = points_4d[:3] / points_4d[3]
        return points_3d.flatten()
    
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
        left_undistorted = self.undistort_points(self.left_point, self.K_left, self.D_left)
        right_undistorted = self.undistort_points(self.right_point, self.K_right, self.D_right)
        
        print(f"去畸变后的左图点: {left_undistorted}")
        print(f"去畸变后的右图点: {right_undistorted}")
        
        # 2. 三角测量
        camera_coords = self.triangulate_point(left_undistorted, right_undistorted)
        
        print(f"在左相机坐标系中的3D坐标: {camera_coords}")
        
        # 3. 转换到世界坐标系
        world_coords = self.transform_to_world_coords(camera_coords)
        
        print(f"在世界坐标系中的3D坐标: {world_coords}")
        
        return {
            'camera_coords': camera_coords,
            'world_coords': world_coords,
            'left_pixel': self.left_point,
            'right_pixel': self.right_point
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
        # 绘制相机位置
        ax3.scatter(0, 0, 0, c='r', marker='o', s=100, label='Left Camera')
        right_cam_pos = -self.R.T @ self.T
        ax3.scatter(right_cam_pos[0], right_cam_pos[1], right_cam_pos[2], c='g', marker='o', s=100, label='Right Camera')
        
        # 绘制3D点
        ax3.scatter(result['camera_coords'][0], result['camera_coords'][1], 
                   result['camera_coords'][2], c='b', marker='*', s=200, label='3D Point')
        
        # 绘制从相机到3D点的连线
        ax3.plot([0, result['camera_coords'][0]], 
                [0, result['camera_coords'][1]], 
                [0, result['camera_coords'][2]], 'r--', alpha=0.5)
        
        ax3.plot([right_cam_pos[0], result['camera_coords'][0]], 
                [right_cam_pos[1], result['camera_coords'][1]], 
                [right_cam_pos[2], result['camera_coords'][2]], 'g--', alpha=0.5)
        
        # 设置坐标轴标签
        ax3.set_xlabel('X (Right)')
        ax3.set_ylabel('Y (Down)')
        ax3.set_zlabel('Z (Forward)')
        
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
    # 1. 准备相机参数文件 (示例格式)
    # 实际使用中，您需要通过相机标定获取这些参数
    # camera_params = {
    #     "left_camera_matrix": [
    #         [5799.282813412017, 0.0, 2991.704735902185],
    #         [0.0, 5987.778255995474, 2025.6304584651796],
    #         [0.0, 0.0, 1.0]
    #     ],
    #     "right_camera_matrix": [
    #         [4670.655811391591, 0.0, 3091.7005955071077],
    #         [0.0, 4579.933448361181, 1982.97803785274],
    #         [0.0, 0.0, 1.0]
    #     ],
    #     "left_distortion": [
    #         -0.14319722350802536,
    #         3.5626766043206164,
    #         0.016632719946127732,
    #         -0.002200769370862237,
    #         0.49720264289550054
    #     ],
    #     "right_distortion": [
    #         0.21670364660716837,
    #         -1.8857240541391913,
    #         -0.0313728220730505,
    #         0.004243346164428998,
    #         8.617802383990066
    #     ],
    #     "rotation_matrix": [
    #         [0.8666182528679105, -0.3384588558941265, 0.36663115888179676],
    #         [0.35204756057172304, 0.9354543725990161, 0.03142661103078563],
    #         [-0.35360333552113077, 0.10183673036886144, 0.9298354485908302]
    #     ],
    #     "translation_vector": [
    #         -5.774539540635336,
    #         1.4909182826354272,
    #         -5.662007167539076
    #     ]  # 单位为米
    # }
    camera_params = {
    "left_camera_matrix": [
        [5628.451582711846, 0.0, 2698.685087632601],
        [0.0, 5675.814867786561, 1983.7792435461297],
        [0.0, 0.0, 1.0]
    ],
    "right_camera_matrix": [
        [4454.837755298474, 0.0, 3046.236025449827],
        [0.0, 4722.968524658102, 2036.9278417837622],
        [0.0, 0.0, 1.0]
    ],
    "left_distortion": [
        -0.10538791331942947,
        3.081316174389013,
        -0.0047049135841466545,
        0.010629607799912915,
        -8.922728894760338
    ],
    "right_distortion": [
        0.10676213200475755,
        -0.20610675074039744,
        0.02802907647105573,
        0.03335005431280584,
        9.683620069249042
    ],
    "rotation_matrix": [
        [0.6777781499444254, -0.7299124568263363, -0.0885696608757976],
        [0.6638004337597015, 0.6592482940497991, -0.35321476601759133],
        [0.3162052554881784, 0.18060867133715203, 0.9313402945430228]
    ],
    "translation_vector": [
        1.4166373585569374,
        9.173870151289641,
        -6.608207161709112
    ]  # 单位为米
}
    
    # 保存相机参数到文件
    with open('camera_params.json', 'w') as f:
        json.dump(camera_params, f, indent=4)
    
    print("相机参数文件已创建。在实际应用中，您需要替换为真实标定参数。")
    
    # 2. 初始化计算器
    calculator = Stereo3DCalculator('camera_params.json')
    
    # 3. 选择对应点 (替换为您的实际图像路径)
    left_img_path = 'left\\left41.JPG'  # 替换为您的左图像路径
    right_img_path = 'right\\right41.JPG'   # 替换为您的右图像路径
    
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
        # 5. [可选] 设置自定义世界坐标系
        # 例如，将世界坐标系原点设置在左相机前方0.5米处
        # calculator.set_world_coordinate_system(
        #     origin_in_left_camera_coords=[0, 0, 0.5],
        #     rotation_matrix=np.eye(3)
        # )
        
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