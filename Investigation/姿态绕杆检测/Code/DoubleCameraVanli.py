def triangulate_point(self, left_pt, right_pt):
        """
        三角测量计算3D点
        """
        # 将2D点转换为正确的格式 (2, 1) 而不是 (3, 1)
        # cv2.triangulatePoints期望的是2D点坐标，而不是齐次坐标
        left_pt_array = np.array([[left_pt[0]], [left_pt[1]]])
        right_pt_array = np.array([[right_pt[0]], [right_pt[1]]])
        
        # 三角测量
        points_4d = cv2.triangulatePoints(
            self.P_left, 
            self.P_right, 
            left_pt_array, 
            right_pt_array
        )
        
        # 转换为3D欧几里得坐标
        points_3d = points_4d[:3] / points_4d[3]
        return points_3d.flatten()

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
        ax3.set_xlabel('X')
        ax3.set_ylabel('Y')
        ax3.set_zlabel('Z')
        
        # 修正坐标轴方向，使Z轴正方向向上
        # 在相机坐标系中，通常Z轴指向场景前方，Y轴指向下
        # 但我们希望在可视化中看到更直观的方向
        ax3.invert_zaxis()  # 反转Z轴，使正方向向上
        
        ax3.set_title('3D Reconstruction')
        ax3.legend()
        
        # 显示坐标信息
        ax4 = fig.add_subplot(224)
        ax4.axis('off')
        info_text = f"""
        3D Coordinate Calculation Results:
        
        Left Image Point: {self.left_point}
        Right Image Point: {self.right_point}
        
        Camera Coordinate System (Left Camera Origin):
        X: {result['camera_coords'][0]:.3f}
        Y: {result['camera_coords'][1]:.3f}
        Z: {result['camera_coords'][2]:.3f}
        
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