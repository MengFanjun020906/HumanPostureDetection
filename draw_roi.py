import cv2
import numpy as np
import yaml
import os
from pathlib import Path

class ROIDrawer:
    def __init__(self, roi_file_path):
        """
        初始化ROI绘制器
        :param roi_file_path: 包含ROI配置的YAML文件路径
        """
        # 读取ROI配置
        self.roi_data = self._load_roi_config(roi_file_path)
        print(f"✅ 加载ROI配置: {roi_file_path}")
        print(f"左相机ROI: x={self.roi_data['valid_roi_left']['x']}, y={self.roi_data['valid_roi_left']['y']}, "
              f"width={self.roi_data['valid_roi_left']['width']}, height={self.roi_data['valid_roi_left']['height']}")
        print(f"右相机ROI: x={self.roi_data['valid_roi_right']['x']}, y={self.roi_data['valid_roi_right']['y']}, "
              f"width={self.roi_data['valid_roi_right']['width']}, height={self.roi_data['valid_roi_right']['height']}")
    
    def _load_roi_config(self, file_path):
        """
        加载ROI配置文件
        """
        with open(file_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def draw_roi_on_image(self, image_path, is_left_camera=True, save_output=False, output_dir='output_roi'):
        """
        在图像上绘制ROI框
        :param image_path: 图像文件路径
        :param is_left_camera: 是否为左相机图像
        :param save_output: 是否保存输出图像
        :param output_dir: 输出目录
        :return: 绘制了ROI的图像
        """
        # 读取图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"❌ 无法读取图像: {image_path}")
            return None
        
        # 获取对应的ROI配置
        roi_key = 'valid_roi_left' if is_left_camera else 'valid_roi_right'
        roi = self.roi_data[roi_key]
        
        # 绘制ROI框
        color = (0, 255, 0)  # 绿色
        thickness = 2
        
        # 计算ROI的左上角和右下角坐标
        pt1 = (roi['x'], roi['y'])
        pt2 = (roi['x'] + roi['width'], roi['y'] + roi['height'])
        
        # 获取图像尺寸
        img_height, img_width = image.shape[:2]
        
        # 边界检查：确保ROI在图像范围内
        visible_pt1 = (max(0, min(img_width - 1, pt1[0])), max(0, min(img_height - 1, pt1[1])))
        visible_pt2 = (max(0, min(img_width - 1, pt2[0])), max(0, min(img_height - 1, pt2[1])))
        
        # 绘制矩形框
        image_with_roi = image.copy()
        cv2.rectangle(image_with_roi, visible_pt1, visible_pt2, color, thickness)
        
        # 显示调整信息
        if visible_pt1 != pt1 or visible_pt2 != pt2:
            print(f"⚠️  ROI超出图像边界，已调整为可见区域")
            print(f"   原始ROI: {pt1} -> {pt2}")
            print(f"   可见ROI: {visible_pt1} -> {visible_pt2}")
            print(f"   图像尺寸: {img_width}x{img_height}")
        
        # 添加文本标签
        label = "Left ROI" if is_left_camera else "Right ROI"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_color = (0, 255, 0)
        text_thickness = 1
        
        # 计算文本位置（使用可见的ROI左上角）
        text_size = cv2.getTextSize(label, font, font_scale, text_thickness)[0]
        
        # 确保文本在图像范围内
        text_x = max(0, visible_pt1[0])
        text_y = visible_pt1[1] - 10 if visible_pt1[1] > 20 else visible_pt1[1] + text_size[1] + 10
        text_y = max(text_size[1] + 5, min(img_height - 5, text_y))
        
        text_pt = (text_x, text_y)
        
        # 绘制文本背景
        bg_pt1 = (text_pt[0], text_pt[1] - text_size[1] - 5)
        bg_pt2 = (text_pt[0] + text_size[0], text_pt[1] + 5)
        
        # 确保文本背景在图像范围内
        bg_pt1 = (max(0, bg_pt1[0]), max(0, bg_pt1[1]))
        bg_pt2 = (min(img_width - 1, bg_pt2[0]), min(img_height - 1, bg_pt2[1]))
        
        if bg_pt1[0] < bg_pt2[0] and bg_pt1[1] < bg_pt2[1]:
            cv2.rectangle(image_with_roi, bg_pt1, bg_pt2, (0, 255, 0), -1)
            
            # 绘制文本
            cv2.putText(image_with_roi, label, text_pt, font, font_scale, (0, 0, 0), text_thickness)
        
        # 显示图像信息
        print(f"📷 处理图像: {os.path.basename(image_path)}")
        print(f"   图像尺寸: {image.shape[1]}x{image.shape[0]}")
        print(f"   ROI位置: {pt1} -> {pt2}")
        print(f"   ROI尺寸: {roi['width']}x{roi['height']}")
        
        # 保存输出图像
        if save_output:
            os.makedirs(output_dir, exist_ok=True)
            output_filename = f"{'left' if is_left_camera else 'right'}_{os.path.basename(image_path)}"
            output_path = os.path.join(output_dir, output_filename)
            cv2.imwrite(output_path, image_with_roi)
            print(f"💾 保存带ROI的图像: {output_path}")
        
        return image_with_roi
    
    def show_image(self, image, window_name="Image with ROI"):
        """
        显示图像
        """
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.imshow(window_name, image)
        print(f"👁️  显示窗口: {window_name} (按任意键关闭)")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def main():
    # 配置参数
    ROI_FILE_PATH = "calibration_results_double/calibration_double_qwen.yaml"  # 包含ROI配置的YAML文件
    TEST_IMAGE_PATH = "test_images/left03.JPG"  # 测试图像路径
    SAVE_OUTPUT = True  # 是否保存输出图像
    OUTPUT_DIR = "output_roi"  # 输出目录
    
    try:
        # 初始化ROI绘制器
        drawer = ROIDrawer(ROI_FILE_PATH)
        
        # 处理测试图像
        if os.path.exists(TEST_IMAGE_PATH):
            print(f"\n🔍 找到测试图像: {TEST_IMAGE_PATH}")
            
            # 在测试图像上分别绘制左相机和右相机的ROI
            print(f"\n📁 绘制左相机ROI:")
            image_with_left_roi = drawer.draw_roi_on_image(TEST_IMAGE_PATH, is_left_camera=True, 
                                                         save_output=SAVE_OUTPUT, output_dir=OUTPUT_DIR)
            if image_with_left_roi is not None:
                drawer.show_image(image_with_left_roi, f"Left Camera ROI - {os.path.basename(TEST_IMAGE_PATH)}")
            
            print(f"\n📁 绘制右相机ROI:")
            image_with_right_roi = drawer.draw_roi_on_image(TEST_IMAGE_PATH, is_left_camera=False, 
                                                          save_output=SAVE_OUTPUT, output_dir=OUTPUT_DIR)
            if image_with_right_roi is not None:
                drawer.show_image(image_with_right_roi, f"Right Camera ROI - {os.path.basename(TEST_IMAGE_PATH)}")
        else:
            print(f"❌ 未找到测试图像: {TEST_IMAGE_PATH}")
            print("请将测试图像放在test_images目录下，或修改TEST_IMAGE_PATH参数")
        
        print(f"\n✅ 所有图像处理完成！")
        
    except Exception as e:
        print(f"❌ 发生错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()