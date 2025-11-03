import json
import cv2
import numpy as np
import argparse


def load_calibration(calibration_file):
    with open(calibration_file, 'r') as f:
        calib = json.load(f)
    K = np.array(calib['camera_matrix'], dtype=np.float32)
    dist = np.array(calib['distortion_coefficients'], dtype=np.float32)
    img_w = calib['image_size']['width'] if 'image_size' in calib else None
    img_h = calib['image_size']['height'] if 'image_size' in calib else None
    rms = float(calib['reprojection_error']['rms'])
    return K, dist, (img_w, img_h), rms


def validate_calibration(calibration_file, test_image_path=None, rms_threshold=0.6):
    """
    仅验证标定参数是否可用：
    - 读取 K、dist、RMS 并做阈值与合理性检查
    - 可选：读取一张测试图执行一次畸变校正（不显示/不保存）以确认参数可用
    """
    K, dist, image_size, rms = load_calibration(calibration_file)

    print("=== 标定参数验证(简版) ===")
    print(f"RMS误差: {rms:.4f} 像素  | 阈值: {rms_threshold}")
    print(f"内参矩阵K:\n{K}")
    print(f"畸变系数(共{len(dist)}个): {dist.ravel()}")
    if image_size[0] and image_size[1]:
        print(f"标定图像尺寸: {image_size[0]}x{image_size[1]}")

    # 基本合理性检查
    ok = True
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    if fx <= 0 or fy <= 0:
        print("❌ 无效焦距: fx/fy 应为正")
        ok = False
    if image_size[0] and (cx < 0 or cx > image_size[0]):
        print("❌ 主点cx不在图像宽度范围内")
        ok = False
    if image_size[1] and (cy < 0 or cy > image_size[1]):
        print("❌ 主点cy不在图像高度范围内")
        ok = False

    if rms > rms_threshold:
        print("❌ RMS误差超过阈值，建议重新标定或补充图像")
        ok = False
    else:
        print("✅ RMS误差通过阈值检查")

    # 可选：使用一张测试图做一次畸变校正，验证参数可用性
    if test_image_path:
        img = cv2.imread(test_image_path)
        if img is None:
            print(f"⚠️ 警告：无法读取测试图像: {test_image_path}")
        else:
            h, w = img.shape[:2]
            newK, _ = cv2.getOptimalNewCameraMatrix(K, dist[:5], (w, h), 1.0, (w, h))
            _ = cv2.undistort(img, K, dist[:5], None, newK)
            print("✅ 畸变校正执行成功（未显示/未保存）")

    print("\n验证结果:")
    if ok:
        print("✅ 标定参数可用")
        return 0
    else:
        print("❌ 标定参数不可用或需改进")
        return 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="相机标定参数简洁验证工具(无可视化)")
    parser.add_argument("--calib", default="calibration_results/calibration.json", help="calibration.json 路径")
    parser.add_argument("--test", default=None, help="(可选) 测试图像路径，仅用于尝试一次畸变校正")
    parser.add_argument("--rms", type=float, default=0.6, help="RMS误差阈值，默认0.6像素")
    args = parser.parse_args()

    exit_code = validate_calibration(args.calib, args.test, args.rms)
    raise SystemExit(exit_code)