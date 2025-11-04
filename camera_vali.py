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
    parser = argparse.ArgumentParser(description="相机标定参数验证与距离估计(无可视化)")
    subparsers = parser.add_subparsers(dest="cmd", required=False)

    # 子命令: 参数有效性检查
    p_check = subparsers.add_parser("check", help="仅检查标定参数合理性")
    p_check.add_argument("--calib", default="calibration_results/calibration.json", help="calibration.json 路径")
    p_check.add_argument("--test", default=None, help="(可选) 测试图像路径，用于尝试一次畸变校正")
    p_check.add_argument("--rms", type=float, default=0.6, help="RMS误差阈值，默认0.6像素")

    # 子命令: 计算与篮球的距离
    p_dist = subparsers.add_parser("distance", help="基于篮球像素直径计算距离 Z = fx*D/d_px")
    p_dist.add_argument("--calib", default="calibration_results/calibration.json", help="calibration.json 路径")
    p_dist.add_argument("--image", required=True, help="包含篮球的测试图像路径")
    p_dist.add_argument("--diameter-m", type=float, required=True, help="篮球真实直径(米)，如0.24")
    group = p_dist.add_mutually_exclusive_group(required=True)
    group.add_argument("--pixels", type=float, help="篮球像素直径(像素)。建议来自检测或手动测量")
    group.add_argument("--auto", action="store_true", help="自动Hough圆检测，取最大圆作为篮球")

    args = parser.parse_args()

    if args.cmd == "distance":
        # 加载标定
        K, dist, _, rms = load_calibration(args.calib)
        fx = float(K[0, 0])

        # 读取并去畸变图像
        img = cv2.imread(args.image)
        if img is None:
            print(f"❌ 无法读取图像: {args.image}")
            raise SystemExit(2)
        h, w = img.shape[:2]
        newK, _ = cv2.getOptimalNewCameraMatrix(K, dist[:5], (w, h), 1.0, (w, h))
        undist = cv2.undistort(img, K, dist[:5], None, newK)

        # 获取像素直径
        if args.pixels is not None:
            d_px = float(args.pixels)
        else:
            # 自动检测圆
            gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (9, 9), 2)
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                                       param1=120, param2=40, minRadius=20, maxRadius=0)
            if circles is None or len(circles[0]) == 0:
                print("❌ 自动检测失败，请使用 --pixels 手动提供像素直径")
                raise SystemExit(3)
            circles = np.uint16(np.around(circles))
            # 选择半径最大的圆作为篮球
            r = max(circles[0, :, 2])
            d_px = float(2 * r)

        if d_px <= 0:
            print("❌ 像素直径应为正数")
            raise SystemExit(4)

        Z = fx * float(args.diameter_m) / d_px

        print("=== 距离估计(篮球) ===")
        print(f"RMS误差: {rms:.4f} 像素 (仅供质量参考)")
        print(f"fx: {fx:.3f} 像素")
        print(f"篮球像素直径: {d_px:.2f} 像素")
        print(f"篮球实际直径: {args.diameter_m:.3f} 米")
        print(f"估计相机到篮球距离 Z: {Z:.3f} 米  (公式: Z = fx*D/d_px)")
        raise SystemExit(0)

    # 默认走参数检查
    calib = args.calib if hasattr(args, 'calib') else "calibration_results/calibration.json"
    test = args.test if hasattr(args, 'test') else None
    rms_th = args.rms if hasattr(args, 'rms') else 0.6
    exit_code = validate_calibration(calib, test, rms_th)
    raise SystemExit(exit_code)