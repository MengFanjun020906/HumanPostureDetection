import json
import cv2
import numpy as np
import argparse


def load_calibration(calibration_file):
    """
    加载相机标定结果
    
    参数:
        calibration_file: 标定结果JSON文件路径
    
    返回:
        K: 相机内参矩阵 (3x3)
        dist: 畸变系数向量
        image_size: 图像尺寸 (width, height)
        rms: 重投影误差RMS值
    """
    with open(calibration_file, 'r') as f:
        calib = json.load(f)
    # 从标定结果中提取相机内参矩阵
    K = np.array(calib['camera_matrix'], dtype=np.float32)
    # 提取畸变系数
    dist = np.array(calib['distortion_coefficients'], dtype=np.float32)
    # 提取标定使用的图像尺寸
    img_w = calib['image_size']['width'] if 'image_size' in calib else None
    img_h = calib['image_size']['height'] if 'image_size' in calib else None
    # 提取重投影误差RMS值，用于评估标定质量
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
            # 计算优化后的相机矩阵，用于畸变校正
            newK, _ = cv2.getOptimalNewCameraMatrix(K, dist[:5], (w, h), 1.0, (w, h))
            # 执行畸变校正，验证标定参数的可用性
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

    # 子命令: 从像素坐标反推3D位置
    p_pos = subparsers.add_parser("position", help="从像素坐标(u,v)反推物体3D位置 (X,Y,Z)")
    p_pos.add_argument("--calib", default="calibration_results/calibration.json", help="calibration.json 路径")
    p_pos.add_argument("--u", type=float, required=True, help="像素坐标u (水平方向，从左到右)")
    p_pos.add_argument("--v", type=float, required=True, help="像素坐标v (垂直方向，从上到下)")
    p_pos.add_argument("--z", type=float, help="深度Z值(米)。如果未提供，可通过物体尺寸计算")
    p_pos.add_argument("--image", help="测试图像路径(用于畸变校正，如果像素坐标来自原始图像)")
    # 如果未提供Z，可以通过物体尺寸计算
    p_pos.add_argument("--diameter-m", type=float, help="物体真实直径(米)，用于计算深度Z")
    p_pos.add_argument("--diameter-px", type=float, help="物体像素直径(像素)，配合--diameter-m使用")
    p_pos.add_argument("--camera-height", type=float, default=2.5, help="相机安装高度(米)，默认2.5米。用于篮球绕杆检测等应用场景")

    args = parser.parse_args()

    if args.cmd == "distance":
        # 加载标定参数
        K, dist, _, rms = load_calibration(args.calib)
        fx = float(K[0, 0])  # 提取x方向焦距

        # 读取并进行畸变校正
        img = cv2.imread(args.image)
        if img is None:
            print(f"❌ 无法读取图像: {args.image}")
            raise SystemExit(2)
        h, w = img.shape[:2]
        # 计算优化后的相机矩阵
        newK, _ = cv2.getOptimalNewCameraMatrix(K, dist[:5], (w, h), 1.0, (w, h))
        # 使用标定参数进行畸变校正
        undist = cv2.undistort(img, K, dist[:5], None, newK)

        # 获取篮球的像素直径
        if args.pixels is not None:
            d_px = float(args.pixels)
        else:
            # 自动检测篮球：转换为灰度图并进行高斯模糊
            gray = cv2.cvtColor(undist, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (9, 9), 2)
            # 使用Hough变换检测圆
            circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1.2, minDist=50,
                                       param1=120, param2=40, minRadius=20, maxRadius=0)
            if circles is None or len(circles[0]) == 0:
                print("❌ 自动检测失败，请使用 --pixels 手动提供像素直径")
                raise SystemExit(3)
            circles = np.uint16(np.around(circles))
            # 选择半径最大的圆作为篮球（假设篮球是最大的圆形物体）
            r = max(circles[0, :, 2])
            d_px = float(2 * r)

        if d_px <= 0:
            print("❌ 像素直径应为正数")
            raise SystemExit(4)

        # 使用相机标定参数计算距离：Z = fx * D / d_px
        # 其中 fx 是焦距，D 是真实直径，d_px 是像素直径
        Z = fx * float(args.diameter_m) / d_px

        print("=== 距离估计(篮球) ===")
        print(f"RMS误差: {rms:.4f} 像素 (仅供质量参考)")
        print(f"fx: {fx:.3f} 像素")
        print(f"篮球像素直径: {d_px:.2f} 像素")
        print(f"篮球实际直径: {args.diameter_m:.3f} 米")
        # fx：相机焦距 D：篮球真实直径 d_px: 篮球像素直径
        print(f"估计相机到篮球距离 Z: {Z:.3f} 米  (公式: Z = fx*D/d_px)")
        raise SystemExit(0)

    if args.cmd == "position":
        # 加载标定参数
        K, dist, _, rms = load_calibration(args.calib)
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])

        # 获取像素坐标
        u, v = args.u, args.v

        # 如果提供了图像，说明像素坐标来自原始图像，需要进行畸变校正
        if args.image:
            img = cv2.imread(args.image)
            if img is None:
                print(f"⚠️ 警告：无法读取图像: {args.image}，将使用原始像素坐标")
            else:
                h, w = img.shape[:2]
                # 畸变校正
                newK, _ = cv2.getOptimalNewCameraMatrix(K, dist[:5], (w, h), 1.0, (w, h))
                
                # 将原始像素坐标转换为校正后的坐标
                # 注意：需要先转换为归一化坐标，再反向投影
                point = np.array([[[u, v]]], dtype=np.float32)
                # 使用undistortPoints进行点坐标校正
                corrected_point = cv2.undistortPoints(point, K, dist[:5], None, newK)
                u = corrected_point[0, 0, 0] * newK[0, 0] + newK[0, 2]
                v = corrected_point[0, 0, 1] * newK[1, 1] + newK[1, 2]
                print(f"原始像素坐标: ({args.u:.1f}, {args.v:.1f})")
                print(f"校正后像素坐标: ({u:.1f}, {v:.1f})")

        # 确定深度Z值
        if args.z is not None:
            Z = float(args.z)
        elif args.diameter_m is not None and args.diameter_px is not None:
            # 通过物体尺寸计算深度
            Z = fx * float(args.diameter_m) / float(args.diameter_px)
            print(f"通过物体尺寸计算深度: Z = fx*D/d_px = {fx:.3f}*{args.diameter_m:.3f}/{args.diameter_px:.1f} = {Z:.3f}米")
        else:
            print("❌ 错误：必须提供深度Z值(--z)或物体尺寸(--diameter-m和--diameter-px)")
            raise SystemExit(5)

        # 计算归一化坐标（相机坐标系下的归一化坐标）
        xn = (u - cx) / fx
        yn = (v - cy) / fy

        # 计算3D位置（相机坐标系：X右，Y下，Z前）
        X = xn * Z
        Y = yn * Z

        # 获取相机安装高度
        camera_height = args.camera_height
        
        # 计算物体相对于地面的绝对高度（Y向下为正，所以物体高度 = 相机高度 - Y）
        object_height_above_ground = camera_height - Y

        print("\n=== 3D位置反推 ===")
        print(f"RMS误差: {rms:.4f} 像素 (仅供质量参考)")
        print(f"像素坐标: ({u:.1f}, {v:.1f})")
        print(f"相机内参: fx={fx:.3f}, fy={fy:.3f}, cx={cx:.3f}, cy={cy:.3f}")
        print(f"归一化坐标: xn={xn:.4f}, yn={yn:.4f}")
        
        # 计算距离
        distance = np.sqrt(X*X + Y*Y + Z*Z)
        
        print(f"\n【坐标系说明】")
        print(f"相机坐标系定义：")
        print(f"  - 原点O(0, 0, 0)：相机光心位置（安装高度 {camera_height:.2f} 米）")
        print(f"  - X轴：向右为正方向（水平）")
        print(f"  - Y轴：向下为正方向（垂直）")
        print(f"  - Z轴：向前为正方向（深度，垂直于图像平面）")
        print(f"  - 地面高度：0米")
        
        print(f"\n【位置信息 - 相机坐标系】")
        print(f"相机位置: ({0:.3f}, {0:.3f}, {0:.3f}) 米")
        print(f"  → 相机安装高度: {camera_height:.3f} 米（相对于地面）")
        print(f"物体位置: ({X:.3f}, {Y:.3f}, {Z:.3f}) 米")
        print(f"物体相对于相机:")
        if abs(X) < 0.01:
            x_desc = "正前方"
        elif X > 0:
            x_desc = f"右侧 {abs(X):.3f}米"
        else:
            x_desc = f"左侧 {abs(X):.3f}米"
            
        if abs(Y) < 0.01:
            y_desc = "水平"
        elif Y > 0:
            y_desc = f"下方 {abs(Y):.3f}米"
        else:
            y_desc = f"上方 {abs(Y):.3f}米"
            
        print(f"  - 水平方向: {x_desc}")
        print(f"  - 垂直方向: {y_desc}")
        print(f"  - 深度距离: {Z:.3f} 米")
        print(f"  - 直线距离: {distance:.3f} 米")
        
        print(f"\n【位置信息 - 地面坐标系】")
        print(f"相机安装高度: {camera_height:.3f} 米")
        print(f"物体高度（相对于地面）: {object_height_above_ground:.3f} 米")
        if object_height_above_ground < 0:
            print(f"  ⚠️  物体位于地面以下，可能计算有误或物体确实在地面下")
        elif object_height_above_ground < 0.1:
            print(f"  ✓ 物体接近地面")
        else:
            print(f"  ✓ 物体位于地面以上")
        print(f"物体水平位置: X={X:.3f}米, Z={Z:.3f}米（相对于相机正下方在地面的投影点）")
        
        print(f"\n【计算公式】")
        print(f"  xn = (u - cx) / fx = ({u:.1f} - {cx:.3f}) / {fx:.3f} = {xn:.4f}")
        print(f"  yn = (v - cy) / fy = ({v:.1f} - {cy:.3f}) / {fy:.3f} = {yn:.4f}")
        print(f"  X = xn × Z = {xn:.4f} × {Z:.3f} = {X:.3f} 米")
        print(f"  Y = yn × Z = {yn:.4f} × {Z:.3f} = {Y:.3f} 米")
        raise SystemExit(0)

    # 默认走参数检查
    calib = args.calib if hasattr(args, 'calib') else "calibration_results/calibration.json"
    test = args.test if hasattr(args, 'test') else None
    rms_th = args.rms if hasattr(args, 'rms') else 0.6
    exit_code = validate_calibration(calib, test, rms_th)
    raise SystemExit(exit_code)

'''
python camera_vali.py position --u 1000 --v 300 --diameter-m 0.24 --diameter-px 150 --camera-height 1.5
'''