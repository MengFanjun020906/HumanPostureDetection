import argparse
import shutil
import subprocess
import sys
from pathlib import Path
import os

import numpy as np

'''
D:/anaconda3/envs/retinaface_env/python.exe video_pipeline.py <video_path> -o <output_path> -d <device> --fps <fps> -c
'''
# 确保utils模块在Python路径中
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

try:
    from utils import convert_to_images, convert_to_video, draw_trajectory
except ImportError as e:
    print(f"导入utils模块失败: {e}")
    print("请确保utils.py文件在当前目录或Python路径中")
    sys.exit(1)

def check_required_files():
    """检查必需的文件是否存在"""
    required_files = [
        current_dir / "yolov5" / "detect.py",
        current_dir / "models" / "yolov5s_basketball.pt"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not file_path.exists():
            missing_files.append(file_path)
            print(f"❌ 文件不存在: {file_path}")
    
    if missing_files:
        print("\n=== 解决方案 ===")
        print("1. 对于yolov5/detect.py:")
        print("   git clone https://github.com/ultralytics/yolov5")
        print("   pip install -r yolov5/requirements.txt")
        print("2. 对于models/yolov5s_basketball.pt:")
        print("   请确保模型文件已下载到models目录")
        sys.exit(1)
    else:
        print("✅ 所有必需文件存在")

def get_python_command():
    """获取正确的Python命令"""
    # 尝试使用sys.executable获取当前Python解释器
    python_cmd = sys.executable
    if python_cmd:
        return python_cmd
    
    # 回退到系统命令
    if sys.platform == "win32":
        return "python.exe"
    else:
        return "python3"

def process_video(
    video_path: Path,
    output_path: Path,
    device: str,
    fps: int,
    clean: bool,
    enable_trajectory: bool = False,  # 添加轨迹绘制控制参数
) -> None:
    # 检查输入视频是否存在
    if not video_path.exists():
        raise FileNotFoundError(f"输入视频不存在: {video_path}")
    
    output_path = output_path / video_path.stem
    output_path.mkdir(parents=True, exist_ok=True)

    images_raw_path = output_path / "images_raw"
    images_draw_path = output_path / "images_draw"

    images_raw_path.mkdir(parents=True, exist_ok=True)
    images_draw_path.mkdir(parents=True, exist_ok=True)

    # copy video file with error handling
    try:
        shutil.copyfile(video_path, output_path / video_path.name)
        print(f"✅ 视频文件已复制到: {output_path / video_path.name}")
    except Exception as e:
        print(f"⚠️  复制视频文件时出错: {e}")
        # 继续执行，不影响主要流程

    # convert video to images
    print("🎬 正在将视频转换为图像...")
    try:
        convert_to_images(video_path, images_raw_path, video_stride=1)
        print(f"✅ 视频转换完成，图像保存在: {images_raw_path}")
    except Exception as e:
        print(f"❌ 视频转换失败: {e}")
        sys.exit(1)

    # detect balls using YOLO
    python_cmd = get_python_command()
    detect_cmd = [
        python_cmd,
        str(current_dir / "yolov5" / "detect.py"),
        "--weights",
        str(current_dir / "models" / "yolov5s_basketball.pt"),
        "--source",
        str(images_raw_path),  # 使用Path对象自动处理路径
        "--save-txt",
        "--save-conf",
        "--nosave",
        "--project",
        str(output_path.parent),
        "--name",
        video_path.stem,
        "--exist-ok",
        "--device",
        device,
    ]
    
    print("\n🔍 开始篮球检测...")
    print("执行命令:", " ".join(detect_cmd))
    
    try:
        # 使用capture_output获取详细错误信息
        result = subprocess.run(
            detect_cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd=str(current_dir)  # 设置正确的工作目录
        )
        print("✅ 检测成功完成")
        # print("标准输出:", result.stdout[:500])  # 只显示前500字符
    except subprocess.CalledProcessError as e:
        print(f"❌ 检测失败，退出代码: {e.returncode}")
        print("标准错误输出:")
        print(e.stderr)
        
        # 详细诊断
        print("\n=== 诊断信息 ===")
        print(f"Python命令: {python_cmd}")
        print(f"当前工作目录: {os.getcwd()}")
        print(f"detect.py路径: {current_dir / 'yolov5' / 'detect.py'}")
        print(f"模型路径: {current_dir / 'models' / 'yolov5s_basketball.pt'}")
        
        # 检查CUDA
        if "CUDA" in e.stderr or "cuda" in e.stderr.lower():
            print("CUDA错误，尝试回退到CPU...")
            detect_cmd[-1] = "cpu"  # 修改device参数为cpu
            print("使用CPU重新尝试:", " ".join(detect_cmd))
            try:
                subprocess.run(detect_cmd, check=True)
                print("✅ CPU检测成功")
            except Exception as cpu_e:
                print(f"❌ CPU检测也失败: {cpu_e}")
                sys.exit(1)
        else:
            sys.exit(1)
    except Exception as e:
        print(f"❌ 意外错误: {e}")
        sys.exit(1)

    # draw trajectory
    labels_path = output_path / "labels"
    if not labels_path.exists():
        print(f"⚠️  labels目录不存在: {labels_path}")
        print("可能检测没有生成任何结果")
        # 创建空轨迹文件
        np.savetxt(output_path / "trajectory.txt", np.array([]))
    else:
        print("\n📊 正在绘制轨迹...")
        try:
            trajectory = draw_trajectory(
                labels_path,
                images_raw_path,
                images_draw_path,
                ball_conf=0.5,
                max_distance=30,
                enable_trajectory=enable_trajectory,  # 传递轨迹绘制控制参数
            )
            trajectory = np.array(trajectory)
            np.savetxt(output_path / "trajectory.txt", trajectory, fmt="%4d %4d")
            print(f"✅ 轨迹数据已保存: {output_path / 'trajectory.txt'}")
            print(f"轨迹点数量: {len(trajectory)}")
        except Exception as e:
            print(f"❌ 绘制轨迹失败: {e}")
            # 创建空轨迹文件
            np.savetxt(output_path / "trajectory.txt", np.array([]))

    # make video
    output_video_path = output_path / f"output_{video_path.stem}.avi"
    print("\n🎥 正在生成输出视频...")
    try:
        convert_to_video(
            images_draw_path,
            output_video_path,
            fps=fps,
        )
        print(f"✅ 输出视频已生成: {output_video_path}")
    except Exception as e:
        print(f"❌ 生成视频失败: {e}")

    # clean up
    if clean:
        print("\n🧹 清理临时文件...")
        try:
            if images_raw_path.exists():
                shutil.rmtree(images_raw_path)
                print(f"✅ 已删除: {images_raw_path}")
            
            if images_draw_path.exists():
                shutil.rmtree(images_draw_path)
                print(f"✅ 已删除: {images_draw_path}")
            
            video_file = output_path / video_path.name
            if video_file.exists():
                video_file.unlink()
                print(f"✅ 已删除: {video_file}")
        except Exception as e:
            print(f"⚠️  清理时出错: {e}")

    print(f"\n🎉 处理完成！结果保存在: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='篮球轨迹检测与可视化')

    parser.add_argument("video", type=str, help="输入视频文件的路径")
    parser.add_argument(
        "-o", "--output", type=str, default="output", help="输出目录 (默认: output)"
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cuda:0",
        help="YOLOv5模型使用的设备 (默认: cuda:0, 可替换为 cpu)",
    )
    parser.add_argument("--fps", type=int, default=30, help="输出视频的FPS (默认: 30)")
    parser.add_argument(
        "-c",
        "--clean",
        action="store_true",
        help="清理中间文件 (原始图像、绘制图像、复制的视频)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式，显示更多详细信息",
    )
    parser.add_argument(
        "--no-trajectory",
        action="store_true",
        help="不绘制轨迹线",
    )

    args = parser.parse_args()

    # 环境检查
    print("=== 环境检查 ===")
    print(f"Python版本: {sys.version.split()[0]}")
    print(f"平台: {sys.platform}")
    print(f"当前工作目录: {os.getcwd()}")
    print(f"脚本位置: {current_dir}")
    
    # 检查必需文件
    check_required_files()

    # 处理视频
    try:
        process_video(
            Path(args.video),
            Path(args.output),
            args.device,
            args.fps,
            args.clean,
            not args.no_trajectory,  # 将轨迹绘制选项传递给process_video函数
        )
        print("✅ 程序成功完成！")
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)