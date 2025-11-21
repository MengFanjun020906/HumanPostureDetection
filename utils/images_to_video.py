from pathlib import Path

import cv2
from tqdm import tqdm


def convert_to_video(images_path: Path, output_path: Path, fps: int = 240) -> None:
    filenames = list(images_path.glob("*.jpg"))
    if not filenames:
        print(f"警告: 在目录 {images_path} 中未找到任何.jpg文件")  # 修改：打印警告并返回而不是抛出异常
        return
    
    filenames = sorted(filenames, key=lambda x: int(x.stem))

    img = cv2.imread(str(filenames[0]))
    if img is None:
        raise ValueError(f"无法读取图像文件: {filenames[0]}")
        
    height, width, _ = img.shape
    out = cv2.VideoWriter(
        str(output_path), cv2.VideoWriter_fourcc(*"DIVX"), fps, (width, height)
    )

    progress_bar = tqdm(filenames)
    for filename in progress_bar:
        progress_bar.set_description("Making a video")
        img = cv2.imread(str(filename))
        if img is None:
            print(f"警告: 无法读取图像文件 {filename}，跳过...")
            continue
        out.write(img)

    out.release()