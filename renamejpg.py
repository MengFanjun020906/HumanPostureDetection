import os

def rename_images(folder_path, prefix="right", reverse=False):
    """
    重命名文件夹内所有图片文件
    按文件修改时间排序
    Args:
        folder_path: 文件夹路径
        prefix: 文件名前缀
        reverse: 是否按修改时间倒序（True表示最新的文件排在前面）
    """
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp']
    
    try:
        files = os.listdir(folder_path)
    except FileNotFoundError:
        print(f"错误：文件夹 '{folder_path}' 不存在")
        return
    
    image_files = []
    for f in files:
        if os.path.splitext(f)[1].lower() in image_extensions:
            file_path = os.path.join(folder_path, f)
            mtime = os.path.getmtime(file_path)
            image_files.append((f, mtime))
    
    if not image_files:
        print("未找到图片文件")
        return
    
    image_files.sort(key=lambda x: x[1], reverse=reverse)
    
    num_len = len(str(len(image_files)))
    format_str = f"{{:0{num_len}d}}"
    
    for i, (filename, _) in enumerate(image_files, 1):
        file_ext = os.path.splitext(filename)[1]
        new_name = f"{prefix}{format_str.format(i)}{file_ext}"
        old_path = os.path.join(folder_path, filename)
        new_path = os.path.join(folder_path, new_name)
        
        try:
            os.rename(old_path, new_path)
            print(f"重命名: {filename} -> {new_name}")
        except Exception as e:
            print(f"重命名 {filename} 时出错: {e}")

# 使用示例
if __name__ == "__main__":
    folder_path = input("请输入文件夹路径: ").strip()
    rename_images(folder_path)