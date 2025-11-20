import os
import shutil
from pathlib import Path


def merge_and_renumber_images(source_dir, output_dir, prefix='img', start_index=1):
    """
    将source_dir下所有子文件夹中的图片合并到output_dir,并重新编号
    
    Args:
        source_dir: 源图片目录 (包含子文件夹)
        output_dir: 输出目录
        prefix: 新文件名前缀,默认为'img'
        start_index: 起始编号,默认为1
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}
    
    # 收集所有图片文件
    image_files = []
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(os.path.join(root, file))
    
    print(f"找到 {len(image_files)} 张图片")
    
    # 复制并重命名图片
    current_index = start_index
    for img_path in sorted(image_files):
        # 获取原始文件扩展名
        ext = Path(img_path).suffix
        
        # 生成新文件名
        new_filename = f"{prefix}_{current_index:06d}{ext}"
        new_path = os.path.join(output_dir, new_filename)
        
        # 复制文件
        shutil.copy2(img_path, new_path)
        print(f"已复制: {os.path.basename(img_path)} -> {new_filename}")
        
        current_index += 1
    
    print(f"\n完成! 总共处理了 {current_index - start_index} 张图片")
    print(f"输出目录: {output_dir}")


if __name__ == '__main__':
    # 源目录
    source_directory = r"C:\Users\29115\yolov8\yolov11-seg\datasets_20k\package_image"
    
    # 输出目录
    output_directory = r"C:\Users\29115\yolov8\yolov11-seg\datasets_20k\images"
    
    # 执行合并和重编号
    merge_and_renumber_images(
        source_dir=source_directory,
        output_dir=output_directory,
        prefix='img',  # 可以修改前缀
        start_index=1   # 可以修改起始编号
    )
