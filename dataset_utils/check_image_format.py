import os
import shutil
from pathlib import Path
from PIL import Image


def check_and_move_invalid_images(source_dir, corrupted_dir=None):
    """
    检查指定目录中的所有图片格式,将损坏或非JPEG格式的图片移动到损坏文件夹
    
    参数:
        source_dir (str): 源图片目录路径
        corrupted_dir (str, optional): 损坏文件夹路径,默认为源目录下的 'corrupted_images'
    
    返回:
        dict: 包含处理统计信息的字典
    """
    # 设置源目录和损坏文件夹路径
    source_path = Path(source_dir)
    if corrupted_dir is None:
        corrupted_path = source_path.parent / "corrupted_images2"
    else:
        corrupted_path = Path(corrupted_dir)
    
    # 创建损坏文件夹
    corrupted_path.mkdir(exist_ok=True)
    
    # 统计信息
    stats = {
        "total": 0,
        "valid_jpeg": 0,
        "invalid_format": 0,
        "corrupted": 0,
        "moved": []
    }
    
    # 支持的图片扩展名
    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.gif', '.tiff', '.webp']
    
    # 遍历目录中的所有文件
    for file_path in source_path.iterdir():
        if not file_path.is_file():
            continue
            
        # 检查文件扩展名
        if file_path.suffix.lower() not in image_extensions:
            continue
        
        stats["total"] += 1
        should_move = False
        reason = ""
        
        try:
            # 尝试打开图片
            with Image.open(file_path) as img:
                # 验证图片
                img.verify()
            
            # 重新打开图片以检查格式(verify()后图片会被关闭)
            with Image.open(file_path) as img:
                img.load()  # 确保图片可以完全加载
                
                # 检查是否为JPEG格式
                if img.format not in ['JPEG', 'JPG']:
                    should_move = True
                    reason = f"非JPEG格式 ({img.format})"
                    stats["invalid_format"] += 1
                else:
                    stats["valid_jpeg"] += 1
                    print(f"✓ {file_path.name} - 有效的JPEG图片")
                    
        except (IOError, OSError, Image.UnidentifiedImageError) as e:
            # 图片损坏或无法识别
            should_move = True
            reason = f"损坏或无法识别: {str(e)}"
            stats["corrupted"] += 1
        except Exception as e:
            # 其他错误
            should_move = True
            reason = f"未知错误: {str(e)}"
            stats["corrupted"] += 1
        
        # 移动无效图片
        if should_move:
            try:
                dest_path = corrupted_path / file_path.name
                # 如果目标文件已存在，添加序号
                counter = 1
                while dest_path.exists():
                    dest_path = corrupted_path / f"{file_path.stem}_{counter}{file_path.suffix}"
                    counter += 1
                
                shutil.move(str(file_path), str(dest_path))
                stats["moved"].append(file_path.name)
                print(f"✗ {file_path.name} - {reason} -> 已移动到 {dest_path}")
            except Exception as e:
                print(f"✗ 移动文件 {file_path.name} 时出错: {str(e)}")
    
    return stats


def print_summary(stats):
    """打印处理摘要"""
    print("\n" + "="*60)
    print("处理摘要:")
    print("="*60)
    print(f"总共检查的图片: {stats['total']}")
    print(f"有效的JPEG图片: {stats['valid_jpeg']}")
    print(f"非JPEG格式图片: {stats['invalid_format']}")
    print(f"损坏的图片: {stats['corrupted']}")
    print(f"已移动的图片总数: {len(stats['moved'])}")
    print("="*60)
    
    if stats['moved']:
        print("\n已移动的文件:")
        for filename in stats['moved']:
            print(f"  - {filename}")


if __name__ == "__main__":
    # 设置源目录路径
    source_directory = r"C:\Users\29115\yolov8\yolov11-seg\datasets_20k\corrupted_images"
    
    print(f"开始检查目录: {source_directory}")
    print("-" * 60)
    
    # 执行检查和移动操作
    statistics = check_and_move_invalid_images(source_directory)
    
    # 打印摘要
    print_summary(statistics)
