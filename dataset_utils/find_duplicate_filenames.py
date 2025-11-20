"""
递归查找文件夹中所有重复文件名的文件路径
"""
import os
from collections import defaultdict
from pathlib import Path


def find_duplicate_filenames(root_dir):
    """
    递归查找指定目录下所有重复的文件名
    
    Args:
        root_dir: 要搜索的根目录路径
        
    Returns:
        dict: 键为文件名,值为该文件名对应的所有路径列表
    """
    # 使用defaultdict存储文件名和对应的路径列表
    filename_dict = defaultdict(list)
    
    # 递归遍历目录
    for root, dirs, files in os.walk(root_dir):
        for filename in files:
            # 获取完整路径
            full_path = os.path.join(root, filename)
            # 将路径添加到对应文件名的列表中
            filename_dict[filename].append(full_path)
    
    # 只返回出现多次的文件名
    duplicates = {name: paths for name, paths in filename_dict.items() if len(paths) > 1}
    
    return duplicates


def print_duplicates(duplicates):
    """
    格式化打印重复文件信息
    
    Args:
        duplicates: 重复文件字典
    """
    if not duplicates:
        print("未找到重复的文件名")
        return
    
    print(f"找到 {len(duplicates)} 个重复的文件名:\n")
    
    for filename, paths in sorted(duplicates.items()):
        print(f"文件名: {filename}")
        print(f"出现次数: {len(paths)}")
        print("路径:")
        for path in paths:
            print(f"  - {path}")
        print()


def save_to_file(duplicates, output_file="duplicate_files.txt"):
    """
    将重复文件信息保存到文件
    
    Args:
        duplicates: 重复文件字典
        output_file: 输出文件名
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(f"找到 {len(duplicates)} 个重复的文件名:\n\n")
        
        for filename, paths in sorted(duplicates.items()):
            f.write(f"文件名: {filename}\n")
            f.write(f"出现次数: {len(paths)}\n")
            f.write("路径:\n")
            for path in paths:
                f.write(f"  - {path}\n")
            f.write("\n")
    
    print(f"结果已保存到: {output_file}")


if __name__ == "__main__":
    # 设置要搜索的目录(当前目录)
    search_dir = r"C:\Users\29115\yolov8\yolov11-seg\datasets_20k\labeled"
    
    # 你也可以指定其他目录,例如:
    # search_dir = r"c:\Users\29115\yolov8\yolov11-seg"
    
    print(f"正在搜索目录: {os.path.abspath(search_dir)}")
    print("请稍候...\n")
    
    # 查找重复文件名
    duplicates = find_duplicate_filenames(search_dir)
    
    # 打印结果
    print_duplicates(duplicates)
    
    # 保存结果到文件
    if duplicates:
        save_to_file(duplicates)
