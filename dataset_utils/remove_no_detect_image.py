#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
删除没有 labelme 标注文件的图片

使用方法:
    python remove_no_detect_image.py <图片目录路径>

说明:
    - 遍历指定目录下的所有图片文件
    - 检查是否存在对应的 labelme JSON 标注文件
    - 如果不存在标注文件，则删除该图片
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Set


# 支持的图片格式
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp'}


def get_image_files(directory: str) -> List[Path]:
    """
    获取目录下所有图片文件
    
    Args:
        directory: 图片目录路径
        
    Returns:
        图片文件路径列表
    """
    dir_path = Path(directory)
    if not dir_path.exists():
        raise FileNotFoundError(f"目录不存在: {directory}")
    
    if not dir_path.is_dir():
        raise NotADirectoryError(f"不是一个目录: {directory}")
    
    image_files = []
    for file_path in dir_path.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in IMAGE_EXTENSIONS:
            image_files.append(file_path)
    
    return image_files


def has_labelme_annotation(image_path: Path) -> bool:
    """
    检查图片是否有对应的 labelme 标注文件
    
    Args:
        image_path: 图片文件路径
        
    Returns:
        如果存在对应的 JSON 标注文件返回 True，否则返回 False
    """
    json_path = image_path.with_suffix('.json')
    return json_path.exists()


def remove_unlabeled_images(directory: str, dry_run: bool = False) -> tuple:
    """
    删除没有标注的图片
    
    Args:
        directory: 图片目录路径
        dry_run: 如果为 True，只显示将要删除的文件，不实际删除
        
    Returns:
        (删除的文件数量, 保留的文件数量)
    """
    image_files = get_image_files(directory)
    
    if not image_files:
        print(f"在目录 {directory} 中没有找到图片文件")
        return 0, 0
    
    print(f"找到 {len(image_files)} 个图片文件")
    
    removed_count = 0
    kept_count = 0
    removed_files = []
    
    for image_path in image_files:
        if has_labelme_annotation(image_path):
            kept_count += 1
        else:
            removed_files.append(image_path)
            if not dry_run:
                try:
                    image_path.unlink()
                    print(f"已删除: {image_path.name}")
                    removed_count += 1
                except Exception as e:
                    print(f"删除失败 {image_path.name}: {e}")
            else:
                print(f"将删除: {image_path.name}")
                removed_count += 1
    
    return removed_count, kept_count


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='删除没有 labelme 标注文件的图片',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    # 删除指定目录下未标注的图片
    python remove_no_detect_image.py /path/to/images
    
    # 预览模式（不实际删除）
    python remove_no_detect_image.py /path/to/images --dry-run
        """
    )
    
    parser.add_argument(
        'directory',
        type=str,
        help='包含图片的目录路径'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='预览模式，只显示将要删除的文件，不实际删除'
    )
    
    # 如果没有提供参数，显示帮助信息
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(0)
    
    args = parser.parse_args()
    
    try:
        print(f"{'=' * 60}")
        print(f"目录: {args.directory}")
        print(f"模式: {'预览模式（不会实际删除文件）' if args.dry_run else '删除模式'}")
        print(f"{'=' * 60}\n")
        
        removed, kept = remove_unlabeled_images(args.directory, args.dry_run)
        
        print(f"\n{'=' * 60}")
        print(f"处理完成!")
        print(f"{'将删除' if args.dry_run else '已删除'}: {removed} 个文件")
        print(f"保留: {kept} 个文件（有标注）")
        print(f"{'=' * 60}")
        
        if args.dry_run and removed > 0:
            print("\n提示: 这是预览模式，没有实际删除文件")
            print("如需实际删除，请去掉 --dry-run 参数")
        
    except Exception as e:
        print(f"错误: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
