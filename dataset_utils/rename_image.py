import os
import argparse
from pathlib import Path


def rename_file_extensions(folder_path, old_ext, new_ext, dry_run=False):
    """
    批量重命名文件夹中的文件后缀
    
    Args:
        folder_path: 目标文件夹路径
        old_ext: 原始文件后缀 (例如: '.txt', 'txt')
        new_ext: 新的文件后缀 (例如: '.label', 'label')
        dry_run: 如果为True,只显示将要执行的操作而不实际重命名
    """
    # 规范化后缀格式
    if not old_ext.startswith('.'):
        old_ext = '.' + old_ext
    if not new_ext.startswith('.'):
        new_ext = '.' + new_ext
    
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"错误: 文件夹 '{folder_path}' 不存在!")
        return
    
    if not folder.is_dir():
        print(f"错误: '{folder_path}' 不是一个文件夹!")
        return
    
    # 查找所有匹配的文件
    files = list(folder.glob(f'*{old_ext}'))
    
    if not files:
        print(f"在 '{folder_path}' 中没有找到后缀为 '{old_ext}' 的文件")
        return
    
    print(f"找到 {len(files)} 个后缀为 '{old_ext}' 的文件")
    
    if dry_run:
        print("\n=== 预览模式 (不会实际重命名) ===")
    else:
        print("\n=== 开始重命名 ===")
    
    success_count = 0
    error_count = 0
    
    for file_path in files:
        new_name = file_path.stem + new_ext
        new_path = file_path.parent / new_name
        
        try:
            if dry_run:
                print(f"  {file_path.name} -> {new_name}")
                success_count += 1
            else:
                if new_path.exists():
                    print(f"  跳过 {file_path.name}: 目标文件 {new_name} 已存在")
                    error_count += 1
                else:
                    file_path.rename(new_path)
                    print(f"  ✓ {file_path.name} -> {new_name}")
                    success_count += 1
        except Exception as e:
            print(f"  ✗ 重命名 {file_path.name} 失败: {e}")
            error_count += 1
    
    print(f"\n{'预览' if dry_run else '完成'}: 成功 {success_count} 个, 失败 {error_count} 个")


def main():
    parser = argparse.ArgumentParser(
        description='批量重命名文件夹中的文件后缀',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
使用示例:
  # 预览将 .txt 改为 .label (不实际修改)
  python rename_image.py -f ./datasets/labels -o txt -n label --dry-run
  
  # 实际执行重命名
  python rename_image.py -f ./datasets/labels -o txt -n label
  
  # 使用完整后缀格式
  python rename_image.py -f ./datasets/labels -o .txt -n .label
        '''
    )
    
    parser.add_argument('-f', '--folder', required=True,
                        help='目标文件夹路径')
    parser.add_argument('-o', '--old-ext', required=True,
                        help='原始文件后缀 (例如: txt 或 .txt)')
    parser.add_argument('-n', '--new-ext', required=True,
                        help='新的文件后缀 (例如: label 或 .label)')
    parser.add_argument('--dry-run', action='store_true',
                        help='预览模式,不实际重命名文件')
    
    args = parser.parse_args()
    
    rename_file_extensions(
        folder_path=args.folder,
        old_ext=args.old_ext,
        new_ext=args.new_ext,
        dry_run=args.dry_run
    )


if __name__ == '__main__':
    main()
