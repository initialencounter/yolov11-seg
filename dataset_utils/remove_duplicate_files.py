import os
import hashlib
from pathlib import Path


def calculate_file_hash(file_path):
    """计算文件的MD5哈希值"""
    hash_md5 = hashlib.md5()
    try:
        with open(file_path, 'rb') as f:
            # 分块读取文件，避免大文件占用过多内存
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {e}")
        return None


def find_duplicate_files(folder_path, recursive=True):
    """
    查找文件夹中内容相同的文件
    
    参数:
        folder_path: 文件夹路径
        recursive: 是否递归查找子文件夹 (默认True)
    """
    hash_dict = {}  # 存储 hash -> [文件路径列表]
    duplicates = []  # 存储重复文件组
    
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"文件夹不存在: {folder_path}")
        return []
    
    # 遍历文件夹中的所有文件
    print(f"正在{'递归' if recursive else '非递归'}扫描文件夹: {folder_path}")
    file_count = 0
    
    # 根据recursive参数选择遍历方式
    file_iterator = folder.rglob('*') if recursive else folder.glob('*')
    
    for file_path in file_iterator:
        if file_path.is_file():
            file_count += 1
            # 显示相对路径,更清晰
            relative_path = file_path.relative_to(folder)
            print(f"正在处理 ({file_count}): {relative_path}", end='\r')
            
            file_hash = calculate_file_hash(file_path)
            if file_hash:
                if file_hash in hash_dict:
                    hash_dict[file_hash].append(file_path)
                else:
                    hash_dict[file_hash] = [file_path]
    
    print(f"\n总共扫描了 {file_count} 个文件")
    
    # 找出重复的文件
    for file_hash, file_list in hash_dict.items():
        if len(file_list) > 1:
            duplicates.append(file_list)
    
    return duplicates


def delete_duplicate_files(folder_path, keep_first=True, dry_run=True, recursive=True):
    """
    删除重复文件
    
    参数:
        folder_path: 文件夹路径
        keep_first: True表示保留第一个文件,删除其他;False表示保留最后一个
        dry_run: True表示仅显示将要删除的文件,不实际删除
        recursive: 是否递归查找子文件夹 (默认True)
    """
    duplicates = find_duplicate_files(folder_path, recursive=recursive)
    
    if not duplicates:
        print("没有找到重复文件")
        return
    
    print(f"\n找到 {len(duplicates)} 组重复文件:")
    
    total_deleted = 0
    total_size_saved = 0
    
    for i, file_group in enumerate(duplicates, 1):
        print(f"\n=== 重复组 {i} ===")
        
        # 显示所有重复文件
        for j, file_path in enumerate(file_group):
            file_size = file_path.stat().st_size
            size_mb = file_size / (1024 * 1024)
            marker = "[保留]" if (keep_first and j == 0) or (not keep_first and j == len(file_group) - 1) else "[删除]"
            print(f"{marker} {file_path} ({size_mb:.2f} MB)")
        
        # 确定要删除的文件
        if keep_first:
            files_to_delete = file_group[1:]
        else:
            files_to_delete = file_group[:-1]
        
        # 删除文件
        for file_path in files_to_delete:
            file_size = file_path.stat().st_size
            total_size_saved += file_size
            
            if dry_run:
                print(f"  [模拟] 将删除: {file_path}")
            else:
                try:
                    file_path.unlink()
                    print(f"  [已删除] {file_path}")
                    total_deleted += 1
                except Exception as e:
                    print(f"  [错误] 删除失败: {file_path}, 原因: {e}")
    
    print(f"\n{'='*60}")
    if dry_run:
        print(f"[模拟模式] 将删除 {sum(len(g)-1 for g in duplicates)} 个重复文件")
        print(f"[模拟模式] 将节省空间: {total_size_saved / (1024*1024):.2f} MB")
        print("\n提示: 设置 dry_run=False 以实际执行删除操作")
    else:
        print(f"成功删除 {total_deleted} 个重复文件")
        print(f"节省空间: {total_size_saved / (1024*1024):.2f} MB")


if __name__ == "__main__":
    # 设置要处理的文件夹路径
    folder_path = r'C:\Users\29115\yolov8\yolov11-seg\datasets_20k'
    
    # recursive=True: 递归查找所有子文件夹
    # recursive=False: 仅查找当前文件夹
    
    # 首先运行模拟模式查看将要删除的文件
    delete_duplicate_files(folder_path, keep_first=True, dry_run=False, recursive=True)
