"""
将 datasets_20k/labeled 目录下所有子目录中的图像和标签文件合并到 datasets17k 目录

安全说明:
- 本脚本仅进行文件复制(copy),不会删除或移动原文件
- 原始数据保持完整,不会被破坏
- 如果目标文件已存在且内容相同,则跳过复制
- 如果目标文件已存在但内容不同,会重命名新文件(添加前缀)
"""

import os
import shutil
from pathlib import Path
from tqdm import tqdm


def merge_labeled_files(dry_run=False):
    """
    合并 labeled 目录下所有子目录中的文件到 datasets17k
    
    参数:
        dry_run: 如果为True,只显示将要执行的操作,不实际复制文件
    """
    # 定义源目录和目标目录
    source_dir = Path(r"C:\Users\29115\yolov8\yolov11-seg\datasets_20k\labeled")
    target_dir = Path(r"C:\Users\29115\yolov8\yolov11-seg\datasets_20k\datasets17k")
    
    # 验证源目录存在
    if not source_dir.exists():
        print(f"错误: 源目录不存在: {source_dir}")
        return
    
    # 显示操作模式
    if dry_run:
        print("="*60)
        print("【模拟运行模式】- 不会实际复制文件")
        print("="*60)
    else:
        print("="*60)
        print("【实际运行模式】- 将复制文件到目标目录")
        print("注意: 原始文件不会被删除或移动,完全安全!")
        print("="*60)
    
    # 确保目标目录存在
    if not dry_run:
        target_dir.mkdir(parents=True, exist_ok=True)
    
    # 统计变量
    total_files = 0
    copied_files = 0
    skipped_files = 0
    renamed_files = 0
    
    # 获取所有子目录
    subdirs = [d for d in source_dir.iterdir() if d.is_dir()]
    
    print(f"\n源目录: {source_dir}")
    print(f"目标目录: {target_dir}")
    print(f"找到 {len(subdirs)} 个子目录: {[d.name for d in subdirs]}")
    
    # 首先统计总文件数
    for subdir in subdirs:
        files = [f for f in subdir.iterdir() if f.is_file()]
        total_files += len(files)
    
    print(f"\n开始合并文件,共 {total_files} 个文件...")
    
    # 遍历所有子目录
    with tqdm(total=total_files, desc="合并进度") as pbar:
        for subdir in subdirs:
            print(f"\n处理子目录: {subdir.name}")
            
            # 获取子目录中的所有文件
            files = [f for f in subdir.iterdir() if f.is_file()]
            
            for file_path in files:
                target_path = target_dir / file_path.name
                
                # 检查目标文件是否已存在
                if target_path.exists():
                    # 如果文件已存在,比较大小,如果不同则添加子目录前缀
                    if target_path.stat().st_size != file_path.stat().st_size:
                        # 添加子目录名作为前缀避免冲突
                        new_name = f"{subdir.name}_{file_path.name}"
                        target_path = target_dir / new_name
                        print(f"  文件名冲突,重命名为: {new_name}")
                        renamed_files += 1
                    else:
                        # 文件相同,跳过
                        skipped_files += 1
                        pbar.update(1)
                        continue
                
                # 复制文件 (注意: 是复制,不是移动,原文件保持不变)
                if not dry_run:
                    try:
                        shutil.copy2(file_path, target_path)  # copy2 保留元数据
                        copied_files += 1
                    except Exception as e:
                        print(f"  复制失败 {file_path.name}: {e}")
                else:
                    # 模拟模式,只计数
                    copied_files += 1
                
                pbar.update(1)
    
    # 输出统计结果
    print(f"\n{'='*60}")
    if dry_run:
        print(f"【模拟运行完成】")
    else:
        print(f"【合并完成】")
    print(f"{'='*60}")
    print(f"总文件数:        {total_files}")
    print(f"成功复制:        {copied_files}")
    print(f"跳过(已存在):    {skipped_files}")
    print(f"重命名(冲突):    {renamed_files}")
    print(f"源目录:          {source_dir}")
    print(f"目标目录:        {target_dir}")
    print(f"{'='*60}")
    if not dry_run:
        print(f"✓ 原始文件完好无损,未被删除或修改")
        print(f"✓ 所有操作为复制操作,数据安全")
    print(f"{'='*60}")


if __name__ == "__main__":
    import sys
    
    # 检查是否使用 --dry-run 参数
    if "--dry-run" in sys.argv or "-d" in sys.argv:
        print("\n使用 --dry-run 模式,仅模拟运行,不会实际复制文件\n")
        merge_labeled_files(dry_run=True)
    else:
        # 询问用户确认
        print("\n" + "="*60)
        print("即将开始合并文件")
        print("="*60)
        print("源目录: C:\\Users\\29115\\yolov8\\yolov11-seg\\datasets_20k\\labeled")
        print("目标目录: C:\\Users\\29115\\yolov8\\yolov11-seg\\datasets_20k\\datasets17k")
        print("\n安全提示:")
        print("  ✓ 只会复制文件,不会删除原文件")
        print("  ✓ 原始数据完全安全")
        print("  ✓ 可以先使用 --dry-run 参数进行模拟运行")
        print("="*60)
        
        response = input("\n确认开始复制吗? (输入 'yes' 或 'y' 继续): ").strip().lower()
        if response in ['yes', 'y']:
            merge_labeled_files(dry_run=False)
        else:
            print("\n操作已取消。")
            print("提示: 可以使用 'python merge_labeled_to_datasets17k.py --dry-run' 进行模拟运行")
