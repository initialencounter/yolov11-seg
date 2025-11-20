"""
分离人工标注和模型标注的数据
人工标注的数据通常points数量较少（小于等于10个点），放入val目录
模型标注的数据points数量较多（大于10个点），放入train目录
"""
import json
import os
import shutil
from pathlib import Path


def check_annotation_type(json_file, manual_threshold=10):
    """
    检查JSON文件是否为人工标注
    人工标注：所有shapes的points数量都小于等于阈值
    模型标注：任何一个shape的points数量大于阈值
    
    Args:
        json_file: JSON标注文件路径
        manual_threshold: 判断人工标注的阈值
        
    Returns:
        (is_manual, max_points, min_points) 元组
        is_manual: True表示人工标注，False表示模型标注
        max_points: 最大点数
        min_points: 最小点数
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if 'shapes' not in data or not data['shapes']:
            return None, 0, 0
        
        # 检查所有shape的点数
        points_list = []
        for shape in data['shapes']:
            if 'points' in shape:
                num_points = len(shape['points'])
                points_list.append(num_points)
        
        if not points_list:
            return None, 0, 0
        
        max_points = max(points_list)
        min_points = min(points_list)
        
        # 只有所有shape的points都小于等于阈值，才认为是人工标注
        is_manual = all(p <= manual_threshold for p in points_list)
        
        return is_manual, max_points, min_points
    except Exception as e:
        print(f"Error reading {json_file}: {e}")
        return None, 0, 0


def split_annotations(dataset_dir, manual_threshold=10):
    """
    将标注数据分为人工标注和模型标注
    
    Args:
        dataset_dir: 数据集目录路径
        manual_threshold: 判断人工标注的阈值，小于等于此值的认为是人工标注
    """
    dataset_path = Path(dataset_dir)
    
    # 创建输出目录
    train_dir = dataset_path / 'train'
    val_dir = dataset_path / 'val'
    train_dir.mkdir(exist_ok=True)
    val_dir.mkdir(exist_ok=True)
    
    # 统计信息
    manual_count = 0
    auto_count = 0
    error_count = 0
    
    # 遍历所有JSON文件
    json_files = list(dataset_path.glob('*.json'))
    
    print(f"Found {len(json_files)} JSON files in {dataset_dir}")
    print(f"Manual annotation threshold: points <= {manual_threshold}")
    print("-" * 60)
    
    for json_file in json_files:
        # 跳过已经在train或val目录中的文件
        if 'train' in str(json_file) or 'val' in str(json_file):
            continue
        
        # 检查标注类型和统计点数
        is_manual, max_points, min_points = check_annotation_type(json_file, manual_threshold)
        print(f"Processing {json_file.name}: max points = {max_points}, min points = {min_points}")
        if is_manual is None:
            error_count += 1
            print(f"Skipping {json_file.name}: No valid annotations")
            continue
        
        # 获取对应的图片文件
        image_name = json_file.stem  # 去掉.json后缀
        possible_extensions = ['.jpeg', '.jpg', '.png', '.bmp']
        image_file = None
        
        for ext in possible_extensions:
            potential_image = json_file.parent / f"{image_name}{ext}"
            if potential_image.exists():
                image_file = potential_image
                break
        
        # 判断是人工标注还是模型标注
        if is_manual:
            # 人工标注 -> val
            target_dir = val_dir
            manual_count += 1
            label = "Manual"
        else:
            # 模型标注 -> train
            target_dir = train_dir
            auto_count += 1
            label = "Auto"
        
        # 移动JSON文件
        target_json = target_dir / json_file.name
        shutil.move(str(json_file), str(target_json))
        print(f"{label:6} | {json_file.name:30} | points: {max_points:3} -> {target_dir.name}/")
        
        # 如果存在对应的图片文件，也移动
        if image_file and image_file.exists():
            target_image = target_dir / image_file.name
            shutil.move(str(image_file), str(target_image))
    
    print("-" * 60)
    print(f"\nSummary:")
    print(f"  Manual annotations (val):  {manual_count} files")
    print(f"  Auto annotations (train):  {auto_count} files")
    print(f"  Errors/Skipped:            {error_count} files")
    print(f"  Total processed:           {manual_count + auto_count} files")
    print(f"\nOutput directories:")
    print(f"  Train: {train_dir}")
    print(f"  Val:   {val_dir}")


if __name__ == '__main__':
    # 数据集目录
    dataset_directory = r"C:\Users\29115\yolov8\yolov11-seg\datasets17k"
    
    # 人工标注的阈值（points数量小于等于此值认为是人工标注）
    threshold = 10
    
    # 执行分离
    split_annotations(dataset_directory, manual_threshold=threshold)
