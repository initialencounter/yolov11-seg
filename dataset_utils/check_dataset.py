"""
YOLOv11分割数据集格式检测脚本
检测数据集是否符合YOLO格式要求,输出详细的错误信息
优化版本: 使用多进程和缓存提升大数据集检查速度
"""

import os
import yaml
from pathlib import Path
from PIL import Image
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager, cpu_count
from tqdm import tqdm
import struct

def get_image_size_fast(image_path):
    """
    快速获取图像尺寸,不完全加载图像
    
    Args:
        image_path: 图像文件路径
        
    Returns:
        (width, height) 或 None
    """
    try:
        with Image.open(image_path) as img:
            return img.size
    except:
        return None


def check_single_label(args):
    """
    检查单个标签文件 (用于多进程)
    
    Args:
        args: (label_path, img_width, img_height, num_classes)
        
    Returns:
        dict: 包含检查结果的字典
    """
    label_path, img_width, img_height, num_classes = args
    result = {
        'errors': [],
        'warnings': [],
        'is_empty': False,
        'is_invalid': False
    }
    
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except Exception as e:
        result['errors'].append(f"❌ 无法读取标签文件 {label_path.name}: {e}")
        result['is_invalid'] = True
        return result
    
    # 检查是否为空文件
    if len(lines) == 0:
        result['warnings'].append(f"⚠ 空标签文件: {label_path.name}")
        result['is_empty'] = True
        return result
    
    # 检查每一行
    for line_num, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        
        # 检查格式
        if len(parts) < 7:
            result['errors'].append(
                f"❌ 标签格式错误 {label_path.name}:{line_num} - "
                f"坐标点数不足 (需要至少3个点，当前: {(len(parts)-1)//2}个点)"
            )
            result['is_invalid'] = True
            continue
        
        try:
            # 检查类别ID
            class_id = int(parts[0])
            if class_id < 0 or class_id >= num_classes:
                result['errors'].append(
                    f"❌ 标签文件 {label_path.name}:{line_num} - "
                    f"类别ID {class_id} 超出范围 [0, {num_classes-1}]"
                )
                result['is_invalid'] = True
            
            # 检查坐标点数是否为偶数
            coords = parts[1:]
            if len(coords) % 2 != 0:
                result['errors'].append(
                    f"❌ 标签文件 {label_path.name}:{line_num} - "
                    f"坐标数量为奇数 ({len(coords)})"
                )
                result['is_invalid'] = True
                continue
            
            # 检查每个坐标值
            for i, coord in enumerate(coords):
                coord_val = float(coord)
                if coord_val < 0 or coord_val > 1:
                    result['errors'].append(
                        f"❌ 标签文件 {label_path.name}:{line_num} - "
                        f"坐标值 {coord_val} 超出范围 [0, 1] (索引: {i})"
                    )
                    result['is_invalid'] = True
                    break
        
        except ValueError as e:
            result['errors'].append(
                f"❌ 标签文件 {label_path.name}:{line_num} - "
                f"数值格式错误: {e}"
            )
            result['is_invalid'] = True
    
    return result


class DatasetChecker:
    def __init__(self, yaml_path, num_workers=None):
        """
        初始化数据集检测器
        
        Args:
            yaml_path: 数据集配置文件路径
            num_workers: 工作进程数,默认为CPU核心数
        """
        self.yaml_path = yaml_path
        self.num_workers = num_workers or max(1, cpu_count() - 1)
        self.errors = []
        self.warnings = []
        self.stats = {
            'total_images': 0,
            'total_labels': 0,
            'missing_labels': 0,
            'missing_images': 0,
            'invalid_labels': 0,
            'empty_labels': 0
        }
        
    def check_yaml(self):
        """检查YAML配置文件"""
        print("\n=== 检查YAML配置文件 ===")
        
        if not os.path.exists(self.yaml_path):
            self.errors.append(f"❌ YAML文件不存在: {self.yaml_path}")
            return None
        
        try:
            with open(self.yaml_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
            print(f"✓ YAML文件读取成功")
        except Exception as e:
            self.errors.append(f"❌ YAML文件读取失败: {e}")
            return None
        
        # 检查必需字段
        required_fields = ['path', 'train', 'names']
        for field in required_fields:
            if field not in data:
                self.errors.append(f"❌ YAML缺少必需字段: {field}")
        
        # 检查路径
        if 'path' in data:
            dataset_path = Path(data['path'])
            if not dataset_path.exists():
                self.errors.append(f"❌ 数据集根目录不存在: {dataset_path}")
            else:
                print(f"✓ 数据集根目录: {dataset_path}")
        
        # 检查类别
        if 'names' in data:
            if isinstance(data['names'], dict):
                num_classes = len(data['names'])
                print(f"✓ 类别数量: {num_classes}")
                print(f"  类别列表: {data['names']}")
            else:
                self.errors.append(f"❌ names字段格式错误，应为字典格式")
        
        return data
    
    def check_images_and_labels(self, dataset_config):
        """检查图像和标签文件 (优化版本)"""
        print("\n=== 检查图像和标签文件 ===")
        print(f"使用 {self.num_workers} 个工作进程")
        
        if dataset_config is None:
            return
        
        dataset_path = Path(dataset_config['path'])
        splits = ['train', 'val']
        num_classes = len(dataset_config['names'])
        
        for split in splits:
            if split not in dataset_config or not dataset_config[split]:
                self.warnings.append(f"⚠ 未配置 {split} 数据集")
                continue
            
            print(f"\n--- 检查 {split} 数据集 ---")
            
            # 图像目录
            img_dir = dataset_path / dataset_config[split]
            if not img_dir.exists():
                self.errors.append(f"❌ {split} 图像目录不存在: {img_dir}")
                continue
            
            # 标签目录
            label_dir = dataset_path / dataset_config[split].replace('images', 'labels')
            if not label_dir.exists():
                self.errors.append(f"❌ {split} 标签目录不存在: {label_dir}")
                continue
            
            print(f"✓ 图像目录: {img_dir}")
            print(f"✓ 标签目录: {label_dir}")
            
            # 获取所有图像文件 (优化: 使用rglob一次性获取)
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
            print("正在扫描图像文件...")
            image_files = [
                f for f in img_dir.iterdir() 
                if f.suffix.lower() in image_extensions
            ]
            
            print(f"✓ 找到 {len(image_files)} 张图像")
            self.stats['total_images'] += len(image_files)
            
            if len(image_files) == 0:
                self.errors.append(f"❌ {split} 数据集中没有找到图像文件")
                continue
            
            # 构建图像文件名到路径的映射 (优化: O(1)查找)
            image_stem_to_path = {img.stem: img for img in image_files}
            
            # 获取所有标签文件
            print("正在扫描标签文件...")
            label_files = list(label_dir.glob('*.txt'))
            
            # 检查缺失的标签文件
            print("检查缺失的标签文件...")
            for img_path in tqdm(image_files, desc="检查标签存在性", disable=len(image_files)<100):
                label_path = label_dir / (img_path.stem + '.txt')
                if not label_path.exists():
                    self.errors.append(f"❌ 缺少标签文件: {img_path.name}")
                    self.stats['missing_labels'] += 1
            
            # 找出存在的图像-标签对
            valid_pairs = []
            for img_path in image_files:
                label_path = label_dir / (img_path.stem + '.txt')
                if label_path.exists():
                    valid_pairs.append((img_path, label_path))
            
            self.stats['total_labels'] += len(valid_pairs)
            
            # 使用多进程并行检查标签格式
            if valid_pairs:
                print(f"使用多进程检查 {len(valid_pairs)} 个标签文件...")
                
                # 准备任务参数
                tasks = []
                for img_path, label_path in valid_pairs:
                    # 快速获取图像尺寸
                    img_size = get_image_size_fast(img_path)
                    if img_size is None:
                        self.errors.append(f"❌ 无法打开图像: {img_path.name}")
                        continue
                    
                    img_width, img_height = img_size
                    tasks.append((label_path, img_width, img_height, num_classes))
                
                # 多进程处理
                with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                    futures = {executor.submit(check_single_label, task): task for task in tasks}
                    
                    for future in tqdm(as_completed(futures), total=len(tasks), desc="检查标签格式"):
                        try:
                            result = future.result()
                            self.errors.extend(result['errors'])
                            self.warnings.extend(result['warnings'])
                            if result['is_empty']:
                                self.stats['empty_labels'] += 1
                            if result['is_invalid']:
                                self.stats['invalid_labels'] += 1
                        except Exception as e:
                            self.errors.append(f"❌ 处理标签时出错: {e}")
            
            # 检查多余的标签文件
            print("检查多余的标签文件...")
            for label_path in label_files:
                if label_path.stem not in image_stem_to_path:
                    self.warnings.append(f"⚠ 标签文件没有对应的图像: {label_path.name}")
                    self.stats['missing_images'] += 1
    
    def print_summary(self):
        """打印检测摘要"""
        print("\n" + "="*60)
        print("数据集检测摘要")
        print("="*60)
        
        print(f"\n统计信息:")
        print(f"  总图像数: {self.stats['total_images']}")
        print(f"  总标签数: {self.stats['total_labels']}")
        print(f"  缺少标签: {self.stats['missing_labels']}")
        print(f"  缺少图像: {self.stats['missing_images']}")
        print(f"  无效标签: {self.stats['invalid_labels']}")
        print(f"  空标签文件: {self.stats['empty_labels']}")
        
        if self.warnings:
            print(f"\n警告 ({len(self.warnings)}):")
            for warning in self.warnings[:20]:  # 只显示前20条
                print(f"  {warning}")
            if len(self.warnings) > 20:
                print(f"  ... 还有 {len(self.warnings) - 20} 条警告")
        
        if self.errors:
            print(f"\n错误 ({len(self.errors)}):")
            for error in self.errors[:30]:  # 只显示前30条
                print(f"  {error}")
            if len(self.errors) > 30:
                print(f"  ... 还有 {len(self.errors) - 30} 条错误")
        
        print("\n" + "="*60)
        if len(self.errors) == 0:
            print("✓ 数据集格式检查通过！")
        else:
            print("✗ 数据集存在错误，请修复后再训练")
        print("="*60 + "\n")
    
    def run(self):
        """运行完整的检测流程"""
        print("开始检测数据集格式...")
        print(f"YAML配置文件: {self.yaml_path}")
        
        # 1. 检查YAML配置
        dataset_config = self.check_yaml()
        
        # 2. 检查图像和标签
        self.check_images_and_labels(dataset_config)
        
        # 3. 打印摘要
        self.print_summary()
        
        return len(self.errors) == 0


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='YOLOv11分割数据集格式检测工具 (优化版)')
    parser.add_argument('--yaml', type=str, 
                       default=r"C:\Users\29115\yolov8\yolov11-seg\datasets17k_yolo\yolo11n-seg.yaml",
                       help='数据集YAML配置文件路径')
    parser.add_argument('--workers', type=int, default=None,
                       help='工作进程数,默认为CPU核心数-1')
    
    args = parser.parse_args()
    
    # 创建检测器并运行
    print(f"使用配置文件: {args.yaml}")
    if args.workers:
        print(f"指定工作进程数: {args.workers}")
    
    checker = DatasetChecker(args.yaml, num_workers=args.workers)
    success = checker.run()
    
    # 返回状态码
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
