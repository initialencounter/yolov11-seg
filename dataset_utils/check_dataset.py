"""
YOLOv11分割数据集格式检测脚本
检测数据集是否符合YOLO格式要求,输出详细的错误信息
"""

import os
import yaml
from pathlib import Path
from PIL import Image
import numpy as np


class DatasetChecker:
    def __init__(self, yaml_path):
        """
        初始化数据集检测器
        
        Args:
            yaml_path: 数据集配置文件路径
        """
        self.yaml_path = yaml_path
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
        """检查图像和标签文件"""
        print("\n=== 检查图像和标签文件 ===")
        
        if dataset_config is None:
            return
        
        dataset_path = Path(dataset_config['path'])
        splits = ['train', 'val']
        
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
            
            # 标签目录 (通常是images -> labels)
            label_dir = dataset_path / dataset_config[split].replace('images', 'labels')
            if not label_dir.exists():
                self.errors.append(f"❌ {split} 标签目录不存在: {label_dir}")
                continue
            
            print(f"✓ 图像目录: {img_dir}")
            print(f"✓ 标签目录: {label_dir}")
            
            # 获取所有图像文件
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
            image_files = []
            for ext in image_extensions:
                image_files.extend(list(img_dir.glob(f'*{ext}')))
                image_files.extend(list(img_dir.glob(f'*{ext.upper()}')))
            
            print(f"✓ 找到 {len(image_files)} 张图像")
            self.stats['total_images'] += len(image_files)
            
            if len(image_files) == 0:
                self.errors.append(f"❌ {split} 数据集中没有找到图像文件")
                continue
            
            # 检查每张图像和对应的标签
            for img_path in image_files:
                # 对应的标签文件
                label_path = label_dir / (img_path.stem + '.txt')
                
                # 检查标签文件是否存在
                if not label_path.exists():
                    self.errors.append(f"❌ 缺少标签文件: {img_path.name} -> {label_path.name}")
                    self.stats['missing_labels'] += 1
                    continue
                
                self.stats['total_labels'] += 1
                
                # 检查图像文件
                try:
                    img = Image.open(img_path)
                    img_width, img_height = img.size
                    img.close()
                except Exception as e:
                    self.errors.append(f"❌ 无法打开图像: {img_path.name}, 错误: {e}")
                    continue
                
                # 检查标签文件格式
                self.check_label_file(label_path, img_path.name, img_width, img_height, 
                                     len(dataset_config['names']))
            
            # 检查是否有多余的标签文件（没有对应的图像）
            label_files = list(label_dir.glob('*.txt'))
            image_stems = {img.stem for img in image_files}
            for label_path in label_files:
                if label_path.stem not in image_stems:
                    self.warnings.append(f"⚠ 标签文件没有对应的图像: {label_path.name}")
                    self.stats['missing_images'] += 1
    
    def check_label_file(self, label_path, image_name, img_width, img_height, num_classes):
        """
        检查单个标签文件的格式
        
        Args:
            label_path: 标签文件路径
            image_name: 对应的图像文件名
            img_width: 图像宽度
            img_height: 图像高度
            num_classes: 类别数量
        """
        try:
            with open(label_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
        except Exception as e:
            self.errors.append(f"❌ 无法读取标签文件 {label_path.name}: {e}")
            self.stats['invalid_labels'] += 1
            return
        
        # 检查是否为空文件
        if len(lines) == 0:
            self.warnings.append(f"⚠ 空标签文件: {label_path.name} (图像: {image_name})")
            self.stats['empty_labels'] += 1
            return
        
        # 检查每一行
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split()
            
            # 检查格式：class_id x1 y1 x2 y2 ... (分割格式至少需要6个点，即12个坐标)
            if len(parts) < 7:  # class_id + 至少3个点(6个坐标)
                self.errors.append(
                    f"❌ 标签格式错误 {label_path.name}:{line_num} - "
                    f"坐标点数不足 (需要至少3个点，当前: {(len(parts)-1)//2}个点)"
                )
                self.stats['invalid_labels'] += 1
                continue
            
            try:
                # 检查类别ID
                class_id = int(parts[0])
                if class_id < 0 or class_id >= num_classes:
                    self.errors.append(
                        f"❌ 标签文件 {label_path.name}:{line_num} - "
                        f"类别ID {class_id} 超出范围 [0, {num_classes-1}]"
                    )
                    self.stats['invalid_labels'] += 1
                
                # 检查坐标点数是否为偶数
                coords = parts[1:]
                if len(coords) % 2 != 0:
                    self.errors.append(
                        f"❌ 标签文件 {label_path.name}:{line_num} - "
                        f"坐标数量为奇数 ({len(coords)})"
                    )
                    self.stats['invalid_labels'] += 1
                    continue
                
                # 检查每个坐标值
                for i, coord in enumerate(coords):
                    coord_val = float(coord)
                    # YOLO格式坐标应该是归一化的 [0, 1]
                    if coord_val < 0 or coord_val > 1:
                        self.errors.append(
                            f"❌ 标签文件 {label_path.name}:{line_num} - "
                            f"坐标值 {coord_val} 超出范围 [0, 1] (索引: {i})"
                        )
                        self.stats['invalid_labels'] += 1
                        break
                
            except ValueError as e:
                self.errors.append(
                    f"❌ 标签文件 {label_path.name}:{line_num} - "
                    f"数值格式错误: {e}"
                )
                self.stats['invalid_labels'] += 1
    
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
    # 数据集配置文件路径
    yaml_path = "datasets/yolo11n-seg.yaml"
    
    # 创建检测器并运行
    checker = DatasetChecker(yaml_path)
    success = checker.run()
    
    # 返回状态码
    exit(0 if success else 1)


if __name__ == "__main__":
    main()
