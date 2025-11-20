"""
多边形区域切割脚本
从labelme的JSON文件中读取多边形标注,从原图中切割出对应区域并保存
方便批量检查标注质量
"""

import json
import os
from pathlib import Path
import cv2
import numpy as np
from tqdm import tqdm


def crop_polygon_from_image(image_path, json_path, output_dir, padding=10):
    """
    从图像中根据JSON文件的多边形标注切割出区域
    
    Args:
        image_path: 原图路径
        json_path: labelme JSON文件路径
        output_dir: 输出目录
        padding: 切割时的边距(像素)
    
    Returns:
        切割出的图像数量
    """
    # 读取JSON文件
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 读取图像
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"警告: 无法读取图像 {image_path}")
        return 0
    
    # 获取原图文件名(不含扩展名)
    base_name = Path(image_path).stem
    
    count = 0
    # 遍历所有标注的形状
    for idx, shape in enumerate(data.get('shapes', [])):
        if shape['shape_type'] != 'polygon':
            continue
        
        # 获取多边形点
        points = np.array(shape['points'], dtype=np.int32)
        
        # 获取多边形的边界框
        x_min = max(0, int(points[:, 0].min()) - padding)
        y_min = max(0, int(points[:, 1].min()) - padding)
        x_max = min(image.shape[1], int(points[:, 0].max()) + padding)
        y_max = min(image.shape[0], int(points[:, 1].max()) + padding)
        
        # 切割图像
        cropped = image[y_min:y_max, x_min:x_max].copy()
        
        # 创建掩码 - 调整坐标到切割后的图像
        mask = np.zeros(cropped.shape[:2], dtype=np.uint8)
        adjusted_points = points.copy()
        adjusted_points[:, 0] -= x_min
        adjusted_points[:, 1] -= y_min
        cv2.fillPoly(mask, [adjusted_points], 255)
        
        # 应用掩码(可选:使背景变为白色或透明)
        # 这里我们保留整个矩形区域,方便查看
        
        # 在图像上绘制多边形轮廓,方便查看
        cv2.polylines(cropped, [adjusted_points], True, (0, 255, 0), 2)
        
        # 获取标签和置信度信息
        label = shape.get('label', 'unknown')
        description = shape.get('description', '')
        
        # 构建输出文件名
        if description:
            # 提取置信度
            confidence = description.replace('confidence: ', '')
            output_name = f"{base_name}_obj{idx}_{label}_conf{confidence}.jpg"
        else:
            output_name = f"{base_name}_obj{idx}_{label}.jpg"
        
        output_path = os.path.join(output_dir, output_name)
        
        # 保存切割后的图像
        cv2.imwrite(output_path, cropped)
        count += 1
    
    return count


def process_directory(input_dir, output_dir, padding=10):
    """
    批量处理目录下的所有图像和JSON文件
    
    Args:
        input_dir: 输入目录(包含图像和JSON文件)
        output_dir: 输出目录
        padding: 切割时的边距
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    
    # 创建输出目录
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 查找所有JSON文件
    json_files = list(input_path.glob('*.json'))
    
    if not json_files:
        print(f"在 {input_dir} 中未找到JSON文件")
        return
    
    print(f"找到 {len(json_files)} 个JSON文件")
    
    total_crops = 0
    success_count = 0
    
    # 处理每个JSON文件
    for json_file in tqdm(json_files, desc="处理进度"):
        # 查找对应的图像文件
        # 尝试多种图像格式
        image_file = None
        for ext in ['.jpeg', '.jpg', '.png', '.bmp']:
            potential_image = json_file.with_suffix(ext)
            if potential_image.exists():
                image_file = potential_image
                break
        
        if image_file is None:
            print(f"\n警告: 未找到 {json_file.stem} 对应的图像文件")
            continue
        
        # 切割多边形区域
        try:
            crops = crop_polygon_from_image(image_file, json_file, output_path, padding)
            total_crops += crops
            if crops > 0:
                success_count += 1
        except Exception as e:
            print(f"\n错误: 处理 {json_file.name} 时出错: {e}")
    
    print(f"\n处理完成!")
    print(f"成功处理: {success_count}/{len(json_files)} 个文件")
    print(f"共切割: {total_crops} 个区域")
    print(f"输出目录: {output_path}")


def main():
    """主函数"""
    # 配置路径
    # 输入目录 - 包含图像和JSON标注文件的目录
    input_dir = r"c:\Users\29115\yolov8\yolov11-seg\datasets_20k\no_detection"
    
    # 输出目录 - 切割后的图像保存位置
    output_dir = r"c:\Users\29115\yolov8\yolov11-seg\datasets_20k\no_detection_cropped_polygons"
    
    # 切割时的边距(像素)
    padding = 10
    
    print("=" * 60)
    print("多边形区域切割工具")
    print("=" * 60)
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    print(f"边距设置: {padding} 像素")
    print("=" * 60)
    
    # 处理目录
    process_directory(input_dir, output_dir, padding)


if __name__ == "__main__":
    main()
