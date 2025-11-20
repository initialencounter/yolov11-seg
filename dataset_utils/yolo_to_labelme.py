import os
import json
import base64
from pathlib import Path
from PIL import Image
import numpy as np


def yolo_to_labelme(yolo_txt_path, image_path, class_names=None):
    """
    将YOLO分割格式转换为LabelMe JSON格式
    
    Args:
        yolo_txt_path: YOLO标注文件路径 (.txt)
        image_path: 对应的图片路径
        class_names: 类别名称列表,默认为None则使用数字编号
    
    Returns:
        labelme_data: LabelMe格式的字典
    """
    # 读取图片信息
    img = Image.open(image_path)
    img_width, img_height = img.size
    
    # 将图片转为base64 (可选,LabelMe可以不包含imageData)
    # 注释掉可减小JSON文件大小
    # with open(image_path, 'rb') as f:
    #     image_data = base64.b64encode(f.read()).decode('utf-8')
    
    # 初始化LabelMe数据结构
    labelme_data = {
        "version": "5.0.1",
        "flags": {},
        "shapes": [],
        "imagePath": os.path.basename(image_path),
        "imageData": None,  # 设为 None 不包含 base64 数据
        "imageHeight": img_height,
        "imageWidth": img_width
    }
    
    # 读取YOLO标注
    if not os.path.exists(yolo_txt_path):
        return labelme_data
    
    with open(yolo_txt_path, 'r') as f:
        lines = f.readlines()
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        parts = line.split()
        if len(parts) < 3:  # 至少需要类别ID + 一对坐标
            continue
            
        class_id = int(parts[0])
        coords = list(map(float, parts[1:]))
        
        # 检查坐标点数量是否为偶数（成对的x,y坐标）
        if len(coords) % 2 != 0:
            print(f"警告: 坐标点数量不是偶数 (类别 {class_id}, 坐标数: {len(coords)})，丢弃最后一个不完整的坐标")
            coords = coords[:-1]  # 丢弃最后一个单独的数字
        
        # 如果丢弃后没有足够的坐标点，跳过
        if len(coords) < 6:  # 至少需要3个点（6个数字）才能构成多边形
            print(f"警告: 坐标点太少，跳过该标注 (类别 {class_id})")
            continue
        
        # 将归一化坐标转换为实际像素坐标
        points = []
        for i in range(0, len(coords), 2):
            x = coords[i] * img_width
            y = coords[i + 1] * img_height
            points.append([x, y])
        
        # 确定标签名称
        if class_names and class_id < len(class_names):
            label = class_names[class_id]
        else:
            label = f"class_{class_id}"
        
        # 添加形状
        shape = {
            "label": label,
            "points": points,
            "group_id": None,
            "shape_type": "polygon",
            "flags": {}
        }
        labelme_data["shapes"].append(shape)
    
    return labelme_data


def batch_convert_yolo_to_labelme(source_dir, output_dir, class_names=None, include_imagedata=False):
    """
    批量转换YOLO格式到LabelMe格式
    
    Args:
        source_dir: 包含图片和标注文件的源目录
        output_dir: 输出JSON文件的目录
        class_names: 类别名称列表
        include_imagedata: 是否在JSON中包含图片的base64数据(默认False以减小文件大小)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    # 查找所有图片文件
    image_files = []
    for file in os.listdir(source_dir):
        if Path(file).suffix.lower() in image_extensions:
            image_files.append(file)
    
    print(f"找到 {len(image_files)} 张图片")
    
    converted_count = 0
    for img_file in sorted(image_files):
        img_path = os.path.join(source_dir, img_file)
        txt_file = Path(img_file).stem + '.txt'
        txt_path = os.path.join(source_dir, txt_file)
        
        # 检查是否存在对应的标注文件
        if not os.path.exists(txt_path):
            print(f"警告: 未找到标注文件 {txt_file}, 跳过 {img_file}")
            continue
        
        try:
            # 转换格式
            labelme_data = yolo_to_labelme(txt_path, img_path, class_names)
            
            # 检查是否有有效的标注
            if len(labelme_data["shapes"]) == 0:
                print(f"警告: {img_file} 没有有效的标注数据")
            
            # 可选：如果需要imageData，重新读取并编码
            if include_imagedata:
                with open(img_path, 'rb') as f:
                    image_data = base64.b64encode(f.read()).decode('utf-8')
                    labelme_data["imageData"] = image_data
            
            # 保存JSON文件
            json_file = Path(img_file).stem + '.json'
            json_path = os.path.join(output_dir, json_file)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(labelme_data, f, ensure_ascii=False, indent=2)
            
            print(f"已转换: {img_file} -> {json_file} (包含 {len(labelme_data['shapes'])} 个标注)")
            converted_count += 1
            
        except Exception as e:
            print(f"错误: 处理 {img_file} 时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n完成! 成功转换了 {converted_count} 个文件")
    print(f"输出目录: {output_dir}")


if __name__ == '__main__':
    # 定义类别名称 (根据你的实际类别修改)
    class_names = [
        "background",      # 类别0
        "object",          # 类别1
        "package",         # 类别2
        # 添加更多类别...
    ]
    
    # 源目录(包含图片和txt标注文件)
    source_directory = r"C:\Users\29115\yolov8\yolov11-seg\datasets_20k\low_confidence"
    
    # 输出目录
    output_directory = r"C:\Users\29115\yolov8\yolov11-seg\datasets_20k\low_confidence_labelme"
    
    # 批量转换
    # include_imagedata=False 不包含图片base64数据,大幅减小JSON文件大小(推荐)
    # include_imagedata=True 包含图片base64数据,文件会很大但可独立使用
    batch_convert_yolo_to_labelme(
        source_dir=source_directory,
        output_dir=output_directory,
        class_names=class_names,
        include_imagedata=False  # 推荐设为False,只存路径
    )
