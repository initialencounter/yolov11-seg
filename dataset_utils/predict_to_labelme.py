import os
import json
from pathlib import Path
from PIL import Image
from ultralytics import YOLO


def predict_and_save_labelme(model, image_path, output_json_path, class_names=None):
    """
    使用YOLO模型预测单张图片并保存为LabelMe JSON格式
    
    Args:
        model: YOLO模型对象
        image_path: 图片路径
        output_json_path: 输出JSON文件路径
        class_names: 类别名称列表，默认为None则使用数字编号
    
    Returns:
        bool: 是否成功保存
    """
    try:
        # 读取图片信息
        img = Image.open(image_path)
        img_width, img_height = img.size
        
        # 使用模型预测
        results = model(image_path, verbose=False)
        
        # 初始化LabelMe数据结构
        labelme_data = {
            "version": "5.0.1",
            "flags": {},
            "shapes": [],
            "imagePath": os.path.basename(image_path),
            "imageData": None,
            "imageHeight": img_height,
            "imageWidth": img_width
        }
        
        # 处理预测结果
        for result in results:
            if result.masks is None:
                continue
                
            # 获取masks和类别信息
            masks_xy = result.masks.xy  # 多边形格式的掩码
            boxes = result.boxes
            
            for i, mask_points in enumerate(masks_xy):
                # 获取类别信息
                class_id = int(boxes.cls[i])
                confidence = float(boxes.conf[i])
                
                # 确定标签名称
                if class_names and class_id < len(class_names):
                    label = class_names[class_id]
                else:
                    label = f"class_{class_id}"
                
                # 转换点坐标格式 (numpy array -> list of [x, y])
                points = mask_points.tolist()
                
                # 添加形状
                shape = {
                    "label": label,
                    "points": points,
                    "group_id": None,
                    "shape_type": "polygon",
                    "flags": {},
                    "description": f"confidence: {confidence:.4f}"
                }
                labelme_data["shapes"].append(shape)
        
        # 保存JSON文件
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(labelme_data, f, ensure_ascii=False, indent=2)
        
        return True
        
    except Exception as e:
        print(f"错误: 处理 {image_path} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def batch_predict_to_labelme(model_path, image_dir, output_dir, class_names=None):
    """
    批量预测图片并转换为LabelMe格式
    
    Args:
        model_path: YOLO模型权重文件路径
        image_dir: 包含待预测图片的目录
        output_dir: 输出JSON文件的目录
        class_names: 类别名称列表
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载模型
    print(f"正在加载模型: {model_path}")
    model = YOLO(model_path)
    print("模型加载完成!")
    
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    
    # 查找所有图片文件
    image_files = []
    for file in os.listdir(image_dir):
        if Path(file).suffix.lower() in image_extensions:
            image_files.append(file)
    
    print(f"\n找到 {len(image_files)} 张图片")
    print("开始批量预测...\n")
    
    success_count = 0
    fail_count = 0
    
    for idx, img_file in enumerate(sorted(image_files), 1):
        img_path = os.path.join(image_dir, img_file)
        json_file = Path(img_file).stem + '.json'
        json_path = os.path.join(output_dir, json_file)
        
        print(f"[{idx}/{len(image_files)}] 正在处理: {img_file}", end=" ... ")
        
        if predict_and_save_labelme(model, img_path, json_path, class_names):
            # 读取保存的JSON以获取shapes数量
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                num_objects = len(data['shapes'])
            
            print(f"✓ 完成 (检测到 {num_objects} 个对象)")
            success_count += 1
        else:
            print("✗ 失败")
            fail_count += 1
    
    print(f"\n{'='*60}")
    print(f"批量预测完成!")
    print(f"成功: {success_count} 个文件")
    print(f"失败: {fail_count} 个文件")
    print(f"输出目录: {output_dir}")
    print(f"{'='*60}")


if __name__ == '__main__':
    # 模型路径
    model_path = r"C:\Users\29115\yolov8\yolov11-seg\runs\segment_788\train\weights\best.pt"
    
    # 图片目录
    image_directory = r"C:\Users\29115\yolov8\yolov11-seg\datasets_20k\not_package_image"
    
    # 定义类别名称 (根据你的实际类别修改)
    # 如果不确定类别，可以设置为None，会自动使用 class_0, class_1 等
    class_names = [
        '9',
        '9A',
        'BTY',
        'CAO'
    ]
    
    # 批量预测并转换为LabelMe格式
    batch_predict_to_labelme(
        model_path=model_path,
        image_dir=image_directory,
        output_dir=image_directory,
        class_names=class_names
    )
