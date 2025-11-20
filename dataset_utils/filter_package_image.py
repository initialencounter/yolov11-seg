import os
import shutil
from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm


def filter_package_images(
    model_path: str,
    source_dir: str,
    target_dir: str,
    batch_size: int = 32,
    device: str = "0"
):
    """
    预测所有图片,将未检测到目标的图片移动到目标文件夹
    
    Args:
        model_path: 模型路径
        source_dir: 源图片文件夹路径
        target_dir: 目标文件夹路径(未检测到目标的图片)
        batch_size: 批处理大小
        device: GPU设备编号,默认为"0"
    """
    # 创建目标文件夹
    os.makedirs(target_dir, exist_ok=True)
    
    # 加载模型
    print(f"正在加载模型: {model_path}")
    model = YOLO(model_path)
    
    # 获取所有图片文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp', 
                        '*.JPG', '*.JPEG', '*.PNG', '*.BMP', '*.WEBP']
    image_files = []
    for pattern in image_extensions:
        image_files.extend(Path(source_dir).glob(pattern))
    
    # 去重（防止文件系统不区分大小写时重复）
    image_files = list(set(image_files))
    
    print(f"找到 {len(image_files)} 张图片")
    
    if len(image_files) == 0:
        print("未找到任何图片文件")
        return
    
    # 批量预测
    moved_count = 0
    total_batches = (len(image_files) + batch_size - 1) // batch_size
    
    for i in tqdm(range(0, len(image_files), batch_size), total=total_batches, desc="处理进度"):
        batch_files = image_files[i:i + batch_size]
        batch_paths = [str(f) for f in batch_files]
        
        # 批量推理,使用GPU
        results = model.predict(
            batch_paths,
            device=device,
            verbose=False,
            stream=False
        )
        
        # 检查每张图片的结果
        for img_path, result in zip(batch_files, results):
            # 如果没有检测到任何目标
            if result.masks is None or len(result.masks) == 0:
                # 移动图片到目标文件夹
                target_path = os.path.join(target_dir, img_path.name)
                shutil.move(str(img_path), target_path)
                moved_count += 1
    
    print(f"\n完成! 共移动 {moved_count} 张未检测到目标的图片到 {target_dir}")
    print(f"剩余 {len(image_files) - moved_count} 张有目标的图片在 {source_dir}")


if __name__ == '__main__':
    # 配置参数
    model_path = r"C:\Users\29115\yolov8\yolov11-seg\runs\segment_788\train\weights\best.pt"
    source_dir = r"C:\Users\29115\yolov8\yolov11-seg\datasets_20k\package_image"
    target_dir = r"C:\Users\29115\yolov8\yolov11-seg\datasets_20k\not_package_image"
    
    # 执行过滤
    filter_package_images(
        model_path=model_path,
        source_dir=source_dir,
        target_dir=target_dir,
        batch_size=128,  # 批处理大小,可根据GPU内存调整
        device="0"  # 使用第一块GPU
    )