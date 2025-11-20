import os
import shutil
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO


model = YOLO(r"C:\Users\29115\yolov8\yolov11-seg\runs\segment_788\train\weights\best.pt")

# 置信度阈值
conf_threshold = 0.589


# 批量预测并分类
def batch_predict_and_classify(source_dir, output_base_dir, batch_size=128):
  """批量预测并分类图片
  
  Args:
    source_dir: 源图片目录
    output_base_dir: 输出基础目录
    batch_size: 批处理大小,默认128
  """
  # 获取所有图片文件
  image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
  image_files = []
  
  source_path = Path(source_dir)
  for ext in image_extensions:
    # 使用 glob 的不区分大小写匹配,或者收集后去重
    image_files.extend(list(source_path.glob(f'*{ext}')))
    image_files.extend(list(source_path.glob(f'*{ext.upper()}')))
  
  # 去重(防止大小写重复)
  image_files = list(set(image_files))
  
  print(f"找到 {len(image_files)} 张图片")
  print(f"使用批处理大小: {batch_size}")
  
  if len(image_files) == 0:
    print("未找到图片文件!")
    return
  
  # 创建输出目录
  output_dirs = {
    'high_confidence': os.path.join(output_base_dir, 'high_confidence'),
    'medium_confidence': os.path.join(output_base_dir, 'medium_confidence'),
    'low_confidence': os.path.join(output_base_dir, 'low_confidence'),
    'no_detection': os.path.join(output_base_dir, 'no_detection')
  }
  for dir_path in output_dirs.values():
    os.makedirs(dir_path, exist_ok=True)
  
  # 统计结果
  stats = {
    'high_confidence': 0,
    'medium_confidence': 0,
    'low_confidence': 0,
    'no_detection': 0
  }
  
  # 批量预测 - 使用stream=True返回生成器
  # 指定保存目录,方便后续找到标签文件
  save_dir = Path(output_base_dir) / 'predict_temp'
  results = model.predict(
    source=source_dir,
    conf=conf_threshold,
    save_txt=True,  # 保存YOLO格式标签
    save_conf=True,  # 保存置信度
    exist_ok=True,
    batch=batch_size,  # 设置批处理大小
    verbose=True,  # 开启详细输出以显示进度
    stream=True,  # 使用流式处理
    project=str(save_dir.parent),  # 指定项目目录
    name=save_dir.name  # 指定保存名称
  )
  
  # 使用tqdm包装结果迭代器
  for result in tqdm(results, desc="分类图片", unit="张", total=len(image_files)):
    img_path = result.path
    boxes = result.boxes
    
    if boxes is not None and len(boxes) > 0:
      # 获取平均置信度
      avg_conf = boxes.conf.mean().item()
      
      # 根据置信度分类
      if avg_conf >= 0.8:
        confidence_level = 'high_confidence'
      elif avg_conf >= 0.6:
        confidence_level = 'medium_confidence'
      else:
        confidence_level = 'low_confidence'
      
      stats[confidence_level] += 1
      
      # 复制图片到对应目录
      dest_dir = output_dirs[confidence_level]
      shutil.copy2(img_path, dest_dir)
      
      # 复制标签文件
      # 标签文件在 save_dir/labels 目录下
      img_name = Path(img_path).stem
      label_file = save_dir / 'labels' / f'{img_name}.txt'
      if label_file.exists():
        shutil.copy2(label_file, dest_dir)
      else:
        print(f"\n警告: 未找到标签文件 {label_file}")
    else:
      stats['no_detection'] += 1
      # 未检测到的也复制到对应目录
      dest_dir = output_dirs['no_detection']
      shutil.copy2(img_path, dest_dir)
  
  # 打印统计结果
  print("\n处理完成! 统计结果:")
  print(f"  高置信度 (≥0.8): {stats['high_confidence']} 张")
  print(f"  中置信度 (0.6-0.8): {stats['medium_confidence']} 张")
  print(f"  低置信度 (<0.6): {stats['low_confidence']} 张")
  print(f"  未检测到: {stats['no_detection']} 张")
  
  # 清理临时预测目录
  if save_dir.exists():
    shutil.rmtree(save_dir)
    print(f"\n已清理临时目录: {save_dir}")


if __name__ == '__main__':
  # 执行批量处理
  # batch_size参数控制每次推理处理的图片数量,默认128
  # 可以根据显存大小调整: 显存小用32-64, 显存大用128-256
  batch_predict_and_classify(
    r'C:\Users\29115\yolov8\yolov11-seg\datasets_20k\images', 
    r'C:\Users\29115\yolov8\yolov11-seg\datasets_20k',
    batch_size=128  # 批处理大小
  )
