import os
import json
from pathlib import Path

def rename_no_detection_files(folder_path, start_number=17281):
    """
    重命名no_detection文件夹中的图像和标签文件
    
    Args:
        folder_path: no_detection文件夹路径
        start_number: 起始编号，默认17281
    """
    folder = Path(folder_path)
    
    if not folder.exists():
        print(f"错误: 文件夹不存在: {folder_path}")
        return
    
    # 获取所有JSON文件（标签文件）
    json_files = sorted(folder.glob("*.json"))
    
    if not json_files:
        print("没有找到JSON文件")
        return
    
    print(f"找到 {len(json_files)} 个标签文件")
    
    # 创建重命名映射
    rename_map = []
    current_number = start_number
    
    for json_file in json_files:
        # 读取JSON文件获取图像文件扩展名
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                old_image_name = data.get('imagePath', '')
                
                if not old_image_name:
                    print(f"警告: {json_file.name} 中没有找到imagePath字段")
                    continue
                
                # 获取图像文件扩展名
                image_ext = Path(old_image_name).suffix
                old_image_path = folder / old_image_name
                
                # 检查图像文件是否存在
                if not old_image_path.exists():
                    print(f"警告: 图像文件不存在: {old_image_path}")
                    continue
                
                # 生成新文件名
                new_base_name = f"img_{current_number:06d}"
                new_image_name = f"{new_base_name}{image_ext}"
                new_json_name = f"{new_base_name}.json"
                
                rename_map.append({
                    'old_json': json_file,
                    'old_image': old_image_path,
                    'new_json': folder / new_json_name,
                    'new_image': folder / new_image_name,
                    'new_image_name': new_image_name,
                    'data': data
                })
                
                current_number += 1
                
        except Exception as e:
            print(f"错误: 处理 {json_file.name} 时出错: {e}")
            continue
    
    print(f"\n准备重命名 {len(rename_map)} 对文件 (从 img_{start_number:06d} 到 img_{current_number-1:06d})")
    
    # 确认操作
    response = input("是否继续? (y/n): ")
    if response.lower() != 'y':
        print("操作已取消")
        return
    
    # 执行重命名
    success_count = 0
    error_count = 0
    
    for item in rename_map:
        try:
            # 检查新文件名是否已存在
            if item['new_json'].exists() or item['new_image'].exists():
                print(f"警告: 目标文件已存在，跳过: {item['new_json'].name}")
                error_count += 1
                continue
            
            # 更新JSON中的imagePath
            item['data']['imagePath'] = item['new_image_name']
            
            # 重命名图像文件
            item['old_image'].rename(item['new_image'])
            
            # 保存并重命名JSON文件
            with open(item['new_json'], 'w', encoding='utf-8') as f:
                json.dump(item['data'], f, ensure_ascii=False, indent=2)
            
            # 删除旧的JSON文件
            item['old_json'].unlink()
            
            success_count += 1
            
            if success_count % 10 == 0:
                print(f"已处理 {success_count}/{len(rename_map)} 个文件...")
                
        except Exception as e:
            print(f"错误: 重命名 {item['old_json'].name} 时出错: {e}")
            error_count += 1
    
    print(f"\n完成!")
    print(f"成功: {success_count} 个文件")
    print(f"失败: {error_count} 个文件")
    print(f"最终编号范围: img_{start_number:06d} - img_{start_number + success_count - 1:06d}")


if __name__ == "__main__":
    # no_detection文件夹路径
    no_detection_folder = r"C:\Users\29115\yolov8\yolov11-seg\datasets_20k\labeled\no_detection"
    
    # 从17281开始重命名
    rename_no_detection_files(no_detection_folder, start_number=17281)
