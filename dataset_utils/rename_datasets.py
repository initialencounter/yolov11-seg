import os
import json
import glob

def rename_labelme_dataset(directory, start_index, prefix="img_"):
    # 支持的图片格式
    image_extensions = ('.png', '.jpg', '.jpeg')
    
    # 获取目录下的所有文件并筛选出图片
    all_files = os.listdir(directory)
    images = [f for f in all_files if f.lower().endswith(image_extensions)]
    # 排序以保证每次运行的结果一致
    images.sort()
    
    current_index = start_index
    
    for img_name in images:
        old_img_path = os.path.join(directory, img_name)
        
        # 获取文件名和后缀
        file_base, file_ext = os.path.splitext(img_name)
        old_json_name = file_base + '.json'
        old_json_path = os.path.join(directory, old_json_name)
        
        # 如果对应的 json 不存在，可以选择跳过或仅重命名图片
        # 这里假设只有存在对应图片的我们才处理
        
        new_base = f"{prefix}{current_index:06d}"
        new_img_name = new_base + file_ext
        new_json_name = new_base + '.json'
        
        new_img_path = os.path.join(directory, new_img_name)
        new_json_path = os.path.join(directory, new_json_name)
        
        # 重命名图片
        os.rename(old_img_path, new_img_path)
        print(f"Renamed: {img_name} -> {new_img_name}")
        
        # 如果存在对应的 labelme 标注文件，则重命名并修改其内部内容
        if os.path.exists(old_json_path):
            with open(old_json_path, 'r', encoding='utf-8') as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    print(f"Error reading JSON: {old_json_path}")
                    data = None
            
            if data is not None:
                # 修改 json 内部的 imagePath
                data['imagePath'] = new_img_name
                
                # 写回 json 文件并使用新名字保存
                with open(new_json_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, ensure_ascii=False, indent=2)
                
                # 删除旧的 json 文件
                os.remove(old_json_path)
                print(f"Updated and renamed json: {old_json_name} -> {new_json_name}")
                
        current_index += 1

if __name__ == '__main__':
    target_dir = r"C:\Users\29115\yolov8\yolov11-seg\datasets17k_labelme\background_images"
    rename_labelme_dataset(target_dir, start_index=19000)
