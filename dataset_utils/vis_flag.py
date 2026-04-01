import os
import json

label_map = {0: '9', 1: '9A', 2: 'BTY', 3: 'CAO'}

class1_count = 0

def visualize_all_annotations(image_dir):
    for label_file in os.listdir(image_dir):
        if not label_file.endswith(('.json')):
            continue
        
        label_file_path = os.path.join(image_dir, label_file)
        with open(label_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            shapes = data.get('shapes')
            global class1_count
            class1_count += len(shapes)
                
        
visualize_all_annotations(
    r"C:\Users\29115\yolov8\yolov11-seg\datasets17k_labelme\i\img_007326",
)

print(f"Class '9' count: {class1_count}")
