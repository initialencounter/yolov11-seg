import os
import json
import shutil
from pathlib import Path
from ultralytics import YOLO

label_map = {0: "9", 1: "9A", 2: "BTY", 3: "CAO"}
MODEL_PATH  = r"C:\Users\29115\yolov8\yolov11-seg\runs\segment_17k\yolo26n-segment-17k-65epochs\weights\best.pt"
IMG_DIR     = r"C:\Users\29115\yolov8\yolov11-seg\datasets17k_labelme\labeled_5k"
WRONG_IMG_DIR = r"C:\Users\29115\yolov8\yolov11-seg\datasets17k_labelme\labeled_5k_wrong"

def check_annotations():
    os.makedirs(WRONG_IMG_DIR, exist_ok=True)
    model = YOLO(MODEL_PATH)

    for file_name in os.listdir(IMG_DIR):
        if not file_name.endswith(".json"):
            continue
            
        json_path = os.path.join(IMG_DIR, file_name)
        with open(json_path, "r", encoding="utf-8") as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                print(f"无法读取JSON: {json_path}")
                continue
                
        image_name = data.get("imagePath", file_name.replace(".json", ".jpg"))
        img_path = os.path.join(IMG_DIR, image_name)
        if not os.path.exists(img_path):
            img_path = os.path.join(IMG_DIR, file_name.replace(".json", ".png"))
            if not os.path.exists(img_path):
                img_path = os.path.join(IMG_DIR, file_name.replace(".json", ".jpeg"))
                
        if not os.path.exists(img_path):
            print(f"找不到图片: {image_name}")
            continue

        gt_labels = [shape["label"] for shape in data.get("shapes", [])]
        gt_labels.sort()

        results = model.predict(img_path, conf=0.7, verbose=False) 
        
        pred_labels = []
        if len(results) > 0 and results[0].boxes is not None:
            for cls_idx in results[0].boxes.cls.cpu().numpy():
                pred_label = label_map.get(int(cls_idx), str(int(cls_idx)))
                pred_labels.append(pred_label)
        pred_labels.sort()

        if gt_labels != pred_labels:
            print(f"异常或漏标 -> {file_name}")
            print(f"  人工标注 (GT): {gt_labels}")
            print(f"  模型预测 (PD): {pred_labels}")
            
            shutil.copy(json_path, os.path.join(WRONG_IMG_DIR, file_name))
            shutil.copy(img_path, os.path.join(WRONG_IMG_DIR, os.path.basename(img_path)))

if __name__ == "__main__":
    check_annotations()

