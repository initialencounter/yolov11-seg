"""
半自动标注脚本
使用 YOLO 分割模型推理 train4100 目录下所有未标注图片，
将 mask 转为 4 点多边形，输出 LabelMe 格式 JSON。
"""

import os
import json
import base64
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ── 配置 ────────────────────────────────────────────────────────────────
MODEL_PATH  = r"C:\Users\29115\yolov8\yolov11-seg\runs\segment_9k\613epoch\weights\best.pt"
IMG_DIR     = r"C:\Users\29115\yolov8\yolov11-seg\datasets17k_labelme\img_006087"
LABELME_VER = "5.11.3_auto_label"
CONF_THRES  = 0.40  # 置信度阈值，可按需调整

YOLO_CLASSES = {0: '9', 1: '9A', 2: 'BTY', 3: 'CAO'}
# ────────────────────────────────────────────────────────────────────────


def mask_to_4pt_polygon(mask_xy: np.ndarray) -> list[list[float]]:
    """
    将 YOLO 输出的 xy 轮廓点（N×2 float32）转为 4 个顶点的多边形。
    算法：凸包 → Douglas-Peucker 迭代，直到点数 <= 4。
    这样能贴合实际目标轮廓，去除多余突出边角。
    返回 [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]。
    """
    pts = mask_xy.astype(np.float32).reshape(-1, 1, 2)

    # 1. 先求凸包，去除内凹噪点
    hull = cv2.convexHull(pts)

    # 2. Douglas-Peucker：逐步增大 epsilon，直到点数收敛到 4
    perimeter = cv2.arcLength(hull, True)
    epsilon = 0.01 * perimeter
    for _ in range(200):
        approx = cv2.approxPolyDP(hull, epsilon, True)
        if len(approx) <= 4:
            break
        epsilon *= 1.05   # 每次增大 5%

    approx = approx.reshape(-1, 2)

    if len(approx) == 4:
        return [[float(p[0]), float(p[1])] for p in approx]

    # 点数不足 4（极少数退化情况），回退到最小外接矩形
    rect = cv2.minAreaRect(pts)
    box  = cv2.boxPoints(rect)
    return [[float(p[0]), float(p[1])] for p in box]


def image_to_base64(img_path: str) -> str:
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def make_labelme_json(img_path: str, shapes: list, img_h: int, img_w: int) -> dict:
    img_name = Path(img_path).name
    return {
        "version": LABELME_VER,
        "flags": {},
        "shapes": shapes,
        "imagePath": img_name,
        "imageData": image_to_base64(img_path),
        "imageHeight": img_h,
        "imageWidth": img_w,
    }


def make_shape(label: str, points: list[list[float]]) -> dict:
    return {
        "label": label,
        "points": points,
        "group_id": None,
        "description": "",
        "shape_type": "polygon",
        "flags": {},
        "mask": None,
    }


def process_image(model: YOLO, img_path: str) -> int:
    """推理单张图片，写 JSON。返回检测到的目标数量。"""
    results = model(img_path, conf=CONF_THRES, verbose=False)

    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]

    shapes = []
    for result in results:
        if result.masks is None:
            continue

        # masks.xy: list of (N,2) arrays, one per detected instance
        for i, mask_xy in enumerate(result.masks.xy):
            cls_id = int(result.boxes.cls[i].item())
            label  = YOLO_CLASSES.get(cls_id, str(cls_id))

            if len(mask_xy) < 3:
                print(f"  [skip] {Path(img_path).name} instance {i}: mask too small")
                continue

            # points = [[float(p[0]), float(p[1])] for p in mask_xy]
            points = mask_to_4pt_polygon(mask_xy)
            shapes.append(make_shape(label, points))

    json_data = make_labelme_json(img_path, shapes, img_h, img_w)

    json_path = Path(img_path).with_suffix(".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    return len(shapes)


def main():
    img_dir = Path(IMG_DIR)
    all_images = sorted(img_dir.glob("*.jpeg")) + sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))

    # 跳过已有 JSON 的图片
    todo = [p for p in all_images if not p.with_suffix(".json").exists()]
    print(f"共 {len(all_images)} 张图片，其中 {len(all_images)-len(todo)} 张已有标注，待处理 {len(todo)} 张")

    if not todo:
        print("全部已标注，退出。")
        return

    print(f"加载模型：{MODEL_PATH}")
    model = YOLO(MODEL_PATH)

    for idx, img_path in enumerate(todo, 1):
        print(f"[{idx}/{len(todo)}] {img_path.name} ... ", end="", flush=True)
        try:
            n = process_image(model, str(img_path))
            print(f"检测到 {n} 个目标")
        except Exception as e:
            print(f"ERROR: {e}")

    print("完成！")


if __name__ == "__main__":
    main()
