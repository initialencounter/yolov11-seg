import os
import json
import shutil
from pathlib import Path
from shapely.geometry import Polygon

def check_duplicate_polygons(source_dir, output_dir, overlap_threshold=0.5):
    """
    遍历指定目录下的所有LabelMe标注文件(.json)，检查图片中是否存在重叠面积较大的多边形。
    如果任何两个多边形的相交面积占两者中较小多边形面积的比例超过 overlap_threshold (默认50%)，
    则将该JSON文件及关联图片复制到输出目录。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"创建输出目录: {output_dir}")

    # 遍历源目录下的所有文件
    for filename in os.listdir(source_dir):
        if not filename.endswith('.json'):
            continue

        json_path = os.path.join(source_dir, filename)
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"无法读取文件 {filename}: {e}")
            continue

        shapes = data.get('shapes', [])
        # 如果多边形少于2个，不可能重叠，跳过
        if len(shapes) < 2:
            continue

        # 解析包含的合法多边形
        polygons = []
        for shape in shapes:
            if shape.get('shape_type') == 'polygon':
                points = shape.get('points')
                if points and len(points) >= 3:
                    poly = Polygon(points)
                    # 修复可能自相交的无效多边形
                    if not poly.is_valid:
                        poly = poly.buffer(0)
                    polygons.append(poly)

        has_overlap = False
        n = len(polygons)

        # 两两计算多边形的重叠面积
        for i in range(n):
            for j in range(i + 1, n):
                poly1 = polygons[i]
                poly2 = polygons[j]

                # 快速检查是否有交集外接框
                if not poly1.intersects(poly2):
                    continue

                try:
                    intersection_area = poly1.intersection(poly2).area
                    min_area = min(poly1.area, poly2.area)

                    if min_area > 0 and (intersection_area / min_area) > overlap_threshold:
                        has_overlap = True
                        break
                except Exception:
                    continue
            
            if has_overlap:
                break

        # 如果存在重叠大于50%，复制到新文件夹
        if has_overlap:
            print(f"发现多边形重叠并复制: {filename}")
            
            # 定位图片
            image_filename = data.get('imagePath')
            base_name = filename.rsplit('.', 1)[0]
            
            if not image_filename:
                image_filename = base_name + '.jpg'
            
            image_path_src = os.path.join(source_dir, image_filename)
            
            # 兼容找不到原图的情况（排查不同扩展名）
            if not os.path.exists(image_path_src):
                found = False
                for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                    temp_path = os.path.join(source_dir, base_name + ext)
                    if os.path.exists(temp_path):
                        image_path_src = temp_path
                        found = True
                        break
                if not found:
                    print(f"  -> 警告: 找不到对应的图片文件，仅复制了JSON ({base_name})")

            # 复制文件
            shutil.copy(json_path, os.path.join(output_dir, filename))
            if os.path.exists(image_path_src):
                shutil.copy(image_path_src, os.path.join(output_dir, os.path.basename(image_path_src)))

if __name__ == "__main__":
    # 配置此处即可执行
    # source_dir = "../datasets17k_labelme/img_013000"
    # output_dir = "../datasets17k_labelme/overlap_issues"
    
    # 将下方路径修改为你的实际测试路径
    source_folder = r"C:\Users\29115\yolov8\yolov11-seg\datasets17k_labelme\labeled_9k"
    output_folder = os.path.join(Path(source_folder).parent, Path(source_folder).name + "_duplicate_polygons")
    
    check_duplicate_polygons(source_folder, output_folder, overlap_threshold=0.5)
