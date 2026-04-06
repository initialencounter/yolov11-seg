import os
import json
import base64
import io
import random
import numpy as np
from PIL import Image
import argparse
from pathlib import Path
from collections import defaultdict

def read_name_file(name_path):
    if not os.path.exists(name_path):
        # 如果 obj.names 不存在,返回默认类别
        return ['9', '9A', 'BTY', 'CAO']
    with open(name_path, "r") as name_file:
        names = [name.strip() for name in name_file]
    return names

def convert_coor(size, xy):
    dw, dh = size
    x, y = xy
    return x / dw, y / dh

no_image_jsons = []
def convert(file, txt_name=None, img_output_dir=None):
    if txt_name is None:
        txt_name = file.rstrip(".json") + ".txt"

    names = read_name_file('obj.names')
    
    # 读取JSON文件
    with open(file, "r", encoding='utf-8') as txt_file:
        js = json.loads(txt_file.read())
    
    # 保存图片
    if 'imageData' in js and js['imageData']:
        img_data = img_b64_to_arr(js['imageData'])
        img_name = os.path.splitext(os.path.basename(file))[0] + '.png'
    else:
        img_name = js.get("imagePath")
        image_path = Path(file).parent / img_name
        if image_path.exists():
            img_data = image_path.read_bytes()
            img_data = img_b64_to_arr(base64.b64encode(img_data).decode('utf-8'))
        else:
            print(f"Warning: No image data found for {file}, skipping image saving")
            no_image_jsons.append(file)
            return
    
    if img_output_dir:
        img_path = os.path.join(img_output_dir, img_name)
    else:
        img_path = txt_name.replace('.txt', '.png')
    Image.fromarray(img_data).save(img_path)
    
    # 写入标签文件
    with open(txt_name, "w") as txt_outfile:
        for item in js["shapes"]:
            label = item["label"]
            # 查找类别索引
            try:
                cls = str(names.index(label))
            except ValueError:
                print(f"Warning: Label '{label}' not found in obj.names, skipping...")
                continue

            height, width = js["imageHeight"], js["imageWidth"]

            for idx, pt in enumerate(item["points"]):
                if idx == 0:
                    txt_outfile.write(cls)
                x, y = pt
                bb = convert_coor((width, height), [x, y])
                txt_outfile.write(" " + " ".join([str(a) for a in bb]))

            txt_outfile.write("\n")

def img_data_to_pil(img_data):
    f = io.BytesIO()
    f.write(img_data)
    return Image.open(f)

def img_data_to_arr(img_data):
    return np.array(img_data_to_pil(img_data))

def img_b64_to_arr(img_b64):
    return img_data_to_arr(base64.b64decode(img_b64))


def get_labels_from_json(json_path):
    """读取JSON文件，返回该文件包含的label集合。无标注返回 {'__background__'}。"""
    with open(json_path, 'r', encoding='utf-8') as f:
        js = json.load(f)
    labels = set()
    for item in js.get('shapes', []):
        label = item.get('label', '')
        if label:
            labels.add(label)
    if not labels:
        labels.add('__background__')
    return labels


def stratified_split(json_files, val_ratio, seed=42):
    """
    按label进行分层抽样，确保每种label（含background）都按 val_ratio 比例分到验证集。
    
    策略：
    1. 为每个文件确定其"主类别"（label集合）。
    2. 按主类别分组，每组内随机抽取 val_ratio 比例作为验证集。
    3. 这样能保证每种label在验证集中的比例近似 val_ratio。
    """
    random.seed(seed)
    
    # 按label组合分组
    group_map = defaultdict(list)  # key: frozenset of labels, value: list of json_files
    for jf in json_files:
        labels = get_labels_from_json(jf)
        key = frozenset(labels)
        group_map[key].append(jf)
    
    train_files = []
    val_files = []
    
    print(f"\n{'='*60}")
    print(f"数据集分层抽样统计 (val_ratio={val_ratio})")
    print(f"{'='*60}")
    
    for label_key in sorted(group_map.keys(), key=lambda x: sorted(x)):
        files = group_map[label_key]
        random.shuffle(files)
        
        n_val = max(1, round(len(files) * val_ratio))  # 至少1个
        if len(files) <= 1:
            # 只有1个文件时放到训练集
            train_files.extend(files)
            n_val = 0
        else:
            val_files.extend(files[:n_val])
            train_files.extend(files[n_val:])
        
        label_str = ', '.join(sorted(label_key))
        print(f"  [{label_str}]: 总计 {len(files)}, 训练 {len(files) - n_val}, 验证 {n_val}")
    
    print(f"{'='*60}")
    print(f"  总计: {len(json_files)} 文件, 训练 {len(train_files)}, 验证 {len(val_files)}")
    print(f"{'='*60}\n")
    
    return train_files, val_files


def main():
    parser = argparse.ArgumentParser(description="Convert JSON to TXT")
    parser.add_argument('--input', type=str, help="Path to the input JSON file or directory", required=True)
    parser.add_argument('--output', type=str, help="Path to the output TXT file or directory", default=None)
    parser.add_argument('--img-output', type=str, help="Path to the output image directory", default=None)
    parser.add_argument('--val-ratio', type=float, default=None,
                        help="验证集比例 (如 0.1 表示10%%作为验证集)。"
                             "启用后输出到 --output 下的 images/train, images/val, labels/train, labels/val")
    parser.add_argument('--seed', type=int, default=42, help="随机种子，默认42")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    # 如果输入是目录
    if input_path.is_dir():
        from concurrent.futures import ThreadPoolExecutor, as_completed
        import threading
        # 获取所有JSON文件
        json_files = list(input_path.glob('*.json'))
        if not json_files:
            print(f"No JSON files found in {input_path}")
            return

        print(f"Found {len(json_files)} JSON files to convert")

        # ======== 带 val-ratio 的分层抽样模式 ========
        if args.val_ratio is not None:
            if not args.output:
                print("Error: --output is required when using --val-ratio")
                return
            if not (0 < args.val_ratio < 1):
                print("Error: --val-ratio must be between 0 and 1")
                return

            base_output = Path(args.output)
            img_train_dir = base_output / 'images' / 'train'
            img_val_dir   = base_output / 'images' / 'val'
            lbl_train_dir = base_output / 'labels' / 'train'
            lbl_val_dir   = base_output / 'labels' / 'val'
            for d in [img_train_dir, img_val_dir, lbl_train_dir, lbl_val_dir]:
                d.mkdir(parents=True, exist_ok=True)

            # 分层抽样
            train_files, val_files = stratified_split(json_files, args.val_ratio, seed=args.seed)

            def convert_split(json_file, img_dir, lbl_dir):
                txt_name = lbl_dir / (json_file.stem + '.txt')
                try:
                    convert(str(json_file), str(txt_name), str(img_dir))
                    return True
                except Exception as e:
                    print(f"Error converting {json_file.name}: {e}")
                    return False

            max_workers = min(32, os.cpu_count() * 2 or 4)
            converted = 0
            total = len(train_files) + len(val_files)
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for jf in train_files:
                    fut = executor.submit(convert_split, jf, img_train_dir, lbl_train_dir)
                    futures[fut] = ('train', jf)
                for jf in val_files:
                    fut = executor.submit(convert_split, jf, img_val_dir, lbl_val_dir)
                    futures[fut] = ('val', jf)
                for future in as_completed(futures):
                    split_name, jf = futures[future]
                    converted += 1
                    if converted % 100 == 0 or converted == total:
                        print(f"  进度: {converted}/{total}")

            print(f"\n完成! 训练集: {len(train_files)}, 验证集: {len(val_files)}")
            print(f"输出目录: {base_output}")
            return

        # ======== 原有模式（不分割） ========
        # 创建输出目录
        if args.output:
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = input_path

        # 创建图片输出目录
        if args.img_output:
            img_output_dir = Path(args.img_output)
            img_output_dir.mkdir(parents=True, exist_ok=True)
        else:
            img_output_dir = None

        def convert_one(json_file):
            txt_name = output_dir / (json_file.stem + '.txt')
            try:
                convert(str(json_file), str(txt_name), str(img_output_dir) if img_output_dir else None)
                print(f"Converted: {json_file.name}")
            except Exception as e:
                print(f"Error converting {json_file.name}: {e}")

        max_workers = min(32, os.cpu_count() * 2 or 4)
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(convert_one, json_file) for json_file in json_files]
            for future in as_completed(futures):
                pass  # 结果已在convert_one中打印
    
    # 如果输入是单个文件
    elif input_path.is_file():
        convert(str(input_path), args.output, args.img_output)
        print(f"Converted: {input_path.name}")
    else:
        print(f"Error: {input_path} is not a valid file or directory")

if __name__ == "__main__":
    main()
    with open("no_image_jsons.txt", "w") as f:
        for json_file in no_image_jsons:
            f.write(json_file + "\n")