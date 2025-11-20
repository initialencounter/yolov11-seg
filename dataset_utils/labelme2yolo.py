import os
import json
import base64
import io
import numpy as np
from PIL import Image
import argparse
from pathlib import Path

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

def main():
    parser = argparse.ArgumentParser(description="Convert JSON to TXT")
    parser.add_argument('--input', type=str, help="Path to the input JSON file or directory", required=True)
    parser.add_argument('--output', type=str, help="Path to the output TXT file or directory", default=None)
    parser.add_argument('--img-output', type=str, help="Path to the output image directory", default=None)
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    # 如果输入是目录
    if input_path.is_dir():
        # 获取所有JSON文件
        json_files = list(input_path.glob('*.json'))
        if not json_files:
            print(f"No JSON files found in {input_path}")
            return
        
        print(f"Found {len(json_files)} JSON files to convert")
        
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
        
        # 转换每个文件
        for json_file in json_files:
            txt_name = output_dir / (json_file.stem + '.txt')
            try:
                convert(str(json_file), str(txt_name), str(img_output_dir) if img_output_dir else None)
                print(f"Converted: {json_file.name}")
            except Exception as e:
                print(f"Error converting {json_file.name}: {e}")
    
    # 如果输入是单个文件
    elif input_path.is_file():
        convert(str(input_path), args.output, args.img_output)
        print(f"Converted: {input_path.name}")
    else:
        print(f"Error: {input_path} is not a valid file or directory")

if __name__ == "__main__":
    main()