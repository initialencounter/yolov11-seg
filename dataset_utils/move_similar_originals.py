import os
import re
import shutil
from pathlib import Path

def move_similar_originals_from_czkawka(czkawka_log_path, originals_dir, output_dir):
    """
    根据 Czkawka 输出文件，移动相似重复的原图及其 JSON 标注文件
    """
    originals_dir = Path(originals_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 读取 Czkawka 输出
    with open(czkawka_log_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 按 "Found " 分割成多个相似的区块
    blocks = content.split("Found ")[1:]
    
    # 提取需要移动的组（按组保存所有的前缀，包含 Original，方便分文件夹对比）
    groups_to_move = []
    
    for block in blocks:
        lines = block.strip().split('\n')
        
        # 跳过第一行（"X images which have similar friends"）
        image_lines = [line for line in lines[1:] if line.strip().startswith('"')]
        
        group_prefixes = []
        # 将整组图片（包括 Original 和 重复项）都提取出来，以便放入同一个子文件夹复核
        # 若只想移动重复项而保留 Original 在原目录中，请将 `range(0, len(image_lines))` 改为 `range(1, len(image_lines))`
        for i in range(0, len(image_lines)):
            line = image_lines[i]
            match = re.search(r'"([^"]+)"', line)
            if match:
                filepath = match.group(1)
                # 解析裁剪图的名称，提取原图的基础名称 img_xxxxxx
                # 例如：img_009510_obj0_BTY.png -> img_009510
                base_name_match = re.search(r'(img_\d+)', os.path.basename(filepath))
                if base_name_match:
                    base_name = base_name_match.group(1)
                    group_prefixes.append(base_name)
                    
        if len(group_prefixes) > 0:
            groups_to_move.append(group_prefixes)

    print(f"找到 {len(groups_to_move)} 组相似图片。")
    
    # 开始在原图目录移动匹配的图片和对应的 JSON 文件
    moved_count = 0
    for idx, group_prefixes in enumerate(groups_to_move):
        # 针对每一组创建一个子文件夹，命名如 group_001_img_001711
        if not group_prefixes: continue
        group_dir_name = f"group_{idx+1:03d}_{group_prefixes[0]}"
        group_dir = output_dir / group_dir_name
        group_dir.mkdir(parents=True, exist_ok=True)
        
        # 使用 set 去重，防止同一张原图的多个裁剪区在一组内导致重复移动报错
        for prefix in set(group_prefixes):
            # 查找该前缀对应的所有文件 (.jpg, .png, .json 等)
            matched_files = list(originals_dir.glob(f"{prefix}.*"))
            for file_path in matched_files:
                try:
                    dest_path = group_dir / file_path.name
                    shutil.move(str(file_path), str(dest_path))
                    print(f"Moved to {group_dir_name}: {file_path.name}")
                    moved_count += 1
                except Exception as e:
                    print(f"错误: 移动 {file_path.name} 失败: {e}")
                
    print(f"操作完成，共移动了 {moved_count} 个文件到 {output_dir}")

if __name__ == "__main__":
    # 配置你的路径
    # 1. 把你的 Czkawka 终端输出复制保存到一个 txt 文件里面，比如 czkawka_result.txt
    czkawka_log_path = r"C:\Users\29115\yolov8\yolov11-seg\czkawka_result.txt"
    
    # 2. 最初包含了 JSON 和图像原图的文件夹
    originals_dir = r"C:\Users\29115\yolov8\yolov11-seg\datasets17k_labelme\labeled_12k"
    
    # 3. 相似图片需要移动到哪个新文件夹
    output_dir = r"C:\Users\29115\yolov8\yolov11-seg\datasets17k_labelme\similar_duplicates"
    
    # 确保保存了 txt 之后放开执行
    if os.path.exists(czkawka_log_path):
        move_similar_originals_from_czkawka(czkawka_log_path, originals_dir, output_dir)
    else:
        print(f"未找到 {czkawka_log_path}，请先将输出保存为 txt 文件。")