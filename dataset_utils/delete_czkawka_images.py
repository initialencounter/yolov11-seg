import os
import re

def delete_images(txt_file):
    if not os.path.exists(txt_file):
        print(f"找不到文件: {txt_file}")
        return

    count = 0
    with open(txt_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        # 正则表达式匹配被双引号包裹的文件路径
        match = re.search(r'"([^"]+)"', line)
        if match:
            filepath = match.group(1)
            
            # 如果您只想删除重复项而保留标记为 "Original" 的原始文件，
            # 取消注释下面这两行代码即可：
            # if "- Original" in line:
            #     continue
            
            if os.path.exists(filepath):
                try:
                    os.remove(filepath)
                    print(f"已删除: {filepath}")
                    count += 1
                except Exception as e:
                    print(f"删除失败 {filepath}: {e}")
            else:
                print(f"文件不存在，已跳过: {filepath}")
                
    print(f"\n清理完成！总共删除了 {count} 张图片。")

if __name__ == '__main__':
    # 指向您的结果文件
    delete_images('czkawka_result.txt')
