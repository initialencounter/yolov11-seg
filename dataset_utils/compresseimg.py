import os
from PIL import Image
from pathlib import Path

def compress_images_lossless(input_dir, output_dir):
    """
    无损/高保真压缩图片，保持原始尺寸和分辨率
    
    :param input_dir: 原始图片所在目录
    :param output_dir: 压缩后图片保存目录
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    # 常见图片格式
    supported_extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

    # 遍历目录下所有文件
    for file_path in input_path.rglob('*'):
        if file_path.is_file() and file_path.suffix.lower() in supported_extensions:
            try:
                # 构建输出路径以保持原始目录结构
                relative_path = file_path.relative_to(input_path)
                save_path = output_path / relative_path
                
                # 确保输出所属的子文件夹存在
                save_path.parent.mkdir(parents=True, exist_ok=True)

                # 将输出路径的后缀名统一改为 .jpg
                save_path = save_path.with_suffix('.jpg').with_suffix('.jpg')
                if save_path.name.endswith('.jpg'): # just safely .with_suffix('.jpg')
                   pass
                save_path = save_path.with_suffix('.jpg')

                with Image.open(file_path) as img:
                    # 统一将图片转为 RGB 格式 (JPEG不支持透明度或索引色彩)
                    if img.mode in ('RGBA', 'P', 'LA'):
                        img = img.convert('RGBA')
                        background = Image.new('RGB', img.size, (255, 255, 255))
                        background.paste(img, mask=img.split()[3])
                        img = background
                    elif img.mode != 'RGB':
                        img = img.convert('RGB')

                    # 统一保存为 JPEG 格式
                    # quality=95 保证肉眼无损的感官质量，optimize=True 优化哈夫曼表来减小体积
                    img.save(save_path, format="JPEG", optimize=True, quality=95)

                # 计算压缩效果
                original_size = os.path.getsize(file_path) / 1024
                compressed_size = os.path.getsize(save_path) / 1024
                print(f"[成功] {relative_path} | {original_size:.2f}KB -> {compressed_size:.2f}KB")

            except Exception as e:
                print(f"[错误] 处理文件 {file_path.name} 时出错: {str(e)}")

if __name__ == '__main__':
    print("=== 批量图片无损压缩工具 ===")
    src_dir = r"C:\Users\29115\yolov8\yolov11-seg\datasets17k_labelme\img_016001"
    dst_dir = r"C:\Users\29115\yolov8\yolov11-seg\datasets17k_labelme\img_016001_compressed"
    
    if os.path.exists(src_dir):
        print("\n开始处理，请稍候...")
        compress_images_lossless(src_dir, dst_dir)
        print("\n所有图片处理完成！")
    else:
        print(f"找不到输入的目录: {src_dir}")
