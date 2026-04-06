import json
import os


def remove_json_without_image(root_dir: str, dry_run: bool = False) -> None:
    """
    递归遍历指定目录下的所有 labelme JSON 标注文件，
    检查是否存在对应的图片文件，若不存在则删除该 JSON 文件。

    Args:
        root_dir: 要遍历的根目录路径
        dry_run:  若为 True，只打印将要删除的文件，不实际删除
    """
    root_dir = os.path.abspath(root_dir)
    deleted = 0
    skipped = 0

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if not filename.lower().endswith(".json"):
                continue

            json_path = os.path.join(dirpath, filename)

            # 读取 JSON，获取 imagePath 字段
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError) as e:
                print(f"[WARN] 无法读取 {json_path}: {e}")
                skipped += 1
                continue

            image_path_field = data.get("imagePath")
            if not image_path_field:
                print(f"[SKIP] 无 imagePath 字段，跳过: {json_path}")
                skipped += 1
                continue

            # imagePath 为相对路径时，相对于 JSON 文件所在目录解析
            if not os.path.isabs(image_path_field):
                image_abs_path = os.path.join(dirpath, image_path_field)
            else:
                image_abs_path = image_path_field

            if os.path.isfile(image_abs_path):
                continue  # 图片存在，保留 JSON

            # 图片不存在，删除 JSON
            if dry_run:
                print(f"[DRY RUN] 将删除: {json_path}  (图片不存在: {image_abs_path})")
            else:
                os.remove(json_path)
                print(f"[DELETED] {json_path}  (图片不存在: {image_abs_path})")
            deleted += 1

    print(f"\n完成。删除 {deleted} 个 JSON 文件，跳过 {skipped} 个。")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="删除没有对应图片的 labelme JSON 标注文件")
    # parser.add_argument("root_dir", help="要处理的根目录路径")
    # parser.add_argument(
    #     "--dry-run",
    #     action="store_true",
    #     help="仅打印将要删除的文件，不实际删除",
    # )
    args = parser.parse_args()
    
    remove_json_without_image(r"C:\Users\29115\yolov8\yolov11-seg\datasets17k_labelme\labeled_8k", dry_run=False)
