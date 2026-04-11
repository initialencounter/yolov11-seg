import os
from ultralytics.utils.plotting import plot_results

def main():
    # 指向你的 results.csv 绝对路径
    csv_file = r"C:\Users\29115\yolov8\yolov11-seg\runs\segment_9k\250epoch\results.csv"
    
    if not os.path.exists(csv_file):
        print(f"未找到文件: {csv_file}")
        return

    print(f"正在读取 {csv_file} 并重新生成图表...")
    try:
        # segment=True 确保会绘制 Mask 相关的 Loss 和 mAP 指标
        plot_results(file=csv_file)
        print(f"✅ 成功! 重新绘制的图表已保存在 results.csv 所在的文件夹中。")
    except Exception as e:
        print(f"❌ 绘制时发生错误: {e}")

if __name__ == "__main__":
    main()
