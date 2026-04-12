from ultralytics import YOLO
import torch

if __name__ == '__main__':
    model = YOLO("yolo26n-seg.pt")  # load a pretrained model (recommended for training)
    
    base_lr = 0.01
    batch_size = 64
    your_batch_size = 400
    scaled_lr = base_lr * your_batch_size / batch_size  # 线性缩放规则
    print(f"Scaled learning rate: {scaled_lr}")
    
    # Train the model with optimized GPU settings
    results = model.train(
        data="/root/dev/datasets12k/yolo26n-seg.yaml",
        batch=your_batch_size,
        workers=6,
        epochs=1000,
        device=[0, 1, 2, 3], #export MKL_THREADING_LAYER=GNU
        resume=True,
        save_period=10,
        patience=100,
        lr0=scaled_lr,
        optimizer='SGD',
        cos_lr=True,
        augment=True,
    )
    
