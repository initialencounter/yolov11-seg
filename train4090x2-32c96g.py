from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo26n-seg.pt")  # load a pretrained model (recommended for training)

    # Train the model with optimized GPU settings
    results = model.train(
        data="/root/dev/datasets12k/yolo26n-seg.yaml",
        batch=224,
        workers=4,
        epochs=1000,
        device=[0, 1],
        resume=True,
        save_period=10,
        patience=100,
        lr0 = 0.001,
        lrf= 0.001,
        augment=True,
    )
    
