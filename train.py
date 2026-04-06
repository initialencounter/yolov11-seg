from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolo26n-seg.pt")  # load a pretrained model (recommended for training)

    # Train the model with optimized GPU settings
    results = model.train(
        data="datasets9k/yolo26n-seg.yaml",
        epochs=100,
        resume=True,
        save_period=10,
        # patience=100,
        augment=True,
    )
    
