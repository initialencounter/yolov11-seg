from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO("yolo11n-seg.yaml")  # build a new model from YAML
    model = YOLO("yolo11n-seg.pt")  # load a pretrained model (recommended for training)
    model = YOLO("yolo11n-seg.yaml").load("yolo11n.pt")  # build from YAML and transfer weights

    # Train the model
    results = model.train(data="datasets17k_yolo/yolo11n-seg.yaml", epochs=300, imgsz=640)