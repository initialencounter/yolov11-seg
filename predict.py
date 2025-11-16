from ultralytics import YOLO

if __name__ == '__main__':
    # Load a model
    model = YOLO(r"C:\Users\29115\yolov8\yolov11-seg\runs\segment\train\weights\best.pt")  # load a custom model

    # Predict with the model
    results = model(r"datasets\images\val\1716992252.5054588.png")  # predict on an image

    # Access the results
    for result in results:
        xy = result.masks.xy  # mask in polygon format
        xyn = result.masks.xyn  # normalized
        masks = result.masks.data  # mask in matrix format (num_objects x H x W)