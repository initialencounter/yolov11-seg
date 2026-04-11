from ultralytics import YOLO
yolo_classes = {0: '9', 1: '9A', 2: 'BTY', 3: 'CAO'}
if __name__ == '__main__':
    # Load a model
    model = YOLO(r"C:\Users\29115\yolov8\yolov11-seg\runs\segment_17k\yolo26n-seg-17k-42epochs\weights\best.pt")  # load a custom model

    # Predict with the model
    results = model(r"C:\Users\29115\yolov8\yolov11-seg\datasets17k_labelme\train\img_016835.png")  # predict on an image

    for result in results:
        boxes = result.boxes  # Boxes object for bounding box outputs
        masks = result.masks  # Masks object for segmentation masks outputs
        keypoints = result.keypoints  # Keypoints object for pose outputs
        probs = result.probs  # Probs object for classification outputs
        obb = result.obb  # Oriented boxes object for OBB outputs
        result.show()  # display to screen
        result.save(filename="result.jpg")  # save to disk
