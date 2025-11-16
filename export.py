from ultralytics import YOLO

if __name__ == '__main__':
  # Load a model
  model = YOLO(r"C:\Users\29115\yolov8\yolov11-seg\runs\segment\train\weights\best.pt")  # load a custom trained model

  # Export the model
  model.export(format="onnx")