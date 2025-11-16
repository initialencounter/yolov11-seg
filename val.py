from ultralytics import YOLO

if __name__ == '__main__':
  # Load a model
  # model = YOLO("yolo11n-seg.pt")  # load an official model
  model = YOLO(r"C:\Users\29115\yolov8\yolov11-seg\runs\segment\train\weights\best.pt")  # load a custom model

  # Validate the model
  metrics = model.val()  # no arguments needed, dataset and settings remembered
  metrics.box.map  # map50-95(B)
  metrics.box.map50  # map50(B)
  metrics.box.map75  # map75(B)
  metrics.box.maps  # a list containing mAP50-95(B) for each category
  metrics.seg.map  # map50-95(M)
  metrics.seg.map50  # map50(M)
  metrics.seg.map75  # map75(M)
  metrics.seg.maps  # a list containing mAP50-95(M) for each category