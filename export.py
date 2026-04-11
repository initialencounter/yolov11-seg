from ultralytics import YOLO
import shutil

if __name__ == '__main__':
  # Load a model
  path = r"C:\Users\29115\yolov8\yolov11-seg\runs\segment_9k\613epoch\weights"
  model = YOLO(path+"/best.pt")  # load a custom trained model

  # Export the model
  model.export(format="onnx", dynamic=True, simplify=True)  # export to ONNX format
  
  # shutil.copyfile(path+"/best.onnx", r"C:\Users\29115\RustroverProjects\Aircraft\packages\wxt\public\segment.onnx")  # 重命名导出的ONNX文件