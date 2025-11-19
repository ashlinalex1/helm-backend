from ultralytics import YOLO

print("Converting best1.pt...")
model1 = YOLO("best1.pt")
model1.export(format="onnx")

print("Converting best2.pt...")
model2 = YOLO("best2.pt")
model2.export(format="onnx")

print("DONE! Files saved inside /runs/weights/")
