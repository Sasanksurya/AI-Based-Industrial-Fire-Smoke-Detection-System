from ultralytics import YOLO

# Load pretrained YOLO model
model = YOLO("yolov8n.pt")

# Train on fire/smoke dataset
model.train(
    data="data/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="fire_smoke_detector"
)