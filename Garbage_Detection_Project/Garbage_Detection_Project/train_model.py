from ultralytics import YOLO

# Load small YOLO model (fastest)
model = YOLO("yolov8n.pt")

model.train(
    data="data.yaml",
    epochs=10,      # reduce epochs (50 → 10)
    imgsz=320,      # smaller image size (640 → 320)
    batch=8         # small batch for laptop
)