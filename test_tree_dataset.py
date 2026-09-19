from ultralytics import YOLO

model = YOLO("yolo11n.pt")

results = model.predict(
    source="dataset/images/val",
    save=True,
    conf=0.25
)

print("Tree dataset images tested successfully.")