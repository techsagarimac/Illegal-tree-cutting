from ultralytics import YOLO

model = YOLO("models/tree_model.pt")

results = model.predict(
    source="dataset/images/val",
    save=True,
    conf=0.40
)

print("Custom tree model test completed.")