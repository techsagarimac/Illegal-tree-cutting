import cv2
from ultralytics import YOLO

model = YOLO("models/tree_model.pt")

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Could not open webcam")
    exit()

print("🌳 Custom Tree Detection")
print("Press Q to quit")

while True:
    ret, frame = cap.read()

    if not ret:
        print("❌ Failed to read webcam")
        break

    results = model(frame, conf=0.40)

    annotated_frame = results[0].plot()

    cv2.imshow("Forest Monitoring - Tree Detection", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()