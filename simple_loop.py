import cv2
import time
from ultralytics import YOLO

model = YOLO("./../yolo_world/runs/detect/train-2/weights/best.pt")

cap = cv2.VideoCapture("./data/Vid2.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)
frame_time = 1.0 / 10

while True:
    start = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    annotated = results[0].plot()

    cv2.imshow("YOLO", annotated)

    elapsed = time.time() - start
    remaining = max(0, frame_time - elapsed)

    if cv2.waitKey(int(remaining * 1000)) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
