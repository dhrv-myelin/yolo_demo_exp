# polygon_points = [[537, 405], [763, 421], [740, 220], [556, 206], [537, 400]]
# polygon_points = [[566, 408], [745, 410], [740, 175], [558, 165], [558, 399]]


import cv2
import numpy as np
from ultralytics import YOLO

# --------------------------------------------------
# Load model and video
# --------------------------------------------------

model = YOLO("./../yolo_world/runs/detect/train-2/weights/best.pt")

cap = cv2.VideoCapture("./data/Vid2.mp4")

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

frame_time = 1.0 / 10
# frame_time = 1.0 / fps

print(f"Video FPS: {fps}")

ret, first_frame = cap.read()

if not ret:
    raise RuntimeError("Could not read video")

# --------------------------------------------------
# ROI selection
# --------------------------------------------------

# polygon_points = []
# polygon_points = [[537, 405], [763, 421], [740, 220], [556, 206], [537, 400]]
# polygon_points = [[566, 408], [745, 410], [740, 175], [558, 165], [558, 399]]
roi_polygon = None


def mouse_callback(event, x, y, flags, param):
    global polygon_points

    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append((x, y))


cv2.namedWindow("Draw ROI")
cv2.setMouseCallback("Draw ROI", mouse_callback)

print("Instructions:")
print(" - Left click to add ROI points")
print(" - Press 'c' to close polygon")
print(" - Press 's' to start tracking")

while True:
    display = first_frame.copy()

    # draw points
    for pt in polygon_points:
        cv2.circle(display, pt, 5, (0, 0, 255), -1)

    # draw lines between points
    if len(polygon_points) > 1:
        cv2.polylines(
            display,
            [np.array(polygon_points, dtype=np.int32)],
            False,
            (0, 255, 0),
            2,
        )

    cv2.imshow("Draw ROI", display)

    key = cv2.waitKey(20) & 0xFF

    if key == ord("c"):

        if len(polygon_points) >= 3:
            roi_polygon = np.array(
                polygon_points,
                dtype=np.int32,
            )

            print("\nROI Polygon:")
            print(roi_polygon.tolist())

            break

    elif key == 27:  # ESC
        cap.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyWindow("Draw ROI")

# --------------------------------------------------
# Tracking
# --------------------------------------------------

# rewind video
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

object_states = {}

while True:

    ret, frame = cap.read()

    if not ret:
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
    )

    annotated = frame.copy()

    # draw ROI
    cv2.polylines(
        annotated,
        [roi_polygon],
        isClosed=True,
        color=(0, 255, 0),
        thickness=2,
    )

    boxes = results[0].boxes

    if boxes.id is not None:

        ids = boxes.id.cpu().numpy().astype(int)
        coords = boxes.xyxy.cpu().numpy()

        for track_id, box in zip(ids, coords):

            x1, y1, x2, y2 = map(int, box)

            # center point
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            cv2.circle(
                annotated,
                (cx, cy),
                4,
                (0, 0, 255),
                -1,
            )

            inside = (
                cv2.pointPolygonTest(
                    roi_polygon,
                    (float(cx), float(cy)),
                    False,
                )
                >= 0
            )

            previous = object_states.get(track_id)

            if previous is None:
                object_states[track_id] = inside

            else:

                if previous and not inside:
                    print(f"Cash ID {track_id} LEFT ROI")

                elif not previous and inside:
                    print(f"Cash ID {track_id} ENTERED ROI")

                object_states[track_id] = inside

    annotated = results[0].plot(img=annotated)

    cv2.imshow("Cash Tracking", annotated)

    if cv2.waitKey(int(frame_time * 1000)) & 0xFF == ord("q"):
        break
    # if cv2.waitKey(1) & 0xFF == ord("q"):
    #     break

cap.release()
cv2.destroyAllWindows()
