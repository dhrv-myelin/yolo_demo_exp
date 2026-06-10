import cv2
import numpy as np
from ultralytics import YOLO

# --------------------------------------------------
# Load model and RTSP stream
# --------------------------------------------------

model = YOLO("best.pt")

RTSP_URL = "rtsp://readusermyelin:USeR2O26R@192.168.2.5:554/stream1"

cap = cv2.VideoCapture(RTSP_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

frame_time = 1.0 / fps

print(f"Stream FPS: {fps}")

ret, first_frame = cap.read()

if not ret:
    raise RuntimeError("Could not read from RTSP stream")

height, width = first_frame.shape[:2]

# --------------------------------------------------
# ROI selection
# --------------------------------------------------

polygon_points = []

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
print(" - Press ESC to quit")

while True:
    display = first_frame.copy()

    for pt in polygon_points:
        cv2.circle(display, pt, 5, (0, 0, 255), -1)

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

    elif key == 27:
        cap.release()
        cv2.destroyAllWindows()
        exit()

cv2.destroyWindow("Draw ROI")

# --------------------------------------------------
# Re-open stream after ROI selection (connection may have dropped while idle)
# --------------------------------------------------

cap.release()

cap = cv2.VideoCapture(RTSP_URL)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

print("Reconnecting to stream...")

if not cap.isOpened():
    raise RuntimeError("Could not reconnect to RTSP stream")

# --------------------------------------------------
# Tracking
# --------------------------------------------------

object_states = {}

event_history = []
MAX_EVENTS = 5

while True:

    ret, frame = cap.read()

    if not ret:
        print("Stream ended or lost connection")
        break

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
    )

    annotated = frame.copy()

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

                    msg = f"Cash item {track_id} removed from monitored area"

                    print(msg)

                    event_history.append(msg)
                    event_history = event_history[-MAX_EVENTS:]

                object_states[track_id] = inside

    annotated = results[0].plot(img=annotated)

    # --------------------------------------------------
    # Event overlay
    # --------------------------------------------------

    if len(event_history) > 0:

        overlay_height = 40 + len(event_history) * 30

        cv2.rectangle(
            annotated,
            (10, 10),
            (550, overlay_height),
            (0, 0, 0),
            -1,
        )

        cv2.putText(
            annotated,
            "Events",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        for idx, event in enumerate(reversed(event_history)):

            cv2.putText(
                annotated,
                event,
                (20, 70 + idx * 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    # --------------------------------------------------
    # Output
    # --------------------------------------------------

    cv2.imshow("Cash Tracking", annotated)

    if cv2.waitKey(int(frame_time * 1000)) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

print("Done.")
