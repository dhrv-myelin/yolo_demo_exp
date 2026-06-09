import argparse
import json
import logging

import cv2
import numpy as np
from ultralytics import YOLO

# --------------------------------------------------
# Arguments
# --------------------------------------------------

parser = argparse.ArgumentParser()

parser.add_argument(
    "--show",
    action="store_true",
    help="Display video instead of saving to MP4",
)

parser.add_argument(
    "--video",
    required=True,
    help="Path to input video",
)

parser.add_argument(
    "--camera-name",
    required=True,
    help="Camera identifier used in logs/output",
)

parser.add_argument(
    "--roi",
    help="ROI polygon as x,y;x,y;x,y...",
)

args = parser.parse_args()

VIDEO_PATH = args.video
CAMERA_NAME = args.camera_name

# --------------------------------------------------
# Logging
# --------------------------------------------------

logging.basicConfig(
    filename="cash_alerts.log",
    level=logging.INFO,
    format="%(message)s",
)

logger = logging.getLogger(__name__)


def log_alert(camera_name, video_timestamp, message):
    logger.info(
        json.dumps(
            {
                "camera": camera_name,
                "timestamp": video_timestamp,
                "alert": message,
            }
        )
    )


# --------------------------------------------------
# Load model and video
# --------------------------------------------------

model = YOLO("./../yolo_world/runs/detect/train-2/weights/best.pt")

cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)

if fps <= 0:
    fps = 30

frame_time = 1.0 / 30
# frame_time = 1.0 / fps

print(f"Video FPS: {fps}")

ret, first_frame = cap.read()

if not ret:
    raise RuntimeError(f"Could not read video: {VIDEO_PATH}")

height, width = first_frame.shape[:2]

writer = None

output_path = f"{CAMERA_NAME}_tracking.mp4"

if not args.show:
    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    print(f"Saving output to {output_path}")

# --------------------------------------------------
# ROI Selection
# --------------------------------------------------

roi_polygon = None

if args.roi:

    polygon_points = [
        [int(coord) for coord in point.split(",")] for point in args.roi.split(";")
    ]

    roi_polygon = np.array(
        polygon_points,
        dtype=np.int32,
    )

    print("\nUsing ROI from command line:")
    print(roi_polygon.tolist())

else:

    polygon_points = []

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

        # for pt in polygon_points:
        #     cv2.circle(display, pt, 5, (0, 0, 255), -1)

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

            if writer is not None:
                writer.release()

            cv2.destroyAllWindows()
            exit()

    cv2.destroyWindow("Draw ROI")

# --------------------------------------------------
# Tracking
# --------------------------------------------------

cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

object_states = {}

event_history = []
MAX_EVENTS = 5

frame_idx = 0

while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_idx += 1

    current_seconds = frame_idx / fps

    hours = int(current_seconds // 3600)
    minutes = int((current_seconds % 3600) // 60)
    seconds = int(current_seconds % 60)

    timestamp = f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    results = model.track(
        frame,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
    )

    annotated = frame.copy()

    boxes = results[0].boxes

    if boxes.id is not None:

        ids = boxes.id.cpu().numpy().astype(int)
        coords = boxes.xyxy.cpu().numpy()

        for track_id, box in zip(ids, coords):

            x1, y1, x2, y2 = map(int, box)

            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # cv2.circle(
            #     annotated,
            #     (cx, cy),
            #     4,
            #     (0, 0, 255),
            #     -1,
            # )

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

                    msg = "suspicious activity detected"

                    print(f"[{timestamp}] {msg}")

                    log_alert(
                        CAMERA_NAME,
                        timestamp,
                        msg,
                    )

                    event_history.append(f"[{timestamp}] {msg}")

                    event_history = event_history[-MAX_EVENTS:]

                object_states[track_id] = inside

    # annotated = results[0].plot(img=annotated)

    # --------------------------------------------------
    # Timestamp Overlay
    # --------------------------------------------------

    cv2.putText(
        annotated,
        timestamp,
        (20, height - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # --------------------------------------------------
    # Event Overlay
    # --------------------------------------------------

    if len(event_history) > 0:

        overlay_height = 40 + len(event_history) * 30

        cv2.rectangle(
            annotated,
            (10, 10),
            (700, overlay_height),
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

    if args.show:

        cv2.imshow("Cash Tracking", annotated)

        if cv2.waitKey(int(frame_time * 1000)) & 0xFF == ord("q"):
            break

    else:

        writer.write(annotated)

cap.release()

if writer is not None:
    writer.release()

cv2.destroyAllWindows()

print("Done.")
print(f"Output video: {output_path}")
print("Alert log: cash_alerts.log")
