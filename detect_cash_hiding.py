#!/usr/bin/env python3
"""
detect_cash_hiding.py

Two-model hybrid detector:
  1. Custom YOLO  -> "cash"
  2. COCO YOLOv8n -> "cell phone" and "person"

Triggers an alert when cash overlaps a phone near a person for N consecutive frames.
Saves an annotated output video.
"""

import argparse
import cv2
import math
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ── COCO class indices (YOLOv8n official COCO) ──────────────────────────────
CLASS_PERSON = 0
CLASS_CELL_PHONE = 67

# ── Drawing colors (BGR) ────────────────────────────────────────────────────
COLOR_CASH = (0, 255, 0)  # Green
COLOR_PHONE = (255, 0, 0)  # Blue
COLOR_PERSON = (0, 255, 255)  # Yellow
COLOR_ALERT = (0, 0, 255)  # Red

FONT = cv2.FONT_HERSHEY_SIMPLEX


def parse_args():
    p = argparse.ArgumentParser(description="Detect cash being hidden under a phone.")
    p.add_argument("--source", type=str, required=True, help="Input video path")
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output video path (default: <source>_alert.mp4)",
    )
    p.add_argument(
        "--cash-model",
        type=str,
        default="./../yolo_world/runs/detect/train-2/weights/best.pt",
        help="Path to custom cash detector",
    )
    p.add_argument(
        "--coco-model",
        type=str,
        default="yolov8n.pt",
        help="COCO model for phone/person (auto-downloaded if missing)",
    )
    p.add_argument(
        "--cash-conf", type=float, default=0.3, help="Confidence threshold for cash"
    )
    p.add_argument(
        "--coco-conf",
        type=float,
        default=0.2,
        help="Confidence threshold for COCO objects",
    )
    p.add_argument(
        "--iou-thresh",
        type=float,
        default=0.05,
        help="Min IoU between cash and phone to count as overlap",
    )
    p.add_argument(
        "--dist-thresh",
        type=float,
        default=100.0,
        help="Max pixel distance between phone and person centroids",
    )
    p.add_argument(
        "--alert-frames",
        type=int,
        default=4,
        help="Consecutive overlapping frames required to trigger alert",
    )
    p.add_argument(
        "--show", action="store_true", help="Show live OpenCV window while processing"
    )
    return p.parse_args()


def iou(box_a, box_b):
    """Compute IoU between two boxes in xyxy format."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def centroid(box):
    """Return (cx, cy) for an xyxy box."""
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def dist(a, b):
    """Euclidean distance between two points."""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def draw_box(img, box, color, label, thick=2):
    x1, y1, x2, y2 = map(int, box)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thick)
    (tw, th), _ = cv2.getTextSize(label, FONT, 0.6, 1)
    cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(img, label, (x1 + 2, y1 - 4), FONT, 0.6, (0, 0, 0), 1, cv2.LINE_AA)


def main():
    args = parse_args()
    src_path = Path(args.source)
    out_path = (
        Path(args.output)
        if args.output
        else src_path.with_stem(src_path.stem + "_alert")
    )
    out_path = out_path.with_suffix(".mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load models ─────────────────────────────────────────────────────────
    print(f"[1/4] Loading cash model: {args.cash_model}")
    cash_model = YOLO(args.cash_model)

    print(
        f"[2/4] Loading COCO model: {args.coco_model} (will auto-download if missing)"
    )
    coco_model = YOLO(args.coco_model)

    # ── Open video ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {src_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

    # ── State ───────────────────────────────────────────────────────────────
    overlap_counter = 0
    alert_active = False

    print(f"[3/4] Processing {total} frames  ->  {out_path}")
    frame_idx = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        display = frame.copy()

        # ── Inference ───────────────────────────────────────────────────────
        cash_results = cash_model.predict(frame, verbose=False, conf=args.cash_conf)
        coco_results = coco_model.predict(frame, verbose=False, conf=args.coco_conf)

        # Extract boxes
        cash_boxes = []
        if cash_results and cash_results[0].boxes is not None:
            for b in cash_results[0].boxes:
                cash_boxes.append([float(x) for x in b.xyxy[0]])

        phone_boxes = []
        person_boxes = []
        if coco_results and coco_results[0].boxes is not None:
            for b in coco_results[0].boxes:
                cls = int(b.cls)
                box = [float(x) for x in b.xyxy[0]]
                if cls == CLASS_CELL_PHONE:
                    phone_boxes.append(box)
                elif cls == CLASS_PERSON:
                    person_boxes.append(box)

        # ── Logic: find cash-phone overlap with nearby person ───────────────
        best_overlap = 0.0
        overlapping_now = False

        for cb in cash_boxes:
            for pb in phone_boxes:
                overlap = iou(cb, pb)
                if overlap >= args.iou_thresh:
                    # Check if a person is nearby this phone
                    person_nearby = False
                    p_cent = centroid(pb)
                    for prb in person_boxes:
                        if dist(p_cent, centroid(prb)) <= args.dist_thresh:
                            person_nearby = True
                            break
                    if person_nearby:
                        overlapping_now = True
                        best_overlap = max(best_overlap, overlap)

        # ── Temporal smoothing ──────────────────────────────────────────────
        if overlapping_now:
            overlap_counter += 1
        else:
            overlap_counter = 0

        if overlap_counter >= args.alert_frames:
            alert_active = True
        # Keep alert active until overlap is lost for a while (hysteresis)
        if overlap_counter == 0 and alert_active:
            # Optional: require some cool-off; here we just turn off immediately
            alert_active = False

        # ── Draw ────────────────────────────────────────────────────────────
        for cb in cash_boxes:
            draw_box(display, cb, COLOR_CASH, "cash")
        for pb in phone_boxes:
            draw_box(display, pb, COLOR_PHONE, "phone")
        for prb in person_boxes:
            draw_box(display, prb, COLOR_PERSON, "person")

        # HUD
        hud = f"Frame: {frame_idx}/{total} | Cash: {len(cash_boxes)} | Phone: {len(phone_boxes)} | Person: {len(person_boxes)}"
        cv2.putText(display, hud, (10, 30), FONT, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

        if alert_active:
            # Big red border
            cv2.rectangle(display, (0, 0), (w - 1, h - 1), COLOR_ALERT, 10)
            cv2.putText(
                display,
                "ALERT: CASH HIDDEN",
                (w // 2 - 280, h // 2),
                FONT,
                1.8,
                COLOR_ALERT,
                4,
                cv2.LINE_AA,
            )
            status = "OVERLAP + ALERT"
        elif overlapping_now:
            status = f"OVERLAP ({overlap_counter}/{args.alert_frames})"
            cv2.putText(
                display, status, (10, 65), FONT, 0.7, (0, 165, 255), 2, cv2.LINE_AA
            )
        else:
            status = "monitoring"
            cv2.putText(
                display, status, (10, 65), FONT, 0.7, (200, 200, 200), 1, cv2.LINE_AA
            )

        writer.write(display)

        if args.show:
            cv2.imshow("Cash Hiding Detector", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                print("  Interrupted by user.")
                break

        # Console progress
        if frame_idx % 30 == 0 or frame_idx == total:
            print(f"  Frame {frame_idx:4d}/{total}  {status}")

    # ── Cleanup ─────────────────────────────────────────────────────────────
    cap.release()
    writer.release()
    cv2.destroyAllWindows()
    print(f"[4/4] Done. Output saved to: {out_path.absolute()}")


if __name__ == "__main__":
    main()
