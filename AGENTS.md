# YOLO Cash Detection — Agent Notes

## Project Goal
Detect cash being hidden under phones in convenience store CCTV footage.  
Primary test videos: `data/Vid2.mp4`, `data/Vid4.mp4`.

## Current Stack
- **Framework**: Ultralytics YOLO (8.4.60)
- **Runtime**: Python 3.13, PyTorch 2.12, CUDA 12.3 (RTX 3070 Laptop)
- **Dependency Manager**: `uv` / `.venv`
- **Models**: YOLO26 (custom cash), YOLOv8s-worldv2 (experimental), COCO YOLOv8n (phone/person)
- **Dataset**: Roboflow "Cash Detection v1-augmented" (single class: `cash`)

## Current Status
- Custom `cash` model trained (YOLO26m, 50 epochs, 640px). Validation mAP50=0.889.
- **Problem**: Model generalizes poorly to test videos due to domain shift:
  - Overhead camera angle (training data is mostly flat/frontal).
  - Severe occlusion by hands and phones.
  - Motion blur during fast hand movements.
- **Latest attempt**: Hybrid two-model script (`detect_cash_hiding.py`) combining custom cash detector + COCO phone/person detector with IoU overlap logic.
- **Current blocker**: COCO model cannot reliably detect phones from overhead angle (out-of-distribution for COCO).

## Key Paths
- `data/Cash Detection.v1-augmented.yolo26/` — training data
- `runs/detect/train-2/weights/best.pt` — current best cash model
- `detect_cash_hiding.py` — latest inference script (hybrid approach)

## User Preferences
- **Self-run**: User runs scripts themselves; do not execute scripts on their behalf.
- **No live demos unless asked**: Provide commands, don't run them.
- **Prefer working solutions over perfect ones**: Quick iteration is valued.

## Potential Next Steps
1. **YOLO-World Test** (5 min): Try zero-shot `yolov8s-worldv2.pt` with text prompts like "cash", "paper money" to see if foundation models handle the domain shift.
2. **Hard Negative Mining** (medium): Extract difficult frames from `Vid2.mp4`/`Vid4.mp4`, annotate occluded/moving cash, retrain custom model.
3. **ROI-Based Logic** (robust): Define a Region-of-Interest on the counter where the phone sits. Track cash entering that zone and disappearing — no phone detection required.
4. **Action Recognition** (hard): Train a temporal model to classify "hiding cash" as an action clip rather than frame-by-frame object detection.
