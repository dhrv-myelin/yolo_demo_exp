from ultralytics import YOLO

# Load a pretrained YOLO model (recommended for training)
model = YOLO("./runs/detect/train-2/weights/best.pt")

# model = YOLO("./runs/detect/train/weights/best.pt")

# Train the model using the 'coco8.yaml' dataset for 3 epochs
# results = model.train(
#     data="./data/Cash Detection.v1-augmented.yolo26/data.yaml", epochs=50
# )

# Evaluate the model's performance on the validation set
# results = model.val()

# Perform object detection on an image using the model
results = model.predict(
    source="./data/Vid4.mp4",
    show=True,
    vid_stride=2,  # every other frame
    imgsz=960,
    # conf=0.4,
)

# # Export the model to ONNX format
# success = model.export(format="onnx")
