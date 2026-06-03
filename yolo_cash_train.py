from ultralytics import YOLO

# Load a pretrained YOLO model (recommended for training)
model = YOLO("yolo26m.pt")

# model = YOLO("./runs/detect/train/weights/best.pt")

# Train the model using the 'coco8.yaml' dataset for 3 epochs
results = model.train(
    data="./data/Cash Detection.v1-augmented.yolo26/data.yaml", epochs=50
)

# Evaluate the model's performance on the validation set
results = model.val()

# Perform object detection on an image using the model
# results = model.track(
#     source="./data/Vid3.mp4",
#     show=True,
#     conf=0.25,
# )

# # Export the model to ONNX format
# success = model.export(format="onnx")
