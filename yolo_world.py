from ultralytics import YOLOWorld

# Load a pretrained YOLOv8s-worldv2 model
model = YOLOWorld("yolov8s-worldv2.pt")

# Define custom classes
model.set_classes(
    [
        "cashier",
    ]
)
#
#
# results = model.train(
#     data="./data/cash detection.v1-augmented.yolo26/data.yaml", epochs=100
# )

# results = model.predict(source="./data/Vid2.mp4", show=True)
results = model.track(source="./data/Vid4.mp4", show=True)
