from ultralytics import YOLO

# Load a model
model = YOLO("yolo26n-seg.pt")  # load an official model


#
# results = model.train(
#     data="./data/cash detection.v1-augmented.yolo26/data.yaml", epochs=100
# )

# results = model.predict(source="./data/Vid2.mp4", show=True)
results = model.track(source="./data/drop1.mp4", show=True)
