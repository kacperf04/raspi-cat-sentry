import os
import pathlib
os.environ["YOLO_CONFIG_DIR"] = os.getcwd()
from ultralytics import YOLO

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = BASE_DIR / "data" / "data.yaml"

model = YOLO("yolo26n.pt")
results = model.train(
    data=CONFIG_PATH,
    epochs=60,
    imgsz=640,
    device=0
)

metrics = model.val()
print(metrics)

model.export(format="onnx")