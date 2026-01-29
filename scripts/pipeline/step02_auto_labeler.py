import os
import pathlib
import shutil
from ultralytics import YOLO

MODEL_NAME = "yolo26x.pt"
LABEL_CLASS = 15 # corresponds to cat
LABEL_NAME = "cat"


def label_images(tmp_dir: pathlib.Path, destination_dir: pathlib.Path) -> None:
    images_dir = destination_dir / "images"
    labels_dir = destination_dir / "labels"
    model_path = pathlib.Path(__file__).parent / MODEL_NAME

    model = YOLO(str(model_path))
    results = model(str(tmp_dir), stream=True, conf=0.4)

    for result in results:
        found_cat = False
        lines_to_write = []

        for box in result.boxes:
            if int(box.cls[0]) == LABEL_CLASS:
                x, y, w, h = box.xywhn[0].tolist()
                lines_to_write.append(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                found_cat = True

        if found_cat:
            image_file_name = pathlib.Path(result.path).name
            label_file_name = pathlib.Path(result.path).stem + ".txt"

            with open(labels_dir / label_file_name, "w") as f:
                f.writelines(lines_to_write)

            shutil.copy(result.path, images_dir / image_file_name)

    with open(labels_dir / "classes.txt", "w") as f:
        f.write(LABEL_NAME)

    print(f"Auto-labeling completed. Labels saved to {labels_dir}")