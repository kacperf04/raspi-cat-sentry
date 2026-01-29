import os
from pathlib import Path
import step01_frame_splitter as s1
import step02_auto_labeler as s2
import step03_train_val_split as s3

BASE_DIR = Path(__file__).resolve().parent.parent
INPUT_DIR = BASE_DIR / "data" / "raw_data"
TMP_DIR = BASE_DIR / "data" / "tmp_images"
FINAL_DATASET_DIR = BASE_DIR / "data" / "training_data"

def run_pipeline():
    prepare_directories()
    s1.split_frames(str(INPUT_DIR), str(TMP_DIR), truncate_output_dir=True)
    s2.label_images(TMP_DIR, FINAL_DATASET_DIR)
    s3.train_val_split(FINAL_DATASET_DIR)


def prepare_directories():
    images_dir = FINAL_DATASET_DIR / "images"
    labels_dir = FINAL_DATASET_DIR / "labels"

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(images_dir / "train", exist_ok=True)
    os.makedirs(images_dir / "val", exist_ok=True)
    os.makedirs(labels_dir / "train", exist_ok=True)
    os.makedirs(labels_dir / "val", exist_ok=True)

    for file in os.listdir(TMP_DIR):
        os.remove(TMP_DIR / file)

    for element in os.listdir(images_dir):
        if os.path.isfile(images_dir / element):
            os.remove(images_dir / element)
        else:
            for file in os.listdir(images_dir / element):
                os.remove(images_dir / element / file)

    for element in os.listdir(labels_dir):
        if os.path.isfile(labels_dir / element):
            os.remove(labels_dir / element)
        else:
            for file in os.listdir(labels_dir / element):
                os.remove(labels_dir / element / file)


if __name__ == "__main__":
    run_pipeline()