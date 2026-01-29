import os
import pathlib
import shutil

from sklearn.model_selection import train_test_split


def train_val_split(input_dir: pathlib.Path) -> None:
    images = get_images(input_dir / "images")

    train_images, val_images = train_test_split(images, test_size=0.2, random_state=42)
    print(f"Train images: {len(train_images)}")
    print(f"Val images: {len(val_images)}")

    copy_images(True, input_dir, train_images)
    copy_images(False, input_dir, val_images)

    shutil.copy(input_dir / "labels" / "classes.txt", input_dir / "labels" / "train" / "classes.txt")
    shutil.copy(input_dir / "labels" / "classes.txt", input_dir / "labels" / "val" / "classes.txt")

    clean_up(input_dir)


def copy_images(train: bool, input_dir: pathlib.Path, images: list[str]) -> None:
    output_images_dir = input_dir / "images" / ("train" if train else "val")
    output_labels_dir = input_dir / "labels" / ("train" if train else "val")

    for image in images:
        path_to_image = input_dir / "images" / image
        path_to_label = input_dir / "labels" / image.replace(".jpg", ".txt")
        shutil.copy(path_to_image, output_images_dir)
        shutil.copy(path_to_label, output_labels_dir)


def get_images(images_dir: pathlib.PurePath) -> list[str]:
    images = []
    for file in os.listdir(images_dir):
        path = images_dir / file
        if os.path.isfile(path): images.append(file)
    return images


def clean_up(input_dir: pathlib.Path) -> None:
    images_dir = input_dir / "images"
    labels_dir = input_dir / "labels"
    for element in os.listdir(images_dir):
        if os.path.isfile(images_dir / element): os.remove(images_dir / element)
    for element in os.listdir(labels_dir):
        if os.path.isfile(labels_dir / element): os.remove(labels_dir / element)


'''subprocess.run(["cp", os.path.join(labels_folder, "classes.txt"), os.path.join(output_labels_folder, "val", "classes.txt")])
subprocess.run(["cp", os.path.join(labels_folder, "classes.txt"), os.path.join(output_labels_folder, "train", "classes.txt")])'''


