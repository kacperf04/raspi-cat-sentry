import os
import subprocess

from ultralytics import YOLO

model = YOLO("yolo11x.pt")

input_folder = "../../dataset_ready"
images_folder = "../../training_data/images"
labels_folder = "../../training_data/labels"
os.makedirs(images_folder, exist_ok=True)
os.makedirs(labels_folder, exist_ok=True)

for file in os.listdir(images_folder):
    os.remove(os.path.join(images_folder, file))

for file in os.listdir(labels_folder):
    os.remove(os.path.join(labels_folder, file))

print("Starting auto-labeling...")
results = model(input_folder, stream=True, conf=0.4)

for result in results:
    file_name = os.path.basename(result.path)
    txt_name = os.path.splitext(file_name)[0] + ".txt"
    label_save_path = os.path.join(labels_folder, txt_name)
    image_save_path = os.path.join(images_folder, file_name)

    with open(label_save_path, "w") as f:
        for box in result.boxes:
            if int(box.cls[0]) == 15:
                x, y, w, h = box.xywhn[0].tolist()
                f.write(f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
                subprocess.run(["cp", os.path.join(input_folder, file_name), os.path.join(images_folder, file_name)])

with open(os.path.join(labels_folder, "classes.txt"), "w") as f:
    f.write("label_cat")

print(f"Auto-labeling completed. Labels saved to {labels_folder}")