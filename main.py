import cv2

from src.utils import frame_splitter


def main():
    raw_data_path = "raw_data/"
    output_path = "dataset_ready/"

    frame_splitter.split_frames(raw_data_path, output_path, truncate_output_dir=True)


if __name__ == "__main__":
    main()