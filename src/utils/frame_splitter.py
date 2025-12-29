import os
from typing import Tuple, Dict
import cv2
import numpy as np

BLUR_THRESHOLD = 100
DARK_THRESHOLD = 15
DUPLICATE_THRESHOLD = 5
VALID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

def split_frames(raw_videos_path: str, output_path: str, truncate_output_dir: bool, target_total_images: int = 200,
                 target_porch_with_cat_percentage: float = 0.7, target_empty_porch_percentage: float = 0.3) -> None:
    """
    Extracts frames from raw videos and saves them in a specified output directory
    :param raw_videos_path -- path to the directory containing raw videos
    :param output_path -- path to the output directory
    :param truncate_output_dir -- whether to truncate the output directory before writing new files
    :param target_total_images -- total number of frames to extract from all videos
    :param target_porch_with_cat_percentage -- percentage of frames to extract from porch videos with a cat
    :param target_empty_porch_percentage -- percentage of frames to extract from empty porch videos
    :return -- None
    """
    total_saved_frames = 0
    total_processed_frames = 0

    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print("Created output directory")

    if truncate_output_dir:
        for file in os.listdir(output_path):
            os.remove(os.path.join(output_path, file))

    frame_count_dict = count_total_frames(raw_videos_path)

    frames_to_extract_porch_cat = int(target_total_images * target_porch_with_cat_percentage)
    frames_to_extract_empty_porch = int(target_total_images * target_empty_porch_percentage)

    frame_info_porch_cat = extract_frames(raw_videos_path, output_path, "porch_cat", frame_count_dict, frames_to_extract_porch_cat)
    frame_info_empty_porch = extract_frames(raw_videos_path, output_path, "empty_porch", frame_count_dict, frames_to_extract_empty_porch)

    total_processed_frames += frame_info_porch_cat[0] + frame_info_empty_porch[0]
    total_saved_frames += frame_info_porch_cat[1] + frame_info_empty_porch[1]
    total_video_count = frame_count_dict["porch_cat"]["number_of_videos"] + frame_count_dict["empty_porch"]["number_of_videos"]

    print(f"\nTotal number of processed frames: {total_processed_frames}")
    print(f"Total number of saved frames: {total_saved_frames}")
    print(f"Total number of videos processed: {total_video_count}")


def extract_frames(video_path: str, output_path: str, video_type: str, frame_count_dict: dict, frames_to_extract) -> Tuple[int, int]:
    """
    Extracts frames from a video and saves them in a specified output directory
    :param video_path -- path to the directory containing raw videos
    :param output_path -- path to the output directory
    :param video_type -- type of video to extract frames from (porch_cat, empty_porch, other)
    :param frame_count_dict -- dictionary containing information about the total number of frames in each video type
    :param frames_to_extract -- number of frames to extract from each video
    :return -- a tuple containing the total number of processed frames and the total number of saved frames
    """
    total_saved_frames = 0
    total_processed_frames = 0

    total_available_frames = frame_count_dict[video_type]["total_frames"]
    if total_available_frames == 0: return 0, 0

    iterator_step = total_available_frames // frames_to_extract
    files = list(frame_count_dict[video_type]["videos"].keys())

    global_frame_count = 0
    last_saved_image = None

    for file in files:
        full_path = os.path.join(video_path, file)
        print(f"-- Processing {file} --")

        cap = cv2.VideoCapture(full_path)
        if not cap.isOpened():
            print(f"Error opening video file: {full_path}")
            continue

        while True:
            ret, image = cap.read()
            if not ret: break

            global_frame_count += 1
            total_processed_frames += 1

            if global_frame_count % iterator_step == 0:
                if not is_image_representative(last_saved_image, image): continue

                save_name = f"{video_type}_{total_saved_frames}.jpg"
                cv2.imwrite(os.path.join(output_path, save_name), image)
                last_saved_image = image
                total_saved_frames += 1

                if total_saved_frames >= frames_to_extract:
                    print(f"Target reached for {video_type}")
                    return total_processed_frames, total_saved_frames

        cap.release()

    return total_processed_frames, total_saved_frames


def count_total_frames(video_path: str) -> Dict[str, Dict[str, int | Dict[str, int]]]:
    """
    Counts the total number of frames in all input videos
    :param video_path -- path to the directory containing raw videos
    :return -- a tuple ([total_porch_with_cat_frames], [total_empty_porch_frames], [total_other_frames])])
    """
    frame_count_dict = {
        "porch_cat": {
            "total_frames": 0,
            "number_of_videos": 0,
            "videos": {}
        },
        "empty_porch": {
            "total_frames": 0,
            "number_of_videos": 0,
            "videos": {}
        }
    }

    all_files = [ f for f in os.listdir(video_path) if f.lower().endswith(VALID_EXTENSIONS) ]

    for file in all_files:
        category = None
        if file.startswith("porch_cat"): category = "porch_cat"
        elif file.startswith("empty_porch"): category = "empty_porch"

        if category:
            cap = cv2.VideoCapture(os.path.join(video_path, file))
            if cap.isOpened():
                count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frame_count_dict[category]["videos"][file] = count
                frame_count_dict[category]["total_frames"] += count
                frame_count_dict[category]["number_of_videos"] += 1
            cap.release()

    return frame_count_dict


def is_blurry(image: cv2.Mat) -> bool:
    """
    Heuristic to determine whether an image is blurry or not
    :param image -- input image
    :return -- True if the image is blurry, False otherwise
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray_image, cv2.CV_64F).var()
    return score < BLUR_THRESHOLD


def is_dark_or_empty(image: cv2.Mat):
    """
    Heuristic to determine whether an image is dark or empty
    :param image -- input image
    :return -- True if the image is dark or empty, False otherwise
    """
    mean, std_dev = cv2.meanStdDev(image)
    return std_dev[0][0] < DARK_THRESHOLD


def is_duplicate(prev_image: cv2.Mat, curr_image: cv2.Mat) -> bool:
    """
    Heuristic to determine whether two images are duplicates or not based on MSE of their grayscale representations
    :param prev_image -- previous image
    :param curr_image -- current image
    :return -- True if the images are duplicates, False otherwise
    """
    if prev_image is None: return False

    gray_curr = cv2.cvtColor(curr_image, cv2.COLOR_BGR2GRAY)
    gray_prev = cv2.cvtColor(prev_image, cv2.COLOR_BGR2GRAY)
    err = np.sum((gray_curr.astype("float") - gray_prev.astype("float")) ** 2)
    err /= float(gray_curr.shape[0] * gray_curr.shape[1])
    return err < DUPLICATE_THRESHOLD

def is_image_representative(prev_image: cv2.Mat, curr_image: cv2.Mat) -> bool:
    """
    Based on the heuristics above, determines whether an image is representative in terms of quality or not
    :param prev_image -- previous image
    :param curr_image -- current image
    :return -- True if the image is representative, False otherwise
    """
    if curr_image is None: return False
    if is_dark_or_empty(curr_image): return False
    if prev_image is None:
        return not is_blurry(curr_image)
    return not is_duplicate(prev_image, curr_image) and not is_blurry(curr_image)