import os
import cv2
import numpy as np

BLUR_THRESHOLD = 100
DARK_THRESHOLD = 30
DUPLICATE_THRESHOLD = 10
VALID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")
CATEGORIES = ["porch_cat_day", "porch_cat_night"]

def split_frames(raw_videos_path: str, output_path: str, truncate_output_dir: bool, target_total_images: int = 400,
                target_day_percentage: float = 0.6) -> None:
    """
    Extracts frames from raw videos and saves them in a specified output directory
    :param raw_videos_path -- path to the directory containing raw videos
    :param output_path -- path to the output directory
    :param truncate_output_dir -- whether to truncate the output directory before writing new files
    :param target_total_images -- total number of frames to extract from all videos
    :param target_day_percentage -- percentage of frames to extract from videos during the day (frames from night videos are a complementary subset of frames from day videos)
    :return -- None
    """
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print("Created output directory")

    if truncate_output_dir:
        for file in os.listdir(output_path):
            os.remove(os.path.join(output_path, file))

    frame_count_dict = count_total_frames(raw_videos_path)

    targets = {
        "porch_cat_day": int(target_day_percentage * target_total_images),
        "porch_cat_night": int((1 - target_day_percentage) * target_total_images)
    }
    total_stats = {"processed": 0, "saved": 0}

    for category in CATEGORIES:
        num_videos = frame_count_dict[category]["number_of_videos"]
        target_for_category = targets.get(category, 0)

        if num_videos == 0 or target_for_category == 0: continue

        base_quota, remainder = divmod(target_for_category, num_videos)

        print(f"--- Extracting {target_for_category} frames for {category} videos ---")

        processed, saved = extract_frames(
            raw_videos_path, output_path, category,
            frame_count_dict, base_quota, remainder
        )

        total_stats["processed"] += processed
        total_stats["saved"] += saved

    print(f"{"="*30}")
    print(f"DONE.")
    print(f"Total processed: {total_stats['processed']}")
    print(f"Total saved: {total_stats['saved']} (Target: {target_total_images})")
    print(f"{"="*30}")


def extract_frames(video_path: str, output_path: str, category: str,
                   frame_count_dict: dict, base_quota: int, remainder: int) -> tuple[int, int]:
    """
    Extracts frames from a video and saves them in a specified output directory
    :param video_path -- path to the directory containing raw videos
    :param output_path -- path to the output directory
    :param category -- type of video to extract frames from (porch_cat, empty_porch, other)
    :param frame_count_dict -- dictionary containing information about the total number of frames in each video type
    :param base_quota -- number of frames to extract from each video
    :param remainder -- number of frames to extract from videos with fewer frames than the target
    :return -- a tuple containing the total number of processed frames and the total number of saved frames
    """
    total_processed = 0
    total_saved = 0
    video_files = list(frame_count_dict[category]["videos"].keys())

    for idx, file in enumerate(video_files):
        current_video_target = base_quota + (1 if idx < remainder else 0)

        if current_video_target == 0: continue

        full_path = os.path.join(video_path, file)
        total_frames_in_video = frame_count_dict[category]["videos"][file]

        print(f"Processing {file} (Target: {current_video_target})")

        cap = cv2.VideoCapture(full_path)
        if not cap.isOpened():
            print(f"Failed to open {full_path}")
            continue

        saved_count = 0
        last_saved_img = None
        current_frame_idx = 0

        while saved_count < current_video_target:
            remaining_needed = current_video_target - saved_count
            remaining_video = total_frames_in_video - current_frame_idx

            if remaining_video <= 0:
                print(f"Reached end of video {file}")
                break

            step = max(1, remaining_video // remaining_needed)
            target_idx = current_frame_idx + step

            if step > 5:
                cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
                ret, image = cap.read()
                current_frame_idx = target_idx
            else:
                ret = False
                for _ in range(step):
                    ret, image = cap.read()
                    current_frame_idx += 1

            if not ret: break

            total_processed += 1

            if is_image_representative(last_saved_img, image):
                save_name = f"{category}_{file.split('.')[0]}_{saved_count}.jpg"
                cv2.imwrite(os.path.join(output_path, save_name), image)
                last_saved_img = image
                saved_count += 1
                total_saved += 1
            else:
                pass

        cap.release()

    return total_processed, total_saved


def count_total_frames(video_path: str) -> dict[str, dict[str, int | dict[str, int]]]:
    """
    Counts the total number of frames in all input videos
    :param video_path -- path to the directory containing raw videos
    :return -- a tuple ([total_porch_with_cat_frames], [total_empty_porch_frames], [total_other_frames])])
    """
    frame_count_dict = {cat: {"total_frames": 0, "number_of_videos": 0, "videos": {}} for cat in CATEGORIES}

    all_files = [ f for f in os.listdir(video_path) if f.lower().endswith(VALID_EXTENSIONS) ]

    for file in all_files:
        category = None

        for cat in CATEGORIES:
            if file.startswith(cat):
                category = cat
                break

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


def is_dark_or_empty(image: cv2.Mat) -> np.ndarray:
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