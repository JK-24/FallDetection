import os
import json
import cv2
import numpy as np

# ==============================
# CONFIG
# ==============================

DEPTH_ROOT = r"PATH_TO_DEPTH_ROOT"
STAGE1_JSON_DIR = r"PATH_TO_STAGE1_JSON"
STAGE2_JSON_DIR = r"PATH_TO_STAGE2_JSON"
STAGE3_JSON_DIR = r"PATH_TO_STAGE3_JSON"
STAGE4_OUTPUT_DIR = r"PATH_TO_STAGE4_OUTPUT"

WINDOW_SIZE = 5

os.makedirs(STAGE4_OUTPUT_DIR, exist_ok=True)


# ==============================
# HELPERS
# ==============================

def get_torso_centroid(keypoints, image_height):
    torso_ids = [5, 6, 11, 12]

    ys = []
    for idx in torso_ids:
        y_norm = keypoints[idx][1]
        ys.append(y_norm * image_height)

    return np.median(ys)


def compute_depth_metrics(depth_sequence, selected_index):
    """
    depth_sequence: list of torso depths for clip_indices order
    selected_index: index inside clip_indices (0–31)
    """

    # Determine reference window
    start = max(0, selected_index - WINDOW_SIZE)
    reference_indices = list(range(start, selected_index))

    # Ensure minimum reference frames
    if len(reference_indices) < 3:
        reference_indices = list(range(0, selected_index))

    if len(reference_indices) == 0:
        depth_drop = 0.0
    else:
        d_ref = np.mean([depth_sequence[i] for i in reference_indices])
        depth_drop = d_ref - depth_sequence[selected_index]

    return depth_drop


# ==============================
# MAIN STAGE 4 FUNCTION
# ==============================

def process_video(stage1_path, stage2_path, stage3_path, category, activity, collection):

    with open(stage1_path, "r") as f:
        stage1_data = json.load(f)

    with open(stage2_path, "r") as f:
        stage2_data = json.load(f)

    with open(stage3_path, "r") as f:
        stage3_data = json.load(f)

    clip_indices = stage1_data["clip_indices"]

    keyframe_indices = stage3_data["keyframe_indices"]
    attention_weights = stage3_data["attention_weights"]

    # Select highest attention frame
    k_star_idx = int(np.argmax(attention_weights))
    selected_original_idx = keyframe_indices[k_star_idx]

    # Map original frame index → depth frame index
    depth_frame_index = clip_indices.index(selected_original_idx)

    # Prepare depth sequence
    depth_sequence = []

    # Build lookup from frame_idx to keypoints
    frame_lookup = {fd["frame_idx"]: fd for fd in stage2_data["frame_data"]}

    video_folder = "video000000"  # Assuming single clip per video

    depth_folder_path = os.path.join(
        DEPTH_ROOT,
        category,
        activity,
        collection,
        video_folder
    )

    for i, original_idx in enumerate(clip_indices):

        depth_image_path = os.path.join(depth_folder_path, f"{i:03d}.jpg")
        depth_img = cv2.imread(depth_image_path, cv2.IMREAD_GRAYSCALE)

        if depth_img is None:
            depth_sequence.append(0.0)
            continue

        depth_img = depth_img.astype(np.float32) / 255.0

        if original_idx in frame_lookup:
            keypoints = frame_lookup[original_idx]["keypoints"]
            h, w = depth_img.shape

            torso_y = get_torso_centroid(keypoints, h)
            torso_x = np.median([kp[0] * w for kp in keypoints])

            x = int(torso_x)
            y = int(torso_y)

            if 0 <= x < w and 0 <= y < h:
                depth_value = depth_img[y, x]
            else:
                depth_value = 0.0
        else:
            depth_value = 0.0

        depth_sequence.append(depth_value)

    selected_depth = depth_sequence[depth_frame_index]

    # Height ratio
    selected_keypoints = frame_lookup[selected_original_idx]["keypoints"]
    image_height = depth_img.shape[0]
    torso_y_pixel = get_torso_centroid(selected_keypoints, image_height)
    height_ratio = torso_y_pixel / image_height

    depth_drop = compute_depth_metrics(depth_sequence, depth_frame_index)

    # Write Stage 4 JSON
    output_data = {
        "video_name": stage3_data["video_name"],
        "selected_original_frame": selected_original_idx,
        "selected_depth_frame_index": depth_frame_index,
        "attention_weight": attention_weights[k_star_idx],
        "torso_depth": float(selected_depth),
        "height_ratio": float(height_ratio),
        "depth_drop": float(depth_drop)
    }

    output_path = os.path.join(
        STAGE4_OUTPUT_DIR,
        os.path.basename(stage3_path)
    )

    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)

    print(f"Stage 4 completed for {stage3_data['video_name']}")


print("Stage 4 module ready.")