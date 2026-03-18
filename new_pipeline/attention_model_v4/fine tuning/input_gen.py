import os
import json
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# =========================================================
# CONFIG
# =========================================================

ROOT_DIR = r"D:\fall_detection\new_pipeline\attention_model_v4\fine tuning\stage3_output"  # directory with 100 JSON files
OUTPUT_CSV = r"D:\fall_detection\new_pipeline\attention_model_v4\fine tuning\sequence_features_caucafall.csv"
NUM_WORKERS = os.cpu_count()

FALL_CLASSES = [
    "FallBackward",
    "FallRight",
    "FallLeft",
    "FallForward"
]

# =========================================================
# COLLECT JSON FILES
# =========================================================

def collect_tasks(root_dir):
    tasks = []

    for file in os.listdir(root_dir):
        if file.endswith(".json"):
            full_path = os.path.join(root_dir, file)
            unique_id = os.path.splitext(file)[0]
            tasks.append((full_path, unique_id))

    return tasks


# =========================================================
# FEATURE EXTRACTION
# =========================================================

def extract_sequence_data(task):
    json_path, unique_id = task

    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        # Label from filename
        filename = os.path.basename(json_path)
        label = 1 if any(f in filename for f in FALL_CLASSES) else 0

        frame_data = sorted(
            data.get("frame_data", []),
            key=lambda x: x["frame_idx"]
        )

        if not frame_data:
            return []

        n = len(frame_data)

        # =========================
        # Raw arrays
        # =========================
        tilt = np.array([float(f["features"].get("tilt_angle", 0.0)) for f in frame_data])
        vel = np.array([float(f["features"].get("vertical_velocity", 0.0)) for f in frame_data])
        ground = np.array([float(f["features"].get("ground_proximity", 0.0)) for f in frame_data])
        depth = np.array([float(f["features"].get("torso_depth", 0.0)) for f in frame_data])

        # =========================
        # Derivatives
        # =========================
        tilt_vel = np.diff(tilt, prepend=tilt[0])
        tilt_acc = np.diff(tilt_vel, prepend=tilt_vel[0])
        depth_vel = np.diff(depth, prepend=depth[0])
        depth_acc = np.diff(depth_vel, prepend=depth_vel[0])

        # =========================
        # Relative metrics
        # =========================
        tilt_rel = tilt - tilt[0]
        depth_rel_ratio = (depth - depth[0]) / (depth[0] + 1e-9)
        height_drop_ratio = (np.max(ground) - ground) / (np.max(ground) + 1e-9)

        # =========================
        # Energy & motion stats
        # =========================
        motion_energy = np.sum(vel**2) / n
        peak_motion = np.max(np.abs(vel))
        accel_energy = np.sum(depth_acc**2) / n

        peak_idx = np.argmax(np.abs(vel))
        peak_time_ratio = peak_idx / n

        threshold = 0.6 * peak_motion
        duration_high_motion = np.sum(np.abs(vel) > threshold) / n

        half = n // 2
        first_half_mean = np.mean(np.abs(vel[:half])) if half > 0 else 0
        second_half_mean = np.mean(np.abs(vel[half:])) if half > 0 else 0
        descent_recovery_ratio = first_half_mean / (second_half_mean + 1e-9)

        last_portion = int(n * 0.3)
        post_stillness = np.mean(np.abs(vel[-last_portion:])) if last_portion > 0 else 0
        stillness_ratio = np.sum(np.abs(vel[-last_portion:]) < 0.1) / (last_portion + 1e-9)

        max_depth_drop_ratio = (np.max(depth) - np.min(depth)) / (np.max(depth) + 1e-9)
        depth_peak_idx = np.argmax(np.abs(depth_rel_ratio))
        depth_peak_time_ratio = depth_peak_idx / n

        rows = []

        for i in range(n):
            rows.append({
                "video_id": unique_id,
                "frame_index": i,
                "label": label,

                # Frame-level features
                "tilt_angle": tilt[i],
                "vertical_velocity": vel[i],
                "ground_proximity": ground[i],
                "torso_depth": depth[i],

                "tilt_velocity": tilt_vel[i],
                "tilt_acceleration": tilt_acc[i],
                "depth_velocity": depth_vel[i],
                "depth_acceleration": depth_acc[i],

                "tilt_relative_change": tilt_rel[i],
                "depth_relative_ratio": depth_rel_ratio[i],
                "height_drop_ratio": height_drop_ratio[i],

                # Sequence-level (repeated per frame)
                "motion_energy": motion_energy,
                "peak_motion": peak_motion,
                "acceleration_energy": accel_energy,
                "peak_time_ratio": peak_time_ratio,
                "duration_high_motion": duration_high_motion,
                "descent_recovery_ratio": descent_recovery_ratio,
                "post_impact_stillness": post_stillness,
                "stillness_ratio": stillness_ratio,
                "max_depth_drop_ratio": max_depth_drop_ratio,
                "depth_peak_time_ratio": depth_peak_time_ratio,
            })

        return rows

    except Exception as e:
        print("Error in:", json_path)
        print(e)
        return []


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    tasks = collect_tasks(ROOT_DIR)
    print("Total JSON files found:", len(tasks))

    all_rows = []

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(executor.map(extract_sequence_data, tasks))

    for r in results:
        all_rows.extend(r)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print("✅ sequence_features_caucafall.csv created")
    print("Total Frames:", len(df))
    print("Unique Videos:", df["video_id"].nunique())