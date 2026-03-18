import os
import json
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor

# =========================================================
# CONFIG
# =========================================================
ROOT_DIR = r"E:\JK\misc\New pipeline run 1\stage3_output"
OUTPUT_CSV = "sequence_features.csv"
NUM_WORKERS = os.cpu_count()

# =========================================================
# Collect JSON paths and generate Unique IDs
# =========================================================
def collect_tasks(root_dir):
    tasks = []
    for label_folder in ["fall", "no_fall"]:
        label_path = os.path.join(root_dir, label_folder)
        if not os.path.exists(label_path):
            continue

        for root, _, files in os.walk(label_path):
            for file in files:
                if file.endswith(".json"):
                    full_path = os.path.join(root, file)
                    # Create ID: "fall/folder_name/filename" (minus .json)
                    rel_path = os.path.relpath(full_path, root_dir)
                    unique_id = os.path.splitext(rel_path)[0].replace("\\", "/")
                    
                    tasks.append((full_path, unique_id))
    return tasks

# =========================================================
# Worker function
# =========================================================
def extract_sequence_data(task):
    json_path, unique_id = task  # Unpack tuple
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        label = data.get("label")
        frame_data = data.get("frame_data", [])
        
        if not frame_data:
            return []

        # Ensure frames are sorted
        frame_data = sorted(frame_data, key=lambda x: x["frame_idx"])

        # Extract depth values for derivatives
        depth_values = [float(f.get("features", {}).get("torso_depth", 0.0)) for f in frame_data]
        depth_values = np.array(depth_values, dtype=np.float64)

        # Compute depth derivatives
        depth_velocity = np.zeros_like(depth_values)
        depth_acceleration = np.zeros_like(depth_values)

        for i in range(1, len(depth_values)):
            depth_velocity[i] = depth_values[i] - depth_values[i - 1]
        for i in range(1, len(depth_velocity)):
            depth_acceleration[i] = depth_velocity[i] - depth_velocity[i - 1]

        depth_relative_change = depth_values - depth_values[0]

        rows = []
        for idx, frame in enumerate(frame_data):
            features = frame.get("features", {})
            rows.append({
                "video_id": unique_id,  # Now uses the path-based ID
                "frame_index": idx,
                "label": label,
                "tilt_angle": float(features.get("tilt_angle", 0.0)),
                "vertical_velocity": float(features.get("vertical_velocity", 0.0)),
                "h_w_ratio": float(features.get("h_w_ratio", 0.0)),
                "ground_proximity": float(features.get("ground_proximity", 0.0)),
                "torso_depth": depth_values[idx],
                "depth_velocity": depth_velocity[idx],
                "depth_acceleration": depth_acceleration[idx],
                "depth_relative_change": depth_relative_change[idx],
            })
        return rows

    except Exception as e:
        print(f"Error in {json_path}: {e}")
        return []

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":
    print("Mapping JSON files to unique path-based IDs...")
    tasks = collect_tasks(ROOT_DIR)
    print(f"Total tasks created: {len(tasks)}")

    all_rows = []
    print(f"Processing with {NUM_WORKERS} workers...")

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        results = list(executor.map(extract_sequence_data, tasks))

    for video_rows in results:
        all_rows.extend(video_rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(OUTPUT_CSV, index=False)

    print("\n" + "═"*40)
    print("✅ sequence_features.csv CREATED")
    print(f"Total Frames:      {len(df)}")
    print(f"Unique Video IDs:  {df['video_id'].nunique()}")
    print(f"Saved as:          {OUTPUT_CSV}")
    print("═"*40)