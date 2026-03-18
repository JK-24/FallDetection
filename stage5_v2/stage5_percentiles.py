import os
import json
import numpy as np
import pandas as pd

# =========================================================
# CONFIG
# =========================================================
STAGE4_DIR = r"E:\JK\misc\New pipeline run 1\stage4_output"
OUTPUT_FILE = "stage5_percentiles.json"

POSE_FEATURES = [
    "tilt_std",
    "ground_proximity_range",
    "tilt_total_variation",
    "ground_proximity_iqr",
    "tilt_abs_delta"
]

DEPTH_FEATURES = [
    "depth_variance",
    "depth_iqr",
    "depth_mean_abs_diff",
    "depth_second_half_std",
    "depth_abs_delta"
]

ALL_FEATURES = POSE_FEATURES + DEPTH_FEATURES


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    print("Scanning Stage4 directory...")
    print("Root:", STAGE4_DIR)

    rows = []
    total_json = 0
    valid_json = 0

    for root, _, files in os.walk(STAGE4_DIR):
        for file in files:
            if file.endswith(".json"):

                total_json += 1
                path = os.path.join(root, file)

                try:
                    with open(path, "r") as f:
                        data = json.load(f)

                    # Validate expected structure
                    if "stage2_summary" not in data or "stage3_summary" not in data:
                        continue

                    stage2 = data["stage2_summary"]
                    stage3 = data["stage3_summary"]

                    row = {}

                    for f in POSE_FEATURES:
                        row[f] = stage2.get(f, 0.0)

                    for f in DEPTH_FEATURES:
                        row[f] = stage3.get(f, 0.0)

                    rows.append(row)
                    valid_json += 1

                except Exception as e:
                    print(f"Error reading {path}: {e}")

    print("\n========== SANITY CHECK ==========")
    print("Total JSON files found:", total_json)
    print("Valid JSON files used:", valid_json)
    print("==================================")

    if valid_json == 0:
        raise RuntimeError("No valid Stage4 JSON files found.")

    df = pd.DataFrame(rows)

    print("\nComputing percentiles from", len(df), "videos...")

    percentiles = {}

    for feature in ALL_FEATURES:

        if feature not in df.columns:
            raise ValueError(f"Feature missing: {feature}")

        p5 = np.percentile(df[feature], 5)
        p95 = np.percentile(df[feature], 95)

        percentiles[feature] = {
            "p5": float(p5),
            "p95": float(p95)
        }

        print(f"{feature:30s} | p5={p5:.6f} | p95={p95:.6f}")

    with open(OUTPUT_FILE, "w") as f:
        json.dump(percentiles, f, indent=4)

    print("\n✅ Percentiles saved to:", OUTPUT_FILE)