import os
import json
import pandas as pd
from concurrent.futures import ProcessPoolExecutor
import os

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
ROOT_DIR = r"E:\JK\misc\New pipeline run 1\stage3_output"
OUTPUT_CSV = r"stage23_master_features.csv"
NUM_WORKERS = os.cpu_count()


# -------------------------------------------------------
# Collect JSON paths in directory order
# -------------------------------------------------------
def collect_stage3_paths(root_dir):
    json_paths = []

    for label_folder in ["fall", "no_fall"]:
        label_path = os.path.join(root_dir, label_folder)

        if not os.path.exists(label_path):
            continue

        for root, dirs, files in os.walk(label_path):
            for file in files:
                if file.endswith(".json"):
                    json_paths.append(os.path.join(root, file))

    return json_paths


# -------------------------------------------------------
# Worker: Extract ONLY required fields
# -------------------------------------------------------
def extract_required_fields(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)

        row = {}

        # Required metadata
        row["video_name"] = data.get("video_name")
        row["label"] = data.get("label")

        # Stage 2 Summary
        stage2 = data.get("stage2_summary", {})
        for k, v in stage2.items():
            row[k] = v

        # Stage 3 Summary
        stage3 = data.get("stage3_summary", {})
        for k, v in stage3.items():
            row[k] = v

        return row

    except Exception as e:
        print(f"Error in {json_path}: {e}")
        return None


# -------------------------------------------------------
# MAIN
# -------------------------------------------------------
if __name__ == "__main__":

    print("Collecting stage3_output.json files...")
    json_paths = collect_stage3_paths(ROOT_DIR)
    print(f"Total files found: {len(json_paths)}")

    print("Starting parallel extraction...")

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # preserves input order
        results = list(executor.map(extract_required_fields, json_paths))

    # Remove failed entries
    results = [r for r in results if r is not None]

    print("Creating DataFrame...")
    df = pd.DataFrame(results)

    df.to_csv(OUTPUT_CSV, index=False)

    print("\n════════════════════════════════════")
    print("✅ CSV CREATED SUCCESSFULLY")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Saved as:", OUTPUT_CSV)
    print("════════════════════════════════════")

