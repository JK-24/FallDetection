import os
import json
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# =========================================================
# CONFIG
# =========================================================
STAGE4_DIR = r"E:\JK\misc\New pipeline run 1\stage4_output"
PERCENTILE_FILE = r"stage5_v2\stage5_percentiles.json"
OUTPUT_CSV = "stage5_output.csv"

BASE_THRESHOLD = 0.44

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

# =========================================================
# LOAD FROZEN PERCENTILES
# =========================================================
with open(PERCENTILE_FILE, "r") as f:
    percentiles = json.load(f)

def normalize(val, feature):
    p5 = percentiles[feature]["p5"]
    p95 = percentiles[feature]["p95"]
    if p95 - p5 == 0:
        return 0.0
    return float(np.clip((val - p5) / (p95 - p5), 0.0, 1.0))

def compute_pose_score(row):
    return np.mean([normalize(row[f], f) for f in POSE_FEATURES])

def compute_depth_score(row):
    return np.mean([normalize(row[f], f) for f in DEPTH_FEATURES])

# =========================================================
# STAGE 5 RULE ENGINE (REFINED V6)
# =========================================================
def stage5_rule(row):

    ensemble_P = row["ensemble_probability"]
    max_model_P = max(
        row["prob_model_bed"],
        row["prob_model_chair"],
        row["prob_model_stand"]
    )

    # Probability fusion
    fused_P = 0.7 * ensemble_P + 0.3 * max_model_P

    pose_score = compute_pose_score(row)
    depth_score = compute_depth_score(row)

    # Base weighted score
    final_score = (
        0.75 * fused_P +
        0.20 * pose_score +
        0.05 * depth_score
    )

   # -----------------------------------------------------
    # SOFT SCORE SHAPING (refined)
    # -----------------------------------------------------

    # Strong motion reinforcement (tightened)
    if normalize(row["tilt_std"], "tilt_std") > 0.75:
        final_score += 0.015

    if normalize(row["depth_variance"], "depth_variance") > 0.65:
        final_score += 0.015

    # Confidence gap correction (reduced boost)
    confidence_gap = max_model_P - ensemble_P
    if confidence_gap > 0.30:
        final_score += 0.015

    # Borderline rescue (tightened conditions)
    if (
        0.42 < fused_P < 0.55 and
        pose_score > 0.60 and
        depth_score > 0.45
    ):
        final_score += 0.02

     # -----------------------------------------------------
    # FINAL DECISION
    # -----------------------------------------------------

    threshold = BASE_THRESHOLD
    decision = 1 if final_score >= threshold else 0

    return decision, final_score   

# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    print("Scanning Stage4 directory...")
    print("Root:", STAGE4_DIR)

    rows = []
    total_json = 0

    for root, _, files in os.walk(STAGE4_DIR):
        for file in files:
            if file.endswith(".json"):

                total_json += 1
                path = os.path.join(root, file)

                try:
                    with open(path, "r") as f:
                        data = json.load(f)

                    stage2 = data["stage2_summary"]
                    stage3 = data["stage3_summary"]
                    stage4 = data["stage4"]

                    row = {
                        "video_id": data.get("video_name", file.replace(".json","")),
                        "label": data.get("label", None),
                        "prob_model_bed": stage4["prob_model_bed"],
                        "prob_model_chair": stage4["prob_model_chair"],
                        "prob_model_stand": stage4["prob_model_stand"],
                        "ensemble_probability": stage4["ensemble_probability"]
                    }

                    for f in POSE_FEATURES:
                        row[f] = stage2.get(f, 0.0)

                    for f in DEPTH_FEATURES:
                        row[f] = stage3.get(f, 0.0)

                    rows.append(row)

                except Exception as e:
                    print(f"Error reading {path}: {e}")

    print("Total JSON files processed:", total_json)

    df = pd.DataFrame(rows)

    print("Running Stage5 rule engine...")

    decisions = []
    scores = []

    for _, r in df.iterrows():
        d, s = stage5_rule(r)
        decisions.append(d)
        scores.append(s)

    df["stage5_score"] = scores
    df["stage5_decision"] = decisions

    df.to_csv(OUTPUT_CSV, index=False)

    print("\nStage5 CSV saved:", OUTPUT_CSV)

    # =====================================================
    # CONFUSION MATRIX
    # =====================================================
    if "label" in df.columns and df["label"].notna().all():

        tn, fp, fn, tp = confusion_matrix(
            df["label"], df["stage5_decision"]
        ).ravel()

        total = tp + tn + fp + fn

        accuracy = (tp + tn) / total * 100
        recall = tp / (tp + fn + 1e-8) * 100
        precision = tp / (tp + fp + 1e-8) * 100
        fnr = fn / (tp + fn + 1e-8) * 100
        fpr = fp / (fp + tn + 1e-8) * 100

        print("\n========== CONFUSION MATRIX ==========")
        print(f"TN: {tn}")
        print(f"FP: {fp}")
        print(f"FN: {fn}")
        print(f"TP: {tp}")

        print("\n========== METRICS (%) ==========")
        print(f"Accuracy : {accuracy:.2f}%")
        print(f"Recall   : {recall:.2f}%")
        print(f"Precision: {precision:.2f}%")
        print(f"FNR      : {fnr:.2f}%")
        print(f"FPR      : {fpr:.2f}%")
        print("=================================")

    else:
        print("\nNo labels found. Inference completed for unseen dataset.")