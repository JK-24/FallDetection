# =========================================
# COMPUTE STAGE-5 THRESHOLDS FROM DATASET
# =========================================

import pandas as pd
import numpy as np

CSV_PATH =r"E:\JK\FallVision\stage4_analysis.csv"

df = pd.read_csv(CSV_PATH)

fall_df = df[df["label"] == 1]

FEATURES = [
    "abs_delta_tilt",
    "abs_velocity",
    "tilt_over_hwr",
    "max_tilt",
    "abs_delta_hwr",
    "orientation_velocity_coupling"
]

THRESHOLDS = {}

for f in FEATURES:
    THRESHOLDS[f] = float(np.median(fall_df[f]))

print("\nComputed Stage-5 Thresholds:\n")
for k, v in THRESHOLDS.items():
    print(f"{k:30s} : {v:.6f}")