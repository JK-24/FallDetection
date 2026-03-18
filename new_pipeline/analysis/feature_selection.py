import pandas as pd
import numpy as np

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
AUC_CSV = r"new_pipeline\analysis\auc_scores.csv"
CORR_CSV = r"new_pipeline\analysis\feature_correlation.csv"
CORR_THRESHOLD = 0.85
TOP_K = 10


# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------
auc_df = pd.read_csv(AUC_CSV)
corr_df = pd.read_csv(CORR_CSV, index_col=0)

# -------------------------------------------------------
# FEATURE GROUPING
# -------------------------------------------------------
pose_prefixes = ["tilt_", "velocity_", "h_w_ratio_", "ground_proximity_"]
depth_prefixes = ["depth_"]

pose_features = []
depth_features = []

for feature in auc_df["feature"]:
    if any(feature.startswith(p) for p in pose_prefixes):
        pose_features.append(feature)
    elif any(feature.startswith(p) for p in depth_prefixes):
        depth_features.append(feature)


# -------------------------------------------------------
# FUNCTION: Select least correlated top AUC features
# -------------------------------------------------------
def select_top_features(feature_list, group_name):
    selected = []

    # Sort by descending AUC
    group_auc = auc_df[auc_df["feature"].isin(feature_list)]
    group_auc = group_auc.sort_values("auc_score", ascending=False)

    for feature in group_auc["feature"]:
        if len(selected) >= TOP_K:
            break

        is_correlated = False

        for sel in selected:
            if abs(corr_df.loc[feature, sel]) > CORR_THRESHOLD:
                is_correlated = True
                break

        if not is_correlated:
            selected.append(feature)

    print(f"\nSelected {group_name} features:")
    for f in selected:
        score = auc_df.loc[auc_df["feature"] == f, "auc_score"].values[0]
        print(f"{f}  |  AUC = {score:.4f}")

    return selected


# -------------------------------------------------------
# SELECT FEATURES
# -------------------------------------------------------
top_pose = select_top_features(pose_features, "POSE")
top_depth = select_top_features(depth_features, "DEPTH")

# -------------------------------------------------------
# SAVE SELECTED FEATURES
# -------------------------------------------------------
selected_df = pd.DataFrame({
    "pose_features": pd.Series(top_pose),
    "depth_features": pd.Series(top_depth)
})

selected_df.to_csv("selected_features.csv", index=False)

print("\nSaved selected features → selected_features.csv")
print("════════════════════════════════════")
print("✅ FEATURE SELECTION COMPLETE")
print("════════════════════════════════════")