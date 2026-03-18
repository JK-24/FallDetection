import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations

# =========================================================
# CONFIG
# =========================================================
CSV_PATH = r"new_pipeline\attention_model_v4\sequence_features_v4.csv"
CORR_THRESHOLD = 0.90

# =========================================================
# LOAD DATA
# =========================================================
print("Loading CSV...")
df = pd.read_csv(CSV_PATH)

print("Total rows:", len(df))
print("Unique videos:", df["video_id"].nunique())

# =========================================================
# CLEAN FEATURE MATRIX
# =========================================================
drop_cols = ["video_id", "frame_index"]
features_df = df.drop(columns=drop_cols)

# Convert label to numeric if needed
if features_df["label"].dtype == object:
    features_df["label"] = features_df["label"].map({"fall": 1, "no_fall": 0})

feature_columns = [c for c in features_df.columns if c != "label"]

print("\nTotal Features:", len(feature_columns))

# =========================================================
# CORRELATION MATRIX
# =========================================================
print("\nComputing correlation matrix...")
corr_matrix = features_df[feature_columns].corr()

plt.figure(figsize=(20, 18))
sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.show()

# =========================================================
# FIND HIGHLY CORRELATED PAIRS
# =========================================================
print("\nScanning for highly correlated feature pairs...")

high_corr_pairs = []

for f1, f2 in combinations(feature_columns, 2):
    corr_val = corr_matrix.loc[f1, f2]
    if abs(corr_val) > CORR_THRESHOLD:
        high_corr_pairs.append((f1, f2, corr_val))

if not high_corr_pairs:
    print("No highly correlated pairs found.")
else:
    print(f"\nHighly Correlated Feature Pairs (>|{CORR_THRESHOLD}|):")
    for f1, f2, val in sorted(high_corr_pairs, key=lambda x: -abs(x[2])):
        print(f"{f1}  <-->  {f2}  | Corr = {val:.4f}")

# =========================================================
# SUGGEST FEATURES TO DROP
# =========================================================
print("\nSuggested features to drop (keep first occurrence):")

to_drop = set()
seen = set()

for f1, f2, _ in high_corr_pairs:
    if f1 not in seen and f2 not in seen:
        to_drop.add(f2)
        seen.add(f1)
        seen.add(f2)

if to_drop:
    for f in to_drop:
        print("-", f)
else:
    print("No features suggested for removal.")

# =========================================================
# FEATURE vs LABEL CORRELATION
# =========================================================
print("\nComputing feature-to-label correlation...")

label_corr = features_df.corr()["label"].drop("label")
label_corr = label_corr.sort_values(key=lambda x: abs(x), ascending=False)

print("\nTop 15 Most Correlated Features With Label:")
print(label_corr.head(15))

# =========================================================
# LOW VARIANCE CHECK
# =========================================================
print("\nChecking low variance features...")

variance = features_df[feature_columns].var()
low_var = variance[variance < 1e-5]

if len(low_var) > 0:
    print("Low variance features detected:")
    print(low_var)
else:
    print("No near-constant features found.")

# =========================================================
# SAVE REDUCED FEATURE LIST
# =========================================================
final_features = [f for f in feature_columns if f not in to_drop]

pd.Series(final_features).to_csv("selected_features_after_corr.csv", index=False)

print("\nSaved recommended feature list to: selected_features_after_corr.csv")
print("Final feature count:", len(final_features))

print("\nCorrelation analysis complete.")