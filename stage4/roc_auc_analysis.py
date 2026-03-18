import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

# ===============================
# CONFIG
# ===============================
CSV_PATH = r"E:\JK\FallVision\stage4_analysis.csv"
CORR_THRESHOLD = 0.85

# ===============================
# LOAD DATA
# ===============================
df = pd.read_csv(CSV_PATH)

if "label" not in df.columns:
    raise ValueError("CSV must contain 'label' column.")

y = df["label"].values

# Remove metadata
non_feature_cols = ["video_name", "label"]
feature_cols = [c for c in df.columns if c not in non_feature_cols]

# Keep only numeric features
feature_cols = [
    c for c in feature_cols
    if np.issubdtype(df[c].dtype, np.number)
]

print(f"\nFound {len(feature_cols)} numeric features.\n")

# ===============================
# AUC COMPUTATION
# ===============================
auc_results = []

for feature in feature_cols:
    values = df[feature].values

    if np.std(values) < 1e-8:
        continue

    try:
        auc = roc_auc_score(y, values)

        if auc < 0.5:
            auc = roc_auc_score(y, -values)
            direction = "inverted"
        else:
            direction = "normal"

        auc_results.append((feature, auc, direction))

    except:
        continue

auc_results.sort(key=lambda x: x[1], reverse=True)

print("===== FEATURE RANKING (by AUC) =====\n")
for rank, (f, auc, d) in enumerate(auc_results, 1):
    print(f"{rank:02d}. {f:30s} | AUC: {auc:.4f} | {d}")

# ===============================
# CORRELATION MATRIX
# ===============================
print("\n===== COMPUTING CORRELATION MATRIX =====\n")

top_features = [f[0] for f in auc_results[:15]]  # analyze top 15
corr_matrix = df[top_features].corr().abs()

plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap="coolwarm", annot=False)
plt.title("Feature Correlation Matrix (Top 15)")
plt.tight_layout()
plt.show()

# ===============================
# FIND HIGHLY CORRELATED PAIRS
# ===============================
print("\n===== HIGH CORRELATION PAIRS (> {:.2f}) =====\n".format(CORR_THRESHOLD))

high_corr_pairs = []

for i in range(len(top_features)):
    for j in range(i+1, len(top_features)):
        f1 = top_features[i]
        f2 = top_features[j]
        corr_val = corr_matrix.loc[f1, f2]

        if corr_val > CORR_THRESHOLD:
            high_corr_pairs.append((f1, f2, corr_val))
            print(f"{f1:25s} <-> {f2:25s} | Corr: {corr_val:.4f}")

if not high_corr_pairs:
    print("No highly correlated feature pairs found.")

# ===============================
# SUGGEST INDEPENDENT SET
# ===============================
print("\n===== SUGGESTED INDEPENDENT FEATURES =====\n")

selected = []
for feature, auc, _ in auc_results:
    keep = True
    for s in selected:
        if corr_matrix.loc[feature, s] > CORR_THRESHOLD:
            keep = False
            break
    if keep:
        selected.append(feature)

print("Recommended feature set (low redundancy + high AUC):\n")
for f in selected[:6]:
    print(f)