import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score

# -------------------------------------------------------
# CONFIG
# -------------------------------------------------------
INPUT_CSV = r"new_pipeline\analysis\stage23_master_features.csv"
AUC_OUTPUT_CSV = "auc_scores.csv"
CORR_OUTPUT_CSV = "feature_correlation.csv"
CORR_IMAGE = "feature_correlation.png"
AUC_PLOT_DIR = "auc_plots"

os.makedirs(AUC_PLOT_DIR, exist_ok=True)


# -------------------------------------------------------
# LOAD DATA
# -------------------------------------------------------
print("Loading dataset...")
df = pd.read_csv(INPUT_CSV)

# Label handling
# If label is 0/1 → use directly
# If label is string → map it
if df["label"].dtype == object:
    df["label_binary"] = df["label"].map({"fall": 1, "no_fall": 0})
else:
    df["label_binary"] = df["label"]

# Remove non-feature columns
ignore_cols = ["video_name", "label", "label_binary"]
feature_cols = [c for c in df.columns if c not in ignore_cols]

print(f"Total features: {len(feature_cols)}")

# -------------------------------------------------------
# AUC COMPUTATION + PLOTTING
# -------------------------------------------------------
print("Generating AUC curves...")

auc_results = []

for feature in feature_cols:
    try:
        y_true = df["label_binary"]
        y_score = df[feature]

        # Skip constant features
        if y_score.nunique() <= 1:
            continue

        auc_score = roc_auc_score(y_true, y_score)
        fpr, tpr, _ = roc_curve(y_true, y_score)

        auc_results.append((feature, auc_score))

        # Plot ROC
        plt.figure()
        plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curve - {feature}")
        plt.legend()
        plt.tight_layout()

        # Save image
        safe_name = feature.replace("/", "_")
        plt.savefig(os.path.join(AUC_PLOT_DIR, f"{safe_name}.png"))
        plt.close()

    except Exception as e:
        print(f"Skipped {feature}: {e}")

# Save AUC scores
auc_df = pd.DataFrame(auc_results, columns=["feature", "auc_score"])
auc_df = auc_df.sort_values("auc_score", ascending=False)
auc_df.to_csv(AUC_OUTPUT_CSV, index=False)

print(f"Saved AUC scores → {AUC_OUTPUT_CSV}")
print(f"AUC plots saved in → {AUC_PLOT_DIR}")


# -------------------------------------------------------
# CORRELATION MATRIX
# -------------------------------------------------------
print("Computing correlation matrix...")

corr_matrix = df[feature_cols].corr()

# Save CSV
corr_matrix.to_csv(CORR_OUTPUT_CSV)

# Plot heatmap
plt.figure(figsize=(18, 15))
plt.imshow(corr_matrix, cmap="coolwarm", aspect="auto")
plt.colorbar()
plt.title("Feature Correlation Matrix")
plt.tight_layout()
plt.savefig(CORR_IMAGE)
plt.close()

print(f"Saved correlation matrix CSV → {CORR_OUTPUT_CSV}")
print(f"Saved correlation heatmap → {CORR_IMAGE}")

print("\n════════════════════════════════════")
print("✅ FEATURE ANALYSIS COMPLETE")
print("════════════════════════════════════")