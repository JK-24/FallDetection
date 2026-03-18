import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve

# ==========================
# CONFIG
# ==========================
CSV_PATH = r"E:\JK\FallVision\stage3_analysis.csv"  # <-- change if needed


# ==========================
# LOAD CSV
# ==========================
df = pd.read_csv(CSV_PATH)

# Safety checks
required_cols = ["label", "stage3_fall_prob"]
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in CSV.")

y_true = df["label"].values
y_score = df["stage3_fall_prob"].values

# ==========================
# BASIC STATS
# ==========================
print("\n===== BASIC STATISTICS =====")
print(df["stage3_fall_prob"].describe())

print("\n===== CLASS-WISE MEAN PROBABILITY =====")
print(df.groupby("label")["stage3_fall_prob"].mean())

# ==========================
# ROC AUC
# ==========================
auc = roc_auc_score(y_true, y_score)
print(f"\n===== ROC-AUC =====")
print(f"AUC: {auc:.4f}")

# ==========================
# ROC CURVE PLOT
# ==========================
fpr, tpr, thresholds = roc_curve(y_true, y_score)

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Stage3 Fall Probability")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# ==========================
# HISTOGRAM PLOT
# ==========================
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="stage3_fall_prob", hue="label",
             bins=30, stat="density", common_norm=False)

plt.title("Probability Distribution (Fall vs NoFall)")
plt.xlabel("Stage3 Fall Probability")
plt.ylabel("Density")
plt.tight_layout()
plt.show()

# ==========================
# INTERPRETATION GUIDE
# ==========================
print("\n===== INTERPRETATION GUIDE =====")
if auc > 0.85:
    print("Excellent separation.")
elif auc > 0.75:
    print("Good separation.")
elif auc > 0.65:
    print("Moderate separation.")
elif auc > 0.55:
    print("Weak separation.")
else:
    print("Model likely failing (near random).")


from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(y_true, y_score)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]