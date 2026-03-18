import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


def plot_confusion_matrix(tp, tn, fp, fn, save_path="confusion_matrix.png"):

    # Confusion matrix layout
    cm = np.array([
        [tp, fn],
        [fp, tn]
    ], dtype=float)

    # Row-wise normalization (actual class percentage)
    cm_percent = cm / cm.sum(axis=1, keepdims=True) * 100

    # Custom mild blue → white colormap
    cmap = LinearSegmentedColormap.from_list(
        "mild_blue",
        ["#ffffff", "#d6e6ff", "#7fb3ff"]  # white → light blue → soft blue
    )

    fig, ax = plt.subplots(figsize=(6,5))

    im = ax.imshow(cm_percent, cmap=cmap)

    ax.set_xticks([0,1])
    ax.set_yticks([0,1])

    ax.set_xticklabels(["Pred: Fall", "Pred: No Fall"])
    ax.set_yticklabels(["Actual: Fall", "Actual: No Fall"])

    # Write percentage values
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm_percent[i, j]:.2f}%",
                    ha="center", va="center", fontsize=12)

    ax.set_title("Confusion Matrix (%)")

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

    print(f"Saved to {save_path}")


# Example
plot_confusion_matrix(tp=28, tn=31, fp=9, fn=2)