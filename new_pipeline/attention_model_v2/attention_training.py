import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve
)
from torch.utils.data import Dataset, DataLoader

# =========================================================
# CONFIG
# =========================================================
INPUT_CSV = r"new_pipeline\attention_model_v2\sequence_features.csv"
SAVE_DIR = r"new_pipeline\attention_model_v2"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_COLUMNS = [
    "tilt_angle",
    "vertical_velocity",
    "h_w_ratio",
    "ground_proximity",
    "torso_depth",
    "depth_velocity",
    "depth_acceleration",
    "depth_relative_change",
]

INPUT_DIM = 8
HIDDEN_DIM = 32
BATCH_SIZE = 16
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
EARLY_STOP_PATIENCE = 5
FEATURE_NOISE_STD = 0.02
MIXUP_ALPHA = 0.4
TEMPERATURE = 1.5

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================================================
# SE BLOCK
# =========================================================
class SEBlock(nn.Module):
    def __init__(self, channel_dim, reduction=8):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel_dim, channel_dim // reduction),
            nn.ReLU(),
            nn.Linear(channel_dim // reduction, channel_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        z = x.mean(dim=1)
        weights = self.fc(z)
        return x * weights.unsqueeze(1)


# =========================================================
# MODEL
# =========================================================
class BiLSTMAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(INPUT_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(HIDDEN_DIM * 2)
        self.se = SEBlock(HIDDEN_DIM * 2)
        self.attn = nn.Linear(HIDDEN_DIM * 2, 1)

        self.fc = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1)
        )

    def forward(self, x, mask):
        lstm_out, _ = self.lstm(x)
        lstm_out = self.norm(lstm_out)
        lstm_out = self.se(lstm_out)

        attn_scores = self.attn(lstm_out).squeeze(-1)
        attn_scores = attn_scores.masked_fill(~mask, -1e9)
        attn_weights = F.softmax(attn_scores, dim=1)

        context = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)
        logits = self.fc(context)
        return logits.squeeze()


# =========================================================
# DATASET + COLLATE
# =========================================================
class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


def collate_fn(batch):
    sequences, labels = zip(*batch)
    max_len = max(seq.shape[0] for seq in sequences)

    padded, mask = [], []

    for seq in sequences:
        pad_len = max_len - seq.shape[0]
        padded_seq = F.pad(seq, (0, 0, 0, pad_len))
        padded.append(padded_seq)

        m = torch.zeros(max_len, dtype=torch.bool)
        m[:seq.shape[0]] = 1
        mask.append(m)

    return torch.stack(padded), torch.stack(mask), torch.tensor(labels, dtype=torch.float32)


# =========================================================
# MIXUP
# =========================================================
def mixup_data(x, y, alpha):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


# =========================================================
# LOAD CSV
# =========================================================
df = pd.read_csv(INPUT_CSV)
df = df.sort_values(["video_id", "frame_index"])
df["environment"] = df["video_id"].apply(lambda x: x.split("/")[1])

environments = ["bed", "chair", "stand"]
summary_results = []

for TEST_ENV in environments:

    print(f"\n=========== Testing on {TEST_ENV} ===========")

    train_df = df[df["environment"] != TEST_ENV].copy()
    val_df = df[df["environment"] == TEST_ENV].copy()

    # ---- FIX SCALER LEAKAGE ----
    scaler = StandardScaler()
    train_df[FEATURE_COLUMNS] = scaler.fit_transform(train_df[FEATURE_COLUMNS])
    val_df[FEATURE_COLUMNS] = scaler.transform(val_df[FEATURE_COLUMNS])

    # Build sequences
    sequences_train, labels_train = [], []
    sequences_val, labels_val = [], []

    for vid, group in train_df.groupby("video_id"):
        sequences_train.append(torch.tensor(group[FEATURE_COLUMNS].values, dtype=torch.float32))
        labels_train.append(group["label"].iloc[0])

    for vid, group in val_df.groupby("video_id"):
        sequences_val.append(torch.tensor(group[FEATURE_COLUMNS].values, dtype=torch.float32))
        labels_val.append(group["label"].iloc[0])

    num_pos = sum(labels_train)
    num_neg = len(labels_train) - num_pos
    pos_weight = torch.tensor([num_neg / num_pos]).to(DEVICE)

    train_loader = DataLoader(
        SequenceDataset(sequences_train, labels_train),
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        SequenceDataset(sequences_val, labels_val),
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = BiLSTMAttention().to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    best_recall = 0
    patience_counter = 0
    fold_logs = []

    for epoch in range(EPOCHS):

        # -------- TRAIN --------
        model.train()
        train_loss = 0

        for x, mask, y in train_loader:
            x = x + torch.randn_like(x) * FEATURE_NOISE_STD
            x, mask, y = x.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)

            x, y_a, y_b, lam = mixup_data(x, y, MIXUP_ALPHA)

            optimizer.zero_grad()
            logits = model(x, mask)
            loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        # -------- VALIDATION --------
        model.eval()
        val_probs, val_true = [], []

        with torch.no_grad():
            for x, mask, y in val_loader:
                x, mask, y = x.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)
                logits = model(x, mask)
                probs = torch.sigmoid(logits)
                val_probs.extend(probs.cpu().numpy())
                val_true.extend(y.cpu().numpy())

        fpr, tpr, thresholds = roc_curve(val_true, val_probs)
        optimal_idx = np.argmax(tpr - fpr)
        threshold = thresholds[optimal_idx]

        preds = (np.array(val_probs) > threshold).astype(int)

        val_acc = accuracy_score(val_true, preds)
        val_prec = precision_score(val_true, preds, zero_division=0)
        val_rec = recall_score(val_true, preds)
        val_f1 = f1_score(val_true, preds)
        val_auc = roc_auc_score(val_true, val_probs)

        tp = sum((np.array(val_true) == 1) & (preds == 1))
        tn = sum((np.array(val_true) == 0) & (preds == 0))
        fp = sum((np.array(val_true) == 0) & (preds == 1))
        fn = sum((np.array(val_true) == 1) & (preds == 0))

        print(f"Epoch {epoch+1} | Recall: {val_rec:.4f} | AUC: {val_auc:.4f}")

        fold_logs.append({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "val_acc": val_acc,
            "val_precision": val_prec,
            "val_recall": val_rec,
            "val_f1": val_f1,
            "val_auc": val_auc,
            "val_tp": tp,
            "val_tn": tn,
            "val_fp": fp,
            "val_fn": fn,
            "threshold": threshold
        })

        if val_rec > best_recall:
            best_recall = val_rec
            patience_counter = 0

            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"model_{TEST_ENV}.pth"))
            joblib.dump(scaler, os.path.join(SAVE_DIR, f"scaler_{TEST_ENV}.pkl"))

            with open(os.path.join(SAVE_DIR, f"config_{TEST_ENV}.json"), "w") as f:
                json.dump({
                    "threshold": float(threshold),
                    "temperature": TEMPERATURE
                }, f)

        else:
            patience_counter += 1

        if patience_counter >= EARLY_STOP_PATIENCE:
            print("Early stopping.")
            break

    pd.DataFrame(fold_logs).to_csv(
        os.path.join(SAVE_DIR, f"training_logs_{TEST_ENV}.csv"),
        index=False
    )

    summary_results.append({
        "environment_tested": TEST_ENV,
        "best_recall": best_recall,
        "best_auc": val_auc
    })

# =========================================================
# SUMMARY REPORT
# =========================================================
recalls = [r["best_recall"] for r in summary_results]
aucs = [r["best_auc"] for r in summary_results]

summary_df = pd.DataFrame(summary_results)
summary_df.to_csv(os.path.join(SAVE_DIR, "env_training_logs.csv"), index=False)

print("\n=========== FINAL RESULTS ===========")
print("Recall Mean ± Std:", np.mean(recalls), "±", np.std(recalls))
print("AUC Mean ± Std:", np.mean(aucs), "±", np.std(aucs))
print("=====================================")