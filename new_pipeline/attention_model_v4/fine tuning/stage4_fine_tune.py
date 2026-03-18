# =========================================================
# STAGE-4 v2 (UNIFIED + FP REDUCTION + SAFE BACKBONE LOAD)
# =========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import joblib
import json
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    matthews_corrcoef,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

from torch.utils.data import Dataset, DataLoader

# =========================================================
# METRICS FUNCTION
# =========================================================

def compute_metrics(labels, probs, threshold):

    preds = (probs >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, zero_division=0)
    recall = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    auc = roc_auc_score(labels, probs)
    mcc = matthews_corrcoef(labels, preds)

    specificity = tn / (tn + fp + 1e-8)

    fpr = fp / (fp + tn + 1e-8)
    fnr = fn / (fn + tp + 1e-8)

    tpr = recall
    tnr = specificity

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": f1,
        "auc": auc,
        "mcc": mcc,
        "fpr": fpr,
        "fnr": fnr,
        "tpr": tpr,
        "tnr": tnr,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn
    }

# =========================================================
# CONFIG
# =========================================================

ORIGINAL_CSV = r"new_pipeline\attention_model_v4\sequence_features_v4.csv"
CAUCA_CSV    = r"new_pipeline\attention_model_v4\fine tuning\sequence_features_caucafall.csv"

PRETRAINED_MODEL = r"new_pipeline\attention_model_v4\model.pth"
SAVE_DIR = r"new_pipeline\attention_model_v4\fine tuning"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

FEATURE_COLUMNS = [
    "max_depth_drop_ratio",
    "height_drop_ratio",
    "peak_time_ratio",
    "ground_proximity",
    "peak_motion",
    "tilt_angle",
    "post_impact_stillness",
    "depth_peak_time_ratio",
    "tilt_relative_change",
    "motion_energy",
    "acceleration_energy",
    "duration_high_motion",
]

INPUT_DIM = 12
HIDDEN_DIM = 64
BATCH_SIZE = 16
EPOCHS = 50
LR = 1e-4
WEIGHT_DECAY = 1e-4
EARLY_STOP_PATIENCE = 10

os.makedirs(SAVE_DIR, exist_ok=True)

# =========================================================
# LOAD DATA
# =========================================================

df_original = pd.read_csv(ORIGINAL_CSV)
df_cauc = pd.read_csv(CAUCA_CSV)

df_original["video_id"] = "orig_" + df_original["video_id"].astype(str)
df_cauc["video_id"] = "cauc_" + df_cauc["video_id"].astype(str)

df = pd.concat([df_original, df_cauc], ignore_index=True)
df = df.sort_values(["video_id", "frame_index"])

video_labels = df.groupby("video_id")["label"].first().reset_index()

print("Class distribution:")
print(video_labels["label"].value_counts())

train_videos, val_videos = train_test_split(
    video_labels,
    test_size=0.2,
    stratify=video_labels["label"],
    random_state=42
)

train_df = df[df["video_id"].isin(train_videos["video_id"])].copy()
val_df   = df[df["video_id"].isin(val_videos["video_id"])].copy()

# =========================================================
# SCALER
# =========================================================

scaler = StandardScaler()

train_df.loc[:, FEATURE_COLUMNS] = scaler.fit_transform(train_df[FEATURE_COLUMNS])
val_df.loc[:, FEATURE_COLUMNS]   = scaler.transform(val_df[FEATURE_COLUMNS])

joblib.dump(scaler, os.path.join(SAVE_DIR, "scaler.pkl"))

# =========================================================
# DATASET
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


def build_sequences(dataframe):

    sequences, labels = [], []

    for vid, group in dataframe.groupby("video_id"):

        sequences.append(
            torch.tensor(group[FEATURE_COLUMNS].values, dtype=torch.float32)
        )

        labels.append(group["label"].iloc[0])

    return sequences, labels


sequences_train, labels_train = build_sequences(train_df)
sequences_val, labels_val     = build_sequences(val_df)

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

# =========================================================
# MODEL
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


class BiLSTMAttention(nn.Module):

    def __init__(self):
        super().__init__()

        self.lstm = nn.LSTM(INPUT_DIM, HIDDEN_DIM,
                            batch_first=True,
                            bidirectional=True)

        self.norm = nn.LayerNorm(HIDDEN_DIM * 2)

        self.se = SEBlock(HIDDEN_DIM * 2)

        self.attn = nn.Linear(HIDDEN_DIM * 2, 1)

        self.fc = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1)
        )

    def forward(self, x, mask):

        lstm_out, _ = self.lstm(x)

        lstm_out = self.norm(lstm_out)

        lstm_out = self.se(lstm_out)

        attn_scores = self.attn(lstm_out).squeeze(-1)

        attn_scores = attn_scores.masked_fill(~mask, -1e9)

        attn_weights = F.softmax(attn_scores, dim=1)

        context = torch.sum(
            lstm_out * attn_weights.unsqueeze(-1),
            dim=1
        )

        logits = self.fc(context)

        return logits.squeeze()


model = BiLSTMAttention().to(DEVICE)

# =========================================================
# SAFE BACKBONE LOAD
# =========================================================

pretrained_dict = torch.load(PRETRAINED_MODEL, map_location=DEVICE)

model_dict = model.state_dict()

filtered_dict = {
    k: v for k, v in pretrained_dict.items()
    if k in model_dict and not k.startswith("fc")
}

model_dict.update(filtered_dict)

model.load_state_dict(model_dict)

print("Backbone loaded. New FC head initialized.")

# =========================================================
# FOCAL LOSS
# =========================================================

class FocalLoss(nn.Module):

    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):

        bce = F.binary_cross_entropy_with_logits(
            logits,
            targets,
            reduction='none'
        )

        probs = torch.sigmoid(logits)

        pt = torch.where(targets == 1, probs, 1 - probs)

        focal = self.alpha * (1 - pt) ** self.gamma * bce

        return focal.mean()


criterion = FocalLoss(alpha=0.25, gamma=2)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

# =========================================================
# THRESHOLD SCAN
# =========================================================

def scan_thresholds(probs, labels):

    thresholds = np.linspace(0.3, 0.9, 200)

    best_mcc = -1
    best_t = 0.5

    for t in thresholds:

        preds = (probs >= t).astype(int)

        mcc = matthews_corrcoef(labels, preds)

        if mcc > best_mcc:
            best_mcc = mcc
            best_t = t

    print(f"🔥 Best Threshold: {best_t:.3f} | MCC: {best_mcc:.4f}")

    return best_t

# =========================================================
# TRAIN
# =========================================================

best_auc = 0
patience = 0
best_threshold = 0.5

training_logs = []

for epoch in range(EPOCHS):

    model.train()

    train_loss = 0

    for x, mask, y in train_loader:

        x, mask, y = x.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()

        logits = model(x, mask)

        loss = criterion(logits, y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        train_loss += loss.item()

    train_loss /= len(train_loader)

    model.eval()

    val_probs, val_true = [], []

    with torch.no_grad():

        for x, mask, y in val_loader:

            x, mask = x.to(DEVICE), mask.to(DEVICE)

            logits = model(x, mask)

            probs = torch.sigmoid(logits)

            val_probs.extend(probs.cpu().numpy())
            val_true.extend(y.numpy())

    val_probs = np.array(val_probs)
    val_true  = np.array(val_true)

    val_auc = roc_auc_score(val_true, val_probs)

    metrics = compute_metrics(val_true, val_probs, best_threshold)

    training_logs.append({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        **metrics
    })

    print(
        f"Epoch {epoch+1} | "
        f"Train Loss: {train_loss:.4f} | "
        f"AUC: {metrics['auc']:.4f} | "
        f"F1: {metrics['f1']:.4f} | "
        f"Recall: {metrics['recall']:.4f}"
    )

    if val_auc > best_auc:

        best_auc = val_auc
        patience = 0

        best_threshold = scan_thresholds(val_probs, val_true)

        torch.save(
            model.state_dict(),
            os.path.join(SAVE_DIR, "model.pth")
        )

        with open(os.path.join(SAVE_DIR, "config.json"), "w") as f:

            json.dump({
                "threshold": float(best_threshold)
            }, f)

    else:

        patience += 1

    if patience >= EARLY_STOP_PATIENCE:

        print("Early stopping triggered.")

        break

# =========================================================
# SAVE TRAINING LOGS
# =========================================================

pd.DataFrame(training_logs).to_csv(
    os.path.join(SAVE_DIR, "training_logs.csv"),
    index=False
)

print("\nStage-4 v2 training complete.")