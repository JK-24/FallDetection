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
    recall_score,
    precision_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    confusion_matrix,
    matthews_corrcoef
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
INPUT_CSV = r"new_pipeline\attention_model_v4\sequence_features_v4.csv"
SAVE_DIR = r"new_pipeline\attention_model_v4"
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
EPOCHS = 30
LR = 5e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP = 1.0
EARLY_STOP_PATIENCE = 5

FEATURE_NOISE_STD = 0.02
SCALING_JITTER_RANGE = (0.9, 1.1)
FEATURE_JITTER_STD = 0.01
MIXUP_ALPHA = 0.4
TEMPERATURE = 1.5
LABEL_SMOOTHING = 0.05

MARGIN = 0.7
MARGIN_WEIGHT = 0.2

os.makedirs(SAVE_DIR, exist_ok=True)

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

        self.lstm = nn.LSTM(INPUT_DIM, HIDDEN_DIM, batch_first=True, bidirectional=True)
        self.norm = nn.LayerNorm(HIDDEN_DIM * 2)
        self.se = SEBlock(HIDDEN_DIM * 2)
        self.attn = nn.Linear(HIDDEN_DIM * 2, 1)

        self.fc = nn.Sequential(
            nn.Linear(HIDDEN_DIM * 2, 32),
            nn.ReLU(),
            nn.Dropout(0.6),
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
# MIXUP + LABEL SMOOTHING
# =========================================================
def mixup_data(x, y, alpha):

    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]

    y_a, y_b = y, y[index]

    return mixed_x, y_a, y_b, lam


def smooth_labels(y, smoothing=0.05):

    return y * (1 - smoothing) + 0.5 * smoothing

# =========================================================
# MARGIN LOSS
# =========================================================
def margin_loss(logits, targets):

    pos_mask = targets == 1
    neg_mask = targets == 0

    pos_logits = logits[pos_mask]
    neg_logits = logits[neg_mask]

    loss_pos = F.relu(MARGIN - pos_logits).mean() if pos_logits.numel() > 0 else 0
    loss_neg = F.relu(MARGIN + neg_logits).mean() if neg_logits.numel() > 0 else 0

    return loss_pos + loss_neg

# =========================================================
# THRESHOLD SCAN
# =========================================================
def scan_thresholds(probs, labels):

    thresholds = np.linspace(0.30, 0.80, 101)

    best_mcc = -1
    best_t = 0.5

    for t in thresholds:

        preds = (probs >= t).astype(int)

        mcc = matthews_corrcoef(labels, preds)

        if mcc > best_mcc:
            best_mcc = mcc
            best_t = t

    print(f"🔥 Best Threshold (MCC): {best_t:.3f} | MCC: {best_mcc:.4f}")

    return best_t

# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv(INPUT_CSV)

df = df.sort_values(["video_id", "frame_index"])

video_labels = df.groupby("video_id")["label"].first().reset_index()

train_videos, val_videos = train_test_split(
    video_labels,
    test_size=0.20,
    stratify=video_labels["label"],
    random_state=42
)

train_df = df[df["video_id"].isin(train_videos["video_id"])].copy()
val_df = df[df["video_id"].isin(val_videos["video_id"])].copy()

# =========================================================
# SCALER
# =========================================================
scaler = StandardScaler()

train_df.loc[:, FEATURE_COLUMNS] = scaler.fit_transform(train_df[FEATURE_COLUMNS])
val_df.loc[:, FEATURE_COLUMNS] = scaler.transform(val_df[FEATURE_COLUMNS])

# =========================================================
# BUILD SEQUENCES
# =========================================================
def build_sequences(dataframe):

    sequences, labels = [], []

    for vid, group in dataframe.groupby("video_id"):

        sequences.append(torch.tensor(group[FEATURE_COLUMNS].values, dtype=torch.float32))
        labels.append(group["label"].iloc[0])

    return sequences, labels


sequences_train, labels_train = build_sequences(train_df)
sequences_val, labels_val = build_sequences(val_df)

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
# TRAINING
# =========================================================
model = BiLSTMAttention().to(DEVICE)

criterion = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR,
    weight_decay=WEIGHT_DECAY
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='max',
    factor=0.5,
    patience=2
)

best_auc = 0
patience_counter = 0
training_logs = []

best_threshold = 0.5

for epoch in range(EPOCHS):

    model.train()

    train_loss_epoch = 0

    for x, mask, y in train_loader:

        x = x + torch.randn_like(x) * FEATURE_NOISE_STD

        scale = torch.empty(1).uniform_(*SCALING_JITTER_RANGE).item()
        x = x * scale

        x = x + torch.randn_like(x) * FEATURE_JITTER_STD

        if torch.rand(1).item() > 0.5:
            x = torch.roll(x, shifts=1, dims=1)

        x, mask, y = x.to(DEVICE), mask.to(DEVICE), y.to(DEVICE)

        x, y_a, y_b, lam = mixup_data(x, y, MIXUP_ALPHA)

        y_a = smooth_labels(y_a, LABEL_SMOOTHING)
        y_b = smooth_labels(y_b, LABEL_SMOOTHING)

        optimizer.zero_grad()

        logits = model(x, mask)

        bce = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)

        m_loss = margin_loss(logits, y)

        loss = bce + MARGIN_WEIGHT * m_loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)

        optimizer.step()

        train_loss_epoch += loss.item()

    train_loss_epoch /= len(train_loader)

    # ======================
    # VALIDATION
    # ======================
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
    val_true = np.array(val_true)

    val_auc = roc_auc_score(val_true, val_probs)

    scheduler.step(val_auc)

    # compute metrics
    metrics_dict = compute_metrics(val_true, val_probs, best_threshold)

    training_logs.append({
        "epoch": epoch + 1,
        "train_loss": train_loss_epoch,
        **metrics_dict
    })

    print(
        f"Epoch {epoch+1} | "
        f"Loss {train_loss_epoch:.4f} | "
        f"AUC {metrics_dict['auc']:.4f} | "
        f"F1 {metrics_dict['f1']:.4f} | "
        f"Recall {metrics_dict['recall']:.4f}"
    )

    # ======================
    # SAVE BEST MODEL
    # ======================
    if val_auc > best_auc:

        best_auc = val_auc
        patience_counter = 0

        # threshold tuning
        model.eval()

        train_probs, train_true = [], []

        with torch.no_grad():

            for x, mask, y in train_loader:

                x, mask = x.to(DEVICE), mask.to(DEVICE)

                logits = model(x, mask)

                probs = torch.sigmoid(logits)

                train_probs.extend(probs.cpu().numpy())
                train_true.extend(y.numpy())

        best_threshold = scan_thresholds(
            np.array(train_probs),
            np.array(train_true)
        )

        torch.save(model.state_dict(), os.path.join(SAVE_DIR, "model.pth"))

        joblib.dump(
            scaler,
            os.path.join(SAVE_DIR, "scaler.pkl")
        )

        with open(os.path.join(SAVE_DIR, "config.json"), "w") as f:

            json.dump({
                "threshold": float(best_threshold),
                "temperature": TEMPERATURE,
                "input_dim": INPUT_DIM
            }, f)

    else:

        patience_counter += 1

    if patience_counter >= EARLY_STOP_PATIENCE:

        print("Early stopping triggered.")
        break

# =========================================================
# SAVE LOGS
# =========================================================
pd.DataFrame(training_logs).to_csv(
    os.path.join(SAVE_DIR, "training_logs.csv"),
    index=False
)

print("\nTraining complete.")