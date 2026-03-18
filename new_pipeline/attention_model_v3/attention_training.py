import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import recall_score, roc_auc_score, roc_curve
from torch.utils.data import Dataset, DataLoader

# =========================================================
# CONFIG
# =========================================================
INPUT_CSV = r"new_pipeline\attention_model_v3\sequence_features.csv"
SAVE_DIR = r"new_pipeline\attention_model_v3"
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

# 🔥 NEW
MARGIN = 0.7
MARGIN_WEIGHT = 0.2

os.makedirs(SAVE_DIR, exist_ok=True)

from sklearn.metrics import confusion_matrix, matthews_corrcoef

def scan_thresholds(probs, labels):

    thresholds = np.linspace(0.30, 0.80, 101)
    results = []

    for t in thresholds:
        preds = (probs >= t).astype(int)

        tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()

        recall = tp / (tp + fn + 1e-9)
        specificity = tn / (tn + fp + 1e-9)
        fpr = fp / (fp + tn + 1e-9)
        fnr = fn / (fn + tp + 1e-9)
        mcc = matthews_corrcoef(labels, preds)

        results.append((t, recall, specificity, fpr, fnr, mcc))

    results = sorted(results, key=lambda x: x[5], reverse=True)
    best = results[0]

    print("\n🔥 Threshold Optimization (Train Set)")
    print(f"Best Threshold (Max MCC): {best[0]:.3f}")
    print(f"Recall: {best[1]:.4f}")
    print(f"Specificity: {best[2]:.4f}")
    print(f"FPR: {best[3]:.4f}")
    print(f"FNR: {best[4]:.4f}")
    print(f"MCC: {best[5]:.4f}")

    return best[0]

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
# MIXUP
# =========================================================
def mixup_data(x, y, alpha):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def smooth_labels(y, smoothing=0.05):
    return y * (1 - smoothing) + 0.5 * smoothing

# 🔥 Confidence Margin Regularization
def margin_loss(logits, targets):
    pos_mask = targets == 1
    neg_mask = targets == 0

    pos_logits = logits[pos_mask]
    neg_logits = logits[neg_mask]

    loss_pos = F.relu(MARGIN - pos_logits).mean() if pos_logits.numel() > 0 else 0
    loss_neg = F.relu(MARGIN + neg_logits).mean() if neg_logits.numel() > 0 else 0

    return loss_pos + loss_neg

# =========================================================
# LOAD DATA
# =========================================================
df = pd.read_csv(INPUT_CSV)
df = df.sort_values(["video_id", "frame_index"])
df["environment"] = df["video_id"].apply(lambda x: x.split("/")[1])

environments = ["bed", "chair", "stand"]
summary_results = []

# =========================================================
# 3-FOLD ENVIRONMENT ROTATION
# =========================================================
for TEST_ENV in environments:

    print(f"\n=========== Testing on {TEST_ENV} ===========")

    train_df = df[df["environment"] != TEST_ENV].copy()
    val_df = df[df["environment"] == TEST_ENV].copy()

    scaler = StandardScaler()
    train_df[FEATURE_COLUMNS] = scaler.fit_transform(train_df[FEATURE_COLUMNS])
    val_df[FEATURE_COLUMNS] = scaler.transform(val_df[FEATURE_COLUMNS])

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

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=2,
        verbose=True
    )

    best_recall = 0
    patience_counter = 0

    for epoch in range(EPOCHS):

        model.train()
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

            bce_loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
            m_loss = margin_loss(logits, y)

            loss = bce_loss + MARGIN_WEIGHT * m_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()

        # Validation
        model.eval()
        val_probs, val_true = [], []

        with torch.no_grad():
            for x, mask, y in val_loader:
                x, mask = x.to(DEVICE), mask.to(DEVICE)
                logits = model(x, mask)
                probs = torch.sigmoid(logits)

                val_probs.extend(probs.cpu().numpy())
                val_true.extend(y.numpy())

        val_auc = roc_auc_score(val_true, val_probs)
        scheduler.step(val_auc)

        fpr, tpr, thresholds = roc_curve(val_true, val_probs)
        threshold = thresholds[np.argmax(tpr - fpr)]
        preds = (np.array(val_probs) > threshold).astype(int)
        val_rec = recall_score(val_true, preds)

        print(f"Epoch {epoch+1} | Recall: {val_rec:.4f} | AUC: {val_auc:.4f}")

        if val_rec > best_recall:
            best_recall = val_rec
            patience_counter = 0

            # -------------------------------------------------
            # Collect TRAIN probabilities for threshold tuning
            # -------------------------------------------------
            model.eval()
            train_probs, train_true = [], []

            with torch.no_grad():
                for x, mask, y in train_loader:
                    x, mask = x.to(DEVICE), mask.to(DEVICE)
                    logits = model(x, mask)
                    probs = torch.sigmoid(logits)

                    train_probs.extend(probs.cpu().numpy())
                    train_true.extend(y.numpy())

            train_probs = np.array(train_probs)
            train_true = np.array(train_true)

            # -------------------------------------------------
            # Threshold Optimization (Max MCC)
            # -------------------------------------------------
            best_threshold = scan_thresholds(train_probs, train_true)

            # -------------------------------------------------
            # Save model + scaler + config
            # -------------------------------------------------
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, f"model_{TEST_ENV}.pth"))
            joblib.dump(scaler, os.path.join(SAVE_DIR, f"scaler_{TEST_ENV}.pkl"))

            with open(os.path.join(SAVE_DIR, f"config_{TEST_ENV}.json"), "w") as f:
                json.dump({
                    "threshold": float(best_threshold),
                    "temperature": TEMPERATURE,
                    "logit_mean": float(np.mean(train_probs)),
                    "logit_std": float(np.std(train_probs))
                }, f)

        if patience_counter >= EARLY_STOP_PATIENCE:
            print("Early stopping.")
            break

    summary_results.append({"environment_tested": TEST_ENV, "best_recall": best_recall})

print("\n=========== FINAL RESULTS ===========")
print(summary_results)
print("=====================================")