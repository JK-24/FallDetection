import os
import json
import glob
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    confusion_matrix, balanced_accuracy_score, roc_auc_score
)
from sklearn.linear_model import LogisticRegression

# =====================================================
# CONFIGURATION
# =====================================================
DATA_PATH = os.path.join("stage3", "input", "stage2_output")

SEQ_LEN = 24
NUM_FEATURES = 38
NUM_FOLDS = 3
EPOCHS = 15
BATCH_SIZE = 32
LEARNING_RATE = 0.0003

LOG_DIR = r"stage3\training_logs"
MODEL_DIR = r"stage3\saved_models"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# =====================================================
# LOAD ALL FILES
# =====================================================
all_files = []
all_labels = []

for label_str, val in [('Fall', 1), ('No_Fall', 0)]:
    root = os.path.join(DATA_PATH, label_str)
    for path in glob.glob(os.path.join(root, "**", "*.json"), recursive=True):
        all_files.append(path)
        all_labels.append(val)

print("Total Falls:", sum(all_labels))
print("Total No_Falls:", len(all_labels) - sum(all_labels))

# =====================================================
# COMPUTE FEATURE NORMALIZATION STATS
# =====================================================
print("Computing dataset mean/std...")

all_features = []

for file in all_files:
    with open(file, 'r') as f:
        doc = json.load(f)
    frames = doc.get("frame_data", [])
    for fdata in frames:
        kp = np.array(fdata["keypoints"], dtype=np.float32).flatten()
        feat = fdata.get("features", {})
        f_vec = [
            float(feat.get("tilt_angle", 0)) / 180.0,
            float(feat.get("vertical_velocity", 0)) / 100.0,
            float(feat.get("h_w_ratio", 0)) / 5.0,
            float(feat.get("ground_proximity", 0))
        ]
        combined = np.nan_to_num(np.concatenate([kp, f_vec]), nan=0.0)
        all_features.append(combined)

all_features = np.array(all_features)
feat_mean = np.mean(all_features, axis=0)
feat_std = np.std(all_features, axis=0) + 1e-6

# =====================================================
# DATASET
# =====================================================
class FallDataset(Dataset):
    def __init__(self, file_list, labels):
        self.file_list = file_list
        self.labels = labels

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        with open(self.file_list[idx], 'r') as f:
            doc = json.load(f)

        frames = doc.get("frame_data", [])
        sequence = []

        for fdata in frames:
            kp = np.array(fdata["keypoints"], dtype=np.float32).flatten()
            feat = fdata.get("features", {})
            f_vec = [
                float(feat.get("tilt_angle", 0)) / 180.0,
                float(feat.get("vertical_velocity", 0)) / 100.0,
                float(feat.get("h_w_ratio", 0)) / 5.0,
                float(feat.get("ground_proximity", 0))
            ]
            combined = np.nan_to_num(np.concatenate([kp, f_vec]), nan=0.0)
            combined = (combined - feat_mean) / feat_std
            sequence.append(combined)

        sequence = np.array(sequence, dtype=np.float32)

        if len(sequence) < SEQ_LEN:
            pad = np.zeros((SEQ_LEN - len(sequence), NUM_FEATURES))
            sequence = np.vstack([sequence, pad])
        else:
            sequence = sequence[:SEQ_LEN]

        return torch.FloatTensor(sequence), torch.FloatTensor([self.labels[idx]])

# =====================================================
# MODEL
# =====================================================
class FallTransformerModel(nn.Module):
    def __init__(self):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=NUM_FEATURES,
            nhead=2,
            dim_feedforward=128,
            dropout=0.3,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.temporal_attention = nn.Linear(NUM_FEATURES, 1)

        self.fc = nn.Sequential(
            nn.Linear(NUM_FEATURES, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.encoder(x)
        attn_weights = torch.softmax(self.temporal_attention(x), dim=1)
        x = torch.sum(attn_weights * x, dim=1)
        logits = self.fc(x)
        return logits

# =====================================================
# TRAINING
# =====================================================
skf = StratifiedKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=42)

for fold, (train_idx, test_idx) in enumerate(skf.split(all_files, all_labels)):

    print(f"\n{'='*20} FOLD {fold+1} {'='*20}")

    train_files = [all_files[i] for i in train_idx]
    train_labels = [all_labels[i] for i in train_idx]
    test_files  = [all_files[i] for i in test_idx]
    test_labels = [all_labels[i] for i in test_idx]

    train_loader = DataLoader(FallDataset(train_files, train_labels),
                              batch_size=BATCH_SIZE, shuffle=True)

    test_loader = DataLoader(FallDataset(test_files, test_labels),
                             batch_size=BATCH_SIZE, shuffle=False)

    model = FallTransformerModel().to(device)

    num_pos = sum(train_labels)
    num_neg = len(train_labels) - num_pos
    pos_weight = torch.tensor([num_neg / num_pos]).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.5
    )

    best_acc = 0

    for epoch in range(EPOCHS):

        # TRAIN
        model.train()
        train_loss = 0

        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)

            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item()

        # VALIDATION
        model.eval()
        val_loss = 0
        v_logits = []
        v_truths = []

        with torch.no_grad():
            for bx, by in test_loader:
                bx, by = bx.to(device), by.to(device)

                logits = model(bx)
                loss = criterion(logits, by)

                val_loss += loss.item()

                v_logits.extend(logits.cpu().numpy().flatten())
                v_truths.extend(by.cpu().numpy().flatten())

        v_logits_np = np.array(v_logits).reshape(-1,1)
        v_truths_np = np.array(v_truths)

        # ======================
        # PLATT SCALING
        # ======================
        platt = LogisticRegression()
        platt.fit(v_logits_np, v_truths_np)
        calibrated_probs = platt.predict_proba(v_logits_np)[:,1]

        v_preds = (calibrated_probs > 0.5).astype(int)

        acc = accuracy_score(v_truths_np, v_preds)
        prec = precision_score(v_truths_np, v_preds, zero_division=0)
        rec = recall_score(v_truths_np, v_preds, zero_division=0)
        f1 = f1_score(v_truths_np, v_preds, zero_division=0)
        b_acc = balanced_accuracy_score(v_truths_np, v_preds)

        try:
            auc = roc_auc_score(v_truths_np, calibrated_probs)
        except:
            auc = 0.5

        scheduler.step(acc)

        print(f"\nEpoch {epoch+1:02d}")
        print(f"Logits range: {np.min(v_logits_np):.3f} / {np.max(v_logits_np):.3f}")
        print(f"Calibrated Prob range: {np.min(calibrated_probs):.3f} / {np.max(calibrated_probs):.3f}")
        print("Class means:")
        print("NoFall:", np.mean(calibrated_probs[v_truths_np==0]))
        print("Fall  :", np.mean(calibrated_probs[v_truths_np==1]))
        print(f"Val-Acc: {acc:.4f} | AUC: {auc:.4f}")

        # PLOT DISTRIBUTION
        plt.figure(figsize=(6,4))
        plt.hist(calibrated_probs[v_truths_np==0], bins=30, alpha=0.6, label="NoFall")
        plt.hist(calibrated_probs[v_truths_np==1], bins=30, alpha=0.6, label="Fall")
        plt.legend()
        plt.title(f"Fold {fold+1} Epoch {epoch+1}")
        plt.xlabel("Calibrated Probability")
        plt.ylabel("Count")
        plt.tight_layout()
        plt.show()

        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(),
                       os.path.join(MODEL_DIR, f"best_fold_{fold+1}.pth"))

    print(f"[✓] Fold {fold+1} Best Acc: {best_acc:.4f}")