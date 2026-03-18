# =========================================================
# STAGE-4 v2 INFERENCE SCRIPT
# =========================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import joblib
import json
import os

# =========================================================
# CONFIG
# =========================================================

INPUT_CSV   = r"new_pipeline\attention_model_v4\fine tuning\sequence_features_caucafall.csv"
MODEL_PATH  = r"new_pipeline\attention_model_v4\fine tuning\model.pth"
SCALER_PATH = r"new_pipeline\attention_model_v4\fine tuning\scaler.pkl"
CONFIG_PATH = r"new_pipeline\attention_model_v4\fine tuning\config.json"

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

# =========================================================
# MODEL DEFINITION (MUST MATCH TRAINING)
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

        context = torch.sum(lstm_out * attn_weights.unsqueeze(-1), dim=1)
        logits = self.fc(context)
        return logits.squeeze()


# =========================================================
# LOAD MODEL + SCALER + CONFIG
# =========================================================

model = BiLSTMAttention().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

scaler = joblib.load(SCALER_PATH)

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

threshold = config["threshold"]

print(f"Loaded model. Threshold = {threshold:.3f}")

# =========================================================
# LOAD INPUT CSV
# =========================================================

df = pd.read_csv(INPUT_CSV)
df = df.sort_values(["video_id", "frame_index"])

# Scale features
df.loc[:, FEATURE_COLUMNS] = scaler.transform(df[FEATURE_COLUMNS])

# =========================================================
# BUILD SEQUENCES
# =========================================================

def build_sequences(dataframe):
    sequences = {}
    for vid, group in dataframe.groupby("video_id"):
        sequences[vid] = torch.tensor(
            group[FEATURE_COLUMNS].values,
            dtype=torch.float32
        )
    return sequences

sequences = build_sequences(df)

# =========================================================
# RUN INFERENCE
# =========================================================

results = []

with torch.no_grad():
    for vid, seq in sequences.items():

        seq = seq.unsqueeze(0).to(DEVICE)
        mask = torch.ones((1, seq.shape[1]), dtype=torch.bool).to(DEVICE)

        logits = model(seq, mask)
        prob = torch.sigmoid(logits).item()

        prediction = 1 if prob >= threshold else 0

        results.append({
            "video_id": vid,
            "probability_fall": prob,
            "prediction": prediction
        })

# =========================================================
# OUTPUT
# =========================================================

results_df = pd.DataFrame(results)
print("\n===== INFERENCE RESULTS =====")
print(results_df)

# Optional: save results
results_df.to_csv("stage4_v2_predictions.csv", index=False)
print("\nSaved predictions to stage4_v2_predictions.csv")