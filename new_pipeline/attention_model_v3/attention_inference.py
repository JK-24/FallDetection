import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
import joblib
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# =========================================================
# CONFIG
# =========================================================
SEQUENCE_CSV = r"new_pipeline\attention_model_v2\sequence_features.csv"
STAGE3_DIR = r"E:\JK\misc\New pipeline run 1\stage3_output"
STAGE4_DIR = r"E:\JK\misc\New pipeline run 1\stage4_output"
MODEL_DIR = r"new_pipeline\attention_model_v2"

DEVICE = torch.device("cpu")
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1)

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
ENVIRONMENTS = ["bed", "chair", "stand"]

# =========================================================
# MODEL ARCHITECTURE
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
# GLOBALS FOR WORKERS
# =========================================================
MODELS = {}
SCALERS = {}
TEMPERATURES = {}


def init_worker():
    global MODELS, SCALERS, TEMPERATURES

    for env in ENVIRONMENTS:
        model = BiLSTMAttention()
        model.load_state_dict(torch.load(
            os.path.join(MODEL_DIR, f"model_{env}.pth"),
            map_location="cpu"
        ))
        model.eval()

        MODELS[env] = model
        SCALERS[env] = joblib.load(
            os.path.join(MODEL_DIR, f"scaler_{env}.pkl")
        )

        with open(os.path.join(MODEL_DIR, f"config_{env}.json"), "r") as f:
            config = json.load(f)

        TEMPERATURES[env] = config["temperature"]


# =========================================================
# WORKER FUNCTION
# =========================================================
def process_video(task):
    video_id, features_np = task

    probs = []

    for env in ENVIRONMENTS:
        # Pass DataFrame with column names to match how scaler was fitted
        scaled = SCALERS[env].transform(pd.DataFrame(features_np, columns=FEATURE_COLUMNS))
        seq = torch.tensor(scaled, dtype=torch.float32).unsqueeze(0)
        mask = torch.ones(seq.shape[:2], dtype=torch.bool)

        with torch.no_grad():
            logits = MODELS[env](seq, mask).item()

        calibrated = logits / TEMPERATURES[env]
        prob = torch.sigmoid(torch.tensor(calibrated)).item()
        probs.append(prob)

    ensemble_prob = float(np.mean(probs))
    final_decision = int(ensemble_prob > 0.5)

    relative_path = video_id + ".json"
    input_json_path = os.path.join(STAGE3_DIR, relative_path)
    output_json_path = os.path.join(STAGE4_DIR, relative_path)

    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    if not os.path.exists(input_json_path):
        return f"Missing JSON: {video_id}"

    with open(input_json_path, "r") as f:
        data = json.load(f)

    data["stage4"] = {
        "prob_model_bed": probs[0],
        "prob_model_chair": probs[1],
        "prob_model_stand": probs[2],
        "ensemble_probability": ensemble_prob,
        "final_decision": final_decision
    }

    with open(output_json_path, "w") as f:
        json.dump(data, f, indent=4)

    return f"Processed: {video_id}"

# =========================================================
# MAIN EXECUTION
# =========================================================
def main():

    print("\n========== SANITY CHECKS ==========")

    # Check model files
    for env in ENVIRONMENTS:
        print(f"\nChecking files for {env}:")
        print("  Model:", os.path.exists(os.path.join(MODEL_DIR, f"model_{env}.pth")))
        print("  Scaler:", os.path.exists(os.path.join(MODEL_DIR, f"scaler_{env}.pkl")))
        print("  Config:", os.path.exists(os.path.join(MODEL_DIR, f"config_{env}.json")))

    # Load CSV
    df = pd.read_csv(SEQUENCE_CSV)
    df = df.sort_values(["video_id", "frame_index"])
    video_groups = df.groupby("video_id")

    print("\nTotal videos in sequence CSV:", len(video_groups))

    # Count Stage3 JSON
    json_count = 0
    for root, _, files in os.walk(STAGE3_DIR):
        json_count += len([f for f in files if f.endswith(".json")])

    print("Total JSON files in stage3:", json_count)
    print("Multiprocessing workers:", NUM_WORKERS)

    print("\n===================================")
    input("Press ENTER to continue with inference...")

    # Prepare tasks
    tasks = [
        (video_id, group[FEATURE_COLUMNS].values)
        for video_id, group in video_groups
    ]

    print("\nStarting parallel inference...\n")

    with ProcessPoolExecutor(
        max_workers=NUM_WORKERS,
        initializer=init_worker
    ) as executor:

        for result in executor.map(process_video, tasks):
            print(result)

    print("\nStage4 inference completed successfully.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()