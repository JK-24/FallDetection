"""
run_stage3_stage4.py
====================
Stage-3 & Stage-4 batch inference for FallVision dataset.

Fill in the PATH CONFIGURATION section below before running.

Usage:
    python run_stage3_stage4.py
"""

import os
import sys
import json
import csv
import time
import logging
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2


# ═════════════════════════════════════════════════════════════════════════════
# PATH CONFIGURATION  —  fill these in before running
# ═════════════════════════════════════════════════════════════════════════════

# Root of FallVision dataset  (must contain fall/ and no_fall/ subdirs)
DATA_ROOT = ""

# Root directory containing existing Stage-1 JSON files
S1_ROOT = ""

# Root directory containing existing Stage-2 JSON files
S2_ROOT = ""

# Path to Stage-3 FallAttentionModel checkpoint  (.pth file)
S3_CKPT = ""

# Output root where Stage-3 JSON files will be written
S3_ROOT = ""

# Output root where Stage-4 JSON files will be written
S4_ROOT = ""

# Output path for stage5_input.csv
CSV_OUT = ""

# Set to True to run MiDaS depth estimation (slower but complete features)
# Set to False to skip MiDaS (depth fields will be 0.0, faster)
USE_MIDAS = True

# Number of parallel worker threads
NUM_WORKERS = 4

# ═════════════════════════════════════════════════════════════════════════════


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("inference_log.txt", mode="w"),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
SEQ_LEN      = 16
NUM_FEATURES = 38   # 17 keypoints x 2 coords + 4 physics features
CLIP_LEN     = 32
LABEL_MAP    = {"fall": 1, "no_fall": 0}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL  —  Stage-3 UPDATED TRANSFORMER MODEL
# ─────────────────────────────────────────────────────────────────────────────

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

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )

        self.temporal_attention = nn.Linear(NUM_FEATURES, 1)

        self.fc = nn.Sequential(
            nn.Linear(NUM_FEATURES, 64),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(64, 1)
        )

    def forward(self, x, mask=None):

        x = self.encoder(x, src_key_padding_mask=mask)

        attn_weights = torch.softmax(self.temporal_attention(x), dim=1)

        if mask is not None:
            valid_mask = (~mask).unsqueeze(-1).float()
            attn_weights = attn_weights * valid_mask
            attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-8)

        x = torch.sum(attn_weights * x, dim=1)

        logits = self.fc(x)

        return logits, attn_weights
    

# ─────────────────────────────────────────────────────────────────────────────
# HELPER  —  build reverse frame index map
# ─────────────────────────────────────────────────────────────────────────────
def build_reverse_clip_map(clip_indices, sorted_image_paths):
    """
    How frame indices work (verified against real JSON files):

      clip_indices = [0, 5, 10, ..., 62, 67, 72, ...]
                                      ^ pos 12
      Position 12 in clip_indices has value 62.
      Position 12 in sorted_image_paths is 012.jpg.

      So frame_idx=62 in Stage-2 JSON  ->  reverse_map[62]  ->  012.jpg

    Returns dict:  original_frame_number -> image_file_path
    """
    reverse_map = {}
    for position, orig_frame_num in enumerate(clip_indices):
        if position < len(sorted_image_paths):
            reverse_map[int(orig_frame_num)] = sorted_image_paths[position]
    return reverse_map


# ─────────────────────────────────────────────────────────────────────────────
# STAGE-3  —  build padded sequence tensor from Stage-2 JSON
# ─────────────────────────────────────────────────────────────────────────────
def build_sequence_tensor(stage2_data, device):
    """
    Converts stage2 frame_data into a padded tensor for FallAttentionModel.
    Returns: (tensor, pad_mask, frame_entries, frame_indices, real_len)
    """
    frames        = stage2_data.get("frame_data", [])
    sequence      = []
    frame_indices = []

    for fdata in frames:
        frame_indices.append(fdata.get("frame_idx", -1))
        kp   = np.array(fdata["keypoints"], dtype=np.float32).flatten()  # 34-dim
        feat = fdata.get("features", {})
        fvec = [
            float(feat.get("tilt_angle",        0) or 0) / 180.0,
            float(feat.get("vertical_velocity", 0) or 0) / 100.0,
            float(feat.get("h_w_ratio",         0) or 0) / 5.0,
            float(feat.get("ground_proximity",  0) or 0),
        ]
        combined = np.nan_to_num(np.concatenate([kp, fvec]), nan=0.0)  # 38-dim
        sequence.append(combined)

    real_len = len(sequence)
    pad_mask = torch.zeros(SEQ_LEN, dtype=torch.bool)

    if real_len == 0:
        seq_np      = np.zeros((SEQ_LEN, NUM_FEATURES), dtype=np.float32)
        pad_mask[:] = True
        real_len    = 0
    else:
        seq_np = np.array(sequence, dtype=np.float32)
        if real_len < SEQ_LEN:
            pad    = np.zeros((SEQ_LEN - real_len, NUM_FEATURES), dtype=np.float32)
            seq_np = np.vstack([seq_np, pad])
            pad_mask[real_len:] = True
        else:
            seq_np   = seq_np[:SEQ_LEN]
            real_len = SEQ_LEN

    tensor = torch.FloatTensor(seq_np).unsqueeze(0).to(device)  # (1, SEQ_LEN, 38)
    mask   = pad_mask.unsqueeze(0).to(device)                   # (1, SEQ_LEN)
    return tensor, mask, frames, frame_indices, real_len


# ─────────────────────────────────────────────────────────────────────────────
# STAGE-4  —  physics helpers
# ─────────────────────────────────────────────────────────────────────────────
def get_torso_centroid(keypoints, H, W):
    torso_ids = [5, 6, 11, 12]
    xs = [keypoints[t][0] * W for t in torso_ids]
    ys = [keypoints[t][1] * H for t in torso_ids]
    return int(np.median(xs)), int(np.median(ys))


def weighted_avg(values, weights):
    v = np.array(values,  dtype=np.float64)
    w = np.array(weights, dtype=np.float64)
    return float(np.sum(v * w) / (np.sum(w) + 1e-9))


def run_midas_on_image(img_rgb, midas, transform, device):
    """Run MiDaS on one RGB image. Returns normalised depth map (H x W, float 0-1)."""
    input_b = transform(img_rgb).to(device)
    with torch.no_grad():
        pred = midas(input_b)
        pred = F.interpolate(
            pred.unsqueeze(1),
            size=img_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()
    depth        = pred.cpu().numpy()
    d_min, d_max = depth.min(), depth.max()
    return (depth - d_min) / (d_max - d_min + 1e-6)


# ─────────────────────────────────────────────────────────────────────────────
# CORE  —  process one video  (Stage-3 + Stage-4)
# ─────────────────────────────────────────────────────────────────────────────
def process_one_video(vk, stage3_model, midas, midas_transform, device):
    """
    vk (video_key) dict:
      label_int, label_str, activity, group, video_name,
      images_dir,
      s1_json_path, s2_json_path,
      s3_json_path, s4_json_path
    """
    video_name = vk["video_name"]

    # Load JSONs
    with open(vk["s1_json_path"], "r") as fh:
        stage1_data = json.load(fh)

    with open(vk["s2_json_path"], "r") as fh:
        stage2_data = json.load(fh)

    clip_indices = stage1_data.get("clip_indices", list(range(CLIP_LEN)))

    # Build frame_idx -> image path reverse map
    img_files = sorted([
        f for f in os.listdir(vk["images_dir"])
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    sorted_img_paths = [os.path.join(vk["images_dir"], f) for f in img_files]
    reverse_map      = build_reverse_clip_map(clip_indices, sorted_img_paths)

    # ── STAGE 3 ───────────────────────────────────────────────────────────────
    tensor, mask, frame_entries, frame_indices, real_len = build_sequence_tensor(
        stage2_data, device
    )

    with torch.no_grad():
        logits, attn_weights = stage3_model(tensor, mask=mask)

    probs = torch.sigmoid(logits)
    fall_prob = float(probs.item())
    risk_level = "HIGH" if fall_prob >= 0.75 else ("MEDIUM" if fall_prob >= 0.45 else "LOW")

    # Compute per-frame attention scores
    aw = attn_weights.cpu().numpy()
    if aw.ndim == 4:
        aw = np.mean(aw, axis=1)   # average over heads -> (1, SEQ, SEQ)
    aw         = aw.squeeze(0)     # (SEQ, SEQ)
    total_attn = np.sum(aw, axis=0)
    attn_real  = total_attn[:real_len]
    attn_norm  = attn_real / (np.sum(attn_real) + 1e-8)

    stage3_out = {
        "video_name":        video_name,
        "fall_probability":  round(fall_prob, 4),
        "risk_level":        risk_level,
        "keyframe_indices":  [int(i) for i in frame_indices[:real_len]],
        "attention_weights": [round(float(w), 4) for w in attn_norm],
    }

    os.makedirs(os.path.dirname(vk["s3_json_path"]), exist_ok=True)
    with open(vk["s3_json_path"], "w") as fh:
        json.dump(stage3_out, fh, indent=4)

    # ── STAGE 4 ───────────────────────────────────────────────────────────────
    keyframe_indices = stage3_out["keyframe_indices"]
    attn_w           = np.array(stage3_out["attention_weights"])
    attn_w           = attn_w / (attn_w.sum() + 1e-9)

    frame_lookup = {fd["frame_idx"]: fd for fd in stage2_data["frame_data"]}

    tilt_list  = []
    hwr_list   = []
    vel_list   = []
    gp_list    = []
    depth_list = []

    for kf_orig_idx in keyframe_indices:
        if kf_orig_idx not in frame_lookup:
            log.debug(f"  frame_idx={kf_orig_idx} not in stage2 frame_lookup ({video_name})")
            continue

        fd       = frame_lookup[kf_orig_idx]
        features = fd["features"]

        tilt_list.append(float(features.get("tilt_angle",        0) or 0))
        hwr_list.append( float(features.get("h_w_ratio",         0) or 0))
        vel_list.append( float(features.get("vertical_velocity", 0) or 0))
        gp_list.append(  float(features.get("ground_proximity",  0) or 0))

        # MiDaS depth
        depth_val = 0.0
        if USE_MIDAS and midas is not None:
            img_path = reverse_map.get(int(kf_orig_idx))
            if img_path is None:
                log.debug(f"  MiDaS: frame_idx={kf_orig_idx} not in reverse_map ({video_name})")
            elif not os.path.exists(img_path):
                log.debug(f"  MiDaS: image not found: {img_path}")
            else:
                img_bgr = cv2.imread(img_path)
                if img_bgr is not None:
                    img_rgb   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    depth_map = run_midas_on_image(img_rgb, midas, midas_transform, device)
                    h, w      = depth_map.shape
                    cx, cy    = get_torso_centroid(fd["keypoints"], h, w)
                    if 0 <= cx < w and 0 <= cy < h:
                        depth_val = float(depth_map[cy, cx])

        depth_list.append(depth_val)

    # Fallback for degenerate clips
    if not tilt_list:
        log.warning(f"  No matching keyframes found in stage2 for {video_name} — using zero fallback.")
        tilt_list  = [0.0]; hwr_list  = [1.0]
        vel_list   = [0.0]; gp_list   = [0.5]
        depth_list = [0.0]

    n     = len(tilt_list)
    w_eff = attn_w[:n]
    w_eff = w_eff / (w_eff.sum() + 1e-9)

    max_tilt   = float(np.max(tilt_list))
    min_hwr    = float(np.min(hwr_list))
    delta_tilt = float(tilt_list[-1] - tilt_list[0])
    delta_hwr  = float(hwr_list[-1]  - hwr_list[0])

    weighted_tilt = weighted_avg(tilt_list, w_eff)
    weighted_vel  = weighted_avg(vel_list,  w_eff)
    weighted_gp   = weighted_avg(gp_list,   w_eff)

    d_arr          = np.array(depth_list, dtype=np.float64)
    depth_variance = float(np.var(d_arr))
    depth_range    = float(np.max(d_arr) - np.min(d_arr))
    depth_drop     = float(np.mean(d_arr[:-1]) - d_arr[-1]) if len(d_arr) > 1 else 0.0

    stage4_out = {
        "video_name":                video_name,
        "label":                     vk["label_int"],
        "label_str":                 vk["label_str"],
        "max_tilt":                  round(max_tilt,        6),
        "delta_tilt":                round(delta_tilt,      6),
        "min_h_w_ratio":             round(min_hwr,         6),
        "delta_h_w_ratio":           round(delta_hwr,       6),
        "weighted_tilt":             round(weighted_tilt,   6),
        "weighted_velocity":         round(weighted_vel,    6),
        "weighted_ground_proximity": round(weighted_gp,     6),
        "depth_drop":                round(depth_drop,      6),
        "depth_variance":            round(depth_variance,  6),
        "depth_range":               round(depth_range,     6),
        "stage3_fall_prob":          round(fall_prob,       6),
    }

    os.makedirs(os.path.dirname(vk["s4_json_path"]), exist_ok=True)
    with open(vk["s4_json_path"], "w") as fh:
        json.dump(stage4_out, fh, indent=4)

    return stage4_out


# ─────────────────────────────────────────────────────────────────────────────
# DATASET DISCOVERY  —  positional JSON matching
# ─────────────────────────────────────────────────────────────────────────────
def discover_videos():
    """
    Walk FallVision/ and pair every video directory with its Stage-1 and
    Stage-2 JSON files using positional (alphabetical order) matching.

    Within each group directory:
      - Sort video subdirs alphabetically
      - Sort Stage-1 JSONs alphabetically
      - Sort Stage-2 JSONs alphabetically
      - Zip all three by position

    Returns: (video_keys list, skipped list)
    """
    video_keys = []
    skipped    = []

    data_root = Path(DATA_ROOT)
    s1_root   = Path(S1_ROOT)
    s2_root   = Path(S2_ROOT)
    s3_root   = Path(S3_ROOT)
    s4_root   = Path(S4_ROOT)

    for label_dir in sorted(data_root.iterdir()):
        if not label_dir.is_dir():
            continue
        label_str = label_dir.name.lower()
        if label_str not in LABEL_MAP:
            log.warning(f"Unknown label dir '{label_dir.name}' — skipping.")
            continue
        label_int = LABEL_MAP[label_str]

        for activity_dir in sorted(label_dir.iterdir()):
            if not activity_dir.is_dir():
                continue
            activity = activity_dir.name.lower()

            for group_dir in sorted(activity_dir.iterdir()):
                if not group_dir.is_dir():
                    continue
                group = group_dir.name

                # Collect video directories (one per video clip)
                video_dirs = sorted([d for d in group_dir.iterdir() if d.is_dir()])
                if not video_dirs:
                    log.warning(f"No video subdirs in {group_dir} — skipping group.")
                    continue

                # Mirror path in JSON trees
                rel_group    = Path(label_str) / activity / group
                s1_group_dir = s1_root / rel_group
                s2_group_dir = s2_root / rel_group

                # Both JSON group dirs must exist
                missing_dirs = []
                if not s1_group_dir.exists():
                    missing_dirs.append(str(s1_group_dir))
                if not s2_group_dir.exists():
                    missing_dirs.append(str(s2_group_dir))

                if missing_dirs:
                    msg = f"SKIP group {rel_group} — JSON dirs not found: {missing_dirs}"
                    log.warning(msg)
                    for vd in video_dirs:
                        skipped.append({"path": str(rel_group / vd.name), "reason": msg})
                    continue

                s1_jsons = sorted(s1_group_dir.glob("*.json"))
                s2_jsons = sorted(s2_group_dir.glob("*.json"))

                # Warn on count mismatches
                for tag, jsons in [("Stage-1", s1_jsons), ("Stage-2", s2_jsons)]:
                    if len(jsons) != len(video_dirs):
                        log.warning(
                            f"Count mismatch — {rel_group}: "
                            f"{len(video_dirs)} video dirs vs {len(jsons)} {tag} JSONs. "
                            f"Pairing up to min count."
                        )

                n_pairs = min(len(video_dirs), len(s1_jsons), len(s2_jsons))

                for pos in range(n_pairs):
                    vid_dir    = video_dirs[pos]
                    s1_json    = s1_jsons[pos]
                    s2_json    = s2_jsons[pos]
                    video_name = vid_dir.name

                    s3_json = s3_root / rel_group / f"{video_name}.json"
                    s4_json = s4_root / rel_group / f"{video_name}.json"

                    video_keys.append({
                        "label_int":    label_int,
                        "label_str":    label_str,
                        "activity":     activity,
                        "group":        group,
                        "video_name":   video_name,
                        "images_dir":   str(vid_dir),
                        "s1_json_path": str(s1_json),
                        "s2_json_path": str(s2_json),
                        "s3_json_path": str(s3_json),
                        "s4_json_path": str(s4_json),
                    })

    return video_keys, skipped


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # Validate that all required paths have been filled in
    required = {
        "DATA_ROOT": DATA_ROOT,
        "S1_ROOT":   S1_ROOT,
        "S2_ROOT":   S2_ROOT,
        "S3_CKPT":   S3_CKPT,
        "S3_ROOT":   S3_ROOT,
        "S4_ROOT":   S4_ROOT,
        "CSV_OUT":   CSV_OUT,
    }
    empty = [name for name, val in required.items() if not val]
    if empty:
        log.error(f"The following path variables are empty — please fill them in: {empty}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info("=" * 62)
    log.info("  Stage-3 & Stage-4 Batch Inference")
    log.info("=" * 62)
    log.info(f"  DATA_ROOT   :  {DATA_ROOT}")
    log.info(f"  S1_ROOT     :  {S1_ROOT}")
    log.info(f"  S2_ROOT     :  {S2_ROOT}")
    log.info(f"  S3_CKPT     :  {S3_CKPT}")
    log.info(f"  S3_ROOT     :  {S3_ROOT}")
    log.info(f"  S4_ROOT     :  {S4_ROOT}")
    log.info(f"  CSV_OUT     :  {CSV_OUT}")
    log.info(f"  USE_MIDAS   :  {USE_MIDAS}")
    log.info(f"  NUM_WORKERS :  {NUM_WORKERS}")
    log.info(f"  Device      :  {device}")
    if torch.cuda.is_available():
        log.info(f"  GPU         :  {torch.cuda.get_device_name(0)}")
    log.info("=" * 62)

   # ───── Stage-3 model loading (UPDATED ARCHITECTURE) ─────
    log.info("Loading Stage-3 FallTransformerModel ...")
    stage3_model = FallTransformerModel().to(device)
    stage3_model.load_state_dict(torch.load(S3_CKPT, map_location=device))
    stage3_model.eval()
    for m in stage3_model.modules():
        if isinstance(m, nn.Dropout):
            m.eval()
    log.info("Stage-3 model loaded.")

    # Load MiDaS (optional)
    midas = midas_transform = None
    if USE_MIDAS:
        log.info("Loading MiDaS ...")
        midas           = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
        midas.to(device).eval()
        _transforms     = torch.hub.load("intel-isl/MiDaS", "transforms")
        midas_transform = _transforms.small_transform
        log.info("MiDaS loaded.")
    else:
        log.info("MiDaS skipped (USE_MIDAS=False) — depth fields will be 0.0")

    # Discover all videos
    log.info(f"Scanning dataset: {DATA_ROOT}")
    video_keys, skipped = discover_videos()
    log.info(f"Videos to process : {len(video_keys)}")
    log.info(f"Skipped groups    : {len(skipped)}")

    if not video_keys:
        log.error(
            "No videos found. Verify DATA_ROOT, S1_ROOT, S2_ROOT point to "
            "the correct directories with the expected structure."
        )
        sys.exit(1)

    # Write skip log
    if skipped:
        skip_log = Path(CSV_OUT).parent / "skipped_videos.json"
        skip_log.parent.mkdir(parents=True, exist_ok=True)
        with open(skip_log, "w") as fh:
            json.dump(skipped, fh, indent=2)
        log.info(f"Skip log -> {skip_log}")


    # ─────────────────────────────────────────────────────────
    # NEW CSV STORAGE LISTS (DOES NOT REMOVE ORIGINAL)
    # ─────────────────────────────────────────────────────────
    stage3_rows = []
    stage4_rows = []

    stage3_header = [
        "video_name",
        "label",
        "stage3_fall_prob",
        "uncertainty",
        "attn_mean",
        "attn_max",
        "attn_variance",
        "attn_entropy"
    ]

    stage4_header = [
        "video_name",
        "label",
        "max_tilt",
        "delta_tilt",
        "min_h_w_ratio",
        "delta_h_w_ratio",
        "weighted_tilt",
        "weighted_velocity",
        "weighted_ground_proximity",
        "depth_drop",
        "depth_variance",
        "depth_range",
        # derived
        "abs_delta_tilt",
        "abs_delta_hwr",
        "abs_velocity",
        "tilt_energy",
        "velocity_energy",
        "combined_energy",
        "orientation_velocity_coupling",
        "horizontal_depth_coupling",
        "tilt_over_hwr",
        "depth_stability_ratio"
    ]

    # Threading state
    CSV_HEADER = [
        "video_name",
        "label_str",
        "activity",
        "group",
        "max_tilt",
        "delta_tilt",
        "min_h_w_ratio",
        "delta_h_w_ratio",
        "weighted_tilt",
        "weighted_velocity",
        "weighted_ground_proximity",
        "depth_drop",
        "depth_variance",
        "depth_range",
        "stage3_fall_prob",
        "LABEL",   # 1 = fall,  0 = no_fall
    ]

    csv_lock = Lock()
    csv_rows = []
    errors   = []
    err_lock = Lock()
    counter  = [0]
    cnt_lock = Lock()
    total    = len(video_keys)
    t_start  = time.time()

    def worker(vk):
        try:
            s4 = process_one_video(vk, stage3_model, midas, midas_transform, device)
            
            # ───── Stage-3 CSV FEATURES ─────
            fall_prob = s4["stage3_fall_prob"]
            uncertainty = abs(fall_prob - 0.5)

            # attention already normalized inside pipeline JSON
            # reload stage3 json to compute stats
            with open(vk["s3_json_path"], "r") as f:
                s3_json = json.load(f)

            attn_vals = np.array(s3_json["attention_weights"])

            attn_mean = float(np.mean(attn_vals))
            attn_max = float(np.max(attn_vals))
            attn_var = float(np.var(attn_vals))
            attn_entropy = float(-np.sum(attn_vals * np.log(attn_vals + 1e-8)))


            # ───── Stage-4 DERIVED FEATURES ─────
            abs_delta_tilt = abs(s4["delta_tilt"])
            abs_delta_hwr  = abs(s4["delta_h_w_ratio"])
            abs_velocity   = abs(s4["weighted_velocity"])

            tilt_energy = s4["max_tilt"] ** 2
            velocity_energy = s4["weighted_velocity"] ** 2
            combined_energy = tilt_energy + velocity_energy

            orientation_velocity_coupling = s4["max_tilt"] * s4["weighted_velocity"]
            horizontal_depth_coupling = (1 / (s4["min_h_w_ratio"] + 1e-6)) * s4["depth_drop"]
            tilt_over_hwr = s4["max_tilt"] / (s4["min_h_w_ratio"] + 1e-6)
            depth_stability_ratio = s4["depth_drop"] / (s4["depth_variance"] + 1e-6)

            row = [
                vk["video_name"],
                vk["label_str"],
                vk["activity"],
                vk["group"],
                s4["max_tilt"],
                s4["delta_tilt"],
                s4["min_h_w_ratio"],
                s4["delta_h_w_ratio"],
                s4["weighted_tilt"],
                s4["weighted_velocity"],
                s4["weighted_ground_proximity"],
                s4["depth_drop"],
                s4["depth_variance"],
                s4["depth_range"],
                s4["stage3_fall_prob"],
                vk["label_int"],
            ]

            with csv_lock:
                csv_rows.append(row)
                
                stage3_rows.append([
                    vk["video_name"],
                    vk["label_int"],
                    fall_prob,
                    uncertainty,
                    attn_mean,
                    attn_max,
                    attn_var,
                    attn_entropy
                ])
                stage4_rows.append([
                    vk["video_name"],
                    vk["label_int"],
                    s4["max_tilt"],
                    s4["delta_tilt"],
                    s4["min_h_w_ratio"],
                    s4["delta_h_w_ratio"],
                    s4["weighted_tilt"],
                    s4["weighted_velocity"],
                    s4["weighted_ground_proximity"],
                    s4["depth_drop"],
                    s4["depth_variance"],
                    s4["depth_range"],
                    abs_delta_tilt,
                    abs_delta_hwr,
                    abs_velocity,
                    tilt_energy,
                    velocity_energy,
                    combined_energy,
                    orientation_velocity_coupling,
                    horizontal_depth_coupling,
                    tilt_over_hwr,
                    depth_stability_ratio
                ])
                
                
            with cnt_lock:
                counter[0] += 1
                n = counter[0]

            elapsed = time.time() - t_start
            eta     = (elapsed / n) * (total - n) if n else 0
            log.info(
                f"[{n:>5}/{total}]  "
                f"{vk['label_str']}/{vk['activity']}/{vk['group']}/{vk['video_name']}"
                f"  P(fall)={s4['stage3_fall_prob']:.4f}"
                f"  ETA {eta/60:.1f}min"
            )

        except Exception as exc:
            with err_lock:
                errors.append({
                    "path":  f"{vk['label_str']}/{vk['activity']}/{vk['group']}/{vk['video_name']}",
                    "error": str(exc),
                    "trace": traceback.format_exc(),
                })
            with cnt_lock:
                counter[0] += 1
                n = counter[0]
            log.error(
                f"[{n:>5}/{total}]  ERROR  "
                f"{vk['label_str']}/{vk['activity']}/{vk['group']}/{vk['video_name']}"
                f"  -- {exc}"
            )

    # GPU ops are serialised by CUDA internally (safe in eval mode).
    # Threads parallelise JSON I/O, image loading, and CPU preprocessing.
    log.info(f"Launching {NUM_WORKERS} worker threads ...")
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = {pool.submit(worker, vk): vk for vk in video_keys}
        for _ in as_completed(futures):
            pass   # progress logged inside worker

    # Write CSV
    csv_path = Path(CSV_OUT)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    csv_rows.sort(key=lambda r: (0 if r[1] == "fall" else 1, r[0]))  # fall first, then no_fall

    with open(csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(CSV_HEADER)
        writer.writerows(csv_rows)

    log.info(f"CSV saved -> {csv_path}  ({len(csv_rows)} rows)")

    # ─────────────────────────────────────────────────────────
    # WRITE NEW ANALYSIS CSVs
    # ─────────────────────────────────────────────────────────

    stage3_csv_path = csv_path.parent / "stage3_analysis.csv"
    with open(stage3_csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(stage3_header)
        writer.writerows(stage3_rows)

    stage4_csv_path = csv_path.parent / "stage4_analysis.csv"
    with open(stage4_csv_path, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(stage4_header)
        writer.writerows(stage4_rows)

    log.info(f"Stage-3 analysis CSV saved -> {stage3_csv_path}")
    log.info(f"Stage-4 analysis CSV saved -> {stage4_csv_path}")

    # Write error log
    if errors:
        err_path = csv_path.parent / "error_log.json"
        with open(err_path, "w") as fh:
            json.dump(errors, fh, indent=2)
        log.warning(f"{len(errors)} errors logged -> {err_path}")

    elapsed = time.time() - t_start
    log.info("=" * 62)
    log.info(f"  COMPLETE  |  Success : {len(csv_rows)}/{total}")
    log.info(f"            |  Errors  : {len(errors)}")
    log.info(f"            |  Time    : {elapsed/60:.1f} min")
    log.info(f"            |  CSV     : {csv_path}")
    log.info("=" * 62)


if __name__ == "__main__":
    main()