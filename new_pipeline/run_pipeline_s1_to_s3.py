"""
run_pipeline_s1_to_s3.py
=========================
Batch pipeline: Stage 1 (3D-CNN saliency) -> Stage 2 (YOLO pose) ->
Stage 3 (MiDaS depth).

Saves two outputs per video:
  stage2_output/<label>/<activity>/<group>/<video_stem>.json
  stage3_output/<label>/<activity>/<group>/<video_stem>.json  <- cumulative

The Stage 3 JSON is the sole input to Stage 4 (attention model).
The summaries contain EVERY derivable feature so an AUC analysis
can be run to decide which features to forward to Stage 4.

════════════════════════════════════════════════════════════════════
FEATURES COMPUTED PER PHYSICS/DEPTH SIGNAL  (43 per signal)
════════════════════════════════════════════════════════════════════

Basic stats (7):
  _max, _min, _mean, _std, _median, _range, _sum

Delta / trend (4):
  _delta              last - first
  _abs_delta          |last - first|
  _norm_delta         delta / range  (how big the change is relative to spread)
  _linear_slope       polyfit degree-1 slope across window

Temporal position (4):
  _peak_pos           normalised argmax position  [0=start, 1=end]
  _trough_pos         normalised argmin position
  _time_to_peak       alias of peak_pos (explicit for readability)
  _peak_trough_gap    |peak_pos - trough_pos|

Acceleration pattern (4):
  _pre_peak_slope     mean frame diff before peak
  _post_peak_slope    mean frame diff after peak
  _slope_ratio        pre / post (captures rise-then-fall asymmetry)
  _curvature          2nd-order polyfit coefficient (parabolic shape)

Half-split (6):
  _first_half_mean, _second_half_mean
  _first_half_std,  _second_half_std
  _half_mean_diff     second_half_mean - first_half_mean
  _half_mean_ratio    second_half_mean / (first_half_mean + 1e-9)

Third-split (3):
  _first_third_mean, _mid_third_mean, _last_third_mean

Energy / distribution (4):
  _energy             sum of squares
  _skewness
  _kurtosis
  _iqr                interquartile range

Crossing / threshold (3):
  _mean_crossings     times the signal crosses its own mean
  _zero_crossings     times the signal crosses zero  (useful for velocity)
  _above_mean_ratio   fraction of frames above mean

Change rate (4):
  _mean_abs_diff      mean |x[i+1] - x[i]|
  _max_diff           max  |x[i+1] - x[i]|
  _total_variation    sum  |x[i+1] - x[i]|
  _rms_diff           sqrt(mean of squared diffs)

Saliency-weighted (4)  — Stage 2 signals only:
  _weighted_mean
  _weighted_std
  _weighted_peak_val   value of signal at saliency-weighted "centre of mass" frame
  _weighted_sum

════════════════════════════════════════════════════════════════════
CUMULATIVE STAGE 3 JSON STRUCTURE
════════════════════════════════════════════════════════════════════
{
  "video_name":       "video001",
  "label":            1,
  "label_str":        "fall",
  "activity":         "bed",
  "group":            "f_raw_b_1",

  // Stage 1 header
  "active_window":    [12, 19],
  "varying_k":        8,
  "detection_status": "Global Event Zone",
  "saliency_weights": [...],        // 32 values
  "clip_indices":     [...],        // 32 original frame numbers

  // Per-frame sequence — read by Stage 4 attention
  // 39 values/frame: 34 keypoint coords + 4 physics + 1 depth
  "frame_data": [
    {
      "frame_idx":       62,
      "keypoints":       [[x,y], ...],   // 17x2 normalised 0-1
      "normalized_bbox": [cx,cy,w,h],
      "score":           0.92,
      "features": {
        "tilt_angle":        7.1,
        "vertical_velocity": 0.03,
        "h_w_ratio":         4.3,
        "ground_proximity":  0.68,
        "torso_depth":       0.42
      }
    }, ...
  ],

  // Stage 2 aggregated summary (~172 features across 4 physics signals)
  "stage2_summary": {
    "tilt_max": ..., "tilt_min": ..., ..., "tilt_weighted_mean": ...,
    "h_w_ratio_max": ..., ...,
    "velocity_max": ..., ...,
    "ground_proximity_max": ..., ...
  },

  // Stage 3 aggregated depth summary (~43 features)
  "stage3_summary": {
    "depth_max": ..., "depth_min": ..., ...,
    "depth_drop": ...,          // explicit extra: mean(all_but_last) - last
    "depth_weighted_mean": ..., // saliency-weighted
    ...
  }
}

════════════════════════════════════════════════════════════════════
DATASET STRUCTURE
════════════════════════════════════════════════════════════════════
FallVision/
  fall/
    bed/
      f_raw_b_1/
        video001.mp4    <- video files directly here, no subdirs
      f_raw_b_2/
      f_raw_b_3/
    chair/
    stand/
  no_fall/
    ...

════════════════════════════════════════════════════════════════════
USAGE
════════════════════════════════════════════════════════════════════
Fill in the PATH CONFIGURATION block below, then:
    python run_pipeline_s1_to_s3.py
"""

import os
import sys
import json
import math
import time
import logging
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

import cv2
import numpy as np
from scipy import stats as scipy_stats
from scipy.ndimage import gaussian_filter1d
from scipy.ndimage import label as nd_label

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO


# ═════════════════════════════════════════════════════════════════════════════
# PATH CONFIGURATION  —  fill these in before running
# ═════════════════════════════════════════════════════════════════════════════

# Root of FallVision dataset  (contains fall/ and no_fall/ subdirs)
DATA_ROOT = ""

# Stage-1 3D-CNN checkpoint  (.pth)
STAGE1_CKPT = ""

# YOLO pose model path  (ultralytics auto-downloads if file not found)
YOLO_MODEL_PATH = "yolov8m-pose.pt"

# MiDaS model type: "MiDaS_small" (fast) or "DPT_Large" (more accurate)
MIDAS_MODEL_TYPE = "MiDaS_small"

# Output root for Stage-2 intermediate JSON files
STAGE2_OUT_ROOT = ""

# Output root for cumulative Stage-3 JSON files
STAGE3_OUT_ROOT = ""

# Video file extensions to scan
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}

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
        logging.FileHandler("pipeline_log.txt", mode="w"),
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
CLIP_LEN   = 32
FRAME_SIZE = 224
NUM_KP     = 17
YOLO_IMGSZ = 640
YOLO_CONF  = 0.1
YOLO_IOU   = 0.5
LABEL_MAP  = {"fall": 1, "no_fall": 0}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL: Stage-1 3D-CNN
# ─────────────────────────────────────────────────────────────────────────────
class Stage1_3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 64, (3,7,7), (1,2,2), (1,3,3), padding_mode="replicate")
        self.bn1   = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d((1,3,3), (1,2,2), (0,1,1))
        self.conv2 = nn.Conv3d(64, 128, 3, padding=1, padding_mode="replicate")
        self.bn2   = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(2, 2)
        self.conv3 = nn.Conv3d(128, 256, 3, padding=1, padding_mode="replicate")
        self.bn3   = nn.BatchNorm3d(256)
        self.conv4 = nn.Conv3d(256, 256, 3, padding=1, padding_mode="replicate")
        self.bn4   = nn.BatchNorm3d(256)
        self.global_pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.dropout     = nn.Dropout(0.5)
        self.fc          = nn.Linear(256, 2)

    def forward(self, x, return_saliency=False):
        x   = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x   = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x   = F.relu(self.bn3(self.conv3(x)))

        sal = x.detach().pow(2).mean(dim=(1, 3, 4))
        T_p = sal.shape[1]
        em  = torch.ones(T_p, device=sal.device)
        em[0]  = 0.05
        em[-1] = 0.5
        sal    = sal * em
        s_min  = sal.min(dim=1, keepdim=True)[0]
        s_max  = sal.max(dim=1, keepdim=True)[0]
        sal    = (sal - s_min) / (s_max - s_min + 1e-6)
        sal    = F.softmax(sal * 5.0, dim=1)

        x      = F.relu(self.bn4(self.conv4(x)))
        x      = self.global_pool(x).flatten(1)
        x      = self.dropout(x)
        logits = self.fc(x)

        if return_saliency:
            return logits, sal
        return logits


# ─────────────────────────────────────────────────────────────────────────────
# EXHAUSTIVE TEMPORAL FEATURE EXTRACTOR
# ─────────────────────────────────────────────────────────────────────────────
def all_temporal_features(values, prefix, saliency_weights=None):
    """
    Derive every possible temporal feature from a 1-D sequence of floats.
    Returns a flat dict with keys like "{prefix}_{stat_name}".

    Args
        values          : list or 1-D array, one value per active-window frame
        prefix          : string prefix for all keys
        saliency_weights: optional array aligned with values for weighted stats
                          (Stage-1 saliency sliced to the active window)

    Returns ~43 features per signal (47 when saliency_weights provided).
    All values rounded to 6 decimal places.
    NaN/Inf are replaced with 0.0.
    """
    arr = np.array(values, dtype=np.float64)
    n   = len(arr)

    def r(v):
        """Round and sanitise a scalar."""
        v = float(v)
        return 0.0 if (math.isnan(v) or math.isinf(v)) else round(v, 6)

    # ── Guard: empty sequence ─────────────────────────────────────────────────
    if n == 0:
        zero_keys = [
            "max","min","mean","std","median","range","sum",
            "delta","abs_delta","norm_delta","linear_slope",
            "peak_pos","trough_pos","time_to_peak","peak_trough_gap",
            "pre_peak_slope","post_peak_slope","slope_ratio","curvature",
            "first_half_mean","second_half_mean",
            "first_half_std","second_half_std",
            "half_mean_diff","half_mean_ratio",
            "first_third_mean","mid_third_mean","last_third_mean",
            "energy","skewness","kurtosis","iqr",
            "mean_crossings","zero_crossings","above_mean_ratio",
            "mean_abs_diff","max_diff","total_variation","rms_diff",
        ]
        out = {f"{prefix}_{k}": 0.0 for k in zero_keys}
        if saliency_weights is not None:
            for k in ["weighted_mean","weighted_std",
                      "weighted_peak_val","weighted_sum"]:
                out[f"{prefix}_{k}"] = 0.0
        return out

    # ── Basic stats ───────────────────────────────────────────────────────────
    sig_range = float(np.max(arr) - np.min(arr))
    delta     = float(arr[-1] - arr[0])

    out = {
        f"{prefix}_max":    r(np.max(arr)),
        f"{prefix}_min":    r(np.min(arr)),
        f"{prefix}_mean":   r(np.mean(arr)),
        f"{prefix}_std":    r(np.std(arr)),
        f"{prefix}_median": r(np.median(arr)),
        f"{prefix}_range":  r(sig_range),
        f"{prefix}_sum":    r(np.sum(arr)),
    }

    # ── Delta / trend ─────────────────────────────────────────────────────────
    norm_delta   = delta / sig_range if sig_range > 1e-9 else 0.0
    lin_coeff    = np.polyfit(np.arange(n), arr, 1)[0] if n > 1 else 0.0

    out.update({
        f"{prefix}_delta":        r(delta),
        f"{prefix}_abs_delta":    r(abs(delta)),
        f"{prefix}_norm_delta":   r(norm_delta),
        f"{prefix}_linear_slope": r(lin_coeff),
    })

    # ── Temporal position ─────────────────────────────────────────────────────
    peak_idx   = int(np.argmax(arr))
    trough_idx = int(np.argmin(arr))
    peak_pos   = peak_idx   / max(n - 1, 1)
    trough_pos = trough_idx / max(n - 1, 1)

    out.update({
        f"{prefix}_peak_pos":        r(peak_pos),
        f"{prefix}_trough_pos":      r(trough_pos),
        f"{prefix}_time_to_peak":    r(peak_pos),      # explicit alias
        f"{prefix}_peak_trough_gap": r(abs(peak_pos - trough_pos)),
    })

    # ── Acceleration pattern ──────────────────────────────────────────────────
    pre_slope  = float(np.mean(np.diff(arr[:peak_idx + 1]))) if peak_idx > 0     else 0.0
    post_slope = float(np.mean(np.diff(arr[peak_idx:])))     if peak_idx < n - 1 else 0.0
    slope_ratio = pre_slope / (post_slope + 1e-9)

    # Curvature: 2nd-order polyfit coefficient (a in ax^2 + bx + c)
    curvature = np.polyfit(np.arange(n), arr, 2)[0] if n >= 3 else 0.0

    out.update({
        f"{prefix}_pre_peak_slope":  r(pre_slope),
        f"{prefix}_post_peak_slope": r(post_slope),
        f"{prefix}_slope_ratio":     r(slope_ratio),
        f"{prefix}_curvature":       r(curvature),
    })

    # ── Half-split ────────────────────────────────────────────────────────────
    mid  = max(n // 2, 1)
    fh   = arr[:mid]
    sh   = arr[mid:] if mid < n else arr[-1:]

    fh_mean = float(np.mean(fh))
    sh_mean = float(np.mean(sh))

    out.update({
        f"{prefix}_first_half_mean":  r(fh_mean),
        f"{prefix}_second_half_mean": r(sh_mean),
        f"{prefix}_first_half_std":   r(np.std(fh)),
        f"{prefix}_second_half_std":  r(np.std(sh)),
        f"{prefix}_half_mean_diff":   r(sh_mean - fh_mean),
        f"{prefix}_half_mean_ratio":  r(sh_mean / (fh_mean + 1e-9)),
    })

    # ── Third-split ───────────────────────────────────────────────────────────
    t1 = max(n // 3, 1)
    t2 = max(2 * n // 3, t1 + 1)

    out.update({
        f"{prefix}_first_third_mean": r(np.mean(arr[:t1])),
        f"{prefix}_mid_third_mean":   r(np.mean(arr[t1:t2])),
        f"{prefix}_last_third_mean":  r(np.mean(arr[t2:])) if t2 < n else r(arr[-1]),
    })

    # ── Energy / distribution ─────────────────────────────────────────────────
    energy   = float(np.sum(arr ** 2))
    skew     = float(scipy_stats.skew(arr))    if n >= 3 else 0.0
    kurt     = float(scipy_stats.kurtosis(arr)) if n >= 4 else 0.0
    iqr      = float(np.percentile(arr, 75) - np.percentile(arr, 25))

    out.update({
        f"{prefix}_energy":   r(energy),
        f"{prefix}_skewness": r(skew),
        f"{prefix}_kurtosis": r(kurt),
        f"{prefix}_iqr":      r(iqr),
    })

    # ── Crossing / threshold ──────────────────────────────────────────────────
    mean_val = float(np.mean(arr))
    centered = arr - mean_val

    # Number of times the signal crosses its own mean
    mean_crossings = int(np.sum(np.diff(np.sign(centered)) != 0))

    # Number of zero crossings (useful for velocity/diff signals)
    zero_crossings = int(np.sum(np.diff(np.sign(arr)) != 0))

    above_mean_ratio = float(np.sum(arr > mean_val) / n)

    out.update({
        f"{prefix}_mean_crossings":   r(mean_crossings),
        f"{prefix}_zero_crossings":   r(zero_crossings),
        f"{prefix}_above_mean_ratio": r(above_mean_ratio),
    })

    # ── Change rate ───────────────────────────────────────────────────────────
    if n > 1:
        diffs     = np.abs(np.diff(arr))
        mean_ad   = float(np.mean(diffs))
        max_d     = float(np.max(diffs))
        tot_var   = float(np.sum(diffs))
        rms_d     = float(np.sqrt(np.mean(diffs ** 2)))
    else:
        mean_ad = max_d = tot_var = rms_d = 0.0

    out.update({
        f"{prefix}_mean_abs_diff":   r(mean_ad),
        f"{prefix}_max_diff":        r(max_d),
        f"{prefix}_total_variation": r(tot_var),
        f"{prefix}_rms_diff":        r(rms_d),
    })

    # ── Saliency-weighted stats (only when weights provided) ──────────────────
    if saliency_weights is not None:
        w = np.array(saliency_weights, dtype=np.float64)
        w = w[:n]                          # trim to actual length
        w = w / (w.sum() + 1e-9)          # normalise to sum=1

        w_mean = float(np.sum(arr * w))
        w_std  = float(np.sqrt(np.sum(w * (arr - w_mean) ** 2)))
        w_sum  = float(np.sum(arr * w))

        # Value at the saliency-weighted centre-of-mass frame
        com_idx     = int(np.round(np.sum(np.arange(n) * w)))
        com_idx     = max(0, min(com_idx, n - 1))
        w_peak_val  = float(arr[com_idx])

        out.update({
            f"{prefix}_weighted_mean":     r(w_mean),
            f"{prefix}_weighted_std":      r(w_std),
            f"{prefix}_weighted_peak_val": r(w_peak_val),
            f"{prefix}_weighted_sum":      r(w_sum),
        })

    return out


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1
# ─────────────────────────────────────────────────────────────────────────────
def extract_frames(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < 2:
        cap.release()
        raise RuntimeError(f"Video too short ({total} frames): {video_path}")

    indices     = np.linspace(0, total - 1, CLIP_LEN, dtype=int)
    frames_full = []
    frames_224  = []

    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            cap.release()
            raise RuntimeError(f"Read failed at frame {idx}: {video_path}")
        frames_full.append(frame.copy())
        small = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
        frames_224.append(cv2.cvtColor(small, cv2.COLOR_BGR2RGB))

    cap.release()
    return frames_full, frames_224, indices.tolist()


def get_active_window(saliency, energy_threshold=0.15):
    thresh    = np.mean(saliency) + 0.5 * np.std(saliency)
    is_active = saliency > thresh
    labs, n   = nd_label(is_active)

    if n == 0:
        return 0, len(saliency) - 1, "Static/Uniform"

    total_e = np.sum(saliency)
    sig_idx = []
    for i in range(1, n + 1):
        mask = labs == i
        if np.sum(saliency[mask]) / total_e > energy_threshold:
            sig_idx.extend(np.where(mask)[0].tolist())

    if not sig_idx:
        pk = int(np.argmax(saliency))
        return max(0, pk - 2), min(len(saliency) - 1, pk + 2), "Peak Only"

    return int(min(sig_idx)), int(max(sig_idx)), "Global Event Zone"


def run_stage1(frames_224, clip_indices, model, device):
    frames_np = np.array(frames_224, dtype=np.float32) / 255.0
    clip = (
        torch.from_numpy(frames_np)
        .permute(3, 0, 1, 2)
        .unsqueeze(0)
        .to(device)
    )
    with torch.no_grad():
        _, saliency = model(clip, return_saliency=True)

    sal_raw    = saliency.squeeze(0).cpu().numpy()
    sal_interp = np.interp(
        np.linspace(0, len(sal_raw) - 1, CLIP_LEN),
        np.arange(len(sal_raw)), sal_raw,
    )
    sal_smooth = gaussian_filter1d(sal_interp, sigma=1.2)
    start, end, status = get_active_window(sal_smooth)

    return {
        "active_window":    [start, end],
        "varying_k":        end - start + 1,
        "detection_status": status,
        "saliency_weights": sal_smooth.tolist(),
        "clip_indices":     clip_indices,
    }


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2
# ─────────────────────────────────────────────────────────────────────────────
def _normalise_kp(kp_px, W, H):
    if kp_px is None:
        return [[0.0, 0.0]] * NUM_KP
    return [[float(x / W), float(y / H)] for x, y in kp_px]


def _interpolate_kp(frame_nums, det_norm):
    L  = len(frame_nums)
    xs = np.full((L, NUM_KP), np.nan)
    ys = np.full((L, NUM_KP), np.nan)
    for i, f in enumerate(frame_nums):
        for j, pt in enumerate(det_norm[f]):
            if pt[0] is not None:
                xs[i, j] = pt[0]
                ys[i, j] = pt[1]
    idx = np.arange(L)
    for j in range(NUM_KP):
        v = ~np.isnan(xs[:, j])
        xs[:, j] = np.interp(idx, idx[v], xs[v, j]) if v.any() else 0.0
        ys[:, j] = np.interp(idx, idx[v], ys[v, j]) if v.any() else 0.0
    return {
        f: [[float(xs[i, j]), float(ys[i, j])] for j in range(NUM_KP)]
        for i, f in enumerate(frame_nums)
    }


def run_stage2(frames_full, clip_indices, s1_meta, yolo_model):
    win_s      = s1_meta["active_window"][0]
    win_e      = s1_meta["active_window"][1]
    pos_list   = list(range(win_s, win_e + 1))
    frame_nums = [clip_indices[p] for p in pos_list]

    det_px   = {}
    bbox_px  = {}
    scores   = {}
    detected = {}

    for pos, fnum in zip(pos_list, frame_nums):
        bgr     = frames_full[pos]
        results = yolo_model.predict(
            source=bgr, imgsz=YOLO_IMGSZ,
            conf=YOLO_CONF, iou=YOLO_IOU, verbose=False,
        )
        if results and len(results[0].boxes) > 0:
            best           = results[0].boxes.conf.argmax()
            kp             = results[0].keypoints.data[best].cpu().numpy()
            bb             = results[0].boxes.xyxy[best].cpu().numpy()
            det_px[fnum]   = kp[:, :2]
            scores[fnum]   = float(results[0].boxes.conf[best])
            bbox_px[fnum]  = [
                float((bb[0]+bb[2])/2), float((bb[1]+bb[3])/2),
                float(bb[2]-bb[0]),     float(bb[3]-bb[1]),
            ]
            detected[fnum] = True
        else:
            det_px[fnum]   = None
            bbox_px[fnum]  = None
            scores[fnum]   = 0.0
            detected[fnum] = False

    H0, W0   = frames_full[pos_list[0]].shape[:2]
    det_norm = {f: _normalise_kp(det_px[f], W0, H0) for f in frame_nums}
    bbox_norm = {
        f: [bbox_px[f][0]/W0, bbox_px[f][1]/H0,
            bbox_px[f][2]/W0, bbox_px[f][3]/H0]
        if bbox_px[f] else [None]*4
        for f in frame_nums
    }
    interp = _interpolate_kp(frame_nums, det_norm)

    frame_data = []
    prev_hip   = None
    for fnum in frame_nums:
        kp   = interp[fnum]
        m_sh = [(kp[5][0]+kp[6][0])/2,   (kp[5][1]+kp[6][1])/2]
        m_hp = [(kp[11][0]+kp[12][0])/2,  (kp[11][1]+kp[12][1])/2]

        tilt     = abs(math.degrees(math.atan2(m_sh[0]-m_hp[0], -(m_sh[1]-m_hp[1]))))
        vel      = float(m_hp[1] - prev_hip[1]) if prev_hip else 0.0
        prev_hip = m_hp
        bx       = bbox_norm[fnum]
        hw       = float(bx[3]/bx[2]) if (bx[2] and bx[2] > 0) else 0.0
        gp       = float(1.0 - m_sh[1])

        frame_data.append({
            "frame_idx":       int(fnum),
            "keypoints":       kp,
            "normalized_bbox": bx,
            "score":           round(scores[fnum], 6),
            "features": {
                "tilt_angle":        round(tilt, 6),
                "vertical_velocity": round(vel,  6),
                "h_w_ratio":         round(hw,   6),
                "ground_proximity":  round(gp,   6),
                # torso_depth added by Stage 3
            },
        })

    return frame_data, detected


def compute_stage2_summary(frame_data, s1_meta):
    """
    Compute all temporal features for each of the 4 physics signals.
    Saliency weights are sliced to the active window for weighted stats.
    """
    sal_full   = np.array(s1_meta["saliency_weights"])
    win_s      = s1_meta["active_window"][0]
    win_e      = s1_meta["active_window"][1]
    sal_window = sal_full[win_s : win_e + 1]

    tilts = [fd["features"]["tilt_angle"]        for fd in frame_data]
    hwrs  = [fd["features"]["h_w_ratio"]         for fd in frame_data]
    vels  = [fd["features"]["vertical_velocity"] for fd in frame_data]
    gps   = [fd["features"]["ground_proximity"]  for fd in frame_data]

    summary = {}
    summary.update(all_temporal_features(tilts, "tilt",              saliency_weights=sal_window))
    summary.update(all_temporal_features(hwrs,  "h_w_ratio",         saliency_weights=sal_window))
    summary.update(all_temporal_features(vels,  "velocity",          saliency_weights=sal_window))
    summary.update(all_temporal_features(gps,   "ground_proximity",  saliency_weights=sal_window))

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3
# ─────────────────────────────────────────────────────────────────────────────
def _torso_centroid(keypoints, H, W):
    ids = [5, 6, 11, 12]
    return (
        int(np.median([keypoints[t][0] * W for t in ids])),
        int(np.median([keypoints[t][1] * H for t in ids])),
    )


def _run_midas(bgr, midas, transform, device):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    inp = transform(rgb).to(device)
    with torch.no_grad():
        pred = midas(inp)
        pred = F.interpolate(
            pred.unsqueeze(1), size=rgb.shape[:2],
            mode="bicubic", align_corners=False,
        ).squeeze()
    d        = pred.cpu().numpy()
    d_min, d_max = d.min(), d.max()
    return (d - d_min) / (d_max - d_min + 1e-6)


def run_stage3(frame_data, detected, frames_full, clip_indices,
               active_window, s1_meta, midas, midas_transform, device):
    """
    Run MiDaS on every active-window frame.
    Person detected -> torso centroid depth.
    Person NOT detected -> image centre depth (MiDaS runs regardless).

    Mutates frame_data in-place (adds torso_depth to features).
    Returns stage3_summary.
    """
    win_s     = active_window[0]
    win_e     = active_window[1]
    pos_map   = {clip_indices[p]: p for p in range(win_s, win_e + 1)}
    depth_vals = []

    for fd in frame_data:
        fnum = fd["frame_idx"]
        pos  = pos_map.get(fnum)

        if pos is None:
            fd["features"]["torso_depth"] = 0.0
            depth_vals.append(0.0)
            continue

        bgr   = frames_full[pos]
        H, W  = bgr.shape[:2]
        dmap  = _run_midas(bgr, midas, midas_transform, device)

        if detected.get(fnum, False):
            cx, cy = _torso_centroid(fd["keypoints"], H, W)
        else:
            cx, cy = W // 2, H // 2     # image centre fallback

        cx    = max(0, min(cx, W - 1))
        cy    = max(0, min(cy, H - 1))
        d_val = float(dmap[cy, cx])

        fd["features"]["torso_depth"] = round(d_val, 6)
        depth_vals.append(d_val)

    # Saliency weights sliced to active window (for weighted depth stats)
    sal_full   = np.array(s1_meta["saliency_weights"])
    sal_window = sal_full[win_s : win_e + 1]

    summary = all_temporal_features(
        depth_vals, "depth", saliency_weights=sal_window
    )

    # Extra depth-specific features not covered by the generic extractor
    d_arr = np.array(depth_vals, dtype=np.float64)
    summary["depth_drop"] = round(
        float(np.mean(d_arr[:-1]) - d_arr[-1]), 6
    ) if len(d_arr) > 1 else 0.0

    # Re-label range and variance explicitly for downstream clarity
    summary["depth_variance"] = summary.get("depth_std", 0.0) ** 2  \
                                 if "depth_std" in summary else 0.0
    summary["depth_variance"] = round(float(np.var(d_arr)), 6)

    return summary


# ─────────────────────────────────────────────────────────────────────────────
# DATASET DISCOVERY
# ─────────────────────────────────────────────────────────────────────────────
def discover_videos():
    video_list = []
    data_root  = Path(DATA_ROOT)

    for label_dir in sorted(data_root.iterdir()):
        if not label_dir.is_dir():
            continue
        label_str = label_dir.name.lower()
        if label_str not in LABEL_MAP:
            log.warning(f"Skipping unknown label dir: '{label_dir.name}'")
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

                vfiles = sorted([
                    f for f in group_dir.iterdir()
                    if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
                ])
                if not vfiles:
                    log.warning(f"No video files in: {group_dir}")
                    continue

                for vf in vfiles:
                    rel = Path(label_str) / activity / group
                    video_list.append({
                        "label_int":  label_int,
                        "label_str":  label_str,
                        "activity":   activity,
                        "group":      group,
                        "video_name": vf.stem,
                        "video_path": str(vf),
                        "s2_json":    str(Path(STAGE2_OUT_ROOT) / rel / f"{vf.stem}.json"),
                        "s3_json":    str(Path(STAGE3_OUT_ROOT) / rel / f"{vf.stem}.json"),
                    })

    return video_list


# ─────────────────────────────────────────────────────────────────────────────
# PER-VIDEO PROCESSING
# ─────────────────────────────────────────────────────────────────────────────
def process_video(vi, stage1_model, yolo_model, midas, midas_transform, device):
    # Extract 32 frames once — reused by all stages
    frames_full, frames_224, clip_indices = extract_frames(vi["video_path"])

    # Stage 1 — saliency
    s1 = run_stage1(frames_224, clip_indices, stage1_model, device)

    # Stage 2 — pose + physics
    frame_data, detected = run_stage2(frames_full, clip_indices, s1, yolo_model)
    s2_summary           = compute_stage2_summary(frame_data, s1)

    # Save Stage-2 JSON
    Path(vi["s2_json"]).parent.mkdir(parents=True, exist_ok=True)
    with open(vi["s2_json"], "w") as fh:
        json.dump({"frame_data": frame_data}, fh, indent=2)

    # Stage 3 — depth (mutates frame_data in-place)
    s3_summary = run_stage3(
        frame_data, detected, frames_full, clip_indices,
        s1["active_window"], s1, midas, midas_transform, device,
    )

    # Cumulative Stage-3 JSON
    cumulative = {
        "video_name":       vi["video_name"],
        "label":            vi["label_int"],
        "label_str":        vi["label_str"],
        "activity":         vi["activity"],
        "group":            vi["group"],
        "active_window":    s1["active_window"],
        "varying_k":        s1["varying_k"],
        "detection_status": s1["detection_status"],
        "saliency_weights": s1["saliency_weights"],
        "clip_indices":     s1["clip_indices"],
        "frame_data":       frame_data,
        "stage2_summary":   s2_summary,
        "stage3_summary":   s3_summary,
    }

    Path(vi["s3_json"]).parent.mkdir(parents=True, exist_ok=True)
    with open(vi["s3_json"], "w") as fh:
        json.dump(cumulative, fh, indent=2)

    return s3_summary


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    required = {
        "DATA_ROOT":       DATA_ROOT,
        "STAGE1_CKPT":     STAGE1_CKPT,
        "STAGE2_OUT_ROOT": STAGE2_OUT_ROOT,
        "STAGE3_OUT_ROOT": STAGE3_OUT_ROOT,
    }
    empty = [k for k, v in required.items() if not v]
    if empty:
        log.error(f"Empty path variables — fill in before running: {empty}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info("=" * 65)
    log.info("  FallVision  Stage 1 -> 2 -> 3  Batch Inference")
    log.info("=" * 65)
    log.info(f"  DATA_ROOT       : {DATA_ROOT}")
    log.info(f"  STAGE1_CKPT     : {STAGE1_CKPT}")
    log.info(f"  YOLO_MODEL_PATH : {YOLO_MODEL_PATH}")
    log.info(f"  MIDAS_MODEL     : {MIDAS_MODEL_TYPE}")
    log.info(f"  STAGE2_OUT_ROOT : {STAGE2_OUT_ROOT}")
    log.info(f"  STAGE3_OUT_ROOT : {STAGE3_OUT_ROOT}")
    log.info(f"  NUM_WORKERS     : {NUM_WORKERS}")
    log.info(f"  Device          : {device}")
    if torch.cuda.is_available():
        log.info(f"  GPU             : {torch.cuda.get_device_name(0)}")
    log.info("=" * 65)

    # Load Stage-1 3D-CNN
    log.info("Loading Stage-1 3D-CNN ...")
    stage1_model = Stage1_3DCNN().to(device)
    stage1_model.load_state_dict(torch.load(STAGE1_CKPT, map_location=device))
    stage1_model.eval()
    for m in stage1_model.modules():
        if isinstance(m, nn.Dropout):
            m.eval()
    log.info("Stage-1 loaded.")

    # Load YOLO
    log.info("Loading YOLO pose model ...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    log.info("YOLO loaded.")

    # Load MiDaS
    log.info(f"Loading MiDaS ({MIDAS_MODEL_TYPE}) ...")
    midas = torch.hub.load("intel-isl/MiDaS", MIDAS_MODEL_TYPE)
    midas.to(device).eval()
    _t              = torch.hub.load("intel-isl/MiDaS", "transforms")
    midas_transform = _t.small_transform if "small" in MIDAS_MODEL_TYPE.lower() \
                      else _t.default_transform
    log.info("MiDaS loaded.")

    # Discover videos
    log.info(f"Scanning: {DATA_ROOT}")
    video_list = discover_videos()
    log.info(f"Found {len(video_list)} videos.")
    if not video_list:
        log.error("No videos found. Check DATA_ROOT and VIDEO_EXTENSIONS.")
        sys.exit(1)

    errors   = []
    err_lock = Lock()
    counter  = [0]
    cnt_lock = Lock()
    total    = len(video_list)
    t_start  = time.time()

    def worker(vi):
        try:
            s3 = process_video(
                vi, stage1_model, yolo_model,
                midas, midas_transform, device,
            )
            with cnt_lock:
                counter[0] += 1
                n = counter[0]
            elapsed = time.time() - t_start
            eta     = (elapsed / n) * (total - n) if n else 0
            log.info(
                f"[{n:>5}/{total}]  "
                f"{vi['label_str']}/{vi['activity']}/{vi['group']}/{vi['video_name']}"
                f"  depth_drop={s3.get('depth_drop', 0):.4f}"
                f"  ETA {eta/60:.1f}min"
            )
        except Exception as exc:
            with err_lock:
                errors.append({
                    "path":  f"{vi['label_str']}/{vi['activity']}/{vi['group']}/{vi['video_name']}",
                    "error": str(exc),
                    "trace": traceback.format_exc(),
                })
            with cnt_lock:
                counter[0] += 1
            log.error(
                f"ERROR  {vi['label_str']}/{vi['activity']}/{vi['group']}"
                f"/{vi['video_name']}  -- {exc}"
            )

    log.info(f"Launching {NUM_WORKERS} worker threads ...")
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as pool:
        futures = {pool.submit(worker, vi): vi for vi in video_list}
        for _ in as_completed(futures):
            pass

    if errors:
        err_path = Path(STAGE3_OUT_ROOT) / "error_log.json"
        err_path.parent.mkdir(parents=True, exist_ok=True)
        with open(err_path, "w") as fh:
            json.dump(errors, fh, indent=2)
        log.warning(f"{len(errors)} errors — see {err_path}")

    elapsed = time.time() - t_start
    log.info("=" * 65)
    log.info(f"  COMPLETE  |  Success : {total - len(errors)}/{total}")
    log.info(f"            |  Errors  : {len(errors)}")
    log.info(f"            |  Time    : {elapsed/60:.1f} min")
    log.info(f"  Stage-2 JSONs -> {STAGE2_OUT_ROOT}")
    log.info(f"  Stage-3 JSONs -> {STAGE3_OUT_ROOT}")
    log.info("=" * 65)


if __name__ == "__main__":
    main()
