import os
import json
import math
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import gaussian_filter1d, label
from ultralytics import YOLO
from multiprocessing import Pool, cpu_count

# =====================================================
# CONFIG & PATHS
# =====================================================
INPUT_ROOT  = r"FallVision"
OUTPUT_ROOT = r"stage3\input"
STAGE1_CKPT = r"stage1_v4\best_model_fold3.pth"
YOLO_MODEL  = r"stage2\yolov8m-pose.pt"

CLIP_LEN   = 32
FRAME_SIZE = 224
NUM_KP     = 17
IMGSZ      = 640
CONF       = 0.1
IOU        = 0.5

# Number of videos to process in parallel.
# Rule of thumb: start with cpu_count()-1, reduce if you run out of RAM.
NUM_WORKERS = max(1, cpu_count() - 1)

# =====================================================
# STAGE 1 MODEL ARCHITECTURE
# =====================================================
class Stage1_3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(3, 64, (3,7,7), (1,2,2), (1,3,3), padding_mode='replicate')
        self.bn1   = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d((1,3,3), (1,2,2), (0,1,1))
        self.conv2 = nn.Conv3d(64, 128, 3, padding=1, padding_mode='replicate')
        self.bn2   = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(2, 2)
        self.conv3 = nn.Conv3d(128, 256, 3, padding=1, padding_mode='replicate')
        self.bn3   = nn.BatchNorm3d(256)
        self.conv4 = nn.Conv3d(256, 256, 3, padding=1, padding_mode='replicate')
        self.bn4   = nn.BatchNorm3d(256)
        self.global_pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, 2)

    def forward(self, x, return_saliency=False):
        x   = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x   = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x   = F.relu(self.bn3(self.conv3(x)))
        sal = x.detach().pow(2).mean(dim=(1, 3, 4))
        T_prime = sal.shape[1]
        mask = torch.ones(T_prime)
        mask[0]  = 0.05
        mask[-1] = 0.5
        sal = sal * mask
        min_val = sal.min(dim=1, keepdim=True)[0]
        max_val = sal.max(dim=1, keepdim=True)[0]
        sal = (sal - min_val) / (max_val - min_val + 1e-6)
        sal = F.softmax(sal * 5.0, dim=1)
        x   = F.relu(self.bn4(self.conv4(x)))
        x   = self.global_pool(x).flatten(1)
        logits = self.fc(self.dropout(x))
        return (logits, sal) if return_saliency else logits


# =====================================================
# SHARED HELPERS  (identical to stage2_final_v.py)
# =====================================================
def get_global_active_window(saliency_values, energy_threshold=0.15):
    threshold = np.mean(saliency_values) + (0.5 * np.std(saliency_values))
    is_active = saliency_values > threshold
    labels, num_features = label(is_active)

    if num_features == 0:
        return 0, len(saliency_values) - 1, "Static/Uniform"

    total_energy = np.sum(saliency_values)
    significant_indices = []
    for i in range(1, num_features + 1):
        mask = (labels == i)
        if (np.sum(saliency_values[mask]) / total_energy) > energy_threshold:
            significant_indices.extend(np.where(mask)[0])

    if not significant_indices:
        peak_idx = np.argmax(saliency_values)
        return max(0, peak_idx-2), min(len(saliency_values)-1, peak_idx+2), "Peak Only"

    return min(significant_indices), max(significant_indices), "Global Event Zone"


def norm_kp(kp, W, H):
    if kp is None:
        return [[None, None]] * NUM_KP
    return [[float(x / W), float(y / H)] for x, y in kp]


def interpolate(frames, dets, num_kp=17):
    L = len(frames)
    xs, ys = np.full((L, num_kp), np.nan), np.full((L, num_kp), np.nan)
    for i, f in enumerate(frames):
        for j, pt in enumerate(dets[f]):
            if pt[0] is not None:
                xs[i, j], ys[i, j] = pt[0], pt[1]
    idx = np.arange(L)
    for j in range(num_kp):
        mask = ~np.isnan(xs[:, j])
        if mask.any():
            xs[:, j] = np.interp(idx, idx[mask], xs[mask, j])
            ys[:, j] = np.interp(idx, idx[mask], ys[mask, j])
    return {f: [[float(xs[i, j]), float(ys[i, j])] for j in range(num_kp)]
            for i, f in enumerate(frames)}


# =====================================================
# PER-VIDEO WORKER  (runs in its own process)
# Each process loads its own model copies — no sharing needed.
# =====================================================
def process_video(args):
    video_path, s1_out, s2_out, json_name = args

    # Load models fresh in each worker process
    device = torch.device("cpu")

    s1_model = Stage1_3DCNN().to(device)
    s1_model.load_state_dict(torch.load(STAGE1_CKPT, map_location=device))
    s1_model.eval()

    yolo = YOLO(YOLO_MODEL)

    pid = os.getpid()
    print(f"[PID {pid}] Starting: {video_path}")

    # ── STAGE 1 ────────────────────────────────────────────────────────────
    cap   = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < CLIP_LEN:
        cap.release()
        print(f"[PID {pid}] Skipping (too short): {video_path}")
        return

    indices = np.linspace(0, total - 1, CLIP_LEN).astype(int)
    frames  = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            break
        f = cv2.resize(frame, (FRAME_SIZE, FRAME_SIZE))
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))

    if len(frames) < CLIP_LEN:
        cap.release()
        print(f"[PID {pid}] Skipping (frame read failed): {video_path}")
        return

    clip_t = (torch.from_numpy(np.array(frames).astype(np.float32) / 255.0)
              .permute(3, 0, 1, 2).unsqueeze(0).to(device))

    with torch.no_grad():
        _, sal_raw = s1_model(clip_t, return_saliency=True)

    saliency = sal_raw.squeeze(0).cpu().numpy()

    saliency_interp = np.interp(
        np.linspace(0, len(saliency) - 1, CLIP_LEN),
        np.arange(len(saliency)),
        saliency
    )
    saliency_smooth = gaussian_filter1d(saliency_interp, sigma=1.2)

    start, end, status = get_global_active_window(saliency_smooth)
    varying_k = end - start + 1

    s1_res = {
        "video_path":       video_path.replace("\\", "/"),
        "active_window":    [int(start), int(end)],
        "varying_k":        int(varying_k),
        "detection_status": status,
        "saliency_weights": saliency_smooth.tolist(),
        "clip_indices":     indices.tolist()
    }
    with open(os.path.join(s1_out, json_name), "w") as f:
        json.dump(s1_res, f, indent=4)

    print(f"[PID {pid}] Stage-1 done | {status} | Window: {start}→{end} | K={varying_k}")

    # ── STAGE 2 ────────────────────────────────────────────────────────────
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_numbers = [int(indices[i]) for i in range(start, end + 1)]

    detections_px = {}
    bboxes_px     = {}
    scores        = {}

    print(f"[PID {pid}] Stage-2: {len(frame_numbers)} frames...")

    for fnum in frame_numbers:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fnum))
        ret, frame = cap.read()
        if not ret:
            detections_px[fnum], bboxes_px[fnum], scores[fnum] = None, None, 0.0
            continue

        results = yolo.predict(source=frame, imgsz=IMGSZ, conf=CONF, iou=IOU, verbose=False)

        if len(results) > 0 and len(results[0].boxes) > 0:
            best_idx = results[0].boxes.conf.argmax()
            kp_data  = results[0].keypoints.data[best_idx].cpu().numpy()  # (17, 3)
            bbox     = results[0].boxes.xyxy[best_idx].cpu().numpy()       # [x1,y1,x2,y2]

            detections_px[fnum] = kp_data[:, :2]
            scores[fnum]        = float(results[0].boxes.conf[best_idx])
            bboxes_px[fnum]     = [
                float((bbox[0] + bbox[2]) / 2), float((bbox[1] + bbox[3]) / 2),
                float(bbox[2] - bbox[0]),        float(bbox[3] - bbox[1])
            ]
        else:
            detections_px[fnum], bboxes_px[fnum], scores[fnum] = None, None, 0.0

    cap.release()

    # Normalize
    detections_norm = {f: norm_kp(detections_px[f], frame_w, frame_h) for f in frame_numbers}
    bboxes_norm = {}
    for f in frame_numbers:
        bx = bboxes_px[f]
        bboxes_norm[f] = (
            [bx[0]/frame_w, bx[1]/frame_h, bx[2]/frame_w, bx[3]/frame_h]
            if bx else [None] * 4
        )

    # Linear interpolation to fill gaps where YOLO missed a person
    interp_results = interpolate(frame_numbers, detections_norm)

    # Compute physics features
    frame_data = []
    prev_hip   = None

    for f in frame_numbers:
        kp = interp_results[f]
        m_sh = [(kp[5][0] + kp[6][0]) / 2,  (kp[5][1] + kp[6][1]) / 2]
        m_hp = [(kp[11][0] + kp[12][0]) / 2, (kp[11][1] + kp[12][1]) / 2]

        tilt = abs(math.degrees(math.atan2(m_sh[0] - m_hp[0], -(m_sh[1] - m_hp[1]))))
        vel  = float(m_hp[1] - prev_hip[1]) if prev_hip else 0.0
        prev_hip = m_hp
        bx   = bboxes_norm[f]
        hw   = float(bx[3] / bx[2]) if bx[2] and bx[2] > 0 else 0.0
        gp   = float(1.0 - m_sh[1])

        frame_data.append({
            "frame_idx":       f,
            "keypoints":       kp,
            "normalized_bbox": bx,
            "score":           scores[f],
            "features": {
                "tilt_angle":        tilt,
                "vertical_velocity": vel,
                "h_w_ratio":         hw,
                "ground_proximity":  gp
            }
        })

    with open(os.path.join(s2_out, json_name), "w") as f:
        json.dump({"frame_data": frame_data}, f, indent=2)

    print(f"[PID {pid}] Stage-2 done | {len(frame_data)} frames → {os.path.join(s2_out, json_name)}")


# =====================================================
# COLLECT ALL JOBS THEN DISPATCH
# =====================================================
def collect_jobs():
    valid_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    jobs = []
    for root, _, files in os.walk(INPUT_ROOT):
        for filename in files:
            if filename.lower().endswith(valid_extensions):
                video_path = os.path.join(root, filename)
                rel_path   = os.path.relpath(root, INPUT_ROOT)
                s1_dir     = os.path.join(OUTPUT_ROOT, "stage1_output", rel_path)
                s2_dir     = os.path.join(OUTPUT_ROOT, "stage2_output", rel_path)
                os.makedirs(s1_dir, exist_ok=True)
                os.makedirs(s2_dir, exist_ok=True)
                json_name  = f"{os.path.splitext(filename)[0]}.json"
                jobs.append((video_path, s1_dir, s2_dir, json_name))
    return jobs


# =====================================================
# ENTRY POINT
# =====================================================
if __name__ == "__main__":
    jobs = collect_jobs()
    print(f"Found {len(jobs)} videos. Running with {NUM_WORKERS} workers.\n")

    # chunksize=1 so each worker picks up one video at a time — fair scheduling
    with Pool(processes=NUM_WORKERS) as pool:
        pool.map(process_video, jobs, chunksize=1)

    print("\nAll done.")