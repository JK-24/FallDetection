# ====== STAGE-2: Pose estimation, annotated visual outputs, stage2_output.json ======
# Paste this immediately after Stage-1 cell (which must save stage1_output.json)
import os, sys, json, math, time
import numpy as np
import cv2

# ---------- CONFIG (edit if needed) ----------
STAGE1_JSON = "stage1_output.json"   # must exist (produced by Stage-1)
STAGE2_JSON = "stage2_output.json"
VIS_DIR = "/content/stage2_vis"
os.makedirs(VIS_DIR, exist_ok=True)

YOLO_MODEL = "yolov8m-pose.pt"   # stronger than yolov8n-pose; will auto-download if missing
IMGSZ = 1024                    # large inference size for small/distant humans
CONF = 0.1                      # low confidence threshold to catch weak detections
IOU  = 0.5

NUM_KP = 17  # COCO keypoints

# ---------- load stage1 json ----------
if not os.path.exists(STAGE1_JSON):
    raise FileNotFoundError(f"{STAGE1_JSON} not found. Run Stage-1 first and save stage1_output.json")

with open(STAGE1_JSON, "r") as f:
    s1 = json.load(f)

VIDEO_PATH = s1["video_path"]
clip_indices = s1["clip_indices"]
active_window = s1["active_window"]
clip_start, clip_end = int(active_window[0]), int(active_window[1])
frame_numbers = [int(clip_indices[i]) for i in range(clip_start, clip_end+1)]
print("Stage-2: Video:", VIDEO_PATH)
print("Stage-2: Frames to process:", frame_numbers)

# ---------- ensure ultralytics installed & imported ----------
try:
    from ultralytics import YOLO
except Exception:
    print("Installing ultralytics...")
    !{sys.executable} -m pip install -q ultralytics
    from ultralytics import YOLO

# ---------- load model ----------
print("Loading YOLO model:", YOLO_MODEL)
model = YOLO(YOLO_MODEL)  # will auto-download if necessary
print("Model loaded.")

# ---------- helper to parse result robustly ----------
def parse_result(res_obj):
    dets = []
    try:
        kpts = getattr(res_obj, "keypoints", None)
        boxes = getattr(res_obj, "boxes", None)
        if kpts is not None:
            karr = np.asarray(kpts)  # (n,17,3)
            for i in range(karr.shape[0]):
                kp = karr[i][:,:2].astype(float)
                confs = karr[i][:,2]
                score = float(np.nanmean(confs))
                bbox = None
                try:
                    if boxes is not None and getattr(boxes, "xywh", None) is not None:
                        bx = np.asarray(boxes.xywh.cpu())
                        if i < bx.shape[0]:
                            bbox = bx[i].tolist()
                except Exception:
                    bbox = None
                if bbox is None:
                    x1,y1 = np.nanmin(kp[:,0]), np.nanmin(kp[:,1])
                    x2,y2 = np.nanmax(kp[:,0]), np.nanmax(kp[:,1])
                    bbox = [float((x1+x2)/2.0), float((y1+y2)/2.0), float(x2-x1), float(y2-y1)]
                dets.append({"keypoints": kp, "bbox": bbox, "score": score})
            return dets
    except Exception:
        pass
    # fallback try boxes.keypoints
    try:
        if hasattr(res_obj, "boxes") and getattr(res_obj.boxes, "keypoints", None) is not None:
            klist = res_obj.boxes.keypoints
            bx = np.asarray(res_obj.boxes.xywh.cpu())
            for i in range(len(klist)):
                kp = np.asarray(klist[i])[:,:2].astype(float)
                score = float(np.nanmean(np.asarray(klist[i])[:,2]))
                b = bx[i].tolist() if i < len(bx) else None
                dets.append({"keypoints": kp, "bbox": b, "score": score})
            return dets
    except Exception:
        pass
    return dets

# ---------- run inference and save overlays ----------
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError("Cannot open video: " + VIDEO_PATH)

frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print("Video resolution:", frame_w, "x", frame_h)

detections_px = {}
bboxes_px = {}
scores = {}

for fnum in frame_numbers:
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(fnum))
    ret, frame = cap.read()
    if not ret:
        print("Frame read failed:", fnum)
        detections_px[fnum] = None
        bboxes_px[fnum] = None
        scores[fnum] = 0.0
        continue

    print(f"Processing frame {fnum}, shape: {frame.shape}")
    res = model.predict(source=frame, imgsz=IMGSZ, conf=CONF, iou=IOU, verbose=False)
    dets = []
    if len(res) > 0:
        dets = parse_result(res[0])
    print(f" - detections found: {len(dets)}")

    if len(dets) == 0:
        detections_px[fnum] = None
        bboxes_px[fnum] = None
        scores[fnum] = 0.0
    else:
        best = max(dets, key=lambda x: x.get("score",0.0))
        kp = np.array(best["keypoints"], dtype=float)
        if kp.shape[0] != NUM_KP:
            tmp = np.full((NUM_KP,2), np.nan)
            n = min(NUM_KP, kp.shape[0])
            tmp[:n] = kp[:n]
            kp = tmp
        detections_px[fnum] = kp
        bboxes_px[fnum] = best["bbox"]
        scores[fnum] = float(best.get("score",0.0))

    # --- make overlay image and save ---
    vis = frame.copy()
    if detections_px[fnum] is not None:
        for (x,y) in detections_px[fnum]:
            if not (np.isnan(x) or np.isnan(y)):
                cv2.circle(vis, (int(round(x)), int(round(y))), 3, (0,255,0), -1)
    # draw bbox if available
    if bboxes_px[fnum] is not None:
        cx,cy,w,h = bboxes_px[fnum]
        x1 = int(cx - w/2); y1 = int(cy - h/2); x2 = int(cx + w/2); y2 = int(cy + h/2)
        cv2.rectangle(vis, (x1,y1), (x2,y2), (0,0,255), 2)
    save_path = os.path.join(VIS_DIR, f"frame_{fnum:05d}.png")
    cv2.imwrite(save_path, vis)
    print(" - saved overlay:", save_path)

cap.release()

# ---------- normalize kps and bboxes & interpolate ----------
def norm_kps_px(kp_px, W, H):
    if kp_px is None: return None
    out = []
    for x,y in kp_px:
        if np.isnan(x) or np.isnan(y):
            out.append([None,None])
        else:
            out.append([float(x/W), float(y/H)])
    return out

detections_norm = {}
bboxes_norm = {}
for fnum in frame_numbers:
    detections_norm[fnum] = norm_kps_px(detections_px.get(fnum, None), frame_w, frame_h)
    bx = bboxes_px.get(fnum)
    if bx is None:
        bboxes_norm[fnum] = [None,None,None,None]
    else:
        bboxes_norm[fnum] = [float(bx[0]/frame_w), float(bx[1]/frame_h), float(bx[2]/frame_w), float(bx[3]/frame_h)]

# linear interpolation across frames per keypoint coordinate
def interp_kps(frames, dets_norm, num_kp=NUM_KP):
    frames = sorted(frames)
    L = len(frames)
    xs = np.full((L,num_kp), np.nan)
    ys = np.full((L,num_kp), np.nan)
    for i,f in enumerate(frames):
        kp = dets_norm.get(f)
        if kp is None: continue
        for j in range(num_kp):
            v = kp[j]
            if v is None: continue
            xs[i,j] = v[0]; ys[i,j] = v[1]
    idx = np.arange(L)
    for j in range(num_kp):
        valid = ~np.isnan(xs[:,j])
        if valid.sum() == 0: continue
        xs[:,j] = np.interp(idx, idx[valid], xs[valid,j])
        ys[:,j] = np.interp(idx, idx[valid], ys[valid,j])
    final = {}
    for i,f in enumerate(frames):
        arr = []
        for j in range(num_kp):
            xv = xs[i,j]; yv = ys[i,j]
            if np.isnan(xv) or np.isnan(yv):
                arr.append([None,None])
            else:
                arr.append([float(xv), float(yv)])
        final[f] = arr
    return final

interp_result = interp_kps(frame_numbers, detections_norm, NUM_KP)

# ---------- compute features ----------
L_SH, R_SH, L_HIP, R_HIP = 5,6,11,12
frame_data = []
prev_mid_hip = None

def mid(a,b):
    if a is None or b is None: return None
    if a[0] is None or b[0] is None: return None
    return [(a[0]+b[0])/2.0, (a[1]+b[1])/2.0]

def tilt_angle(mid_hip, mid_sh):
    if mid_hip is None or mid_sh is None: return None
    vx = mid_sh[0] - mid_hip[0]; vy = mid_sh[1] - mid_hip[1]
    return abs(math.degrees(math.atan2(vx, -vy)))

for f in frame_numbers:
    kplist = interp_result.get(f, [[None,None]]*NUM_KP)
    left_sh = kplist[L_SH]; right_sh = kplist[R_SH]; left_hip = kplist[L_HIP]; right_hip = kplist[R_HIP]
    m_sh = mid(left_sh, right_sh); m_hip = mid(left_hip, right_hip)
    tilt = tilt_angle(m_hip, m_sh)
    vert_v = None
    if prev_mid_hip is not None and m_hip is not None:
        vert_v = float(m_hip[1] - prev_mid_hip[1])
    prev_mid_hip = m_hip if m_hip is not None else prev_mid_hip
    bboxn = bboxes_norm.get(f, [None,None,None,None])
    h_w = None
    if bboxn[2] is not None and bboxn[2] > 1e-9:
        h_w = float(bboxn[3]/bboxn[2])
    gprox = None
    if m_sh is not None:
        gprox = float(1.0 - m_sh[1])
    frame_data.append({
        "frame_idx": int(f),
        "keypoints": kplist,
        "normalized_bbox": bboxn,
        "score": float(scores.get(f, 0.0)),
        "features": {"tilt_angle": tilt, "vertical_velocity": vert_v, "h_w_ratio": h_w, "ground_proximity": gprox}
    })

# ---------- save JSON ----------
with open(STAGE2_JSON, "w") as f:
    json.dump({"frame_data": frame_data}, f, indent=2)
print("Saved Stage-2 JSON:", STAGE2_JSON)
print("Annotated frames saved to:", VIS_DIR)


# -------- CELL 2: Stage-1 training (no CSV) using folder paths; optional quick Stage-1 inference --------
# Edit the two PATH variables below before running.
import os, random, time, json
from glob import glob
import numpy as np
import cv2
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# -------------------- EDIT THESE PATHS --------------------
# Colab example: CAUCA already at /content/CAUCA_FALL
CAUCA_PATH      = "/content/CAUCA_FALL"                       # <-- adjust if running locally
 # <-- set to your local FallVision path
# --------------------------------------------------------

CKPT_OUT = "trained_stage1_local.pth"
CLIP_LEN = 32
FRAME_SIZE = 224
BATCH_SIZE = 4
EPOCHS = 4
LR = 1e-4
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 0   # safe cross-platform
GENERATE_SAMPLE_INFERENCE = True   # set False to skip producing stage1_output.json

print("Device:", DEVICE)
print("Checking dataset paths...")

# -------------------- helper: robust recursive collect & label inference --------------------
def infer_label_from_path(path):
    """
    Heuristic: if any path segment contains 'fall' (and doesn't contain 'no' or 'non'),
    label=1 (Fall). If it contains patterns like 'no', 'no_fall', 'nofall', 'nonfall', treat as 0.
    Otherwise return None.
    """
    parts = [p.lower() for p in path.split(os.sep) if p]
    for p in parts[::-1]:  # check leaf-to-root
        if any(x in p for x in ["no fall","no_fall","nofall","nonfall","no-fall","no_fall"]):
            return 0
        if "fall" in p and not any(x in p for x in ["no","non","not","none"]):
            return 1
    return None

def collect_videos(root):
    samples = []
    if not os.path.exists(root):
        print("Path not found:", root)
        return samples
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith((".avi", ".mp4", ".mov", ".MOV")):
                full = os.path.join(dirpath, fn)
                label = infer_label_from_path(dirpath)
                # fallback: if label still None, try parent folder name
                if label is None:
                    label = 1 if "fall" in dirpath.lower() else 0
                samples.append((full, int(label)))
    return samples

# -------------------- gather samples --------------------
samples = []
samples += collect_videos(CAUCA_PATH)
samples += collect_videos(FALLVISION_PATH)

if len(samples) == 0:
    raise RuntimeError(
        "No videos found under the provided paths.\n"
        f"CAUCA_PATH exists: {os.path.exists(CAUCA_PATH)}\n"
        f"FALLVISION_PATH exists: {os.path.exists(FALLVISION_PATH)}\n\n"
        "If you are using Colab and FALLVISION_PATH is a local macOS path, Colab cannot access it.\n"
        "Either run this notebook locally (so the path exists), or move FallVision into Colab (e.g. upload /content/FallVision)."
    )

random.shuffle(samples)
split_idx = max(1, int(0.8*len(samples)))
train_samples = samples[:split_idx]
val_samples   = samples[split_idx:]

print("Total videos:", len(samples))
print("Train:", len(train_samples), "Val:", len(val_samples))

# -------------------- Dataset --------------------
class VideoClipDataset(Dataset):
    def __init__(self, samples, clip_len=CLIP_LEN, frame_size=FRAME_SIZE):
        self.samples = samples
        self.clip_len = clip_len
        self.frame_size = frame_size

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        vpath, label = self.samples[idx]
        cap = cv2.VideoCapture(vpath)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            # return black clip if unreadable
            arr = np.zeros((3,self.clip_len,self.frame_size,self.frame_size), dtype=np.float32)
            return torch.from_numpy(arr), torch.tensor(label, dtype=torch.long)
        indices = np.linspace(0, max(1,total-1), self.clip_len).astype(int)
        frames = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((self.frame_size,self.frame_size,3), dtype=np.uint8)
            else:
                frame = cv2.resize(frame, (self.frame_size,self.frame_size))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame.astype(np.float32)/255.0)
        cap.release()
        arr = np.stack(frames, axis=0)           # (T,H,W,C)
        tensor = torch.from_numpy(arr).permute(3,0,1,2).float()  # (C,T,H,W)
        return tensor, torch.tensor(label, dtype=torch.long)

train_loader = DataLoader(VideoClipDataset(train_samples), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_loader   = DataLoader(VideoClipDataset(val_samples),   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

# -------------------- Model (Stage1_3DCNN) --------------------
class Stage1_3DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv3d(3,64,(3,7,7),(1,2,2),(1,3,3), padding_mode='replicate')
        self.bn1 = nn.BatchNorm3d(64)
        self.pool1 = nn.MaxPool3d((1,3,3),(1,2,2),(0,1,1))
        self.conv2 = nn.Conv3d(64,128,3,padding=1, padding_mode='replicate')
        self.bn2 = nn.BatchNorm3d(128)
        self.pool2 = nn.MaxPool3d(2,2)
        self.conv3 = nn.Conv3d(128,256,3,padding=1, padding_mode='replicate')
        self.bn3 = nn.BatchNorm3d(256)
        self.conv4 = nn.Conv3d(256,256,3,padding=1, padding_mode='replicate')
        self.bn4 = nn.BatchNorm3d(256)
        self.global_pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256,2)

    def forward(self, x, return_saliency=False):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))   # saliency layer
        # compute saliency for optional return
        sal = x.detach().pow(2).mean(dim=(1,3,4))
        T_prime = sal.shape[1]
        mask = torch.ones(T_prime).to(sal.device)
        mask[0] = 0.05; mask[-1] = 0.5
        sal = sal * mask
        min_val = sal.min(dim=1, keepdim=True)[0]
        max_val = sal.max(dim=1, keepdim=True)[0]
        sal = (sal - min_val) / (max_val - min_val + 1e-6)
        sal = F.softmax(sal * 5.0, dim=1)

        x = F.relu(self.bn4(self.conv4(x)))
        x = self.global_pool(x).flatten(1)
        x = self.dropout(x)
        logits = self.fc(x)
        if return_saliency:
            return logits, sal
        return logits

# -------------------- Training loop --------------------
model = Stage1_3DCNN().to(DEVICE)
optimizer = torch.optim.Adam(model.parameters(), lr=LR)
criterion = nn.CrossEntropyLoss()

best_val_loss = float("inf")
for epoch in range(1, EPOCHS+1):
    model.train()
    running_loss = 0.0; total = 0; correct = 0
    t0 = time.time()
    for xb, yb in train_loader:
        xb = xb.to(DEVICE); yb = yb.to(DEVICE)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * xb.size(0)
        preds = logits.argmax(1)
        correct += (preds == yb).sum().item()
        total += xb.size(0)
    train_loss = running_loss / (total if total>0 else 1)
    train_acc  = correct / (total if total>0 else 1)

    # validation
    model.eval()
    val_loss = 0.0; v_total = 0; v_correct = 0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            logits = model(xb)
            loss = criterion(logits, yb)
            val_loss += loss.item() * xb.size(0)
            preds = logits.argmax(1)
            v_correct += (preds == yb).sum().item()
            v_total += xb.size(0)
    val_loss = val_loss / (v_total if v_total>0 else 1)
    val_acc = v_correct / (v_total if v_total>0 else 1)

    print(f"Epoch {epoch}/{EPOCHS} | train_loss={train_loss:.4f} train_acc={train_acc:.3f} | val_loss={val_loss:.4f} val_acc={val_acc:.3f} | time={(time.time()-t0):.1f}s")
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), CKPT_OUT)
        print("Saved best checkpoint ->", CKPT_OUT)

print("Training completed. Best val loss:", best_val_loss)

# -------------------- Optional quick inference to produce stage1_output.json --------------------
if GENERATE_SAMPLE_INFERENCE:
    # choose a sample video (prefer a val sample)
    sample_video = val_samples[0][0] if len(val_samples)>0 else train_samples[0][0]
    print("Running quick Stage-1 inference on:", sample_video)
    try:
        import math
        from scipy.ndimage import gaussian_filter1d
    except Exception:
        print("Installing scipy (for gaussian smoothing)")
        !{sys.executable} -m pip install -q scipy
        from scipy.ndimage import gaussian_filter1d

    def run_stage1_inference(video_path, ckpt_path, out_json="stage1_output.json"):
        device = DEVICE
        inf_model = Stage1_3DCNN().to(device)
        inf_model.load_state_dict(torch.load(ckpt_path, map_location=device))
        inf_model.eval()
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, total_frames-1, CLIP_LEN).astype(int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if not ret:
                frame = np.zeros((FRAME_SIZE,FRAME_SIZE,3), dtype=np.uint8)
            else:
                frame = cv2.resize(frame, (FRAME_SIZE,FRAME_SIZE))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame.astype(np.float32)/255.0)
        cap.release()
        frames_np = np.array(frames, dtype=np.float32)
        clip = torch.from_numpy(frames_np).permute(3,0,1,2).unsqueeze(0).to(device)
        with torch.no_grad():
            logits, saliency = inf_model(clip, return_saliency=True)
        saliency = saliency.squeeze(0).cpu().numpy()
        saliency_interp = np.interp(np.linspace(0, len(saliency)-1, CLIP_LEN), np.arange(len(saliency)), saliency)
        saliency_smooth = gaussian_filter1d(saliency_interp, sigma=1.2)
        start,end,status = 0, CLIP_LEN-1, "unknown"
        # compute active window as before (simple threshold)
        from scipy.ndimage import label
        threshold = np.mean(saliency_smooth) + 0.5 * np.std(saliency_smooth)
        is_active = saliency_smooth > threshold
        labels, num_features = label(is_active)
        if num_features == 0:
            start, end, status = 0, CLIP_LEN-1, "Static"
        else:
            # pick largest energy island
            total_energy = np.sum(saliency_smooth)
            sig_idx = []
            for i in range(1, num_features+1):
                mask = labels==i
                if np.sum(saliency_smooth[mask]) / (total_energy + 1e-9) > 0.15:
                    sig_idx.extend(np.where(mask)[0].tolist())
            if len(sig_idx)==0:
                peak = int(np.argmax(saliency_smooth))
                start, end, status = max(0,peak-2), min(CLIP_LEN-1, peak+2), "peak"
            else:
                start, end, status = min(sig_idx), max(sig_idx), "global"
        out = {
            "video_path": video_path,
            "active_window": [int(start), int(end)],
            "saliency_weights": saliency_smooth.tolist(),
            "clip_indices": indices.tolist()
        }
        with open(out_json, "w") as fh:
            json.dump(out, fh, indent=2)
        print("Saved stage1_output.json for sample video ->", out_json)
        return out

    if os.path.exists(CKPT_OUT):
        run_stage1_inference(sample_video, CKPT_OUT, out_json="stage1_output.json")
    else:
        print("No checkpoint found at", CKPT_OUT, "- cannot run quick inference.")


        # -------- CELL 3: Visualization — combine stage1_output.json + stage2 overlays into combined images --------
# Edit STAGE2_VIS_DIR if your Stage-2 overlays were saved elsewhere.

import os, json, cv2, numpy as np

STAGE1_JSON = "stage1_output.json"
STAGE2_JSON = "stage2_output.json"
STAGE2_VIS_DIR = "/content/stage2_vis"   # change to local folder if you ran Stage-2 locally
OUT_DIR = "stage_results"
os.makedirs(OUT_DIR, exist_ok=True)

if not os.path.exists(STAGE1_JSON):
    raise FileNotFoundError(f"{STAGE1_JSON} not found. Run Stage-1 quick inference (Cell 2 with GENERATE_SAMPLE_INFERENCE=True) or produce stage1_output.json.")

if not os.path.exists(STAGE2_JSON):
    print(f"Warning: {STAGE2_JSON} not found. You can still create combined images if stage2 overlays exist in {STAGE2_VIS_DIR} and you have stage1_output.json.")

with open(STAGE1_JSON, "r") as f:
    s1 = json.load(f)

clip_indices = s1.get("clip_indices")
active_window = s1.get("active_window")
saliency = np.array(s1.get("saliency_weights", [])) if s1.get("saliency_weights") else None

if clip_indices is None or active_window is None:
    raise ValueError("stage1_output.json missing keys clip_indices or active_window.")

clip_start, clip_end = int(active_window[0]), int(active_window[1])
clip_positions = list(range(clip_start, clip_end+1))
frame_numbers = [int(clip_indices[p]) for p in clip_positions]

# load stage2 mapping if available
frame_data_map = {}
if os.path.exists(STAGE2_JSON):
    with open(STAGE2_JSON, "r") as f:
        s2 = json.load(f)
    for entry in s2.get("frame_data", []):
        frame_data_map[int(entry["frame_idx"])] = entry

for cp, frame_num in zip(clip_positions, frame_numbers):
    overlay_path = os.path.join(STAGE2_VIS_DIR, f"frame_{frame_num:05d}.png")
    if not os.path.exists(overlay_path):
        print("Overlay missing for frame", frame_num, "at", overlay_path, "- skipping.")
        continue
    img = cv2.imread(overlay_path)[:,:,::-1]
    h,w,_ = img.shape

    # build small saliency strip (optional)
    if saliency is not None:
        norm = (saliency - saliency.min()) / (saliency.max()-saliency.min()+1e-8)
        strip = 255 * np.ones((80, w, 3), dtype=np.uint8)
        pts = [ (int(j*(w/len(norm))), 40 - int(norm[j]*30)) for j in range(len(norm)) ]
        for j in range(len(pts)-1):
            cv2.line(strip, pts[j], pts[j+1], (255,0,0), 2)
    else:
        strip = 255 * np.ones((80, w, 3), dtype=np.uint8)

    combined = np.vstack([img, strip])
    # annotate with stage2 features if present
    info = frame_data_map.get(frame_num)
    if info:
        feats = info.get("features", {})
        txt = f"Frame {frame_num} | tilt={feats.get('tilt_angle')} v_vel={feats.get('vertical_velocity')} h/w={feats.get('h_w_ratio')} gp={feats.get('ground_proximity')}"
    else:
        txt = f"Frame {frame_num} | no stage2 entry"
    cv2.putText(combined, txt, (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

    out_path = os.path.join(OUT_DIR, f"combined_{frame_num:05d}.png")
    cv2.imwrite(out_path, combined[:,:,::-1])
    print("Saved combined:", out_path)

print("Visualization complete. Check folder:", OUT_DIR)