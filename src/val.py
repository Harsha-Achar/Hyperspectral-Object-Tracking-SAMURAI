import os
import sys
import math
import time
import csv
import argparse
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
import cv2
import pandas as pd

# Add local folder to system path first (so `tracker.py` resolves here)
DHRUV_DIR = os.path.dirname(os.path.abspath(__file__))
if DHRUV_DIR not in sys.path:
    sys.path.insert(0, DHRUV_DIR)

# Project root: HOTC_BASE_DIR env, else this folder (Hack_code)
BASE_DIR = os.path.abspath(os.environ.get("HOTC_BASE_DIR", DHRUV_DIR))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Import tracking backend factory
from track import get_tracker

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
TRAINING_DIR = os.path.join(DATASET_DIR, "training")
# Default to local ranking 2 split (official-style layout with init_rect.txt)
RANKING_DIR = os.path.join(BASE_DIR, "ranking 2")
VAL_DIR = os.environ.get("HOTC_VAL_DIR", RANKING_DIR if os.path.isdir(RANKING_DIR) else os.path.join(DATASET_DIR, "validation"))
TRAIN_CSV = os.path.join(BASE_DIR, "2026training.csv")
SAMPLE_SUB = os.path.join(DATASET_DIR, "sample_submission.csv")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")

SENSOR_FOLDER_MAP = {"vis": "HSI-VIS", "nir": "HSI-NIR", "rednir": "HSI-RedNIR"}
SENSOR_CFG = {
    "vis": {"block": 4, "raw_bands": 16, "keep_bands": 16},
    "nir": {"block": 5, "raw_bands": 25, "keep_bands": 25},
    "rednir": {"block": 4, "raw_bands": 16, "keep_bands": 15},
}
ACTIVE_SENSORS = ["vis", "nir", "rednir"]

USE_GRID_RESAMPLE, GRID_STEP_NM = True, 20.0
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

BAND_WAVELENGTH_NM = {
    "vis": np.linspace(470.0, 630.0, 16).astype(np.float32),
    "nir": np.linspace(675.0, 975.0, 25).astype(np.float32),
    "rednir": np.linspace(600.0, 875.0, 15).astype(np.float32),
}
BAND_SORT = {s: np.argsort(BAND_WAVELENGTH_NM[s]) for s in BAND_WAVELENGTH_NM}
BAND_WL_SORTED = {s: BAND_WAVELENGTH_NM[s][BAND_SORT[s]] for s in BAND_WAVELENGTH_NM}

# -----------------------------------------------------------------------------
# Reusing Hyperspectral Loading Logic from train_dhruv_2.py
# -----------------------------------------------------------------------------
import glob

def _build_grid():
    grid, interp = {}, {}
    for sensor, wl in BAND_WL_SORTED.items():
        lo, hi = float(wl[0]), float(wl[-1])
        n = max(int(round((hi - lo) / GRID_STEP_NM)) + 1, 4)
        dst = np.linspace(lo, hi, n).astype(np.float32)
        grid[sensor] = dst
        matrix = np.zeros((len(dst), len(wl)), dtype=np.float32)
        for i, wavelength in enumerate(dst):
            j = int(np.clip(np.searchsorted(wl, wavelength), 1, len(wl) - 1))
            t = float(
                np.clip(
                    (wavelength - wl[j - 1]) / max(wl[j] - wl[j - 1], 1e-6),
                    0.0,
                    1.0,
                )
            )
            matrix[i, j - 1], matrix[i, j] = 1.0 - t, t
            interp[sensor] = matrix
    return grid, interp

GRID_WL, GRID_INTERP = _build_grid()
GRID_BANDS = {s: len(GRID_WL[s]) for s in GRID_WL}

def resample_to_grid(cube, sensor):
    matrix = GRID_INTERP[sensor]
    return (cube @ matrix.T).astype(np.float32)

def x2cube(img, block, band_number):
    img = img.astype(np.float32)
    height, width = img.shape
    hb, wb = height // block, width // block
    tensor = img[: hb * block, : wb * block].reshape(hb, block, wb, block)
    tensor = np.transpose(tensor, (0, 2, 1, 3)).reshape(hb, wb, block, block)
    return np.transpose(tensor, (0, 1, 3, 2)).reshape(hb, wb, band_number)

def load_cube(frame_path, sensor):
    img = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(frame_path)
    if img.ndim == 3:
        img = img[..., 0]
    cfg = SENSOR_CFG[sensor]
    cube = x2cube(img, cfg["block"], cfg["raw_bands"])[..., : cfg["keep_bands"]]
    cube = np.ascontiguousarray(cube[..., BAND_SORT[sensor]])
    return resample_to_grid(cube, sensor)

def normalize_per_band(cube):
    lo = cube.min(axis=(0, 1), keepdims=True)
    hi = cube.max(axis=(0, 1), keepdims=True)
    return ((cube - lo) / (hi - lo + 1e-6)).astype(np.float32)

# -----------------------------------------------------------------------------
# FalseColorGen exactly as implemented in train_dhruv_2.py
# -----------------------------------------------------------------------------
class FalseColorGen(nn.Module):
    def __init__(self, sensors):
        super().__init__()
        self.logits = nn.ParameterDict()
        for s in sensors:
            B = GRID_BANDS[s]
            init = torch.full((3, B), -2.0)
            third = max(B // 3, 1)
            for c in range(3):
                hi = (c + 1) * third if c < 2 else B
                init[c, c * third:hi] = 2.0
            self.logits[s] = nn.Parameter(init)

    def forward(self, minmax_bands, sensor):
        w = torch.softmax(self.logits[sensor], dim=1)
        rgb = torch.einsum('cb,nbhw->nchw', w, minmax_bands)
        return (rgb - IMAGENET_MEAN.to(rgb.device)) / IMAGENET_STD.to(rgb.device)

# Helper to load FC weights
def load_fc_weights(fc_model, checkpoint_path, device):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print("Using default FalseColorGen weights.")
        return fc_model
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
        fc_state = {}
        for k, v in state.items():
            if k.startswith("fc."):
                fc_state[k[3:]] = v
            elif k.startswith("module.fc."):
                fc_state[k[10:]] = v
        if fc_state:
            fc_model.load_state_dict(fc_state, strict=True)
            print(f"Loaded FalseColorGen weights from {checkpoint_path}")
        else:
            print("Warning: No FalseColorGen weights found in checkpoint. Using default.")
    except Exception as e:
        print(f"Error loading FalseColorGen: {e}. Using default.")
    return fc_model

# -----------------------------------------------------------------------------
# Local Validation Dataset Building
# -----------------------------------------------------------------------------
def discover_clips_train(root, sensor_key):
    folder = os.path.join(root, SENSOR_FOLDER_MAP[sensor_key])
    if not os.path.isdir(folder):
        return {}
    clips = {}
    for name in sorted(os.listdir(folder)):
        directory = os.path.join(folder, name)
        if os.path.isdir(directory):
            img_dir = os.path.join(directory, "img")
            target_dir = img_dir if os.path.isdir(img_dir) else directory
            frames = sorted(glob.glob(os.path.join(target_dir, "*.png")))
            if frames:
                clips[name] = frames
    return clips

def build_local_validation_sequences():
    disk = {s: discover_clips_train(TRAINING_DIR, s) for s in SENSOR_FOLDER_MAP}
    
    if not os.path.exists(TRAIN_CSV):
        print(f"Local train CSV not found at: {TRAIN_CSV}")
        return []
        
    df = pd.read_csv(TRAIN_CSV)
    df["video"] = df["ID"].str.rsplit("_", n=1).str[0]
    df["frame_num"] = df["ID"].str.rsplit("_", n=1).str[1].astype(int)
    df["sensor"] = df["video"].str.split("-").str[0]
    df["clip_name"] = df["video"].str.split("-", n=1).str[1]
    df = df[df.apply(lambda row: row["clip_name"] in disk.get(row["sensor"], {}), axis=1)]
    df = df.sort_values(["video", "frame_num"]).reset_index(drop=True)
    df["local_idx"] = df.groupby("video").cumcount()

    def path_of(row):
        frames = disk[row["sensor"]][row["clip_name"]]
        return frames[row["local_idx"]] if row["local_idx"] < len(frames) else ""

    df["frame_path"] = df.apply(path_of, axis=1)
    df = df[df.sensor.isin(ACTIVE_SENSORS)]
    df = df[df["frame_path"].apply(lambda path: bool(path) and os.path.exists(path))].reset_index(drop=True)
    
    from sklearn.model_selection import GroupKFold
    df["fold"] = -1
    for sensor in ACTIVE_SENSORS:
        videos = df[df.sensor == sensor]["video"].unique()
        if len(videos) < 2:
            df.loc[df.sensor == sensor, "fold"] = 1
            continue
        gkf = GroupKFold(n_splits=min(5, len(videos)))
        for fold, (_, val_idx) in enumerate(gkf.split(videos, groups=videos)):
            for video in videos[val_idx]:
                df.loc[df.video == video, "fold"] = fold
                
    sequences = []
    for video, group in df.groupby("video"):
        group = group.sort_values("frame_num").reset_index(drop=True)
        sequence = [
            {
                "frame_path": row["frame_path"],
                "x": float(row["x"]),
                "y": float(row["y"]),
                "width": float(row["width"]),
                "height": float(row["height"]),
                "sensor": row["sensor"],
                "video": row["video"],
                "fold": int(row["fold"]),
            }
            for _, row in group.iterrows()
        ]
        if len(sequence) >= 2:
            sequences.append(sequence)
            
    val_sequences = [s for s in sequences if s[0]["fold"] == 0]
    return val_sequences

# -----------------------------------------------------------------------------
# Official Validation Split Sequence Building
# -----------------------------------------------------------------------------
def discover_official_val_sequences(val_dir):
    val_path = Path(val_dir)
    if not val_path.exists():
        print(f"Official validation dir not found at: {val_dir}")
        return []
    sensors = ["vis", "nir", "rednir"]
    sensor_folders = {"vis": "HSI-VIS", "nir": "HSI-NIR", "rednir": "HSI-RedNIR"}
    
    sequences = []
    for sensor in sensors:
        folder_name = sensor_folders[sensor]
        sensor_dir = val_path / folder_name
        if not sensor_dir.exists():
            continue
        for seq_dir in sorted(p for p in sensor_dir.iterdir() if p.is_dir()):
            seq_name = seq_dir.name
            frame_paths = sorted(
                p for p in seq_dir.iterdir()
                if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
            )
            if not frame_paths:
                continue
            box_path = None
            for box_name in ("init_rect.txt", "groundtruth_rect.txt"):
                bp = seq_dir / box_name
                if bp.exists():
                    box_path = bp
                    break
            if not box_path:
                continue
            boxes = []
            with box_path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = [p for p in line.replace("\t", ",").replace(" ", ",").split(",") if p]
                    if len(parts) >= 4:
                        boxes.append(tuple(float(v) for v in parts[:4]))
            if not boxes:
                continue
            
            sequences.append({
                "sensor": sensor,
                "name": seq_name,
                "frame_paths": [str(p) for p in frame_paths],
                "init_box": boxes[0],
                "all_gt": boxes if len(boxes) == len(frame_paths) else None
            })
    return sequences

# -----------------------------------------------------------------------------
# Metric Helpers
# -----------------------------------------------------------------------------
def iou_xywh(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x0, y0 = max(ax, bx), max(ay, by)
    x1, y1 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0

# -----------------------------------------------------------------------------
# Parallel Processing Helper
# -----------------------------------------------------------------------------
def process_sequence(seq_info):
    seq, tracker_name, fc_gen_state, mode, device, decimals = seq_info
    
    # Thread-local FalseColorGen and tracker instantiation
    fc_gen = FalseColorGen(ACTIVE_SENSORS).to(device)
    fc_gen.load_state_dict(fc_gen_state)
    fc_gen.eval()
    
    tracker = get_tracker(tracker_name, device=device)
    
    if mode == "local":
        video_name = seq[0]["video"]
        sensor = seq[0]["sensor"]
        frame_paths = [f["frame_path"] for f in seq]
        init_box = (seq[0]["x"], seq[0]["y"], seq[0]["width"], seq[0]["height"])
        gts = [(f["x"], f["y"], f["width"], f["height"]) for f in seq]
    else:
        video_name = seq["name"]
        sensor = seq["sensor"]
        frame_paths = seq["frame_paths"]
        init_box = seq["init_box"]
        gts = seq["all_gt"]
        
    try:
        cube0 = load_cube(frame_paths[0], sensor)
    except Exception as e:
        print(f"Error loading frame 0 for {sensor}-{video_name}: {e}")
        return f"{sensor}-{video_name}", None, None, 0
        
    cube0_mm = normalize_per_band(cube0)
    cube0_tensor = torch.from_numpy(cube0_mm).permute(2, 0, 1).unsqueeze(0).float().to(device)
    
    with torch.inference_mode():
        rgb0_tensor = fc_gen(cube0_tensor, sensor)
        
    rgb0_denorm = rgb0_tensor * IMAGENET_STD.to(device) + IMAGENET_MEAN.to(device)
    rgb0_denorm = torch.clamp(rgb0_denorm, 0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()
    rgb0_uint8 = (rgb0_denorm * 255.0).astype(np.uint8)
    
    tracker.initialize(rgb0_uint8, init_box)
    
    pred_boxes = [init_box]
    ious = []
    center_errors = []
    
    if gts:
        ious.append(1.0)
        center_errors.append(0.0)
        
    for idx in range(1, len(frame_paths)):
        try:
            cube = load_cube(frame_paths[idx], sensor)
        except Exception as e:
            pred_boxes.append(pred_boxes[-1])
            if gts:
                ious.append(0.0)
                center_errors.append(9999.0)
            continue
            
        cube_mm = normalize_per_band(cube)
        cube_tensor = torch.from_numpy(cube_mm).permute(2, 0, 1).unsqueeze(0).float().to(device)
        
        with torch.inference_mode():
            rgb_tensor = fc_gen(cube_tensor, sensor)
            
        rgb_denorm = rgb_tensor * IMAGENET_STD.to(device) + IMAGENET_MEAN.to(device)
        rgb_denorm = torch.clamp(rgb_denorm, 0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()
        rgb_uint8 = (rgb_denorm * 255.0).astype(np.uint8)
        
        _, bbox, _ = tracker.track(rgb_uint8)
        pred_boxes.append(bbox)
        
        if gts:
            gt = gts[idx]
            iou = iou_xywh(bbox, gt)
            ious.append(iou)
            
            gt_cx, gt_cy = gt[0] + gt[2]/2, gt[1] + gt[3]/2
            pred_cx, pred_cy = bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2
            err = float(np.hypot(gt_cx - pred_cx, gt_cy - pred_cy))
            center_errors.append(err)
            
    if hasattr(tracker, "tmp_dir_holder") and tracker.tmp_dir_holder is not None:
        try:
            tracker.tmp_dir_holder.cleanup()
        except Exception:
            pass
            
    metrics = None
    if gts:
        iou_arr = np.array(ious)
        err_arr = np.array(center_errors)
        thresholds = np.arange(0.0, 1.0001, 0.05)
        
        seq_iou = float(iou_arr.mean())
        seq_auc = float(np.mean([(iou_arr > t).mean() for t in thresholds]))
        seq_dp20 = float((err_arr <= 20).mean())
        seq_fail = float((iou_arr <= 0).mean())
        
        metrics = {
            "video": video_name,
            "sensor": sensor,
            "iou": seq_iou,
            "auc": seq_auc,
            "dp20": seq_dp20,
            "fail": seq_fail
        }
        
    video_key = f"{sensor}-{video_name}"
    return video_key, pred_boxes, metrics, len(frame_paths)

# -----------------------------------------------------------------------------
# Main Evaluation Loop
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate tracker on HSI sequences")
    parser.add_argument(
        "--tracker",
        default="samurai_base",
        choices=["samurai_base", "samurai_large"],
        help="samurai_* needs SAMURAI_REPO_DIR + CUDA (via tracker_2)",
    )
    parser.add_argument(
        "--mode",
        default="official",
        choices=["local", "official"],
        help="local fold-0 eval or official/ranking split submission generation",
    )
    parser.add_argument("--val-dir", default=VAL_DIR)
    parser.add_argument("--limit-seqs", type=int, default=0)
    parser.add_argument("--max-frames", type=int, default=0, help="Optional cap on frames per sequence (0 = all)")
    parser.add_argument("--checkpoint", default=os.path.join(CKPT_DIR, "best_hsi_ostrack_dc.pt"), help="Checkpoint to load FalseColorGen weights from")
    parser.add_argument("--decimals", type=int, default=6)
    parser.add_argument("--output", default=os.path.join(BASE_DIR, "submission_ranking2.csv"))
    parser.add_argument("--workers", type=int, default=1, help="Number of parallel workers (sequences processed in parallel)")
    args = parser.parse_args()
    
    print(f"Tracker: {args.tracker}")
    print(f"Mode: {args.mode}")
    print(f"Val dir: {args.val_dir}")
    print(f"Device: {DEVICE}")
    print(f"Workers: {args.workers}")
    
    # Initialize FalseColorGen and load weights
    fc_gen = FalseColorGen(ACTIVE_SENSORS).to(DEVICE)
    fc_gen = load_fc_weights(fc_gen, args.checkpoint, DEVICE)
    fc_gen.eval()
    fc_gen_state = fc_gen.state_dict()
    
    # Load Sequences
    if args.mode == "local":
        sequences = build_local_validation_sequences()
        if not sequences:
            print("No local validation sequences found.")
            return
    else:
        sequences = discover_official_val_sequences(args.val_dir)
        if not sequences:
            print("No official validation sequences found.")
            return
            
    if args.limit_seqs > 0:
        sequences = sequences[:args.limit_seqs]
        print(f"Limiting to first {args.limit_seqs} sequences.")

    if args.max_frames > 0:
        trimmed = []
        for seq in sequences:
            if args.mode == "local":
                trimmed.append(seq[: args.max_frames])
            else:
                seq = dict(seq)
                seq["frame_paths"] = seq["frame_paths"][: args.max_frames]
                if seq.get("all_gt") is not None:
                    seq["all_gt"] = seq["all_gt"][: args.max_frames]
                trimmed.append(seq)
        sequences = trimmed
        print(f"Limiting each sequence to first {args.max_frames} frames.")
        
    print(f"Total sequences to process: {len(sequences)}")
    
    results = []
    submission_rows = {}
    
    if args.workers > 1:
        from concurrent.futures import ThreadPoolExecutor, as_completed
        tasks = [(seq, args.tracker, fc_gen_state, args.mode, DEVICE, args.decimals) for seq in sequences]
        
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_sequence, t): t for t in tasks}
            progress = tqdm(as_completed(futures), total=len(futures), desc="Processing in parallel")
            for fut in progress:
                video_key, pred_boxes, metrics, num_frames = fut.result()
                if pred_boxes is not None:
                    submission_rows[video_key] = pred_boxes
                    if metrics:
                        results.append(metrics)
                        progress.write(f"[{len(results)}/{len(sequences)}] Video {video_key} ({num_frames} frames) done. AUC: {metrics['auc']:.3f} | DP20: {metrics['dp20']:.3f}")
    else:
        # Sequential tracking
        tracker = get_tracker(args.tracker, device=DEVICE)
        for seq_idx, seq in enumerate(sequences):
            if args.mode == "local":
                video_name = seq[0]["video"]
                sensor = seq[0]["sensor"]
                frame_paths = [f["frame_path"] for f in seq]
                init_box = (seq[0]["x"], seq[0]["y"], seq[0]["width"], seq[0]["height"])
                gts = [(f["x"], f["y"], f["width"], f["height"]) for f in seq]
            else:
                video_name = seq["name"]
                sensor = seq["sensor"]
                frame_paths = seq["frame_paths"]
                init_box = seq["init_box"]
                gts = seq["all_gt"]
                
            print(f"\n[{seq_idx+1}/{len(sequences)}] Video: {sensor}-{video_name} ({len(frame_paths)} frames)")
            
            try:
                cube0 = load_cube(frame_paths[0], sensor)
            except Exception as e:
                print(f"Error loading frame 0: {e}. Skipping sequence.")
                continue
                
            cube0_mm = normalize_per_band(cube0)
            cube0_tensor = torch.from_numpy(cube0_mm).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
            
            with torch.inference_mode():
                rgb0_tensor = fc_gen(cube0_tensor, sensor)
                
            rgb0_denorm = rgb0_tensor * IMAGENET_STD.to(DEVICE) + IMAGENET_MEAN.to(DEVICE)
            rgb0_denorm = torch.clamp(rgb0_denorm, 0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()
            rgb0_uint8 = (rgb0_denorm * 255.0).astype(np.uint8)
            
            tracker.initialize(rgb0_uint8, init_box)
            
            pred_boxes = [init_box]
            ious = []
            center_errors = []
            
            if gts:
                ious.append(1.0)
                center_errors.append(0.0)
                
            t0 = time.time()
            for idx in tqdm(range(1, len(frame_paths)), desc="Tracking"):
                try:
                    cube = load_cube(frame_paths[idx], sensor)
                except Exception as e:
                    print(f"Error loading frame {idx}: {e}")
                    pred_boxes.append(pred_boxes[-1])
                    if gts:
                        ious.append(0.0)
                        center_errors.append(9999.0)
                    continue
                    
                cube_mm = normalize_per_band(cube)
                cube_tensor = torch.from_numpy(cube_mm).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
                
                with torch.inference_mode():
                    rgb_tensor = fc_gen(cube_tensor, sensor)
                    
                rgb_denorm = rgb_tensor * IMAGENET_STD.to(DEVICE) + IMAGENET_MEAN.to(DEVICE)
                rgb_denorm = torch.clamp(rgb_denorm, 0.0, 1.0).squeeze(0).permute(1, 2, 0).cpu().numpy()
                rgb_uint8 = (rgb_denorm * 255.0).astype(np.uint8)
                
                _, bbox, _ = tracker.track(rgb_uint8)
                pred_boxes.append(bbox)
                
                if gts:
                    gt = gts[idx]
                    iou = iou_xywh(bbox, gt)
                    ious.append(iou)
                    
                    gt_cx, gt_cy = gt[0] + gt[2]/2, gt[1] + gt[3]/2
                    pred_cx, pred_cy = bbox[0] + bbox[2]/2, bbox[1] + bbox[3]/2
                    err = float(np.hypot(gt_cx - pred_cx, gt_cy - pred_cy))
                    center_errors.append(err)
                    
            elapsed = time.time() - t0
            fps = (len(frame_paths) - 1) / max(elapsed, 1e-3)
            print(f"Done in {elapsed:.1f}s ({fps:.1f} fps)")
            
            video_key = f"{sensor}-{video_name}"
            submission_rows[video_key] = pred_boxes
            
            if gts:
                iou_arr = np.array(ious)
                err_arr = np.array(center_errors)
                thresholds = np.arange(0.0, 1.0001, 0.05)
                
                seq_iou = float(iou_arr.mean())
                seq_auc = float(np.mean([(iou_arr > t).mean() for t in thresholds]))
                seq_dp20 = float((err_arr <= 20).mean())
                seq_fail = float((iou_arr <= 0).mean())
                
                results.append({
                    "video": video_name,
                    "sensor": sensor,
                    "iou": seq_iou,
                    "auc": seq_auc,
                    "dp20": seq_dp20,
                    "fail": seq_fail
                })
                
                print(f"Metrics: AUC={seq_auc:.3f} | DP20={seq_dp20:.3f} | IoU={seq_iou:.3f} | Fail={seq_fail:.3f}")
                
        if hasattr(tracker, "tmp_dir_holder") and tracker.tmp_dir_holder is not None:
            try:
                tracker.tmp_dir_holder.cleanup()
            except Exception:
                pass
                
    # Print aggregate report
    if results:
        print("\n" + "="*80)
        print("EVALUATION REPORT")
        print("="*80)
        print(f"{'video':30s} {'sens':7s} {'AUC':>6s} {'DP20':>6s} {'IoU':>6s} {'fail':>6s}")
        for r in sorted(results, key=lambda x: x["auc"]):
            print(f"{r['video'][:30]:30s} {r['sensor']:7s} {r['auc']:6.3f} {r['dp20']:6.3f} {r['iou']:6.3f} {r['fail']:6.3f}")
        print("-"*80)
        mean_auc = np.mean([r["auc"] for r in results])
        mean_dp = np.mean([r["dp20"] for r in results])
        mean_iou = np.mean([r["iou"] for r in results])
        mean_fail = np.mean([r["fail"] for r in results])
        print(f"{'MEAN':30s} {'all':7s} {mean_auc:6.3f} {mean_dp:6.3f} {mean_iou:6.3f} {mean_fail:6.3f}")
        
        for sensor in ACTIVE_SENSORS:
            sens_res = [r for r in results if r["sensor"] == sensor]
            if sens_res:
                s_auc = np.mean([r["auc"] for r in sens_res])
                s_dp = np.mean([r["dp20"] for r in sens_res])
                s_iou = np.mean([r["iou"] for r in sens_res])
                s_fail = np.mean([r["fail"] for r in sens_res])
                print(f"{'MEAN':30s} {sensor:7s} {s_auc:6.3f} {s_dp:6.3f} {s_iou:6.3f} {s_fail:6.3f}")
        print("="*80)
        
    # Write submission CSV if in official mode
    if args.mode == "official" and args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Generating submission file in {args.output}...")

        sample_path = Path(SAMPLE_SUB)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "x", "y", "width", "height"])

            if sample_path.exists():
                with sample_path.open("r", encoding="utf-8") as sf:
                    sample_rows = list(csv.DictReader(sf))
                for row in sample_rows:
                    row_id = row["ID"]
                    try:
                        video_id, frame_str = row_id.rsplit("_", 1)
                        frame_idx = int(frame_str) - 1
                    except ValueError:
                        writer.writerow([row_id, "", "", "", ""])
                        continue

                    box = None
                    if video_id in submission_rows and frame_idx < len(submission_rows[video_id]):
                        box = submission_rows[video_id][frame_idx]

                    if box is not None:
                        writer.writerow([
                            row_id,
                            round(box[0], args.decimals),
                            round(box[1], args.decimals),
                            round(box[2], args.decimals),
                            round(box[3], args.decimals),
                        ])
                    else:
                        writer.writerow([row_id, "", "", "", ""])
            else:
                # ranking 2 (and similar) often has no sample_submission.csv —
                # emit one row per predicted frame: {sensor}-{name}_{frame:04d}
                print(f"Sample submission {SAMPLE_SUB} not found; writing predictions directly.")
                for video_key in sorted(submission_rows.keys()):
                    for frame_idx, box in enumerate(submission_rows[video_key]):
                        row_id = f"{video_key}_{frame_idx + 1:04d}"
                        writer.writerow([
                            row_id,
                            round(box[0], args.decimals),
                            round(box[1], args.decimals),
                            round(box[2], args.decimals),
                            round(box[3], args.decimals),
                        ])

        print(f"Finished writing submission CSV.")

if __name__ == "__main__":
    main()
