"""
📌 TCTrack++ V5 Inference Engine (The Physicist Edition)

Purpose:
Execute real-time tracking using a physics-aware, multi-hypothesis approach.

Overview:
- Alpha-Beta Kinematic Filter: Predicts velocity & acceleration during occlusions.
- Dynamic Hyperparameters: Adapts penalty_k & window_influence based on motion.
- Isolated Hypotheses: Wide and Shifted grid search without PySOT state poisoning.
- Decision Engine: Strict anti-teleportation and pure-logic state management.
"""

import os
import sys
import math
import random
import numpy as np
import cv2
import torch

# ==========================================
# 🛡️ 1. REPRODUCIBILITY LOCK
# ==========================================
def set_deterministic_seed(seed=42):
    """Locks all random seeds to ensure reproducible evaluation runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_deterministic_seed(42)

# ==========================================
# ⚙️ 2. PYSOT CONFIGURATION & MODEL LOADING
# ==========================================
# Inject PySOT path to allow internal module imports
PYSOT_PATH = "./tctrack"
if PYSOT_PATH not in sys.path:
    sys.path.insert(0, PYSOT_PATH)

from pysot.core.config import cfg
from pysot.models.utile_tctrackplus.model_builder import ModelBuilder_tctrackplus
from pysot.tracker.tctrack_tracker import TCTrackTracker
from pysot.utils.model_load import load_pretrain

CONFIG_PATH = os.path.join(PYSOT_PATH, "experiments/TCTrack++/config.yaml")
WEIGHTS_PATH = "./models/tctrack++.pth"

cfg.merge_from_file(CONFIG_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"🪄 Loading TCTrack++ V5 Engine on {device.upper()}...")

# Initialize Global Tracker Model to prevent reloading per sequence
GLOBAL_MODEL = ModelBuilder_tctrackplus("test")
GLOBAL_MODEL = load_pretrain(GLOBAL_MODEL, WEIGHTS_PATH).to(device).eval()

# Base Hyperparameters: [penalty_k, window_influence, lr]
BASE_HP = [0.05, 0.55, 0.15]

# ==========================================
# 🛠️ 3. UTILITIES & DYNAMIC ENGINE
# ==========================================
def clip_box(box, W, H):
    """Clips bounding box to image boundaries and ensures minimum size."""
    return [
        max(0, min(box[0], W - 1)),
        max(0, min(box[1], H - 1)),
        max(2.0, min(box[2], W - box[0])),
        max(2.0, min(box[3], H - box[1])),
    ]

def soft_clip(val, lo, hi):
    """Clips a value within a specific range gracefully."""
    return max(lo, min(val, hi))

def is_valid_jump(box1, box2):
    """Anti-teleportation filter: Rejects impossible spatial jumps."""
    cx1 = box1[0] + box1[2] / 2
    cy1 = box1[1] + box1[3] / 2
    cx2 = box2[0] + box2[2] / 2
    cy2 = box2[1] + box2[3] / 2

    dist = math.hypot(cx1 - cx2, cy1 - cy2)
    # Jump must not exceed 3x the maximum dimension of the object
    return dist < max(box1[2], box1[3]) * 3

def get_dynamic_hp(state, base_hp):
    """Adapts hyperparameters dynamically based on target motion and scale."""
    penalty_k, window_influence, lr = base_hp
    motion = math.hypot(state.vx, state.vy)
    area = state.stable_w * state.stable_h

    # Fast Motion: Relax window constraint
    if motion > 10.0:
        window_influence *= 0.85
        
    # Tiny Target: Forgive size penalties and relax window
    if area < 1000:
        penalty_k *= 0.8
        window_influence *= 0.9

    # Lost Recovery: Rapidly decay window influence to allow wide free-search
    if state.lost_counter > 0:
        window_influence = max(0.15, window_influence * (0.75 ** state.lost_counter))

    return [penalty_k, window_influence, lr]

# ==========================================
# 🧠 4. KINEMATIC STATE TRACKER
# ==========================================
class TemporalState:
    """Maintains physical state (Velocity/Acceleration) via Alpha-Beta Filtering."""
    def __init__(self, init_bbox):
        self.csv_box = list(init_bbox)
        self.score_history = [0.5] * 5

        self.lost_counter = 0
        
        # Alpha-Beta Kinematics
        self.vx = 0.0
        self.vy = 0.0
        self.ax = 0.0
        self.ay = 0.0

        self.stable_w = init_bbox[2]
        self.stable_h = init_bbox[3]

    def update_score(self, raw_score):
        self.score_history.append(min(1.0, raw_score))
        if len(self.score_history) > 20:
            self.score_history.pop(0)

    def dynamic_threshold(self):
        """Computes activation thresholds based on historical confidence and speed."""
        mean_score = sum(self.score_history) / len(self.score_history)
        motion = math.hypot(self.vx, self.vy)

        th = max(0.15, mean_score * (0.52 / (1.0 + motion / 10.0)))
        max_lost = 12 if mean_score > 0.5 else 25

        if self.stable_w * self.stable_h < 1200:
            th *= 0.9
            max_lost += 3

        return th, max_lost

# ==========================================
# ⚖️ 5. PURE LOGIC DECISION ENGINE
# ==========================================
def decision_engine(state, raw_bbox, raw_score, frame_shape):
    """Evaluates the best hypothesis purely logically without mutating PySOT internals."""
    raw_x, raw_y, raw_w, raw_h = raw_bbox
    H, W = frame_shape

    # Handle invalid output
    if raw_w <= 2 or raw_h <= 2 or any(math.isnan(v) for v in raw_bbox):
        return state.csv_box

    csv_box = state.csv_box
    score = raw_score

    dynamic_th, max_lost = state.dynamic_threshold()

    raw_cx = raw_x + raw_w / 2
    raw_cy = raw_y + raw_h / 2
    prev_cx = csv_box[0] + csv_box[2] / 2
    prev_cy = csv_box[1] + csv_box[3] / 2
    
    dist = math.hypot(raw_cx - prev_cx, raw_cy - prev_cy)
    motion = math.hypot(state.vx, state.vy)

    # Soft Distance Penalty
    dist_th = max(40.0, 2.2 * max(csv_box[2], csv_box[3])) + motion * 2
    if dist > dist_th:
        excess = (dist - dist_th) / dist_th
        score *= max(0.80, 1.0 - 0.15 * excess)

    # Size Control Penalty
    stable_area = max(1.0, state.stable_w * state.stable_h)
    size_ratio = (raw_w * raw_h) / stable_area
    if size_ratio > 2.5 or size_ratio < 0.4:
        score *= 0.85

    score = max(score, raw_score * 0.75) # Score Floor

    # Recovery Boost
    if state.lost_counter > 3:
        score *= 1.05
    if state.lost_counter > 5 and raw_score > dynamic_th * 0.7:
        score = max(score, dynamic_th * 0.85)

    # --- STATE TRANSITION ---
    if score >= dynamic_th:
        # 🟢 TRACK MODE
        state.lost_counter = 0

        raw_vx = soft_clip(raw_x - csv_box[0], -W * 0.05, W * 0.05)
        raw_vy = soft_clip(raw_y - csv_box[1], -H * 0.05, H * 0.05)

        # Alpha-Beta Filter Update
        alpha_kf = 0.40  
        beta_kf = 0.15   
        
        err_vx = raw_vx - state.vx
        err_vy = raw_vy - state.vy

        state.vx += alpha_kf * err_vx
        state.vy += alpha_kf * err_vy
        state.ax = soft_clip(state.ax + beta_kf * err_vx, -5.0, 5.0)
        state.ay = soft_clip(state.ay + beta_kf * err_vy, -5.0, 5.0)

        state.stable_w, state.stable_h = raw_w, raw_h
        alpha = 0.7 if score > 0.6 else 0.55

        csv_box[0] = alpha * raw_x + (1 - alpha) * csv_box[0]
        csv_box[1] = alpha * raw_y + (1 - alpha) * csv_box[1]
        csv_box[2] = alpha * raw_w + (1 - alpha) * csv_box[2]
        csv_box[3] = alpha * raw_h + (1 - alpha) * csv_box[3]

    else:
        # 🔴 RECOVERY MODE
        state.lost_counter += 1

        if state.lost_counter > 3:
            state.vx *= 0.6
            state.vy *= 0.6
            state.ax *= 0.3
            state.ay *= 0.3

        # Predict using Velocity + Acceleration
        pred_cx = csv_box[0] + csv_box[2] / 2 + state.vx * 1.5 + state.ax * 0.5
        pred_cy = csv_box[1] + csv_box[3] / 2 + state.vy * 1.5 + state.ay * 0.5

        if state.lost_counter < max_lost // 2:
            csv_box[0] += state.vx + state.ax * 0.5
            csv_box[1] += state.vy + state.ay * 0.5

        csv_box[2] = 0.85 * csv_box[2] + 0.15 * state.stable_w
        csv_box[3] = 0.85 * csv_box[3] + 0.15 * state.stable_h

    state.csv_box = csv_box
    return csv_box

# ==========================================
# 🚀 6. MAIN INFERENCE LOOP
# ==========================================
def run_tracker(sequence):
    """Executes the multi-hypothesis tracking pipeline on a given sequence."""
    cap = cv2.VideoCapture(sequence["video_path"])
    ret, frame = cap.read()
    if not ret:
        return {"status": "failed", "predictions": []}

    H, W = frame.shape[:2]
    tracker = TCTrackTracker(GLOBAL_MODEL)

    x, y, w, h = map(float, sequence["init_bbox"])
    tracker.init(frame, [x, y, w, h])

    state = TemporalState([x, y, w, h])
    predictions = [{"id": f"{sequence['seq_name']}_0", "x": x, "y": y, "w": w, "h": h}]
    idx = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        dynamic_th, max_lost = state.dynamic_threshold()
        current_hp = get_dynamic_hp(state, BASE_HP)

        # 🛡️ ISOLATE STATE
        center_backup = tracker.center_pos.copy()
        size_backup = tracker.size.copy()

        # --- HYPOTHESIS 1: MAIN ---
        tracker.center_pos = center_backup.copy()
        tracker.size = size_backup.copy()
        outputs_main = tracker.track(frame, current_hp)
        score_main = float(outputs_main.get("best_score", 0))

        best_outputs = outputs_main
        best_score = score_main
        base_score = score_main  

        # 🚨 Conditional Activation: Only run heavy hypotheses during occlusion
        if state.lost_counter > 2 or score_main < dynamic_th:

            # --- HYPOTHESIS 2: WIDE SEARCH ---
            tracker.center_pos = center_backup.copy()
            tracker.size = size_backup.copy() * 1.4
            outputs_wide = tracker.track(frame, current_hp)
            score_wide = float(outputs_wide.get("best_score", 0))

            # --- HYPOTHESIS 3: ACCELERATION SHIFTED GRID ---
            tracker.center_pos = center_backup.copy() + np.array(
                [state.vx * 1.5 + state.ax, state.vy * 1.5 + state.ay]
            )
            tracker.size = size_backup.copy()
            outputs_shift = tracker.track(frame, current_hp)
            score_shift = float(outputs_shift.get("best_score", 0))

            # 🛡️ RESTORE PURE STATE
            tracker.center_pos = center_backup.copy()
            tracker.size = size_backup.copy()

            # --- HYPOTHESIS SELECTION ---
            margin = 0.05 if score_main > 0.5 else 0.03
            
            if score_shift > base_score + margin and is_valid_jump(outputs_main["bbox"], outputs_shift["bbox"]):
                best_outputs = outputs_shift
                best_score = score_shift

            if score_wide > base_score + margin and is_valid_jump(outputs_main["bbox"], outputs_wide["bbox"]):
                if score_wide > best_score:
                    best_outputs = outputs_wide
                    best_score = score_wide

            # Killer Safety Check: Prevent Sudden Distractor Jumps
            if best_score > score_main + 0.25:
                best_outputs = outputs_main
                best_score = score_main

            best_score = max(best_score, score_main * 0.85)

        else:
            # Safe Restore if alternative hypotheses weren't run
            tracker.center_pos = center_backup.copy()
            tracker.size = size_backup.copy()

        # Deep Lost Penalty
        if score_main < 0.3:
            best_score *= 0.9

        # Evaluate via Decision Engine
        final_box = decision_engine(state, best_outputs["bbox"], best_score, frame.shape[:2])
        state.update_score(best_score)
        final_box = clip_box(final_box, W, H)

        # 🔄 MASTER SYNC
        _, max_lost = state.dynamic_threshold()
        if state.lost_counter > max_lost // 2:
            expand = min(1.5 + 0.1 * state.lost_counter, 3.5)
            tracker.size = np.array([state.stable_w * expand, state.stable_h * expand])
        else:
            tracker.size = np.array([final_box[2], final_box[3]])

        tracker.center_pos = np.array([final_box[0] + final_box[2] / 2, final_box[1] + final_box[3] / 2])

        predictions.append({
            "id": f"{sequence['seq_name']}_{idx}",
            "x": round(final_box[0], 2),
            "y": round(final_box[1], 2),
            "w": round(final_box[2], 2),
            "h": round(final_box[3], 2),
        })

        idx += 1

    cap.release()
    return {"status": "done", "predictions": predictions}