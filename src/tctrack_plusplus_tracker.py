"""
📌 TCTrack++ — The Final Submission Edition
✅ Model safe 100% (Sub-50M params)
✅ Decision Layer: Optimized for AlexNet Backbone (No Over-penalization)
✅ Features: Micro-Target Safety Net, Size Explosion Protection, Velocity Kill-Switch
"""

import cv2
import torch
import sys
import math
import numpy as np

# =========================
# PATH CONFIG
# =========================
if "./tctrack" not in sys.path:
    sys.path.insert(0, "./tctrack")

from pysot.core.config import cfg
from pysot.models.utile_tctrackplus.model_builder import ModelBuilder_tctrackplus 
from pysot.tracker.tctrack_tracker import TCTrackTracker
from pysot.utils.model_load import load_pretrain

# تأكد إن المسارات دي مطابقة لبيئة كاجل أو اللوكال عندك
CONFIG_PATH = "./tctrack/experiments/TCTrack++/config.yaml"
WEIGHTS_PATH = "./models/tctrack++.pth" 

cfg.merge_from_file(CONFIG_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"

print("⏳ Loading TCTrack++ (Final Kaggle Baseline Engine)...")

model = ModelBuilder_tctrackplus('test')
model = load_pretrain(model, WEIGHTS_PATH).to(device).eval()

# =========================
# HYPERPARAMETERS
# =========================
TUNED_HP = [0.05, 0.55, 0.15]

# =========================
# UTILS
# =========================
def clip_box(box, W, H):
    return [
        max(0, min(box[0], W - 1)),
        max(0, min(box[1], H - 1)),
        max(2.0, min(box[2], W - box[0])),
        max(2.0, min(box[3], H - box[1]))
    ]

# =========================
# 🧠 TEMPORAL STATE
# =========================
class TemporalState:
    def __init__(self, init_bbox):
        self.csv_box = list(init_bbox)
        self.score_history = [0.5] * 5
        self.lost_counter = 0

        self.vx_ema = 0.0
        self.vy_ema = 0.0

        self.stable_w = init_bbox[2]
        self.stable_h = init_bbox[3]

    def update_score(self, raw_score):
        self.score_history.append(min(1.0, raw_score))
        if len(self.score_history) > 20:
            self.score_history.pop(0)

    def dynamic_threshold(self):
        mean_score = sum(self.score_history) / len(self.score_history)

        th = max(0.22, mean_score * 0.55)
        max_lost = 10 if mean_score > 0.55 else 20

        # Tiny Object Fix: Lower threshold for UAVs and distant targets
        area = self.csv_box[2] * self.csv_box[3]
        if area < 1200:
            th *= 0.85
            max_lost += 5

        return th, max_lost


# =========================
# ⚙️ DECISION ENGINE
# =========================
def decision_engine(state, raw_bbox, raw_score, tracker, frame_shape, init_size):

    raw_x, raw_y, raw_w, raw_h = raw_bbox
    H, W = frame_shape

    csv_box = state.csv_box
    score = raw_score

    dynamic_th, max_lost = state.dynamic_threshold()

    if raw_w <= 2 or raw_h <= 2 or any(math.isnan(v) for v in raw_bbox):
        return csv_box

    # =========================
    # Penalties & Anti-Distractors
    # =========================
    raw_cx = raw_x + raw_w / 2
    raw_cy = raw_y + raw_h / 2

    prev_cx = csv_box[0] + csv_box[2] / 2
    prev_cy = csv_box[1] + csv_box[3] / 2

    dist = math.hypot(raw_cx - prev_cx, raw_cy - prev_cy)

    # 1. Micro-Motion Safety Net
    dist_th = max(50.0, 2.5 * max(csv_box[2], csv_box[3]))
    if state.lost_counter > 5:
        dist_th *= 1.5

    if dist > dist_th:
        score *= 0.85

    # 2. Size Explosion Protection (Anti-Background)
    size_ratio = (raw_w * raw_h) / max(1.0, (state.stable_w * state.stable_h))
    if size_ratio > 3.0 or size_ratio < 0.33:
        score *= 0.8  

    # 3. Trajectory Penalty (Camera Shake Tolerant)
    if state.lost_counter < 3:
        pot_vx = raw_cx - prev_cx
        pot_vy = raw_cy - prev_cy
        current_speed = math.hypot(state.vx_ema, state.vy_ema)
        
        if current_speed > 5.0 and dist > 5.0:
            direction_change = abs(math.atan2(state.vy_ema, state.vx_ema) - math.atan2(pot_vy, pot_vx))
            if direction_change > 2.8:
                score *= 0.75

    # Recovery Boosts
    if state.lost_counter > 8:
        score *= 1.1

    if state.lost_counter > 5 and raw_score > 0.4:
        score = max(score, dynamic_th * 0.9)

    # =========================
    # STATE SWITCH
    # =========================
    if score >= dynamic_th:
        # ===== TRACK MODE =====
        state.lost_counter = 0

        vx = raw_x - csv_box[0]
        vy = raw_y - csv_box[1]

        vx = max(-W * 0.05, min(vx, W * 0.05))
        vy = max(-H * 0.05, min(vy, H * 0.05))

        state.vx_ema = 0.85 * state.vx_ema + 0.15 * vx
        state.vy_ema = 0.85 * state.vy_ema + 0.15 * vy

        raw_w = max(4.0, min(raw_w, init_size[0] * 10))
        raw_h = max(4.0, min(raw_h, init_size[1] * 10))

        tracker.size = np.array([raw_w, raw_h])
        state.stable_w, state.stable_h = raw_w, raw_h

        alpha = 0.75 if score > 0.6 else 0.55

        csv_box[0] = alpha * raw_x + (1 - alpha) * csv_box[0]
        csv_box[1] = alpha * raw_y + (1 - alpha) * csv_box[1]
        csv_box[2] = alpha * raw_w + (1 - alpha) * csv_box[2]
        csv_box[3] = alpha * raw_h + (1 - alpha) * csv_box[3]

    else:
        # ===== RECOVERY MODE =====
        state.lost_counter += 1

        pred_cx = csv_box[0] + csv_box[2]/2 + state.vx_ema * 2.0
        pred_cy = csv_box[1] + csv_box[3]/2 + state.vy_ema * 2.0

        if state.lost_counter < max_lost // 2:
            # Phase 1: Inertia
            csv_box[0] += state.vx_ema
            csv_box[1] += state.vy_ema

            state.vx_ema *= 0.95
            state.vy_ema *= 0.95

            tracker.center_pos = np.array([pred_cx, pred_cy])

        elif state.lost_counter <= max_lost:
            # Phase 2: Expand search
            base_expand = 1.5 + 0.05 * state.lost_counter

            if state.stable_w * state.stable_h < 1500:
                base_expand *= 0.8

            expand = min(base_expand, 3.2)

            new_w = state.stable_w * expand
            new_h = state.stable_h * expand

            tracker.center_pos = np.array([pred_cx, pred_cy])
            tracker.size = np.array([new_w, new_h])
            
        else:
            # Phase 3: Velocity Kill-Switch (Anti-Drift)
            state.vx_ema = 0.0
            state.vy_ema = 0.0
            
            expand = min(3.5, 1.5 + 0.05 * state.lost_counter)
            new_w = state.stable_w * expand
            new_h = state.stable_h * expand
            
            tracker.center_pos = np.array([csv_box[0] + csv_box[2]/2, csv_box[1] + csv_box[3]/2])
            tracker.size = np.array([new_w, new_h])

        # Stabilize size (Anti-explosion)
        csv_box[2] = 0.9 * csv_box[2] + 0.1 * state.stable_w
        csv_box[3] = 0.9 * csv_box[3] + 0.1 * state.stable_h

    state.csv_box = csv_box
    return csv_box


# =========================
# 🚀 MAIN RUN
# =========================
def run_tracker(sequence):

    video_path = sequence["video_path"]
    init_bbox = sequence["init_bbox"]
    seq_name = sequence["seq_name"]

    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        return {"status": "failed", "predictions": []}

    H, W = frame.shape[:2]

    tracker = TCTrackTracker(model)

    x, y, w, h = map(float, init_bbox)
    tracker.init(frame, [x, y, w, h])

    state = TemporalState([x, y, w, h])

    predictions = [{
        "id": f"{seq_name}_0",
        "x": round(x, 2),
        "y": round(y, 2),
        "w": round(w, 2),
        "h": round(h, 2)
    }]

    frame_idx = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        outputs = tracker.track(frame, TUNED_HP)
        raw_bbox = outputs['bbox']
        raw_score = float(outputs.get('best_score', 0))

        final_box = decision_engine(
            state, raw_bbox, raw_score, tracker, frame.shape[:2], (w, h)
        )

        state.update_score(raw_score)
        final_box = clip_box(final_box, W, H)

        predictions.append({
            "id": f"{seq_name}_{frame_idx}",
            "x": round(final_box[0], 2),
            "y": round(final_box[1], 2),
            "w": round(final_box[2], 2),
            "h": round(final_box[3], 2)
        })

        frame_idx += 1

    cap.release()

    return {
        "status": "done",
        "predictions": predictions
    }