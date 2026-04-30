"""
📌 HiFT — The Ultimate Hybrid SOTA Edition
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🏆 Architecture: Hierarchical Feature Transformer (HiFT)
🏆 Post-Processing: Ultimate Hybrid SOTA Decision Engine
  - Smooth Exponential Penalties (No Hard Cliffs)
  - Adaptive Tracking History (consecutive_tracks, recent_max_speed)
  - Asymmetric Size Gating (Growth vs. Shrink)
  - Two-Tier Proportional Tiny Object Boost
  - Score Floor Safety Net
  - Residual Momentum in Deep Lost Phase
"""

import cv2
import torch
import sys
import math
import numpy as np
import os

# =========================
# PATH CONFIG
# =========================
# ضبط المسار بناءً على هيكلة الملفات الجديدة
if "./HiFT" not in sys.path:
    sys.path.insert(0, "./HiFT")

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder 
from pysot.tracker.hift_tracker import HiFTTracker
from pysot.utils.model_load import load_pretrain

CONFIG_PATH = "./HiFT/experiments/config.yaml"
WEIGHTS_PATH = "./models/hift.pth" 

cfg.merge_from_file(CONFIG_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"

print("⏳ Loading HiFT (Transformer Backbone + Ultimate SOTA Engine)...")

model = ModelBuilder()
model = load_pretrain(model, WEIGHTS_PATH).to(device).eval()

# =========================
# UTILS & HELPERS
# =========================
def clip_box(box, W, H):
    return [
        max(0, min(box[0], W - 1)),
        max(0, min(box[1], H - 1)),
        max(2.0, min(box[2], W - box[0])),
        max(2.0, min(box[3], H - box[1]))
    ]

def soft_clip(val, lo, hi):
    return max(lo, min(val, hi))

def soft_penalty(excess_ratio: float, lo: float = 0.6) -> float:
    """Returns a multiplier in [lo, 1.0] using smooth exponential decay."""
    return lo + (1.0 - lo) * math.exp(-excess_ratio)

# =========================
# 🧠 TEMPORAL STATE (Hybrid)
# =========================
class TemporalState:
    def __init__(self, init_bbox):
        self.csv_box = list(init_bbox)
        self.score_history = [0.5] * 5
        self.lost_counter = 0
        self.consecutive_tracks = 0  
        
        self.vx_ema = 0.0
        self.vy_ema = 0.0
        
        self.stable_w = init_bbox[2]
        self.stable_h = init_bbox[3]
        
        self.recent_max_speed = 0.0

    def update_score(self, raw_score):
        self.score_history.append(min(1.0, raw_score))
        if len(self.score_history) > 20:
            self.score_history.pop(0)

    def dynamic_threshold(self):
        mean_score = sum(self.score_history) / max(1, len(self.score_history))
        
        motion_factor = min(2.0, 1.0 + self.recent_max_speed / 15.0)
        th_floor = 0.15 if self.recent_max_speed > 10 else 0.18
        
        th = max(th_floor, mean_score * (0.50 / motion_factor))
        
        base_max_lost = 12 if mean_score > 0.50 else 25
        if self.consecutive_tracks > 10:
            base_max_lost += 5
            
        max_lost = base_max_lost
        area = self.csv_box[2] * self.csv_box[3]
        
        if area < 1200:
            th *= 0.88  
            max_lost += 3  

        return th, max_lost

# =========================
# ⚙️ DECISION ENGINE (Hybrid)
# =========================
def decision_engine(state, raw_bbox, raw_score, tracker, frame_shape, init_size):
    raw_x, raw_y, raw_w, raw_h = raw_bbox
    H, W = frame_shape
    csv_box = state.csv_box
    score = raw_score
    dynamic_th, max_lost = state.dynamic_threshold()

    if raw_w <= 2 or raw_h <= 2 or any(math.isnan(v) for v in raw_bbox):
        return csv_box

    raw_cx = raw_x + raw_w / 2
    raw_cy = raw_y + raw_h / 2
    prev_cx = csv_box[0] + csv_box[2] / 2
    prev_cy = csv_box[1] + csv_box[3] / 2
    box_diag = math.hypot(csv_box[2], csv_box[3])
    current_speed = math.hypot(state.vx_ema, state.vy_ema)

    # ========== 1. TWO-TIER PROPORTIONAL TINY OBJECT BOOST ==========
    area = raw_w * raw_h
    if area < 1000 and raw_score > dynamic_th * 0.8: 
        boost_factor = 1.0 + 0.1 * ((raw_score - dynamic_th * 0.8) / (1.0 - dynamic_th * 0.8))
        score *= min(boost_factor, 1.08) 
    elif area < 400 and raw_score > 0.45:
        score *= 1.03

    # ========== 2. MOTION-ADAPTIVE VELOCITY CONSISTENCY ==========
    if state.lost_counter == 0 and box_diag > 0:
        pred_vx = raw_cx - prev_cx
        pred_vy = raw_cy - prev_cy
        velocity_diff = math.hypot(pred_vx - state.vx_ema, pred_vy - state.vy_ema)
        
        speed_tolerance = box_diag * (1.5 + min(2.0, current_speed / 10.0))
        
        if velocity_diff > speed_tolerance:
            excess = (velocity_diff - speed_tolerance) / speed_tolerance
            score *= soft_penalty(excess, lo=0.75) 
        
        instant_speed = math.hypot(pred_vx, pred_vy)
        state.recent_max_speed = 0.9 * state.recent_max_speed + 0.1 * instant_speed

    # ========== 3. MOTION-AWARE DISTANCE PENALTY ==========
    dist = math.hypot(raw_cx - prev_cx, raw_cy - prev_cy)
    base_dist_th = max(40.0, 2.0 * max(csv_box[2], csv_box[3]))
    dist_th = base_dist_th + current_speed * 2.0 
    
    if state.lost_counter > 5:
        dist_th *= 1.3 
        
    if dist > dist_th:
        dist_excess = (dist - dist_th) / max(dist_th, 1.0)
        score *= soft_penalty(dist_excess, lo=0.75)

    # ========== 4. ASYMMETRIC SIZE EXPLOSION PROTECTION ==========
    stable_area = max(1.0, state.stable_w * state.stable_h)
    size_ratio = (raw_w * raw_h) / stable_area
    
    if size_ratio > 2.5:
        excess = (size_ratio - 2.5) / 2.5
        score *= soft_penalty(excess, lo=0.82) 
    elif size_ratio < 0.4:
        excess = (0.4 - size_ratio) / 0.4
        score *= soft_penalty(excess, lo=0.75) 

    # ========== 5. SMART TRAJECTORY PENALTY ==========
    if state.lost_counter < 3:
        pot_vx = raw_cx - prev_cx
        pot_vy = raw_cy - prev_cy
        
        if 5.0 < current_speed <= 15.0 and dist > 5.0:
            direction_change = abs(math.atan2(state.vy_ema, state.vx_ema) - math.atan2(pot_vy + 1e-9, pot_vx + 1e-9))
            direction_change = min(direction_change, 2 * math.pi - direction_change)
            
            if direction_change > 2.5:
                turn_excess = (direction_change - 2.5) / 2.5
                score *= soft_penalty(turn_excess, lo=0.82)

    # ========== 6. PROGRESSIVE RECOVERY BOOST ==========
    if state.lost_counter > 8:
        score *= 1.08  
        
    if state.lost_counter > 3:
        progressive_boost = 1.0 + min(0.08, (state.lost_counter - 3) * 0.01)
        score *= progressive_boost

    # ========== 7. DYNAMIC RECOVERY GATE & SAFETY FLOOR ==========
    if state.lost_counter > 5 and raw_score > dynamic_th * 0.75:
        score = max(score, dynamic_th * 0.88)
        
    score = max(score, raw_score * 0.70)

    # =========================
    # STATE SWITCH
    # =========================
    if score >= dynamic_th:
        # ===== TRACK MODE =====
        state.lost_counter = 0
        state.consecutive_tracks += 1
        
        vx = raw_x - csv_box[0]
        vy = raw_y - csv_box[1]
        vx = soft_clip(vx, -W * 0.05, W * 0.05)
        vy = soft_clip(vy, -H * 0.05, H * 0.05)
        
        state.vx_ema = 0.80 * state.vx_ema + 0.20 * vx  
        state.vy_ema = 0.80 * state.vy_ema + 0.20 * vy
        
        raw_w = max(4.0, min(raw_w, init_size[0] * 10))
        raw_h = max(4.0, min(raw_h, init_size[1] * 10))
        
        tracker.size = np.array([raw_w, raw_h])
        state.stable_w, state.stable_h = raw_w, raw_h
        
        if score > 0.65:
            alpha = max(0.60, 0.70 - 0.005 * current_speed)
        elif score > 0.50:
            alpha = max(0.50, 0.60 - 0.005 * current_speed)
        else:
            alpha = max(0.40, 0.50 - 0.005 * current_speed)
            
        csv_box[0] = alpha * raw_x + (1 - alpha) * csv_box[0]
        csv_box[1] = alpha * raw_y + (1 - alpha) * csv_box[1]
        csv_box[2] = alpha * raw_w + (1 - alpha) * csv_box[2]
        csv_box[3] = alpha * raw_h + (1 - alpha) * csv_box[3]
        
    else:
        # ===== RECOVERY MODE =====
        state.lost_counter += 1
        state.consecutive_tracks = 0  
        
        pred_cx = csv_box[0] + csv_box[2]/2 + state.vx_ema * 1.5  
        pred_cy = csv_box[1] + csv_box[3]/2 + state.vy_ema * 1.5
        
        if state.lost_counter < max_lost // 2:
            # Phase 1: Inertia
            csv_box[0] += state.vx_ema
            csv_box[1] += state.vy_ema
            
            decay = 0.92 if state.lost_counter < 3 else 0.85
            state.vx_ema *= decay
            state.vy_ema *= decay
            
            tracker.center_pos = np.array([pred_cx, pred_cy])
            
        elif state.lost_counter <= max_lost:
            # Phase 2: Expand Search
            base_expand = 1.4 + 0.08 * state.lost_counter  
            
            if stable_area < 500: 
                base_expand *= 1.10 
            elif stable_area < 1500: 
                base_expand *= 1.05 
                
            expand = min(base_expand, 3.5)  
            
            new_w = state.stable_w * expand
            new_h = state.stable_h * expand
            
            tracker.center_pos = np.array([pred_cx, pred_cy])
            tracker.size = np.array([new_w, new_h])
            
        else:
            # Phase 3: Deep Lost (Residual Momentum)
            state.vx_ema *= 0.35  
            state.vy_ema *= 0.35
            
            pred_cx = csv_box[0] + csv_box[2]/2 + state.vx_ema
            pred_cy = csv_box[1] + csv_box[3]/2 + state.vy_ema
            
            expand = min(4.0, 1.8 + 0.06 * state.lost_counter)  
            new_w = state.stable_w * expand
            new_h = state.stable_h * expand
            
            tracker.center_pos = np.array([pred_cx, pred_cy])
            tracker.size = np.array([new_w, new_h])
            
        csv_box[2] = 0.85 * csv_box[2] + 0.15 * state.stable_w  
        csv_box[3] = 0.85 * csv_box[3] + 0.15 * state.stable_h
        
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
    
    # Initialize HiFT Tracker
    tracker = HiFTTracker(model)
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
            
        # HiFT model track (No TUNED_HP passed)
        outputs = tracker.track(frame)
        raw_bbox = outputs['bbox']
        raw_score = float(outputs['best_score'])
        
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