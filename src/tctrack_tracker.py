"""
📌 TCTrack Tracker (The Diamond Version - Perfected)

Final Enhancements Applied:
1. UAV-Kinematics: Relaxed Velocity & Scale penalties for natural drone movements.
2. Safe Initialization: Pre-filled score history [0.5]*5 to prevent cold-start drops.
3. Crash Protection: Strict NaN and Negative dimension fallbacks.
4. Jitter-Free: Gentle EMA occlusion handling.
"""

import cv2
import torch
import sys
import numpy as np
import os

# PATH HACK
if "./tctrack" not in sys.path:
    sys.path.insert(0, "./tctrack")

from pysot.core.config import cfg
from pysot.models.utile_tctrack.model_builder import ModelBuilder_tctrack
from pysot.tracker.tctrack_tracker import TCTrackTracker
from pysot.utils.model_load import load_pretrain

CONFIG_PATH = "./tctrack/experiments/TCTrack/config.yaml"
WEIGHTS_PATH = "./models/tctrack.pth" 

cfg.merge_from_file(CONFIG_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"

print("⏳ Loading TCTrack model (Weights only)...")
model = ModelBuilder_tctrack('test')
model = load_pretrain(model, WEIGHTS_PATH).to(device).eval()

# 🚀 TUNED HYPERPARAMETERS
TUNED_HP = [0.08, 0.55, 0.25] 

VISUALIZE = False  # Set to False for Kaggle extraction

# HELPERS
def clip_box(box, w, h):
    x, y, bw, bh = box
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    bw = max(1, min(bw, w - x))
    bh = max(1, min(bh, h - y))
    return [x, y, bw, bh]

# TRACKER
def run_tracker(sequence):
    video_path = sequence["video_path"]
    init_bbox = sequence["init_bbox"]
    seq_name = sequence["seq_name"]
    
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        return {"status": "failed", "predictions": []}

    h_img, w_img = frame.shape[:2]

    # Local Tracker Instance (No State Leak)
    tracker = TCTrackTracker(model)
    x, y, w, h = map(float, init_bbox)
    tracker.init(frame, [x, y, w, h])
    
    csv_box = [x, y, w, h]

    if VISUALIZE:
        cv2.namedWindow("TCTrack (Diamond)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("TCTrack (Diamond)", 1150, 750)

    predictions = [{
        "id": f"{seq_name}_0",
        "x": round(x, 2), "y": round(y, 2), "w": round(w, 2), "h": round(h, 2)
    }]

    # =======================================================
    # 📈 STATE VARIABLES & WARMUP (FIX 3)
    # =======================================================
    score_history = [0.5] * 5  # Pre-filled for stability
    prev_center = (x + w/2, y + h/2)
    prev_area = w * h
    frame_idx = 1

    lost_counter = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        outputs = tracker.track(frame, TUNED_HP)
        raw_x, raw_y, raw_w, raw_h = outputs['bbox']
        score = float(outputs.get('best_score', 0))

        # =======================================================
        # 🚨 CRASH PROTECTION: NaN & GARBAGE FALLBACK (FIX 4 & 5)
        # =======================================================
        if raw_w <= 0 or raw_h <= 0 or np.isnan(raw_x) or np.isnan(raw_y) or np.isnan(raw_w) or np.isnan(raw_h):
            raw_x, raw_y, raw_w, raw_h = csv_box
            score = 0.0  # Force occlusion logic

        cx = raw_x + raw_w / 2
        cy = raw_y + raw_h / 2

        # =======================================================
        # 🛸 UAV KINEMATICS: RELAXED PENALTIES (FIX 1 & 2)
        # =======================================================
        # 1. Velocity check
        dx = cx - prev_center[0]
        dy = cy - prev_center[1]
        speed = (dx**2 + dy**2) ** 0.5
        bbox_diagonal = (raw_w**2 + raw_h**2) ** 0.5
        bbox_diagonal = max(bbox_diagonal, 1.0)
        velocity_th = bbox_diagonal * (0.6 + 0.2 * score)
        
        if speed > velocity_th:  # Relaxed for UAV motion
            score *= 0.6

        # 2. Scale check
        curr_area = raw_w * raw_h
        ratio = curr_area / (prev_area + 1e-6)
        if ratio > 2.2 or ratio < 0.45:    # Relaxed for UAV altitude changes
            score *= 0.7

        prev_center = (cx, cy)
        prev_area = curr_area

        # =======================================================
        # 🧠 DYNAMIC THRESHOLD & HISTORY
        # =======================================================
        score_history.append(score)
        if len(score_history) > 20:
            score_history.pop(0)

        dynamic_th = max(0.30, np.mean(score_history) * 0.7)

        # =======================================================
        # 🌊 OCCLUSION HANDLING & EMA
        # =======================================================
        if frame_idx > 1:
            if score >= dynamic_th:
                # Target is visible & confident -> Adaptive EMA
                lost_counter = 0
                ALPHA = max(0.45, min(score, 0.85)) 
                csv_box[0] = ALPHA * raw_x + (1 - ALPHA) * csv_box[0]
                csv_box[1] = ALPHA * raw_y + (1 - ALPHA) * csv_box[1]
                csv_box[2] = ALPHA * raw_w + (1 - ALPHA) * csv_box[2]
                csv_box[3] = ALPHA * raw_h + (1 - ALPHA) * csv_box[3]
            else:
                lost_counter += 1

                if lost_counter < 20:
                    OCC_ALPHA = 0.20
                    csv_box[0] = OCC_ALPHA * raw_x + (1 - OCC_ALPHA) * csv_box[0]
                    csv_box[1] = OCC_ALPHA * raw_y + (1 - OCC_ALPHA) * csv_box[1]
                else:
                    # smooth freeze (anti-drift)
                    DECAY = 0.95
                    csv_box[0] = DECAY * csv_box[0] + (1 - DECAY) * raw_x
                    csv_box[1] = DECAY * csv_box[1] + (1 - DECAY) * raw_y
        else:
            csv_box = [raw_x, raw_y, raw_w, raw_h]

        csv_box = clip_box(csv_box, w_img, h_img)

        predictions.append({
            "id": f"{seq_name}_{frame_idx}",
            "x": round(csv_box[0], 2),
            "y": round(csv_box[1], 2),
            "w": round(csv_box[2], 2),
            "h": round(csv_box[3], 2)
        })

        if VISUALIZE:
            if score >= dynamic_th:
                color = (0, 255, 0)
                state_text = "Tracking"
            elif lost_counter < 20:
                color = (0, 165, 255) # Orange
                state_text = f"Occluded (Blend) - {lost_counter}"
            else:
                color = (0, 0, 255) # Red
                state_text = "LOST (Hard Freeze)"
            vx, vy, vw, vh = map(int, csv_box)
            cv2.rectangle(frame, (vx, vy), (vx + vw, vy + vh), color, 2)

            cv2.putText(frame, f"State: {state_text}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            cv2.putText(frame, f"Score: {score:.2f} | Thresh: {dynamic_th:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            cv2.imshow("TCTrack (Diamond)", frame)
            key = cv2.waitKey(10) & 0xFF
            if key == 27: break
            if key == ord('q'): return {"status": "stop", "predictions": predictions}

        frame_idx += 1

    cap.release()
    if VISUALIZE: cv2.destroyAllWindows()

    return {"status": "done", "predictions": predictions}