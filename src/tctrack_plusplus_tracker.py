"""
📌 TCTrack++

A production-ready single-object aerial tracker built on TCTrack++ (CVPR 2022 / TPAMI)
with zero-overhead kinematic recovery for the MTC-AIC4 competition.

Features:
1. Inertial Motion Prediction (Kalman-lite) — Zero computational overhead
2. Progressive Search Expansion — Modifies internal tracker state for re-detection
3. Sequence-Adaptive Patience — max_lost scales dynamically with sequence difficulty
4. Dual-Mode EMA — Fast response on high-confidence detections, smooth on medium

Author: AeroVision AI Team
Competition: MTC-AIC4 (10th International Competition of MTC)
"""

import cv2
import torch
import sys
import numpy as np

# PATH CONFIGURATION
# Ensure pysot toolkit is discoverable before importing tracker components
if "./tctrack" not in sys.path:
    sys.path.insert(0, "./tctrack")

from pysot.core.config import cfg
from pysot.models.utile_tctrackplus.model_builder import ModelBuilder_tctrackplus 
from pysot.tracker.tctrack_tracker import TCTrackTracker
from pysot.utils.model_load import load_pretrain


# MODEL CONFIGURATION
CONFIG_PATH = "./tctrack/experiments/TCTrack++/config.yaml"
WEIGHTS_PATH = "./models/tctrack++.pth" 

cfg.merge_from_file(CONFIG_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"

print("⏳ Loading TCTrack++ (God-Tier Engine)...")
model = ModelBuilder_tctrackplus('test')
model = load_pretrain(model, WEIGHTS_PATH).to(device).eval()

# Hyperparameters: [window_penalty, learning_rate, penalty_k]
# Tuned for aerial tracking with fast motion and scale variation
TUNED_HP = [0.08, 0.55, 0.25] 
VISUALIZE = False  # ⚡ MUST BE FALSE FOR MAXIMUM INFERENCE SPEED


# UTILITY FUNCTIONS
def clip_box(box, frame_width, frame_height):
    """
    Clamp bounding box coordinates to valid image bounds.
    
    Ensures:
    - Top-left corner (x, y) within [0, w-1] and [0, h-1]
    - Width/height minimum 2 pixels (prevent degenerate boxes)
    - Bottom-right corner does not exceed frame dimensions
    
    Args:
        box: [x, y, width, height] in pixel coordinates
        frame_width: Image width in pixels
        frame_height: Image height in pixels
    
    Returns:
        Clipped box [x, y, width, height]
    """
    return [
        max(0, min(box[0], frame_width - 1)),
        max(0, min(box[1], frame_height - 1)),
        max(2.0, min(box[2], frame_width - box[0])),
        max(2.0, min(box[3], frame_height - box[1]))
    ]



# MAIN TRACKING PIPELINE
def run_tracker(sequence):
    """
    Execute single-object tracking on a video sequence.
    
    Competition Protocol:
    - Tracker initialized with ground-truth bounding box in frame 0
    - Online-only: no access to future frames
    - No re-initialization allowed during sequence
    
    Args:
        sequence: Dictionary containing:
            - video_path: Path to input video file
            - init_bbox: [x, y, width, height] ground-truth in frame 0
            - seq_name: Sequence identifier for output formatting
    
    Returns:
        Dictionary with status and list of per-frame predictions
    """
    video_path = sequence["video_path"]
    init_bbox = sequence["init_bbox"]
    seq_name = sequence["seq_name"]
    
    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    if not ret:
        return {"status": "failed", "predictions": []}

    frame_height, frame_width = frame.shape[:2]

    # Initialize TCTrack++ tracker with pretrained model
    tracker = TCTrackTracker(model)
    x, y, w, h = map(float, init_bbox)
    tracker.init(frame, [x, y, w, h])
    
    # Output bounding box (what gets submitted for evaluation)
    csv_box = [x, y, w, h]
    predictions = [{
        "id": f"{seq_name}_0",
        "x": round(x, 2), "y": round(y, 2), "w": round(w, 2), "h": round(h, 2)
    }]

    # TRACKER STATE VARIABLES    

    # Score history for adaptive threshold computation
    # Maintains rolling window of last 20 confidence scores
    score_history = [0.5] * 5  
    frame_idx = 1
    lost_counter = 0
    
    # KALMAN-LITE MOTION PREDICTION

    # EMA-smoothed velocity estimates for inertial coasting during occlusion
    # vx_ema, vy_ema: velocity in x and y directions (pixels/frame)
    # stable_w, stable_h: last reliable target size before occlusion
    vx_ema, vy_ema = 0.0, 0.0
    stable_w, stable_h = w, h

    # MAIN TRACKING LOOP
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Run TCTrack++ inference
        outputs = tracker.track(frame, TUNED_HP)
        raw_x, raw_y, raw_w, raw_h = outputs['bbox']
        score = float(outputs.get('best_score', 0))

        # 1. LIGHTWEIGHT SAFETY CHECK
        # Reject degenerate or NaN detections immediately
        if raw_w <= 2 or raw_h <= 2 or any(np.isnan([raw_x, raw_y, raw_w, raw_h])):
            raw_x, raw_y, raw_w, raw_h = csv_box
            score = 0.0  

        # 2. DYNAMIC THRESHOLD & ADAPTIVE PATIENCE
        # Maintain rolling score history for robust statistics
        score_history.append(score)
        if len(score_history) > 20:
            score_history.pop(0)

        # Compute mean score (faster than np.mean for small lists)
        mean_score = sum(score_history) / len(score_history)
        
        # Dynamic threshold: adapts to sequence difficulty
        # Lower bound 0.32 ensures recovery triggers on challenging sequences
        dynamic_th = max(0.32, mean_score * 0.7)
        
        # Adaptive patience: shorter for easy sequences, longer for hard ones
        max_lost = 12 if mean_score > 0.55 else 25

        # 3. THE BEAST LOGIC
        #    Dual EMA + Kalman Inertia + Progressive Search
        if frame_idx > 1:
            if score >= dynamic_th:
                # CONFIDENT TRACKING MODE
                lost_counter = 0
                
                # Update stable size (used during recovery)
                stable_w, stable_h = raw_w, raw_h

                # Inertial Velocity Update (high-confidence only)
                # Only update velocity when tracker is very confident
                # Prevents drift from noisy low-confidence detections
                if score > 0.65:
                    # Compute instantaneous velocity from position delta
                    vx = (raw_x - csv_box[0])
                    vy = (raw_y - csv_box[1])
                    
                    # Clamp velocity to prevent insane jumps
                    # Max 5% of frame dimension per frame
                    vx = max(-frame_width * 0.05, min(vx, frame_width * 0.05))
                    vy = max(-frame_height * 0.05, min(vy, frame_height * 0.05))
                    
                    # EMA smoothing: 80% history, 20% new observation
                    vx_ema = 0.8 * vx_ema + 0.2 * vx
                    vy_ema = 0.8 * vy_ema + 0.2 * vy

                # Dual-Mode EMA Smoothing
                # Fast response (α=0.8) when score > 0.6 for agile targets
                # Smooth response (α=0.6) otherwise for stability
                ALPHA = 0.8 if score > 0.6 else 0.6
                
                csv_box[0] = ALPHA * raw_x + (1 - ALPHA) * csv_box[0]
                csv_box[1] = ALPHA * raw_y + (1 - ALPHA) * csv_box[1]
                csv_box[2] = ALPHA * raw_w + (1 - ALPHA) * csv_box[2]
                csv_box[3] = ALPHA * raw_h + (1 - ALPHA) * csv_box[3]

            else:
                # RECOVERY MODE — Target potentially occluded or lost
                lost_counter += 1

                if lost_counter < max_lost:
                    # PHASE 1: Inertial Coasting (Kalman-lite)
                    # Target temporarily hidden — continue motion prediction
                    # rather than freezing in place
                    csv_box[0] += vx_ema
                    csv_box[1] += vy_ema
                    
                    # Decay velocity (friction factor 0.9)
                    # Simulates gradual slowdown when no visual confirmation
                    vx_ema *= 0.90
                    vy_ema *= 0.90

                    # Force tracker to look at predicted location next frame
                    # Modifies PySOT internal state for search center
                    tracker.center_pos = np.array([
                        csv_box[0] + csv_box[2] / 2, 
                        csv_box[1] + csv_box[3] / 2
                    ])

                else:
                    # PHASE 2: Progressive Search Area Expansion
                    # Target fully lost — expand search region to re-acquire
                    expand_factor = min(
                        1.5 + 0.05 * (lost_counter - max_lost), 
                        3.5
                    )
                    
                    # Scale search region (not output box!)
                    new_w = stable_w * expand_factor
                    new_h = stable_h * expand_factor

                    # Update PySOT internal state for wider search
                    tracker.center_pos = np.array([
                        csv_box[0] + csv_box[2] / 2, 
                        csv_box[1] + csv_box[3] / 2
                    ])
                    tracker.size = np.array([new_w, new_h])

        else:
            # First frame after initialization — no smoothing
            csv_box = [raw_x, raw_y, raw_w, raw_h]

        # 4. CLIP & SAVE
        csv_box = clip_box(csv_box, frame_width, frame_height)
        
        predictions.append({
            "id": f"{seq_name}_{frame_idx}",
            "x": round(csv_box[0], 2),
            "y": round(csv_box[1], 2),
            "w": round(csv_box[2], 2),
            "h": round(csv_box[3], 2)
        })

        frame_idx += 1

    cap.release()
    return {"status": "done", "predictions": predictions}