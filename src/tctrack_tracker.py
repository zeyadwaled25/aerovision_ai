"""
📌 TCTrack Tracker (Competition Version)

Purpose:
Harness Temporal Context for robust aerial tracking.
No manual Optical Flow or EMA needed, the Transformer handles occlusion natively!
"""

import cv2
import torch
import sys
import os

# PATH HACK: Make sure to adjust this if your tctrack code is in a different location
if "./tctrack" not in sys.path:
    sys.path.insert(0, "./tctrack")

from pysot.core.config import cfg
from pysot.models.utile_tctrack.model_builder import ModelBuilder_tctrack
from pysot.tracker.tctrack_tracker import TCTrackTracker
from pysot.utils.model_load import load_pretrain

# CONFIG & PATHS
CONFIG_PATH = "./tctrack/experiments/TCTrack/config.yaml"
WEIGHTS_PATH = "./models/tctrack.pth" 

cfg.merge_from_file(CONFIG_PATH)
device = "cuda" if torch.cuda.is_available() else "cpu"

print("⏳ Loading TCTrack model...")
model = ModelBuilder_tctrack('test')
model = load_pretrain(model, WEIGHTS_PATH).to(device).eval()
tracker = TCTrackTracker(model)

# Hyperparameters (From config)
hp = [cfg.TRACK.PENALTY_K, cfg.TRACK.WINDOW_INFLUENCE, cfg.TRACK.LR]

VISUALIZE = True  # Set to False for faster submission

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
    boxes = sequence["boxes"]

    cap = cv2.VideoCapture(video_path)

    ret, frame = cap.read()
    if not ret:
        return {"status": "failed", "predictions": []}

    h_img, w_img = frame.shape[:2]

    # Initialize Tracker
    x, y, w, h = map(float, init_bbox)
    tracker.init(frame, [x, y, w, h])

    if VISUALIZE:
        cv2.namedWindow("TCTrack (Prediction)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("TCTrack (Prediction)", 1150, 750)

    predictions = [{
        "id": f"{seq_name}_0",
        "x": round(x, 2), "y": round(y, 2), "w": round(w, 2), "h": round(h, 2)
    }]

    frame_idx = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # TCTrack Inference (No manual freeze or flow needed!)
        outputs = tracker.track(frame, hp)
        
        bbox = outputs['bbox']
        score = float(outputs.get('best_score', 0))

        x, y, w, h = bbox
        MAX_JUMP = 100  # Max allowed jump in pixels (heuristic)
        if frame_idx > 1:
            lx, ly, lw, lh = last_bbox
            dx = abs(x - lx)
            dy = abs(y - ly)
            
            if dx > MAX_JUMP or dy > MAX_JUMP:
                x, y, w, h = lx, ly, lw, lh
        
        last_bbox = [x, y, w, h]

        if frame_idx > 1:
            lx, ly, lw, lh = last_bbox
            
            # حساب مركز الهدف القديم والجديد
            cx_old, cy_old = lx + lw/2, ly + lh/2
            cx_new, cy_new = x + w/2, y + h/2
            
            # حساب المسافة اللي الهدف اتحركها (السرعة)
            move_dist = ((cx_new - cx_old)**2 + (cy_new - cy_old)**2)**0.5
            
            # لو الهدف فجأة اتحرك مسافة أكبر من نص حجمه في فريم واحد، أو الـ Score وقع
            # ده معناه بنسبة 90% إنه مسك في Distractor (مشتت)
            if move_dist > (lw * 0.5) or score < 0.3:
                # نرفض النطة تماماً ونستخدم مكانه القديم
                x, y, w, h = lx, ly, lw, lh
            else:
                # لو النطة منطقية، نعمل تنعيم بسيط (EMA) عشان نقتل أي رعشة
                ALPHA = 0.8
                x = ALPHA * x + (1 - ALPHA) * lx
                y = ALPHA * y + (1 - ALPHA) * ly
                w = ALPHA * w + (1 - ALPHA) * lw
                h = ALPHA * h + (1 - ALPHA) * lh

        last_bbox = [x, y, w, h]
        
        # Clip Box to stay inside frame
        x, y, w, h = clip_box([x, y, w, h], w_img, h_img)

        # Save
        predictions.append({
            "id": f"{seq_name}_{frame_idx}",
            "x": round(x, 2),
            "y": round(y, 2),
            "w": round(w, 2),
            "h": round(h, 2)
        })

        # Visualization
        if VISUALIZE:
            color = (255, 0, 0)
            # TCTrack is so robust, it usually doesn't "fail" locally, it just updates its context
            cv2.rectangle(frame, (int(x), int(y)), (int(x + w), int(y + h)), color, 2)

            cv2.putText(frame, "TCTrack (Temporal Context)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            cv2.putText(frame, f"Score: {score:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(frame, f"Frame: {frame_idx}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"Seq: {seq_name}", (20, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # GT (optional)
            if boxes is not None and frame_idx < len(boxes):
                xg, yg, wg, hg = map(float, boxes[frame_idx])
                if wg > 0 and hg > 0:
                    cv2.rectangle(frame, (int(xg), int(yg)), (int(xg + wg), int(yg + hg)), (0, 255, 0), 2)

            cv2.imshow("TCTrack (Prediction)", frame)

            key = cv2.waitKey(10) & 0xFF
            if key == 27:
                break
            if key == ord('q'):
                return {"status": "stop", "predictions": predictions}

        frame_idx += 1

    cap.release()
    if VISUALIZE:
        cv2.destroyAllWindows()

    return {"status": "done", "predictions": predictions}