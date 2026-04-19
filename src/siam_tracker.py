"""
📌 SiamRPN Tracker (Competition Version)

Purpose:
Robust SiamRPN tracking with stability improvements.

Enhancements:
- Confidence-based freeze
- Motion constraint
- Size constraint
- EMA smoothing
- Optional visualization (for submission speed)
"""

import cv2
import torch
from torch.amp import autocast
import numpy as np

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

# 🔹 CONFIG
cfg.merge_from_file("models/config.yaml")

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ModelBuilder()
model.load_state_dict(torch.load("models/siamrpn.pth", map_location=device))
model.eval().to(device)

tracker = build_tracker(model)

# 🔹 SETTINGS
SCORE_TH = 0.4
MAX_JUMP = 100
ALPHA = 0.7
VISUALIZE = True  # Set to False for faster submission (no visualization)


# 🔹 HELPERS
def clip_box(box, w, h):
    x, y, bw, bh = box

    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    bw = max(1, min(bw, w - x))
    bh = max(1, min(bh, h - y))

    return [x, y, bw, bh]


# 🔹 TRACKER
def run_tracker(sequence):
    video_path = sequence["video_path"]
    init_bbox = sequence["init_bbox"]
    seq_name = sequence["seq_name"]
    boxes = sequence["boxes"]

    cap = cv2.VideoCapture(video_path)

    ret, frame = cap.read()
    if not ret:
        return {"status": "failed", "predictions": []}

    x, y, w, h = map(int, init_bbox)

    tracker.init(frame, (x, y, w, h))

    last_bbox = [x, y, w, h]

    # optical flow
    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if VISUALIZE:
        cv2.namedWindow("Tracking (Prediction)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tracking (Prediction)", 1150, 750)

    predictions = [{"id": f"{seq_name}_0", "x": x, "y": y, "w": w, "h": h}]
    frame_idx = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h_img, w_img = frame.shape[:2]
        curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # FP16 (Mixed Precision) for faster inference on GPU
        with autocast(device_type=device):
            outputs = tracker.track(frame)
        bbox = outputs["bbox"]
        score = float(outputs.get("best_score", 0))
        x, y, w, h = bbox

        # 1. Motion Constraint
        dx = abs(x - last_bbox[0])
        dy = abs(y - last_bbox[1])

        if dx > MAX_JUMP or dy > MAX_JUMP:
            x, y, w, h = last_bbox

        # 2. Size Constraint
        area = w * h
        last_area = last_bbox[2] * last_bbox[3]

        if last_area > 0:
            ratio = area / last_area
            if ratio > 2 or ratio < 0.5:
                w, h = last_bbox[2], last_bbox[3]

        # 3. Confidence Freeze WITH Camera Motion Compensation
        if score < SCORE_TH:
            tracking_ok = False
            dx_cam, dy_cam = 0, 0

            # Localized Optical Flow
            mask = np.zeros_like(prev_gray)
            lx, ly, lw, lh = map(int, last_bbox)
            
            # Add padding to the last bbox for better feature tracking
            pad_x, pad_y = max(10, lw), max(10, lh)
            y1 = max(0, ly - pad_y)
            y2 = min(h_img, ly + lh + pad_y)
            x1 = max(0, lx - pad_x)
            x2 = min(w_img, lx + lw + pad_x)
            mask[y1:y2, x1:x2] = 255 # Only track features in the vicinity of the last known position

            # extract good features from the previous frame
            p0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.3, minDistance=7)
            
            if p0 is not None:
                # calculate optical flow to find corresponding points in the current frame
                p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None)
                
                if p1 is not None:
                    good_new = p1[st == 1]
                    good_old = p0[st == 1]
                    
                    if len(good_new) > 0:
                        # calculate median movement to estimate camera motion
                        movement = good_new - good_old
                        dx_cam = np.median(movement[:, 0])
                        dy_cam = np.median(movement[:, 1])
            
            # update position based on camera motion
            x = int(last_bbox[0] + dx_cam)
            y = int(last_bbox[1] + dy_cam)
            w, h = last_bbox[2], last_bbox[3]
            
        else:
            tracking_ok = True
            # 4. Adaptive EMA Smoothing
            current_alpha = max(0.2, min(0.9, score))
            x = current_alpha * x + (1 - current_alpha) * last_bbox[0]
            y = current_alpha * y + (1 - current_alpha) * last_bbox[1]
            w = current_alpha * w + (1 - current_alpha) * last_bbox[2]
            h = current_alpha * h + (1 - current_alpha) * last_bbox[3]

        # 🔹 Clip
        x, y, w, h = clip_box([x, y, w, h], w_img, h_img)

        # 🔹 Update last bbox & prev_gray
        last_bbox = [x, y, w, h]
        prev_gray = curr_gray.copy()

        # 🔹 Save
        predictions.append({
            "id": f"{seq_name}_{frame_idx}", 
            "x": round(x, 2), 
            "y": round(y, 2), 
            "w": round(w, 2), 
            "h": round(h, 2)
        })

        # 🔹 Visualization
        if VISUALIZE:
            color = (255, 0, 0) if tracking_ok else (0, 0, 255)
            label = "Tracking" if tracking_ok else "Frozen (Flow)"

            vx, vy, vw, vh = map(int, [x, y, w, h])
            cv2.rectangle(frame, (vx, vy), (vx + vw, vy + vh), color, 2)

            cv2.putText(frame, label,
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        color,
                        2)

            cv2.putText(frame, f"Score: {score:.2f} | Alpha: {current_alpha if tracking_ok else 0:.2f}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.putText(frame, f"Frame: {frame_idx}",
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 0, 0),
                        2)

            cv2.putText(frame, f"Seq: {seq_name}",
                        (20, 160),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2)

            # GT (optional)
            if boxes is not None and frame_idx < len(boxes):
                xg, yg, wg, hg = map(int, boxes[frame_idx])
                if wg > 0 and hg > 0:
                    cv2.rectangle(frame, (xg, yg), (xg + wg, yg + hg), (0, 255, 0), 2)

            cv2.imshow("Tracking (Prediction)", frame)

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