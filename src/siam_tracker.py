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

    if VISUALIZE:
        cv2.namedWindow("Tracking (Prediction)", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Tracking (Prediction)", 1150, 750)

    predictions = [{
        "id": f"{seq_name}_0",
        "x": x, "y": y, "w": w, "h": h
    }]

    frame_idx = 1

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h_img, w_img = frame.shape[:2]

        outputs = tracker.track(frame)

        bbox = outputs["bbox"]
        score = float(outputs.get("best_score", 0))

        x, y, w, h = map(int, bbox)

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

        # 3. Confidence Freeze
        if score < SCORE_TH:
            x, y, w, h = last_bbox
            tracking_ok = False
        else:
            tracking_ok = True

        # 4. EMA Smoothing
        x = int(ALPHA * x + (1 - ALPHA) * last_bbox[0])
        y = int(ALPHA * y + (1 - ALPHA) * last_bbox[1])
        w = int(ALPHA * w + (1 - ALPHA) * last_bbox[2])
        h = int(ALPHA * h + (1 - ALPHA) * last_bbox[3])

        # 🔹 Clip
        x, y, w, h = clip_box([x, y, w, h], w_img, h_img)

        # 🔹 Update last bbox
        last_bbox = [x, y, w, h]

        # 🔹 Save
        predictions.append({
            "id": f"{seq_name}_{frame_idx}",
            "x": x,
            "y": y,
            "w": w,
            "h": h
        })

        # 🔹 Visualization
        if VISUALIZE:
            color = (255, 0, 0) if tracking_ok else (0, 0, 255)
            label = "Tracking" if tracking_ok else "Frozen"

            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            cv2.putText(frame, label,
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        color,
                        2)

            cv2.putText(frame, f"Score: {score:.2f}",
                        (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 255),
                        2)

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