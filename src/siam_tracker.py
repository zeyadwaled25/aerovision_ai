"""
📌 SiamRPN Tracker

Purpose:
Run SiamRPN tracker with production-ready visualization and stable predictions.

Overview:
- Full SiamRPN tracking (no fallback)
- Clean UI (same style)
- Safe bounding boxes (clip)
- Optional GT visualization + metrics
"""

import cv2
import torch

from pysot.core.config import cfg
from pysot.models.model_builder import ModelBuilder
from pysot.tracker.tracker_builder import build_tracker

# 🔹 CONFIG + MODEL
cfg.merge_from_file("models/config.yaml")

device = "cuda" if torch.cuda.is_available() else "cpu"

model = ModelBuilder()
model.load_state_dict(torch.load("models/siamrpn.pth", map_location=device))
model.eval().to(device)

tracker = build_tracker(model)

# 🔹 HELPERS
def clip_box(box, w, h):
    x, y, bw, bh = box

    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    bw = max(1, min(bw, w - x))
    bh = max(1, min(bh, h - y))

    return [x, y, bw, bh]


# 🔹 MAIN TRACKER
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

    # 🔹 Window
    cv2.namedWindow("Tracking (SiamRPN)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tracking (SiamRPN)", 1150, 750)

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
        x, y, w, h = clip_box([x, y, w, h], w_img, h_img)

        # 🔹 status
        tracking_ok = score > 0.4

        color = (255, 0, 0) if tracking_ok else (0, 0, 255)
        label = "Tracking" if tracking_ok else "Uncertain"

        predictions.append({
            "id": f"{seq_name}_{frame_idx}",
            "x": x,
            "y": y,
            "w": w,
            "h": h
        })

        # 🔹 DRAW
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

        # status
        cv2.putText(frame, label,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    color,
                    2)

        # score
        cv2.putText(frame, f"Score: {score:.2f}",
                    (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2)

        # frame index
        cv2.putText(frame, f"Frame: {frame_idx}",
                    (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2)

        # sequence name
        cv2.putText(frame, f"Seq: {seq_name}",
                    (20, 160),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2)

        # 🔹 Ground Truth + Metrics
        if boxes is not None and frame_idx < len(boxes):
            xg, yg, wg, hg = map(int, boxes[frame_idx])

            if wg > 0 and hg > 0:
                cv2.rectangle(frame, (xg, yg), (xg + wg, yg + hg), (0, 255, 0), 2)

                # IoU + Distance
                from src.utils.metrics import compute_iou, center_distance

                iou = compute_iou([x, y, w, h], boxes[frame_idx])
                dist = center_distance([x, y, w, h], boxes[frame_idx])

                cv2.putText(frame, f"IoU: {iou:.2f}",
                            (20, 200),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 0),
                            2)

                cv2.putText(frame, f"Dist: {dist:.1f}",
                            (20, 240),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 0),
                            2)

        cv2.imshow("Tracking (SiamRPN)", frame)

        key = cv2.waitKey(10) & 0xFF

        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            return {"status": "next", "predictions": predictions}

        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return {"status": "stop", "predictions": predictions}

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()

    return {"status": "done", "predictions": predictions}