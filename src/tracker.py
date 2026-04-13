"""
📌 Tracker Module

Purpose:
Perform online object tracking using OpenCV CSRT tracker.

Overview:
- Initialize tracker using first frame + initial bounding box
- Track object frame-by-frame
- Apply smoothing to reduce jitter
- If tracking fails → use last known bbox (freeze)
- Ensure bbox stays inside frame (clipping)
"""

import cv2
from src.utils.metrics import compute_iou, center_distance


def clip_box(box, w, h):
    # 🔹 Keep bbox inside image boundaries
    x, y, bw, bh = box

    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    bw = max(1, min(bw, w - x))
    bh = max(1, min(bh, h - y))

    return [x, y, bw, bh]


def smooth_box(prev_box, curr_box, alpha=0.7):
    # 🔹 Reduce jitter between frames
    return [
        int(alpha * curr_box[i] + (1 - alpha) * prev_box[i])
        for i in range(4)
    ]


def run_tracker(sequence):
    video_path = sequence["video_path"]
    init_bbox = sequence["init_bbox"]
    seq_name = sequence["seq_name"]

    cap = cv2.VideoCapture(video_path)

    # 🔹 Read first frame
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read video")
        return {"predictions": [], "status": "failed"}

    boxes = sequence["boxes"]

    scale = 0.75
    alpha = 0.7

    # 🔹 Resize first frame
    frame_small = cv2.resize(frame, None, fx=scale, fy=scale)

    # 🔹 Scale initial bbox
    x, y, w, h = map(int, init_bbox)
    bbox = (
        int(x * scale),
        int(y * scale),
        int(w * scale),
        int(h * scale),
    )

    # 🔹 Initialize tracker
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame_small, bbox)

    frame_idx = 1
    last_bbox = list(bbox)

    # 🔹 Visualization window
    cv2.namedWindow("Tracking (Prediction)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tracking (Prediction)", 1150, 750)

    predictions = [{
        "id": f"{seq_name}_0",
        "x": int(init_bbox[0]),
        "y": int(init_bbox[1]),
        "w": int(init_bbox[2]),
        "h": int(init_bbox[3])
    }]

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h_orig, w_orig = frame.shape[:2]

        # 🔹 Resize frame
        frame_small = cv2.resize(frame, None, fx=scale, fy=scale)

        success, bbox = tracker.update(frame_small)

        if success:
            bbox = list(map(int, bbox))

            # 🔹 Smooth movement
            bbox = smooth_box(last_bbox, bbox, alpha)
            last_bbox = bbox

            x, y, bw, bh = bbox

            # 🔹 Convert back to original scale
            x = int(x / scale)
            y = int(y / scale)
            bw = int(bw / scale)
            bh = int(bh / scale)

            # 🔹 Ensure valid bbox
            x, y, bw, bh = clip_box([x, y, bw, bh], w_orig, h_orig)

        else:
            # 🔹 Freeze (use last known bbox)
            x, y, bw, bh = last_bbox

            x = int(x / scale)
            y = int(y / scale)
            bw = int(bw / scale)
            bh = int(bh / scale)

            x, y, bw, bh = clip_box([x, y, bw, bh], w_orig, h_orig)

        # 🔹 Save prediction
        predictions.append({
            "id": f"{seq_name}_{frame_idx}",
            "x": x,
            "y": y,
            "w": bw,
            "h": bh
        })

        # 🔹 Draw prediction
        color = (255, 0, 0) if success else (0, 0, 255)
        label = "Tracking" if success else "Lost (Frozen)"

        cv2.rectangle(frame, (x, y), (x + bw, y + bh), color, 2)
        cv2.putText(frame, label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # 🔹 Ground Truth + metrics (only if available)
        if boxes is not None and frame_idx < len(boxes):
            xg, yg, wg, hg = map(int, boxes[frame_idx])

            if wg > 0 and hg > 0:
                cv2.rectangle(frame, (xg, yg), (xg + wg, yg + hg), (0, 255, 0), 2)

                gt_box = boxes[frame_idx]
                pred_box = [x, y, bw, bh]

                iou = compute_iou(pred_box, gt_box)
                dist = center_distance(pred_box, gt_box)

                cv2.putText(frame, f"IoU: {iou:.2f}", (20, 160),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

                cv2.putText(frame, f"Dist: {dist:.1f}", (20, 200),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        # 🔹 Info
        cv2.putText(frame, f"Frame: {frame_idx}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.putText(frame, f"Seq: {seq_name}", (20, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("Tracking (Prediction)", frame)

        key = cv2.waitKey(12) & 0xFF

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