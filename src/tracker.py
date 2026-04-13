"""
📌 Tracker Module (OpenCV CSRT + Re-detection)

Purpose:
Perform online tracking using OpenCV tracker + simple re-detection.

Pipeline:
1. Initialize tracker with first frame + init_bbox
2. Track object frame-by-frame
3. If tracker fails → try re-detection using template matching
"""

import cv2
from src.utils.metrics import compute_iou, center_distance

def run_tracker(sequence):
    video_path = sequence["video_path"]
    init_bbox = sequence["init_bbox"]
    seq_name = sequence["seq_name"]

    cap = cv2.VideoCapture(video_path)

    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read video")
        return {
            "predictions": [],
            "status": "failed"
        }
    
    boxes = sequence["boxes"]

    # SCALE FACTOR
    scale = 0.75

    # Resize first frame
    frame_small = cv2.resize(frame, None, fx=scale, fy=scale)

    # Convert bbox to scaled coordinates
    x, y, w, h = map(int, init_bbox)
    bbox = (int(x * scale), int(y * scale), int(w * scale), int(h * scale))


    # Create tracker
    tracker = cv2.TrackerCSRT_create()

    tracker.init(frame_small, bbox)

    frame_idx = 1
    last_bbox = bbox

    # Window setup
    cv2.namedWindow("Tracking (Prediction)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tracking (Prediction)", 1150, 750)

    # Predictions
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

        # Resize frame for tracking
        frame_small = cv2.resize(frame, None, fx=scale, fy=scale)

        # Predict new bbox
        success, bbox = tracker.update(frame_small)

        if success:
            last_bbox = bbox
            x, y, bw, bh = map(int, bbox)

            # Convert back to original scale
            x = int(x / scale)
            y = int(y / scale)
            bw = int(bw / scale)
            bh = int(bh / scale)

            predictions.append({
                "id": f"{seq_name}_{frame_idx}",
                "x": x,
                "y": y,
                "w": bw,
                "h": bh
            })

            # Draw bbox
            cv2.rectangle(frame, (x, y), (x+bw, y+bh), (255,0,0), 2)

            # bbox values
            cv2.putText(frame, f"x:{x} y:{y} w:{bw} h:{bh}",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255,0,0),
                        1)

            cv2.putText(frame, "Tracking",
                        (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (255,0,0),
                        2)

        else:
            # Freeze using last known bbox (scaled → original)
            x, y, bw, bh = map(int, last_bbox)

            x = int(x / scale)
            y = int(y / scale)
            bw = int(bw / scale)
            bh = int(bh / scale)

            predictions.append({
                "id": f"{seq_name}_{frame_idx}",
                "x": x,
                "y": y,
                "w": bw,
                "h": bh
            })

            cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0,0,255), 2)

            cv2.putText(frame, "Lost (Frozen)",
                        (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,0,255),
                        2)

        # Ground Truth
        if boxes is not None and frame_idx < len(boxes):
            xg, yg, wg, hg = map(int, boxes[frame_idx])

            if wg > 0 and hg > 0:
                cv2.rectangle(frame, (xg, yg), (xg+wg, yg+hg), (0,255,0), 2)

                cv2.putText(frame, "GT",
                            (xg, yg-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0,255,0),
                            1)
            
                gt_box = boxes[frame_idx]
                pred_box = [x, y, bw, bh]

                iou = compute_iou(pred_box, gt_box)
                dist = center_distance(pred_box, gt_box)

                cv2.putText(frame, f"IoU: {iou:.2f}",
                            (20,160),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255,255,0),
                            2)

                cv2.putText(frame, f"Dist: {dist:.1f}",
                            (20,200),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255,255,0),
                            2)

        # Frame index
        cv2.putText(frame, f"Frame: {frame_idx}",
                    (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255,0,0),
                    2)

        # Sequence name
        cv2.putText(frame, f"Seq: {seq_name}",
                    (20,120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255,255,255),
                    2)

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