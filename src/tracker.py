"""
📌 Tracker Module (OpenCV CSRT)

Purpose:
Perform online tracking using OpenCV tracker.

Pipeline:
1. Initialize tracker with first frame + init_bbox
2. For each new frame:
    - Predict new bounding box
    - Draw prediction
3. (Optional later) save predictions for submission
"""

import cv2

def run_tracker(sequence):
    """
    Run OpenCV tracker on a sequence.

    Args:
        sequence (dict): contains video_path + init_bbox
    """

    video_path = sequence["video_path"]
    init_bbox = sequence["init_bbox"]
    seq_name = sequence["seq_name"]

    cap = cv2.VideoCapture(video_path)

    # Read first frame
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read video")
        return

    # Convert bbox to int
    bbox = tuple(map(int, init_bbox))

    # Create tracker
    tracker = cv2.TrackerCSRT_create()

    # Initialize tracker
    tracker.init(frame, bbox)

    frame_idx = 0

    # Window setup
    cv2.namedWindow("Tracking (Prediction)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tracking (Prediction)", 1150, 750)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # Predict new bbox
        success, bbox = tracker.update(frame)

        if success:
            x, y, bw, bh = map(int, bbox)

            # Draw predicted bbox (GREEN)
            cv2.rectangle(frame, (x, y), (x+bw, y+bh), (0,255,0), 2)

            # bbox values (NEW)
            cv2.putText(frame, f"x:{x} y:{y} w:{bw} h:{bh}",
                        (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0,255,0),
                        1)

            cv2.putText(frame, "Tracking",
                        (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,255,0),
                        2)
        else:
            # Tracker lost object
            cv2.putText(frame, "Lost",
                        (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0,0,255),
                        2)

        # Frame index (NEW)
        cv2.putText(frame, f"Frame: {frame_idx}",
                    (20,80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255,0,0),
                    2)

        # Sequence name (NEW)
        cv2.putText(frame, f"Seq: {seq_name}",
                    (20,120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255,255,255),
                    2)

        # Show frame
        cv2.imshow("Tracking (Prediction)", frame)

        key = cv2.waitKey(30) & 0xFF

        # ESC → next sequence
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            return "next"

        # Q → stop all
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return "stop"

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    return "done"