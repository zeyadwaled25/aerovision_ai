"""
📌 Inference / Visualization Module

Purpose:
Visualize video sequences with Ground Truth bounding boxes.

Overview:
- Read video frame-by-frame (streaming)
- Draw Ground Truth bounding boxes
- Display frames with basic info (frame index)
"""

import cv2


def visualize_sequence(sequence):
    """
    Visualize video with Ground Truth boxes.

    Args:
        sequence (dict): Contains video_path and boxes
    """

    video_path = sequence["video_path"]
    boxes = sequence["boxes"]

    cap = cv2.VideoCapture(video_path)

    frame_idx = 0

    # 🔹 Display window
    cv2.namedWindow("Tracking (Ground Truth)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tracking (Ground Truth)", 1150, 750)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # 🔹 Draw Ground Truth bbox (if exists)
        if boxes is not None and frame_idx < len(boxes):
            x, y, w, h = map(int, boxes[frame_idx])

            if w > 0 and h > 0:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                cv2.putText(frame, f"x:{x} y:{y} w:{w} h:{h}",
                            (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            1)

        # 🔹 Frame index
        cv2.putText(frame, f"Frame: {frame_idx}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 0, 0),
                    2)

        cv2.imshow("Tracking (Ground Truth)", frame)

        key = cv2.waitKey(30) & 0xFF

        # 🔹 Controls
        if key == 27:  # ESC → next sequence
            cap.release()
            cv2.destroyAllWindows()
            return "next"

        if key == ord('q'):  # Q → stop بالكامل
            cap.release()
            cv2.destroyAllWindows()
            return "stop"

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    return "done"