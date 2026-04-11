"""
📌 Inference / Visualization Module

Purpose:
Visualize video sequences with Ground Truth bounding boxes.

What it does:
- Reads video frame-by-frame (streaming)
- Draws bounding boxes for each frame
- Displays the video

Notes:
- Memory-efficient (no full video loading)
- This is NOT tracking (just visualization)
"""

import cv2


def visualize_sequence(sequence):
    """
    Visualize video with bounding boxes.

    Args:
        sequence (dict): Contains video_path and boxes
    """

    video_path = sequence["video_path"]
    boxes = sequence["boxes"]

    cap = cv2.VideoCapture(video_path)

    frame_idx = 0

    # Set up display window
    cv2.namedWindow("Tracking (Ground Truth)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Tracking (Ground Truth)", 1150, 750)
    # cv2.setWindowProperty(
    #     "Tracking (Ground Truth)",
    #     cv2.WND_PROP_FULLSCREEN,
    #     cv2.WINDOW_FULLSCREEN
    # )

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Draw Ground Truth bounding box
        if boxes is not None and frame_idx < len(boxes):
            x, y, w, h = map(int, boxes[frame_idx])

            # Skip invalid boxes (0,0,0,0)
            if w > 0 and h > 0:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0,255,0), 2)
                # bbox values
                cv2.putText(frame, f"x:{x} y:{y} w:{w} h:{h}",
                            (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0,255,0),
                            1)

        # Display frame index
        cv2.putText(frame, f"Frame: {frame_idx}",
            (20,40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255,0,0),
            2)

        # Display frame
        cv2.imshow("Tracking (Ground Truth)", frame)

        key = cv2.waitKey(30) & 0xFF

        # ESC = Play next sequence
        if key == 27:
            cap.release()
            cv2.destroyAllWindows()
            return "next"

        # Q = Exit completely
        if key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            return "stop"

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()
    return "done"