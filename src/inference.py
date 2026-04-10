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

        # Display frame
        cv2.imshow("Tracking (Ground Truth)", frame)

        # Press ESC to exit
        if cv2.waitKey(30) & 0xFF == 27:
            break

        frame_idx += 1

    cap.release()
    cv2.destroyAllWindows()