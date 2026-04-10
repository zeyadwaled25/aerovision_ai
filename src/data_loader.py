import os
import cv2
from src.fake_data import generate_fake_video


# 🟢 1. Fake Data Loader
def load_fake_sequence():
    frames, boxes = generate_fake_video()

    seq_data = {
        "frames": frames,
        "init_bbox": boxes[0],
        "seq_name": "fake_seq"
    }

    return [seq_data]  # list of sequences


# 🟢 2. Real Video Loader (لما الداتا تيجي)
def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    return frames