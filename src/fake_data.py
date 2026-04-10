import numpy as np
import cv2

def generate_fake_video(num_frames=50, width=640, height=480):
    frames = []
    boxes = []

    # بداية الجسم
    x, y, w, h = 100, 100, 50, 50

    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        # رسم object (مربع أبيض)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), -1)

        frames.append(frame)
        boxes.append([x, y, w, h])

        # حركة بسيطة
        x += 5
        y += 2

    return frames, boxes