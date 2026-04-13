"""
📌 Data Loader Module

Purpose:
Load dataset sequences from the manifest file.

Overview:
- Read manifest file
- Load video paths and annotations
- Prepare sequence objects for tracking
"""

import json
import os


def load_annotations(annotation_path):
    """
    Load bounding boxes from file.

    Supports:
    - x,y,w,h
    - x y w h
    """

    boxes = []

    with open(annotation_path, "r") as f:
        for line in f:
            line = line.strip()

            # 🔹 Handle comma or space format
            if "," in line:
                x, y, w, h = map(float, line.split(","))
            else:
                x, y, w, h = map(float, line.split())

            boxes.append([x, y, w, h])

    return boxes


def load_sequences(data_dir, split="train"):
    """
    Load sequences from dataset.

    Args:
        data_dir (str): dataset root
        split (str): "train" or "public_lb"

    Returns:
        List of sequence dicts
    """

    manifest_path = os.path.join(data_dir, "metadata", "contestant_manifest.json")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    sequences = []

    # 🔹 Loop only on selected split
    for seq_id, info in manifest[split].items():

        video_path = os.path.join(data_dir, info["video_path"])
        ann_path = info["annotation_path"]

        # 🔹 Load annotations if available
        if ann_path is not None:
            ann_path = os.path.join(data_dir, ann_path)

            if os.path.exists(ann_path):
                boxes = load_annotations(ann_path)
                init_bbox = boxes[0]

                # 🔹 Basic validation (remove invalid boxes)
                valid_boxes = []
                for b in boxes:
                    x, y, w, h = b
                    if x < 0 or y < 0:
                        continue
                    valid_boxes.append(b)

                boxes = valid_boxes

            else:
                boxes = None
                init_bbox = [0, 0, 0, 0]

        else:
            boxes = None
            init_bbox = [0, 0, 0, 0]

        sequences.append({
            "video_path": video_path,
            "boxes": boxes,
            "init_bbox": init_bbox,
            "seq_name": seq_id
        })

    return sequences