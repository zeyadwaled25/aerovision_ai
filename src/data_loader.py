"""
📌 Data Loader Module

Purpose:
Load dataset sequences and annotations from the competition manifest file.
Handles missing annotations gracefully for hidden test evaluation.
"""

import os
import json

def load_annotations(annotation_path):
    """
    Load bounding boxes from a text file.
    Supports comma-separated or space-separated formats (x, y, w, h).
    """
    boxes = []
    with open(annotation_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if "," in line:
                x, y, w, h = map(float, line.split(","))
            else:
                x, y, w, h = map(float, line.split())
            boxes.append([x, y, w, h])
    return boxes

def load_sequences(data_dir, split="public_lb"):
    """
    Load sequence metadata based on the specified split.
    
    Args:
        data_dir (str): Root directory of the dataset.
        split (str): The split to load (e.g., "public_lb", "hidden_test").
        
    Returns:
        list: A list of dictionaries containing sequence info.
    """
    manifest_path = os.path.join(data_dir, "metadata", "contestant_manifest.json")
    
    if not os.path.exists(manifest_path):
        print(f"⚠️ Warning: Manifest file not found at {manifest_path}")
        return []

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    if split not in manifest:
        print(f"⚠️ Warning: Split '{split}' not found in manifest.")
        return []

    sequences = []
    for seq_id, info in manifest[split].items():
        video_path = os.path.join(data_dir, info.get("video_path", ""))
        ann_path = info.get("annotation_path")

        boxes = None
        init_bbox = [0.0, 0.0, 0.0, 0.0]

        if ann_path is not None:
            full_ann_path = os.path.join(data_dir, ann_path)
            if os.path.exists(full_ann_path):
                boxes = load_annotations(full_ann_path)
                
                # Validation: Replace invalid boxes with [0,0,0,0]
                valid_boxes = []
                for b in boxes:
                    x, y, w, h = b
                    if x < 0 or y < 0 or w <= 0 or h <= 0:
                        valid_boxes.append([0.0, 0.0, 0.0, 0.0])
                    else:
                        valid_boxes.append(b)
                boxes = valid_boxes
                
                # Find the first valid bounding box for initialization
                for b in boxes:
                    if b[2] > 0 and b[3] > 0:
                        init_bbox = b
                        break

        sequences.append({
            "video_path": video_path,
            "boxes": boxes,
            "init_bbox": init_bbox,
            "seq_name": seq_id
        })

    return sequences