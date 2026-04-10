import json
import os


def load_annotations(annotation_path):
    """
    Load bounding boxes from annotation file.

    Each line corresponds to one frame:
    Format:
        x,y,w,h  OR  x y w h

    Returns:
        List of bounding boxes
    """
    boxes = []

    with open(annotation_path, "r") as f:
        for line in f:
            line = line.strip()

            # Handle both formats: comma or space
            if "," in line:
                x, y, w, h = map(float, line.split(","))
            else:
                x, y, w, h = map(float, line.split())

            boxes.append([x, y, w, h])

    return boxes


def load_sequences(data_dir):
    """
    Load all sequences from dataset.

    Steps:
    1. Read manifest file
    2. Iterate over sequences
    3. Load video path and annotations
    4. Prepare sequence objects

    Returns:
        List of sequences
    """

    manifest_path = os.path.join(data_dir, "metadata", "contestant_manifest.json")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    sequences = []

    for split in manifest:  # train / public_lb
        for seq_id, info in manifest[split].items():

            video_path = os.path.join(data_dir, info["video_path"])
            ann_path = info["annotation_path"]

            # If annotation exists
            if ann_path is not None:
                ann_path = os.path.join(data_dir, ann_path)

                if os.path.exists(ann_path):
                    boxes = load_annotations(ann_path)
                    init_bbox = boxes[0]  # first frame
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