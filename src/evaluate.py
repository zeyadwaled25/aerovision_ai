"""
📌 Evaluation Module

Purpose:
Evaluate tracking predictions against Ground Truth annotations.
Computes standard Object Tracking metrics (IoU, Distance, AUC, Precision, Robustness).
"""

import numpy as np
from src.utils.metrics import (
    compute_iou,
    center_distance,
    success_curve,
    compute_auc,
)

def compute_precision_at_threshold(distances, threshold=20):
    """Returns the ratio of frames where center distance is <= threshold."""
    distances = np.array(distances)
    return np.mean(distances <= threshold)

def compute_robustness_threshold(ious, threshold=0.2):
    """
    Computes robustness as the failure rate.
    Failure is defined as an IoU drop below the threshold.
    """
    ious = np.array(ious)
    failures = ious < threshold
    return np.mean(failures)

def evaluate(sequence, predictions):
    """
    Evaluates a single sequence's predictions against its Ground Truth.
    """
    boxes = sequence.get("boxes")
    if not boxes:
        return {}

    ious = []
    distances = []

    for i, pred in enumerate(predictions):
        if i >= len(boxes):
            break

        gt = boxes[i]

        # Skip frames where Ground Truth is invalid/absent
        if gt[2] <= 0 or gt[3] <= 0:
            continue

        pred_box = [pred["x"], pred["y"], pred["w"], pred["h"]]

        iou = compute_iou(pred_box, gt)
        dist = center_distance(pred_box, gt)

        ious.append(iou)
        distances.append(dist)

    if not ious:
        return {}

    avg_iou = np.mean(ious)
    avg_dist = np.mean(distances)

    # Compute AUC from Success Curve
    _, success = success_curve(ious)
    auc = compute_auc(success)

    precision_20 = compute_precision_at_threshold(distances, threshold=20)
    robustness = compute_robustness_threshold(ious, threshold=0.2)

    return {
        "avg_iou": avg_iou,
        "avg_dist": avg_dist,
        "auc": auc,
        "precision@20px": precision_20,
        "robustness": robustness,
    }