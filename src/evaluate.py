"""
📌 Evaluation Module

Purpose:
Evaluate tracking predictions against Ground Truth.

Overview:
- Compute IoU and center distance per frame
- Aggregate metrics (IoU, AUC, precision, robustness)
"""

import numpy as np
from src.utils.metrics import (
    compute_iou,
    center_distance,
    success_curve,
    precision_curve,
    compute_auc,
)


def compute_precision_at_threshold(distances, threshold=20):
    # نسبة الفريمات اللي distance فيها أقل من threshold
    distances = np.array(distances)
    return np.mean(distances <= threshold)


def compute_robustness_threshold(ious, threshold=0.2):
    """
        Failure = IoU أقل من threshold
        Robustness = نسبة الفشل
    """
    ious = np.array(ious)
    failures = ious < threshold
    return np.mean(failures)


def evaluate(sequence, predictions):
    boxes = sequence["boxes"]

    ious = []
    distances = []

    # Loop over frames
    for i, pred in enumerate(predictions):

        if boxes is None or i >= len(boxes):
            continue

        gt = boxes[i]

        # Skip invalid GT
        if gt[2] == 0 or gt[3] == 0:
            continue

        pred_box = [pred["x"], pred["y"], pred["w"], pred["h"]]

        iou = compute_iou(pred_box, gt)
        dist = center_distance(pred_box, gt)

        ious.append(iou)
        distances.append(dist)

    if len(ious) == 0:
        return {}

    # Aggregated Metrics
    avg_iou = np.mean(ious)
    avg_dist = np.mean(distances)

    # Success Curve → AUC
    _, success = success_curve(ious)
    auc = compute_auc(success)

    # Precision
    precision_20 = compute_precision_at_threshold(distances, threshold=20)

    # Robustness (IoU threshold-based)
    robustness = compute_robustness_threshold(ious, threshold=0.2)

    return {
        "avg_iou": avg_iou,
        "avg_dist": avg_dist,
        "auc": auc,
        "precision@20px": precision_20,
        "robustness": robustness,
    }