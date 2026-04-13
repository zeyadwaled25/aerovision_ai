import numpy as np
from src.utils.metrics import (
    compute_iou,
    center_distance,
    success_curve,
    precision_curve,
    compute_auc,
    compute_robustness,
)

def evaluate(sequence, predictions):
    boxes = sequence["boxes"]

    ious = []
    distances = []

    for i, pred in enumerate(predictions):
        if boxes is None or i >= len(boxes):
            continue

        gt = boxes[i]

        if gt[2] == 0 or gt[3] == 0:
            continue

        pred_box = [pred["x"], pred["y"], pred["w"], pred["h"]]

        iou = compute_iou(pred_box, gt)
        dist = center_distance(pred_box, gt)

        ious.append(iou)
        distances.append(dist)

    if len(ious) == 0:
        return {}

    # Averages
    avg_iou = np.mean(ious)
    avg_dist = np.mean(distances)

    # Success Curve + AUC
    th_s, success = success_curve(ious)
    auc = compute_auc(success)

    # Precision Curve
    th_p, precision = precision_curve(distances)

    # Robustness
    robustness = compute_robustness(ious)

    return {
        "avg_iou": avg_iou,
        "avg_dist": avg_dist,
        "auc": auc,
        "robustness": robustness,
    }