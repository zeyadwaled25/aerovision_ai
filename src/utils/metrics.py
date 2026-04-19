"""
📌 Metrics Module

Purpose:
Provide evaluation metrics for object tracking.

Overview:
- Compute overlap (IoU)
- Compute center distance
- Generate success and precision curves
- Compute AUC and robustness
"""

import math
import numpy as np


def compute_iou(boxA, boxB):
    # Intersection
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    # Areas
    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    union = boxAArea + boxBArea - interArea

    if union == 0:
        return 0.0

    return interArea / union


def center_distance(boxA, boxB):
    # Compute centers
    cxA = boxA[0] + boxA[2] / 2
    cyA = boxA[1] + boxA[3] / 2

    cxB = boxB[0] + boxB[2] / 2
    cyB = boxB[1] + boxB[3] / 2

    return math.sqrt((cxA - cxB) ** 2 + (cyA - cyB) ** 2)


def success_curve(ious, thresholds=None):
    # IoU thresholds → success rate
    if thresholds is None:
        thresholds = np.linspace(0, 1, 21)

    success = []
    for t in thresholds:
        success.append(np.mean([iou >= t for iou in ious]))

    return thresholds, success


def precision_curve(distances, thresholds=None):
    # Distance thresholds → precision
    if thresholds is None:
        thresholds = np.arange(0, 51, 1)

    precision = []
    for t in thresholds:
        precision.append(np.mean([d <= t for d in distances]))

    return thresholds, precision


def normalized_precision(distances, diag):
    # Normalize distances (optional use)
    return [d / diag for d in distances]


def compute_auc(success_scores):
    # Area under success curve
    return np.mean(success_scores)


def compute_robustness(ious, threshold=0.2):
    # Failure = IoU < threshold
    ious = np.array(ious)
    return np.mean(ious < threshold)