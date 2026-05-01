"""
📌 Metrics Core Module

Purpose:
Core mathematical functions to compute Object Tracking evaluation metrics.
"""

import math
import numpy as np

def compute_iou(boxA, boxB):
    """Computes Intersection over Union (IoU) between two bounding boxes [x, y, w, h]."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
    yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    boxAArea = boxA[2] * boxA[3]
    boxBArea = boxB[2] * boxB[3]

    union = boxAArea + boxBArea - interArea

    if union <= 0:
        return 0.0

    return interArea / union

def center_distance(boxA, boxB):
    """Computes Euclidean distance between the centers of two bounding boxes."""
    cxA = boxA[0] + boxA[2] / 2
    cyA = boxA[1] + boxA[3] / 2

    cxB = boxB[0] + boxB[2] / 2
    cyB = boxB[1] + boxB[3] / 2

    return math.sqrt((cxA - cxB) ** 2 + (cyA - cyB) ** 2)

def success_curve(ious, thresholds=None):
    """Generates the success curve based on varied IoU thresholds."""
    if thresholds is None:
        thresholds = np.linspace(0, 1, 21)

    success = [np.mean([iou >= t for iou in ious]) for t in thresholds]
    return thresholds, success

def precision_curve(distances, thresholds=None):
    """Generates the precision curve based on varied distance thresholds."""
    if thresholds is None:
        thresholds = np.arange(0, 51, 1)

    precision = [np.mean([d <= t for d in distances]) for t in thresholds]
    return thresholds, precision

def compute_auc(success_scores):
    """Computes the Area Under Curve (AUC) from success scores."""
    return np.mean(success_scores)