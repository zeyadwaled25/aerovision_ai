"""
📌 Main Entry Point (CLI Edition for AIC-4 Evaluation)

Purpose:
Run the TCTrack++ (V5 Physicist Edition) tracking pipeline for the AIC-4 aerial tracking task.

Overview:
- Parse CLI arguments (dataset_dir, split, output_csv).
- Load dataset sequences via data_loader.
- Run tracker on each sequence.
- Evaluate performance metrics (if Ground Truth is available).
- Save final predictions to CSV format required by the competition.
"""

import os
import sys
import argparse
from unittest.mock import MagicMock
import numpy as np
import pandas as pd
import torch

# ==========================================
# 🛠️ Legacy Compatibility Patch (PySOT Hack)
# ==========================================
if 'torch._six' not in sys.modules:
    class _Six:
        string_classes = (str,)
        int_classes = (int,)
    sys.modules['torch._six'] = _Six()
    torch._six = _Six()

if 'visdom' not in sys.modules:
    mock_visdom = MagicMock()
    sys.modules['visdom'] = mock_visdom
    sys.modules['visdom.server'] = mock_visdom

# Internal Modules
from src.data_loader import load_sequences
from src.tctrack_plusplus_tracker import run_tracker
from src.evaluate import evaluate

def main(
    dataset_dir: str = "data",
    split: str = "public_lb",
    output_csv: str = "./outputs/predictions.csv",
):
    print(f"🚀 Starting Tracking Pipeline...")
    print(f"📂 Dataset: {dataset_dir} | Split: {split}")
    print(f"💾 Output: {output_csv}\n")

    sequences = load_sequences(dataset_dir, split=split)
    if not sequences:
        print("❌ No sequences found. Aborting.")
        return

    print(f"Number of sequences loaded: {len(sequences)}")

    all_predictions = []
    all_ious, all_dists, all_aucs, all_robustness = [], [], [], []

    try:
        for sequence in sequences:
            sequence_name = sequence["seq_name"]
            print(f"\n▶ Processing Sequence: {sequence_name}")

            boxes = sequence.get("boxes")

            # 🔹 Dataset info (only if Ground Truth is available)
            if boxes is not None:
                total_frames = len(boxes)
                valid_boxes = [b for b in boxes if b[2] > 0 and b[3] > 0]
                visibility = (len(valid_boxes) / total_frames) if total_frames else 0.0
                print(f"Frames: {total_frames} | Visible: {len(valid_boxes)} | Ratio: {visibility:.2f}")
            else:
                print("Frames: Unknown | No Ground Truth provided (Hidden Test mode).")

            # 🔹 Run tracker (V5 Physicist Edition)
            result = run_tracker(sequence)

            if result and isinstance(result.get("predictions"), list):
                all_predictions.extend(result["predictions"])

                # 🔹 Evaluate (Only if Ground Truth exists)
                if boxes is not None:
                    metrics = evaluate(sequence, result["predictions"])
                    if metrics:
                        print(
                            f"IoU: {metrics['avg_iou']:.3f} | "
                            f"Dist: {metrics['avg_dist']:.2f} | "
                            f"AUC: {metrics['auc']:.3f} | "
                            f"P@20: {metrics['precision@20px']:.3f} | "
                            f"Robustness: {metrics['robustness']:.3f}"
                        )
                        all_ious.append(metrics["avg_iou"])
                        all_dists.append(metrics["avg_dist"])
                        all_aucs.append(metrics["auc"])
                        all_robustness.append(metrics["robustness"])

            # 🔹 Stop manually check
            if result.get("status") == "stop":
                print("\n⛔ Tracking stopped by user constraint.")
                break

        # 🔹 Save predictions
        os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
        df = pd.DataFrame(all_predictions)
        df.to_csv(output_csv, index=False)
        print(f"\n✅ Predictions successfully saved to {output_csv}")

        # 🔹 Final aggregated metrics
        if all_ious:
            print("\n====== 🏆 FINAL EVALUATION RESULTS ======")
            print(f"Avg IoU: {np.mean(all_ious):.3f}")
            print(f"Avg Dist: {np.mean(all_dists):.2f}")
            print(f"AUC: {np.mean(all_aucs):.3f}")
            print(f"Robustness: {np.mean(all_robustness):.3f}")

    except KeyboardInterrupt:
        print("\n⛔ Execution interrupted by user (Ctrl+C)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AIC-4 TCTrack++ Evaluation Pipeline")
    parser.add_argument("--dataset_dir", default="data", help="Path to dataset root (default: data)")
    parser.add_argument("--split", default="public_lb", choices=["public_lb", "hidden_test"], help="Dataset split to evaluate")
    parser.add_argument("--output_csv", default="./outputs/predictions.csv", help="Destination path for the final CSV file")
    
    args = parser.parse_args()
    main(dataset_dir=args.dataset_dir, split=args.split, output_csv=args.output_csv)