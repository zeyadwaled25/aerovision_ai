"""
📌 Main Entry Point

Purpose:
Run the tracking pipeline for the AIC-4 aerial tracking task.

Overview:
- Load dataset sequences
- Run tracker on each sequence
- Collect predictions
- Evaluate performance (train only)
- Save results to CSV
"""
import sys
sys.path.append("./pysot")
import numpy as np
import pandas as pd

from src.data_loader import load_sequences
# from src.tracker import run_tracker
from src.siam_tracker import run_tracker
from src.evaluate import evaluate


def main():

    # 🔹 Load dataset (train / public_lb)
    sequences = load_sequences("data", split="train")
    # sequences = load_sequences("data", split="public_lb")

    print(f"Number of sequences: {len(sequences)}")

    all_predictions = []
    all_ious = []
    all_dists = []
    all_aucs = []
    all_robustness = []

    try:
        for sequence in sequences:
            sequence_name = sequence["seq_name"]
            print(f"\n▶ Processing: {sequence_name}")

            boxes = sequence["boxes"]

            # 🔹 Dataset info (only if GT available)
            if boxes is not None:
                total_frames = len(boxes)
                valid_boxes = [b for b in boxes if b[2] > 0 and b[3] > 0]
                visibility = (len(valid_boxes) / total_frames) if total_frames else 0.0

                print(f"Frames: {total_frames} | Visible: {len(valid_boxes)} | Ratio: {visibility:.2f}")

            # 🔹 Run tracker
            result = run_tracker(sequence)

            if result and isinstance(result.get("predictions"), list):
                all_predictions.extend(result["predictions"])

                # 🔹 Evaluate (train only)
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

            # 🔹 Stop manually
            if result["status"] == "stop":
                print("\n⛔ Stopped by user")
                break

        # 🔹 Save predictions
        df = pd.DataFrame(all_predictions)
        df.to_csv("./outputs/predictions.csv", index=False)
        print("✅ predictions.csv saved")

        # 🔹 Final metrics (only if training)
        if all_ious:
            print("\n====== FINAL RESULTS ======")
            print(f"Avg IoU: {np.mean(all_ious):.3f}")
            print(f"Avg Dist: {np.mean(all_dists):.2f}")
            print(f"AUC: {np.mean(all_aucs):.3f}")
            print(f"Robustness: {np.mean(all_robustness):.3f}")

    except KeyboardInterrupt:
        print("\n⛔ Stopped by Ctrl+C")

if __name__ == "__main__":
    main()