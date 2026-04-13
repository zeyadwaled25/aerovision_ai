"""
📌 Main Entry Point

Purpose:
Run the project pipeline.

What it does:
1. Loads dataset sequences
2. Selects one sequence
3. Visualizes Ground Truth bounding boxes

Notes:
- Currently runs only one sequence (for debugging)
- Can be extended to process all sequences
"""

from src.data_loader import load_sequences
from src.tracker import run_tracker
import pandas as pd
from src.evaluate import evaluate

# Load dataset
sequences = load_sequences("data", split="public_lb")
print(f"Number of sequences: {len(sequences)}")

all_predictions = []

try:
    for sequence in sequences:
        sequence_name = sequence["seq_name"]
        print(f"Playing: {sequence_name}")

        boxes = sequence["boxes"]

        if boxes is not None:
            total_frames = len(boxes)
            valid_boxes = [b for b in boxes if b[2] > 0 and b[3] > 0]

            print(f"Total frames: {total_frames}")
            print(f"Frames with object: {len(valid_boxes)}")
            visibility = (len(valid_boxes) / total_frames) if total_frames else 0.0
            print(f"Visibility ratio: {visibility:.2f}")

        action = run_tracker(sequence)

        if action and isinstance(action.get("predictions"), list):
            all_predictions.extend(action["predictions"])
            
        metrics = evaluate(sequence, action["predictions"])

        if metrics:
            print(
                f"IoU: {metrics['avg_iou']:.3f} | "
                f"Dist: {metrics['avg_dist']:.2f} | "
                f"AUC: {metrics['auc']:.3f} | "
                f"Robustness: {metrics['robustness']:.3f}"
            )

        if action["status"] == "stop":
            print("\n👀 Stopped completely by user")
            break

    df = pd.DataFrame(all_predictions)
    df.to_csv("./outputs/predictions.csv", index=False)
    print("✅ predictions.csv saved")

except KeyboardInterrupt:
    print("\n👀 Stopped by Ctrl+C")