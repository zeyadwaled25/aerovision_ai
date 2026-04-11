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
from src.inference import visualize_sequence

# Load dataset
sequences = load_sequences("data")
print(f"Number of sequences: {len(sequences)}")

# Select all sequences
try:
  for sequence in sequences:
    sequence_name = sequence["seq_name"]
    print("Playing:", sequence_name)

    boxes = sequence["boxes"]

    if boxes is not None:
      total_frames = len(boxes)
      valid_boxes = [b for b in boxes if b[2] > 0 and b[3] > 0]

      print(f"Total frames: {total_frames}")
      print(f"Frames with object: {len(valid_boxes)}")
      print(f"Visibility ratio: {len(valid_boxes)/total_frames:.2f}")

    action = visualize_sequence(sequence)

    if action == "stop":
      print("🛑 Stopped completely by user")
      break

except KeyboardInterrupt:
    print("\n🛑 Stopped by Ctrl+C")