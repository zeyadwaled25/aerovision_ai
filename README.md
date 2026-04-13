# AeroVision Tracker

A manifest-driven object tracking pipeline for video datasets. The project loads many sequences from `data/metadata/contestant_manifest.json`, runs OpenCV CSRT tracking per sequence, overlays predictions and ground truth, computes tracking metrics, and exports all predictions to CSV.

## Features

- Loads sequences from a central manifest file.
- Supports annotation files with either `x,y,w,h` or `x y w h` format.
- Tracks objects online with OpenCV CSRT (`cv2.TrackerCSRT_create`).
- Shows real-time visualization with prediction box, ground truth box, IoU, and center distance.
- Computes per-sequence metrics:
  - Average IoU
  - Average center distance
  - Success AUC
  - Robustness (failure ratio)
- Saves aggregated predictions to `outputs/predictions.csv`.

## Project Structure

- `main.py`: Entry point. Loads all sequences, runs tracking, evaluates, and saves CSV output.
- `src/data_loader.py`: Parses manifest and annotations, builds normalized sequence objects.
- `src/tracker.py`: CSRT tracker loop, OpenCV visualization, keyboard controls, prediction collection.
- `src/evaluate.py`: Sequence-level evaluation logic and metric aggregation.
- `src/utils/metrics.py`: IoU, center distance, success curve, precision curve, AUC, robustness helpers.
- `src/inference.py`: Ground-truth visualization utility.
- `data/`: Datasets and `metadata/contestant_manifest.json`.
- `outputs/`: Generated artifacts such as `predictions.csv`.

## Data Contract

Each loaded sequence follows this schema:

```python
{
		"video_path": str,
		"boxes": list[list[float]],   # [x, y, w, h] per frame, or None
		"init_bbox": list[float],     # first valid bbox or [0, 0, 0, 0]
		"seq_name": str
}
```

The loader expects:

- Manifest: `data/metadata/contestant_manifest.json`
- Manifest fields per sequence:
  - `video_path`
  - `annotation_path` (can be `null` for unlabeled/test sequences)

## Installation

Python 3.10+ recommended.

```bash
pip install -r requirements.txt
```

## Run

From project root:

```bash
python main.py
```

Pipeline behavior:

1. Load all sequences from the manifest.
2. For each sequence, print visibility stats based on valid ground-truth boxes.
3. Track frame-by-frame with CSRT.
4. Evaluate predictions against ground truth (when available).
5. Append all predictions and write `outputs/predictions.csv`.

## Keyboard Controls During Tracking

- `Esc`: Skip current sequence and continue to the next one.
- `q`: Stop the full run immediately.

## Output CSV

`outputs/predictions.csv` contains one row per tracked frame:

- `id`: `<sequence_name>_<frame_index>`
- `x`
- `y`
- `w`
- `h`

## Notes

- Tracking runs on resized frames (`scale = 0.75`) for speed, then boxes are mapped back to original resolution.
- If the tracker loses the object in a frame, the last known box is reused (frozen prediction).
- For sequences without annotations, evaluation metrics are skipped.

## Known Limitations

- Single-object tracking only.
- Uses classical CSRT tracking (no deep re-identification).
- No automatic experiment logging beyond printed metrics and CSV output.
