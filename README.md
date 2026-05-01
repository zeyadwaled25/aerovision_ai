# AeroVision Tracker (AIC-4 UAV Tracking)

## 1. Project Overview

This repository provides an inference-only single-object tracking pipeline for the AIC-4 UAV tracking competition.

The current implementation uses:

- TCTrack++ as the tracking backbone
- A custom decision engine for stability and recovery behavior
- A CLI-based execution flow via main.py
- Reproducibility controls and Docker-based execution support

The pipeline loads sequence metadata, runs frame-by-frame tracking, and exports predictions in competition-ready CSV format.

## 2. Key Features

- Efficient inference loop with conditional multi-hypothesis search activated only when confidence drops
- Reproducibility-oriented setup (deterministic seed initialization)
- Conditional inference/evaluation behavior:
  - Tracking always runs
  - Metrics are computed only when ground-truth annotations are present
- Dockerized runtime for consistent execution environments
- Competition-friendly output generation (outputs/predictions.csv)

## 3. Folder Structure

```text
.
├── main.py                          # CLI entry point
├── Dockerfile                       # Container image definition
├── requirements.txt                 # Python dependencies
├── models/                          # Pretrained weights and tracker configs
│   ├── config.yaml
│   └── tctrack++.pth
├── src/
│   ├── data_loader.py               # Sequence and annotation loading from manifest
│   ├── tctrack_plusplus_tracker.py  # TCTrack++ inference + custom decision engine
│   ├── evaluate.py                  # Metrics computation (if GT exists)
│   └── utils/                       # Utility functions
├── outputs/
│   └── predictions.csv              # Generated tracking predictions
└── tctrack/                         # External tracker code dependency
```

## 4. Installation

### Local

1. Create and activate a Python virtual environment (recommended).
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure required pretrained weights are available in models/.

### Docker

Build image:

```bash
docker build -t aerovision-tracker .
```

Run container (example):

```bash
docker run --rm \
  -v /path/to/local/data:/data \
  -v /path/to/local/outputs:/app/outputs \
  aerovision-tracker \
  --dataset_dir /data \
  --split hidden_test \
  --output_csv /app/outputs/predictions.csv
```

## 5. How To Run (CLI)

Basic usage:

```bash
python main.py --dataset_dir data --split public_lb --output_csv ./outputs/predictions.csv
```

Arguments:

- --dataset_dir: Root directory containing metadata/contestant_manifest.json
- --split: Dataset split (public_lb or hidden_test)
- --output_csv: Output CSV path for predictions

Execution behavior:

- Sequences are loaded from the manifest
- Tracking runs for each sequence
- Evaluation metrics are computed only for sequences with available ground truth

## 6. Output Format (CSV)

Predictions are written to outputs/predictions.csv with one row per frame.

Columns:

- id: Frame identifier in the format <sequence*name>*<frame_index>
- x: Top-left x-coordinate of predicted bounding box
- y: Top-left y-coordinate of predicted bounding box
- w: Predicted box width
- h: Predicted box height

## 7. Inference-Only And Competition Compliance Notes

- This repository is designed for inference-only competition execution.
- No training routine is part of the submission pipeline.
- Ground-truth-dependent metrics are optional and only run when annotations exist.
- For hidden test-style evaluation, the pipeline still produces valid prediction CSV output without requiring labels.
- Docker support is included to improve reproducibility and environment consistency across machines.
