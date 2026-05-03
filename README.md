# AeroVision Tracker (AIC-4 UAV Tracking)

## 🚀 Quick Start (Recommended)

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download model (see section below) and place in:
models/tctrack++.pth

# 3. Run inference
python main.py --dataset_dir data --split public_lb --output_csv ./outputs/predictions.csv
```

---

## 📌 Project Overview

This repository provides an **inference-only Single Object Tracking (SOT) pipeline** for the AIC-4 UAV tracking challenge.

The system is built on:

* **TCTrack++ backbone (lightweight CNN < 5M params)**
* **Custom decision engine** for stability, recovery, and drift prevention
* **Conditional multi-hypothesis inference**
* **Dockerized reproducible pipeline**

The pipeline processes video sequences frame-by-frame and generates a **competition-ready CSV output**.

---

## ⚙️ Key Features

* ✅ Lightweight and efficient (< 50M params constraint)
* ✅ Conditional recovery (only ~4.9% of frames → low FLOPs)
* ✅ Deterministic execution (fixed seeds)
* ✅ No training required (inference-only submission)
* ✅ Fully Docker-compatible

---

## 📁 Project Structure

```text
.
├── main.py
├── Dockerfile
├── requirements.txt
├── models/
│   ├── config.yaml
│   └── tctrack++.pth   ← REQUIRED
├── src/
│   ├── data_loader.py
│   ├── tctrack_plusplus_tracker.py
│   ├── evaluate.py
│   └── utils/
├── outputs/
│   └── predictions.csv
└── tctrack/  ← REQUIRED (PySOT dependency)
```

⚠️ **Important:**
The `tctrack/` directory (PySOT-based implementation) must exist in the repository.

---

## 📥 Model Weights

Download pretrained weights:

👉 https://drive.google.com/uc?export=download&id=1UvdwIRlmuCC5A074mX8iAEd1245ZTwmL

Place file here:

```bash
models/tctrack++.pth
```

❌ Missing weights will cause inference failure.

---

## ▶️ Running Inference

### Local

```bash
python main.py \
  --dataset_dir data \
  --split public_lb \
  --output_csv ./outputs/predictions.csv
```

### Hidden Test

```bash
python main.py \
  --dataset_dir /path/to/hidden_test \
  --split hidden_test \
  --output_csv ./outputs/predictions.csv
```

---

## 🐳 Docker Usage

### Build

```bash
docker build -t aerovision_tracker .
```

### Run

```bash
docker run --rm \
  -v /absolute/path/to/data:/app/data \
  -v /absolute/path/to/models:/app/models \
  -v /absolute/path/to/outputs:/app/outputs \
  aerovision_tracker \
  --dataset_dir /app/data \
  --split hidden_test \
  --output_csv /app/outputs/predictions.csv
```

---

## 📊 Output Format

CSV file: `outputs/predictions.csv`

| Column | Description         |
| ------ | ------------------- |
| id     | sequence_frameIndex |
| x      | top-left x          |
| y      | top-left y          |
| w      | width               |
| h      | height              |

---

## 🧠 System Design

The tracker follows a **conditional inference strategy**:

* Normal tracking → single forward pass (fast path)
* Recovery mode → multi-hypothesis search (activated only when needed)

📊 Profiling Results:

* Total Frames: 74,204
* Recovery Frames: 3,614
* Recovery Ratio: **4.9%**
* Latency: **22.89 ms/frame**

---

## 🔁 Reproducibility

* Fixed random seeds (NumPy, Torch, CUDA)
* Deterministic CuDNN configuration
* Docker environment ensures consistent runtime

---

## ⚠️ Notes

* Inference-only pipeline (no training included)
* Works with both labeled and unlabeled datasets
* Metrics computed only if ground truth exists
* Fully compliant with AIC-4 evaluation protocol

---

## 🧩 Requirements

* Python 3.10+
* PyTorch
* OpenCV
* PySOT (included via `tctrack/`)

---

## 🏁 Final Remarks

This implementation is designed to balance:

* Accuracy (robust tracking under UAV conditions)
* Efficiency (low latency + low FLOPs)
* Deployability (Jetson-class hardware compatibility)

---
