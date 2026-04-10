# AeroVision Tracker

A small object-tracking prototype built around synthetic video frames. The current pipeline generates a fake moving object, runs a simple contour-based tracker with OpenCV, and exports frame-level bounding-box predictions to CSV.

## What It Does

- Generates a fake video sequence with a moving white object.
- Detects the object in each frame using grayscale thresholding and contour extraction.
- Saves tracking results to `outputs/predictions.csv`.
- Includes a basic visualization helper for inspecting tracked frames.

## Project Structure

- `main.py` - entry point that loads a sequence, runs tracking, visualizes results, and saves predictions.
- `src/data_loader.py` - loads fake sequences and provides a placeholder video loader.
- `src/fake_data.py` - creates synthetic frames and ground-truth boxes.
- `src/inference.py` - contains the simple tracker, visualization, and CSV export.
- `data/` - data folder and manifest placeholder.
- `models/` - reserved for future model files.
- `notebooks/` - experimentation notebooks.
- `outputs/` - prediction outputs.

## Requirements

Python 3.10+ is recommended. Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Run The Demo

From the project root:

```bash
python main.py
```

This will:

1. Generate a fake sequence.
2. Run the tracker over each frame.
3. Open a visualization window with tracked boxes.
4. Write predictions to `outputs/predictions.csv`.

## Output Format

The exported CSV contains one row per frame with these columns:

- `id`
- `x`
- `y`
- `w`
- `h`

## Notes

- The current implementation is a baseline tracker, not a trained production model.
- `load_video()` is available in `src/data_loader.py` for future real-video support.
- The fake data generator currently produces a single moving object on a black background, which makes the demo easy to validate.

## Next Steps

Potential improvements include:

- plugging in a stronger tracking model,
- adding evaluation metrics,
- supporting real input videos,
- and cleaning up the visualization flow for batch runs.
