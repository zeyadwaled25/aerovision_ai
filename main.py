from src.data_loader import load_fake_sequence
from src.inference import run_simple_tracker, save_results, visualize_tracking

sequences = load_fake_sequence()

all_results = []

for seq in sequences:
    results = run_simple_tracker(seq)
    all_results.extend(results)

visualize_tracking(seq["frames"], results)
save_results(all_results, "outputs/predictions.csv")