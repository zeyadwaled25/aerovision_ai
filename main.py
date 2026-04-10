from src.data_loader import load_sequences
from src.inference import visualize_sequence


# Load dataset
sequences = load_sequences("data")

print(f"Number of sequences: {len(sequences)}")

# Select first sequence
sequence = sequences[0]

print("Playing:", sequence["seq_name"])

# Visualize
visualize_sequence(sequence)