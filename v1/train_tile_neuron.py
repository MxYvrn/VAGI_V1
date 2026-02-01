"""
Offline training script for N² tile neuron.

Generates synthetic training data, trains a logistic regression neuron,
and saves weights to JSON file for runtime use.

Usage:
    python train_tile_neuron.py
"""

import numpy as np
from tile_neuron import TileNeuron, generate_training_set


def main():
    """Train and save N² tile neuron weights."""

    # Configuration
    n_samples = 10000
    tile_size = 4
    learning_rate = 0.5
    epochs = 150
    batch_size = 64
    weights_path = "n2_tile_weights.json"
    seed = 42

    print(f"Generating {n_samples} training samples...")
    patches, labels = generate_training_set(
        n_samples=n_samples,
        tile_size=tile_size,
        seed=seed
    )

    # Count edge vs non-edge samples
    n_edges = np.sum(labels)
    n_non_edges = len(labels) - n_edges
    print(f"  Edge samples: {n_edges}")
    print(f"  Non-edge samples: {n_non_edges}")
    print(f"  Balance: {n_edges/len(labels)*100:.1f}% edges")

    print("\nTraining tile neuron...")
    neuron = TileNeuron(tile_size=tile_size)
    neuron.train(
        patches=patches,
        labels=labels,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        verbose=True
    )

    print(f"\nSaving weights to {weights_path}...")
    neuron.save_weights(weights_path)

    # Validate by computing final accuracy
    print("\nValidation:")
    X_flat = np.array([neuron._flatten_patch(p) for p in patches])
    predictions = neuron._predict_batch(X_flat)
    accuracy = np.mean(predictions == labels)
    print(f"  Final training accuracy: {accuracy*100:.2f}%")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
