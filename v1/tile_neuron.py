"""
Tile Neuron: Single logistic regression neuron for 4x4 edge detection.
Includes training capability and synthetic patch generation.
"""

import numpy as np
import json
from typing import Tuple, List


class TileNeuron:
    """
    Single logistic regression neuron for classifying 4x4 RGB patches as edge/non-edge.

    Model: y_hat = σ(W·x + b) where x ∈ ℝ⁴⁸ (flattened 4×4×3 patch)
    """

    def __init__(self, tile_size: int = 4):
        self.tile_size = tile_size
        self.input_dim = tile_size * tile_size * 3  # 48 for 4x4 RGB
        self.W = None  # weights: shape (48,)
        self.b = None  # bias: scalar

    def _sigmoid(self, z: np.ndarray) -> np.ndarray:
        """Sigmoid activation function with numerical stability."""
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))
    

    def relu(self, z: np.ndarray) -> np.ndarray:
        """ReLU activation function."""
        return np.maximum(0, z)

    def _flatten_patch(self, patch: np.ndarray) -> np.ndarray:
        """Flatten 4x4x3 patch to 48-dim vector, normalize to [0,1]."""
        flat = patch.reshape(-1).astype(np.float32)
        # Normalize to [0, 1] range
        flat = flat / 255.0
        return flat

    def train(self, patches: np.ndarray, labels: np.ndarray,
              learning_rate: float = 0.1, epochs: int = 100,
              batch_size: int = 32, verbose: bool = False):
        """
        Train the neuron using gradient descent on binary cross-entropy loss.

        Args:
            patches: shape (N, 4, 4, 3) - training patches
            labels: shape (N,) - binary labels (0 or 1)
            learning_rate: step size for gradient descent
            epochs: number of training epochs
            batch_size: mini-batch size
            verbose: print training progress
        """
        N = patches.shape[0]

        # Flatten and normalize patches
        X = np.array([self._flatten_patch(p) for p in patches])  # (N, 48)
        y = labels.astype(np.float32)  # (N,)

        # Initialize weights with small random values
        np.random.seed(42)
        self.W = np.random.randn(self.input_dim).astype(np.float32) * 0.01
        self.b = 0.0

        # Training loop
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(N)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_loss = 0.0
            num_batches = 0

            # Mini-batch gradient descent
            for i in range(0, N, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                # Forward pass
                z = np.dot(X_batch, self.W) + self.b
                y_pred = self._sigmoid(z)

                # Compute loss (binary cross-entropy)
                epsilon = 1e-7  # avoid log(0)
                loss = -np.mean(
                    y_batch * np.log(y_pred + epsilon) +
                    (1 - y_batch) * np.log(1 - y_pred + epsilon)
                )
                epoch_loss += loss
                num_batches += 1

                # Backward pass (gradients)
                error = y_pred - y_batch  # derivative of BCE w.r.t. z
                grad_W = np.dot(X_batch.T, error) / len(X_batch)
                grad_b = np.mean(error)

                # Update weights
                self.W -= learning_rate * grad_W
                self.b -= learning_rate * grad_b

            if verbose and (epoch + 1) % 10 == 0:
                avg_loss = epoch_loss / num_batches
                accuracy = self._compute_accuracy(X, y)
                print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

    def _compute_accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Compute classification accuracy."""
        predictions = self._predict_batch(X)
        return np.mean(predictions == y)

    def _predict_batch(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for batch of flattened patches."""
        z = np.dot(X, self.W) + self.b
        y_pred = self._sigmoid(z)
        return (y_pred > 0.5).astype(int)

    def predict_proba(self, patch: np.ndarray) -> float:
        """
        Predict probability that patch is an edge.

        Args:
            patch: shape (4, 4, 3) - single RGB patch

        Returns:
            float in [0, 1] - probability of being an edge
        """
        if self.W is None or self.b is None:
            raise ValueError("Model not trained or loaded. Call train() or load_weights() first.")

        x = self._flatten_patch(patch)
        z = np.dot(self.W, x) + self.b
        return float(self._sigmoid(z))

    def predict_label(self, patch: np.ndarray) -> int:
        """
        Predict binary label (0 or 1) for a patch.

        Args:
            patch: shape (4, 4, 3) - single RGB patch

        Returns:
            0 (non-edge) or 1 (edge)
        """
        prob = self.predict_proba(patch)
        return 1 if prob > 0.5 else 0

    def save_weights(self, path: str):
        """
        Save weights to JSON file.

        Args:
            path: file path for saving (e.g., "n2_tile_weights.json")
        """
        if self.W is None or self.b is None:
            raise ValueError("No weights to save. Train the model first.")

        data = {
            "tile_size": self.tile_size,
            "W": self.W.tolist(),
            "b": float(self.b)
        }

        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    def load_weights(self, path: str):
        """
        Load weights from JSON file.

        Args:
            path: file path to load from
        """
        with open(path, 'r') as f:
            data = json.load(f)

        self.tile_size = data["tile_size"]
        self.input_dim = self.tile_size * self.tile_size * 3
        self.W = np.array(data["W"], dtype=np.float32)
        self.b = float(data["b"])


# ============================================================================
# Synthetic Patch Generation (pixel_filler-style)
# ============================================================================

def generate_edge_patch(orientation: str, tile_size: int = 4,
                       min_contrast: float = 50.0, noise_std: float = 5.0,
                       p_flip: float = 0.3) -> np.ndarray:
    """
    Generate a synthetic 4x4 RGB patch containing an edge.

    Args:
        orientation: "H" (horizontal), "V" (vertical), or "D" (diagonal)
        tile_size: patch size (default 4)
        min_contrast: minimum color difference between sides
        noise_std: standard deviation of Gaussian noise
        p_flip: probability of flipping border pixels for roughness

    Returns:
        patch: shape (tile_size, tile_size, 3) - RGB values in [0, 255]
    """
    # Sample two contrasting colors
    while True:
        c1 = np.random.randint(0, 256, size=3).astype(np.float32)
        c2 = np.random.randint(0, 256, size=3).astype(np.float32)

        # Ensure sufficient contrast
        if np.linalg.norm(c1 - c2) > min_contrast:
            break

    # Initialize patch
    patch = np.zeros((tile_size, tile_size, 3), dtype=np.float32)
    side_mask = np.zeros((tile_size, tile_size), dtype=bool)  # True = side A (c1)

    # Assign base colors based on orientation
    for i in range(tile_size):
        for j in range(tile_size):
            if orientation == "H":
                # Horizontal: top vs bottom
                side_A = (i < tile_size // 2)
            elif orientation == "V":
                # Vertical: left vs right
                side_A = (j < tile_size // 2)
            elif orientation == "D":
                # Diagonal: above vs below diagonal
                side_A = (i <= j)
            else:
                raise ValueError(f"Unknown orientation: {orientation}")

            side_mask[i, j] = side_A
            patch[i, j] = c1 if side_A else c2

    # Add noise to all pixels
    noise = np.random.randn(tile_size, tile_size, 3) * noise_std
    patch += noise

    # Boundary roughening: flip some border pixels
    for i in range(tile_size):
        for j in range(tile_size):
            # Check if this is a border pixel (adjacent to opposite side)
            is_border = False
            for di, dj in [(-1,0), (1,0), (0,-1), (0,1)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < tile_size and 0 <= nj < tile_size:
                    if side_mask[i, j] != side_mask[ni, nj]:
                        is_border = True
                        break

            # Flip with probability p_flip
            if is_border and np.random.rand() < p_flip:
                # Flip to opposite side's color with fresh noise
                opposite_color = c2 if side_mask[i, j] else c1
                fresh_noise = np.random.randn(3) * noise_std
                patch[i, j] = opposite_color + fresh_noise

    # Clip to valid range
    patch = np.clip(patch, 0, 255)

    return patch.astype(np.float32)


def generate_non_edge_patch(tile_size: int = 4, noise_std: float = 5.0) -> np.ndarray:
    """
    Generate a synthetic 4x4 RGB patch with no edge (uniform color).

    Args:
        tile_size: patch size (default 4)
        noise_std: standard deviation of Gaussian noise

    Returns:
        patch: shape (tile_size, tile_size, 3) - RGB values in [0, 255]
    """
    # Sample base color
    base_color = np.random.randint(0, 256, size=3).astype(np.float32)

    # Initialize patch with base color
    patch = np.tile(base_color, (tile_size, tile_size, 1))

    # Add small noise
    noise = np.random.randn(tile_size, tile_size, 3) * noise_std
    patch += noise

    # Clip to valid range
    patch = np.clip(patch, 0, 255)

    return patch.astype(np.float32)


def generate_training_set(n_samples: int, tile_size: int = 4,
                          seed: int = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a balanced training set of edge and non-edge patches.

    Args:
        n_samples: total number of samples to generate
        tile_size: patch size (default 4)
        seed: random seed for reproducibility

    Returns:
        patches: shape (n_samples, tile_size, tile_size, 3)
        labels: shape (n_samples,) - binary labels (0=non-edge, 1=edge)
    """
    if seed is not None:
        np.random.seed(seed)

    # Types: H, V, D are edges (label=1), N is non-edge (label=0)
    # 3 edge types + 3 N gives ~50/50 balance
    types = ["H", "V", "D", "N", "N", "N"]

    patches = []
    labels = []

    for _ in range(n_samples):
        patch_type = np.random.choice(types)

        if patch_type == "N":
            patch = generate_non_edge_patch(tile_size)
            label = 0
        else:
            patch = generate_edge_patch(patch_type, tile_size)
            label = 1

        patches.append(patch)
        labels.append(label)

    return np.array(patches), np.array(labels, dtype=np.int32)
