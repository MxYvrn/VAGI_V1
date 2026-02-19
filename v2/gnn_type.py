"""Type-level GNN module (NumPy-only prototype)."""

from __future__ import annotations

import numpy as np


class TypeLevelGNN:
    """Simple 2-layer message passing network.

    Forward pass:
    H1 = relu(A @ X @ W1)
    H2 = A @ H1 @ W2
    """

    def __init__(self, feature_dim: int, hidden_dim: int, seed: int = 0) -> None:
        self.feature_dim = int(feature_dim)
        self.hidden_dim = int(hidden_dim)
        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / max(1, self.feature_dim))
        scale2 = np.sqrt(2.0 / max(1, self.hidden_dim))
        self.W1 = (rng.standard_normal((self.feature_dim, self.hidden_dim)) * scale1).astype(np.float32)
        self.W2 = (rng.standard_normal((self.hidden_dim, self.feature_dim)) * scale2).astype(np.float32)
        # TODO: PyTorch migration possible here (replace manual params with nn.Parameter).

    @staticmethod
    def relu(x: np.ndarray) -> np.ndarray:
        """Element-wise ReLU activation."""
        return np.maximum(x, 0.0)

    def forward(self, node_features: np.ndarray, adjacency_matrix: np.ndarray) -> np.ndarray:
        """Run one forward pass and return predicted node features.

        Args:
            node_features: shape (N, F)
            adjacency_matrix: shape (N, N)
        """
        x = np.asarray(node_features, dtype=np.float32)
        a = np.asarray(adjacency_matrix, dtype=np.float32)
        n, f = x.shape
        if f != self.feature_dim:
            raise ValueError("node_features feature dimension mismatch.")
        if a.shape != (n, n):
            raise ValueError("adjacency_matrix must be square with same N as node_features.")

        # TODO: PyTorch migration possible here (batched sparse/dense matmul).
        h1 = self.relu(a @ x @ self.W1)
        h2 = a @ h1 @ self.W2
        return h2

    def update_weights(self, dW1: np.ndarray, dW2: np.ndarray, lr: float = 1e-3) -> None:
        """Minimal weight update hook for future training loop integration."""
        if dW1.shape != self.W1.shape or dW2.shape != self.W2.shape:
            raise ValueError("Gradient shapes must match parameter shapes.")
        self.W1 -= np.float32(lr) * dW1.astype(np.float32)
        self.W2 -= np.float32(lr) * dW2.astype(np.float32)


if __name__ == "__main__":
    N, F, H = 5, 8, 16
    model = TypeLevelGNN(feature_dim=F, hidden_dim=H, seed=7)
    X = np.random.rand(N, F).astype(np.float32)
    A = np.eye(N, dtype=np.float32)
    Y = model.forward(X, A)
    print("Predicted node feature shape:", Y.shape)
