"""Long-term memory store for type-level cognitive state.

This module is intentionally decoupled from the current v1 perception pipeline.
"""

from __future__ import annotations

from typing import Dict

import numpy as np


class LongTermMemory:
    """Persistent type-level memory used by the type GNN.

    Stores:
    - type_centroids: (N, F) array
    - adjacency_matrix: (N, N) array
    - type_activation_counts: (N,) array
    """

    def __init__(self, num_types: int, feature_dim: int) -> None:
        self.num_types = int(num_types)
        self.feature_dim = int(feature_dim)
        # TODO: PyTorch migration possible here (Tensor-backed storage, device placement).
        self.type_centroids = np.zeros((self.num_types, self.feature_dim), dtype=np.float32)
        self.adjacency_matrix = np.zeros((self.num_types, self.num_types), dtype=np.float32)
        self.type_activation_counts = np.zeros((self.num_types,), dtype=np.float32)

    def snapshot(self) -> Dict[str, np.ndarray]:
        """Return a copy-safe snapshot for downstream modules."""
        return {
            "type_centroids": self.type_centroids.copy(),
            "adjacency_matrix": self.adjacency_matrix.copy(),
            "type_activation_counts": self.type_activation_counts.copy(),
        }

    def set_state(
        self,
        type_centroids: np.ndarray,
        adjacency_matrix: np.ndarray,
        type_activation_counts: np.ndarray,
    ) -> None:
        """Replace full memory state after shape validation."""
        if type_centroids.shape != (self.num_types, self.feature_dim):
            raise ValueError("type_centroids has invalid shape.")
        if adjacency_matrix.shape != (self.num_types, self.num_types):
            raise ValueError("adjacency_matrix has invalid shape.")
        if type_activation_counts.shape != (self.num_types,):
            raise ValueError("type_activation_counts has invalid shape.")

        self.type_centroids = type_centroids.astype(np.float32, copy=True)
        self.adjacency_matrix = adjacency_matrix.astype(np.float32, copy=True)
        self.type_activation_counts = type_activation_counts.astype(np.float32, copy=True)


if __name__ == "__main__":
    ltm = LongTermMemory(num_types=4, feature_dim=8)
    state = ltm.snapshot()
    print("LTM centroids shape:", state["type_centroids"].shape)
