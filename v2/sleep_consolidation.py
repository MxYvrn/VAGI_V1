"""Sleep-based consolidation from STM into LTM."""

from __future__ import annotations

from typing import Tuple

import numpy as np

from .long_term_memory import LongTermMemory
from .short_term_memory import STMRecord


class SleepConsolidation:
    """Consolidate short-term observations into long-term memory.

    Uses incremental weighted updates (not naive global averaging).
    """

    def __init__(self, edge_decay: float = 0.98, min_confidence: float = 0.0) -> None:
        self.edge_decay = float(edge_decay)
        self.min_confidence = float(min_confidence)

    def consolidate(self, ltm: LongTermMemory, stm_records: list[STMRecord]) -> Tuple[int, int]:
        """Update centroids, counts, and adjacency from a list of STM records.

        Returns:
        - number of accepted records
        - number of edge transitions applied
        """
        if not stm_records:
            return 0, 0

        # TODO: PyTorch migration possible here (vectorized tensor ops for batched consolidation).
        # Optional decay to prevent unbounded edge growth.
        ltm.adjacency_matrix *= self.edge_decay

        accepted = [r for r in stm_records if r.confidence >= self.min_confidence]
        if not accepted:
            return 0, 0

        accepted.sort(key=lambda r: r.timestamp)

        # Incremental centroid update:
        # c_new = c_old + alpha * (x - c_old), alpha = w / (count_old + w)
        for record in accepted:
            t = record.type_id
            if t < 0 or t >= ltm.num_types:
                continue
            w = max(1e-6, float(record.confidence))
            old_count = float(ltm.type_activation_counts[t])
            alpha = w / (old_count + w)
            ltm.type_centroids[t] = ltm.type_centroids[t] + alpha * (record.feature - ltm.type_centroids[t])
            ltm.type_activation_counts[t] = old_count + w

        transitions = 0
        for a, b in zip(accepted[:-1], accepted[1:]):
            ta, tb = a.type_id, b.type_id
            if ta < 0 or ta >= ltm.num_types or tb < 0 or tb >= ltm.num_types:
                continue
            # Weighted directed transition increment.
            delta = 0.5 * (max(0.0, a.confidence) + max(0.0, b.confidence))
            ltm.adjacency_matrix[ta, tb] += delta
            transitions += 1

        return len(accepted), transitions


if __name__ == "__main__":
    from .short_term_memory import ShortTermMemory

    ltm = LongTermMemory(num_types=3, feature_dim=4)
    stm = ShortTermMemory(capacity=10, feature_dim=4)
    stm.add_observation(0, (0.0, 0.0), 1.0, 0.8, np.array([1, 0, 0, 0], dtype=np.float32))
    stm.add_observation(1, (1.0, 1.0), 2.0, 0.9, np.array([0, 1, 0, 0], dtype=np.float32))

    sleeper = SleepConsolidation()
    n_obs, n_edges = sleeper.consolidate(ltm, stm.get_records())
    print("Consolidated:", n_obs, "records,", n_edges, "transitions")
