"""Short-term memory buffer for recent object instances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class STMRecord:
    """A temporary instance-level observation."""

    type_id: int
    x: float
    y: float
    timestamp: float
    confidence: float
    feature: np.ndarray


class ShortTermMemory:
    """FIFO buffer for recent observations.

    Records are intentionally simple and temporary.
    """

    def __init__(self, capacity: int, feature_dim: int) -> None:
        self.capacity = int(capacity)
        self.feature_dim = int(feature_dim)
        self._records: List[STMRecord] = []

    def add_observation(
        self,
        type_id: int,
        position: tuple[float, float],
        timestamp: float,
        confidence: float,
        feature: np.ndarray,
    ) -> None:
        """Append one observation and evict oldest if capacity is exceeded."""
        feature = np.asarray(feature, dtype=np.float32)
        if feature.shape != (self.feature_dim,):
            raise ValueError("feature has invalid shape.")

        record = STMRecord(
            type_id=int(type_id),
            x=float(position[0]),
            y=float(position[1]),
            timestamp=float(timestamp),
            confidence=float(confidence),
            feature=feature,
        )
        self._records.append(record)
        if len(self._records) > self.capacity:
            self._records.pop(0)

    def get_records(self) -> List[STMRecord]:
        """Return a shallow copy of current records."""
        return list(self._records)

    def clear(self) -> None:
        """Reset the temporary buffer (e.g., after sleep consolidation)."""
        self._records.clear()


if __name__ == "__main__":
    stm = ShortTermMemory(capacity=3, feature_dim=4)
    stm.add_observation(
        type_id=1,
        position=(10.0, 20.0),
        timestamp=123.4,
        confidence=0.9,
        feature=np.array([0.2, 0.4, 0.6, 0.8], dtype=np.float32),
    )
    print("STM size:", len(stm.get_records()))
