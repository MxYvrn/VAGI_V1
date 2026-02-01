"""
Stage 6: Obj-KNN (Object Memory)
Simple KNN memory for object prototypes with weighted distance metric.
"""

import numpy as np
from typing import List, Tuple, Optional


class ObjKNN:
    """
    Object memory using KNN with weighted Euclidean distance.

    Stores v_object vectors and assigns proto_ids.
    Distance metric weights shape features and color features separately.
    """

    def __init__(self, shape_weight: float = 1.0, color_weight: float = 0.1):
        """
        Initialize object memory.

        Args:
            shape_weight: Weight for shape features (first 10 dimensions)
            color_weight: Weight for color features (last 3 dimensions)
        """
        self.shape_weight = shape_weight
        self.color_weight = color_weight

        # Storage
        self.prototypes: List[np.ndarray] = []  # List of v_object vectors
        self.proto_ids: List[int] = []          # Corresponding proto_id for each

        # Counter for assigning new proto_ids
        self.next_proto_id = 0

    def add_object(self, v_object: np.ndarray, proto_id: Optional[int] = None) -> int:
        """
        Add an object to memory.

        Args:
            v_object: 13D feature vector
            proto_id: Optional proto_id to assign. If None, auto-assign.

        Returns:
            Assigned proto_id
        """
        if proto_id is None:
            proto_id = self.next_proto_id
            self.next_proto_id += 1

        self.prototypes.append(v_object.copy())
        self.proto_ids.append(proto_id)

        return proto_id

    def query(
        self,
        v_object: np.ndarray,
        k: int = 1,
        distance_threshold: Optional[float] = None
    ) -> List[Tuple[int, float]]:
        """
        Query for nearest neighbors.

        Args:
            v_object: 13D feature vector to query
            k: Number of nearest neighbors to return
            distance_threshold: Optional threshold - if nearest is farther, return empty

        Returns:
            List of (proto_id, distance) tuples, sorted by distance
        """
        if len(self.prototypes) == 0:
            return []

        # Compute distances to all prototypes
        distances = []
        for i, prototype in enumerate(self.prototypes):
            dist = self._weighted_distance(v_object, prototype)
            distances.append((self.proto_ids[i], dist))

        # Sort by distance
        distances.sort(key=lambda x: x[1])

        # Filter by threshold if provided
        if distance_threshold is not None:
            distances = [(pid, d) for pid, d in distances if d <= distance_threshold]

        # Return top k
        return distances[:k]

    def get_or_add(
        self,
        v_object: np.ndarray,
        similarity_threshold: float = 0.5
    ) -> Tuple[int, bool]:
        """
        Get existing proto_id if similar enough, otherwise add as new.

        Args:
            v_object: 13D feature vector
            similarity_threshold: Maximum distance to consider as "same" object

        Returns:
            (proto_id, is_new) where is_new indicates if a new prototype was created
        """
        # Query for nearest neighbor
        neighbors = self.query(v_object, k=1, distance_threshold=similarity_threshold)

        if neighbors:
            # Found similar object
            return neighbors[0][0], False
        else:
            # No similar object, add new
            proto_id = self.add_object(v_object)
            return proto_id, True

    def _weighted_distance(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """
        Compute weighted Euclidean distance between two v_object vectors.

        v_object structure:
        - [0:8]: direction histogram (shape)
        - [8:10]: turn statistics (shape)
        - [10:13]: RGB color

        Args:
            v1, v2: 13D feature vectors

        Returns:
            Weighted distance
        """
        # Split into shape and color components
        shape1 = v1[:10]
        shape2 = v2[:10]
        color1 = v1[10:13]
        color2 = v2[10:13]

        # Compute weighted distances
        shape_dist = np.linalg.norm(shape1 - shape2) * self.shape_weight
        color_dist = np.linalg.norm(color1 - color2) * self.color_weight

        # Combine
        total_dist = np.sqrt(shape_dist**2 + color_dist**2)

        return total_dist

    def get_prototype(self, proto_id: int) -> Optional[np.ndarray]:
        """
        Get the prototype vector for a given proto_id.

        Args:
            proto_id: ID to look up

        Returns:
            v_object vector or None if not found
        """
        for i, pid in enumerate(self.proto_ids):
            if pid == proto_id:
                return self.prototypes[i].copy()
        return None

    def get_all_prototypes(self) -> List[Tuple[int, np.ndarray]]:
        """
        Get all prototypes.

        Returns:
            List of (proto_id, v_object) tuples
        """
        return list(zip(self.proto_ids, self.prototypes))

    def size(self) -> int:
        """Return number of prototypes in memory."""
        return len(self.prototypes)

    def clear(self):
        """Clear all prototypes from memory."""
        self.prototypes.clear()
        self.proto_ids.clear()
        self.next_proto_id = 0
