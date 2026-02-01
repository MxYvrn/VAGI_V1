"""
Stage 7: Img-ID and Img-KNN
Scene representation and scene-level memory.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class SceneObject:
    """Represents a detected object in a scene."""
    proto_id: int           # Object prototype ID from Obj-KNN
    x: float                # X position (tile coordinates)
    y: float                # Y position (tile coordinates)
    scale: float            # Object scale (perimeter)

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            'proto_id': self.proto_id,
            'x': self.x,
            'y': self.y,
            'scale': self.scale
        }


@dataclass
class Scene:
    """Represents a complete scene with multiple objects."""
    objects: List[SceneObject] = field(default_factory=list)
    scene_id: Optional[int] = None

    def to_dict(self) -> dict:
        """Convert to dictionary representation."""
        return {
            'scene_id': self.scene_id,
            'objects': [obj.to_dict() for obj in self.objects]
        }

    def __len__(self):
        return len(self.objects)


def create_scene_from_objects(
    object_list: List[dict],
    obj_knn,
    similarity_threshold: float = 0.5
) -> Scene:
    """
    Create a Scene representation from extracted objects.

    Args:
        object_list: List of object dictionaries from features.extract_objects_from_chains
        obj_knn: ObjKNN instance for proto_id assignment
        similarity_threshold: Threshold for matching to existing prototypes

    Returns:
        Scene object
    """
    scene = Scene()

    for obj in object_list:
        v_object = obj['v_object']
        centroid = obj['centroid']  # (x, y) in tile coordinates
        scale = obj['scale']

        # Get or assign proto_id
        proto_id, is_new = obj_knn.get_or_add(v_object, similarity_threshold)

        # Create scene object
        scene_obj = SceneObject(
            proto_id=proto_id,
            x=centroid[0],
            y=centroid[1],
            scale=scale
        )

        scene.objects.append(scene_obj)

    return scene


class ImgKNN:
    """
    Scene-level KNN memory.

    Stores scenes and allows similarity queries between scenes.
    """

    def __init__(self):
        """Initialize scene memory."""
        self.scenes: List[Scene] = []
        self.next_scene_id = 0

    def add_scene(self, scene: Scene) -> int:
        """
        Add a scene to memory.

        Args:
            scene: Scene object to add

        Returns:
            Assigned scene_id
        """
        scene.scene_id = self.next_scene_id
        self.next_scene_id += 1

        self.scenes.append(scene)

        return scene.scene_id

    def query(
        self,
        query_scene: Scene,
        k: int = 1,
        distance_threshold: Optional[float] = None
    ) -> List[Tuple[int, float]]:
        """
        Query for similar scenes.

        Args:
            query_scene: Scene to query
            k: Number of nearest neighbors
            distance_threshold: Optional threshold for filtering

        Returns:
            List of (scene_id, distance) tuples
        """
        if len(self.scenes) == 0:
            return []

        # Compute distances to all scenes
        distances = []
        for scene in self.scenes:
            dist = self._scene_distance(query_scene, scene)
            distances.append((scene.scene_id, dist))

        # Sort by distance
        distances.sort(key=lambda x: x[1])

        # Filter by threshold
        if distance_threshold is not None:
            distances = [(sid, d) for sid, d in distances if d <= distance_threshold]

        return distances[:k]

    def _scene_distance(self, scene1: Scene, scene2: Scene) -> float:
        """
        Compute distance between two scenes.

        Simple V1 heuristic:
        - For each proto_id in scene1, find closest match in scene2
        - Sum up:
          - Number of unmatched objects
          - Position differences for matched objects
          - Scale differences for matched objects

        Args:
            scene1, scene2: Scenes to compare

        Returns:
            Distance value
        """
        if len(scene1) == 0 and len(scene2) == 0:
            return 0.0

        # Build proto_id to objects mapping for both scenes
        objs1_by_proto = {}
        for obj in scene1.objects:
            if obj.proto_id not in objs1_by_proto:
                objs1_by_proto[obj.proto_id] = []
            objs1_by_proto[obj.proto_id].append(obj)

        objs2_by_proto = {}
        for obj in scene2.objects:
            if obj.proto_id not in objs2_by_proto:
                objs2_by_proto[obj.proto_id] = []
            objs2_by_proto[obj.proto_id].append(obj)

        # Compute symmetric difference
        all_protos = set(objs1_by_proto.keys()) | set(objs2_by_proto.keys())

        total_distance = 0.0
        matched_count = 0

        for proto_id in all_protos:
            objs1 = objs1_by_proto.get(proto_id, [])
            objs2 = objs2_by_proto.get(proto_id, [])

            count_diff = abs(len(objs1) - len(objs2))
            total_distance += count_diff * 10.0  # Penalty for count mismatch

            # Match objects greedily by position
            min_count = min(len(objs1), len(objs2))
            for i in range(min_count):
                # Simple matching: pair by index (could be improved with Hungarian algorithm)
                obj1 = objs1[i]
                obj2 = objs2[i]

                # Position distance
                pos_dist = np.sqrt((obj1.x - obj2.x)**2 + (obj1.y - obj2.y)**2)
                total_distance += pos_dist

                # Scale distance (normalized)
                scale_dist = abs(obj1.scale - obj2.scale) / max(obj1.scale, obj2.scale, 1.0)
                total_distance += scale_dist

                matched_count += 1

        return total_distance

    def get_scene(self, scene_id: int) -> Optional[Scene]:
        """
        Retrieve a scene by ID.

        Args:
            scene_id: ID to look up

        Returns:
            Scene object or None
        """
        for scene in self.scenes:
            if scene.scene_id == scene_id:
                return scene
        return None

    def size(self) -> int:
        """Return number of scenes in memory."""
        return len(self.scenes)

    def clear(self):
        """Clear all scenes from memory."""
        self.scenes.clear()
        self.next_scene_id = 0
