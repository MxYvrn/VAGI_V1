"""
Img-ID and Img-KNN

Scene representation and scene-level memory.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class SceneObject:
    """Represents a detected object in a scene."""
    proto_id: int
    x: float
    y: float
    scale: float


@dataclass
class Scene:
    """Represents a complete scene with multiple objects."""
    objects: List[SceneObject] = field(default_factory=list)
    scene_id: Optional[int] = None

    def __len__(self):
        return len(self.objects)


def build_scene(
    objects: List[dict],
    obj_memory,
    similarity_threshold: float = 0.5
) -> Scene:
    """
    Create a Scene representation from extracted objects.

    Args:
        objects: List of object dictionaries from features.extract_objects_from_chains
        obj_memory: ObjectMemoryKNN instance for proto_id assignment
        similarity_threshold: Threshold for matching to existing prototypes

    Returns:
        Scene object
    """
    scene = Scene()

    for obj in objects:
        v_object = obj['v_object']
        centroid = obj['centroid']
        scale = obj['scale']

        proto_id, is_new = obj_memory.get_or_add(v_object, similarity_threshold)

        scene_obj = SceneObject(
            proto_id=proto_id,
            x=centroid[0],
            y=centroid[1],
            scale=scale
        )

        scene.objects.append(scene_obj)

    return scene


class SceneMemoryKNN:
    """
    Scene-level KNN memory.

    Stores scenes and allows similarity queries between scenes.
    """

    def __init__(self):
        """Initialize scene memory."""
        self.scenes: List[Scene] = []
        self.next_scene_id = 0

    def add(self, scene: Scene) -> int:
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

        distances = []
        for scene in self.scenes:
            dist = self._scene_distance(query_scene, scene)
            distances.append((scene.scene_id, dist))

        distances.sort(key=lambda x: x[1])

        if distance_threshold is not None:
            distances = [(sid, d) for sid, d in distances if d <= distance_threshold]

        return distances[:k]

    def _scene_distance(self, scene1: Scene, scene2: Scene) -> float:
        """
        Compute distance between two scenes.

        Simple heuristic:
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

        all_protos = set(objs1_by_proto.keys()) | set(objs2_by_proto.keys())

        total_distance = 0.0

        for proto_id in all_protos:
            objs1 = objs1_by_proto.get(proto_id, [])
            objs2 = objs2_by_proto.get(proto_id, [])

            count_diff = abs(len(objs1) - len(objs2))
            total_distance += count_diff * 10.0

            min_count = min(len(objs1), len(objs2))
            for i in range(min_count):
                obj1 = objs1[i]
                obj2 = objs2[i]

                pos_dist = np.sqrt((obj1.x - obj2.x)**2 + (obj1.y - obj2.y)**2)
                total_distance += pos_dist

                scale_dist = abs(obj1.scale - obj2.scale) / max(obj1.scale, obj2.scale, 1.0)
                total_distance += scale_dist

        return total_distance

    def get_scene(self, scene_id: int) -> Optional[Scene]:
        """Retrieve a scene by ID."""
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
