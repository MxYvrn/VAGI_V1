"""
Tests for Stage 7: Img-ID and Img-KNN
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import img_id
import obj_knn


def test_scene_creation():
    """Test creating a scene from objects."""
    memory = obj_knn.ObjKNN()

    # Create mock objects
    objects = [
        {
            'v_object': np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0], dtype=float),
            'centroid': (5.0, 10.0),
            'scale': 12.5
        },
        {
            'v_object': np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0], dtype=float),
            'centroid': (15.0, 20.0),
            'scale': 8.3
        }
    ]

    scene = img_id.create_scene_from_objects(objects, memory, similarity_threshold=0.5)

    print(f"\nScene created with {len(scene)} objects")
    print(f"Scene dict: {scene.to_dict()}")

    assert len(scene) == 2, "Scene should have 2 objects"
    assert scene.objects[0].x == 5.0
    assert scene.objects[0].y == 10.0

    print("✓ test_scene_creation passed")


def test_img_knn():
    """Test scene memory and querying."""
    scene_memory = img_id.ImgKNN()

    # Create scenes
    scene1 = img_id.Scene(objects=[
        img_id.SceneObject(proto_id=1, x=5, y=10, scale=10),
        img_id.SceneObject(proto_id=2, x=15, y=20, scale=8)
    ])

    scene2 = img_id.Scene(objects=[
        img_id.SceneObject(proto_id=1, x=6, y=11, scale=10),  # Similar to scene1
        img_id.SceneObject(proto_id=2, x=16, y=21, scale=8)
    ])

    scene3 = img_id.Scene(objects=[
        img_id.SceneObject(proto_id=3, x=30, y=30, scale=15)  # Different
    ])

    # Add to memory
    id1 = scene_memory.add_scene(scene1)
    id2 = scene_memory.add_scene(scene2)
    id3 = scene_memory.add_scene(scene3)

    print(f"\nAdded scenes with IDs: {id1}, {id2}, {id3}")

    # Query for similar to scene1
    results = scene_memory.query(scene1, k=2)
    print(f"Nearest to scene1: {results}")

    # Should find itself first, then scene2
    assert results[0][0] == id1, "Should find itself"
    assert results[0][1] == 0.0, "Distance to itself should be 0"
    assert results[1][0] == id2, "Scene2 should be second nearest"

    print("✓ test_img_knn passed")


def test_scene_distance():
    """Test scene distance computation."""
    scene_memory = img_id.ImgKNN()

    # Identical scenes
    scene1 = img_id.Scene(objects=[
        img_id.SceneObject(proto_id=1, x=5, y=10, scale=10)
    ])

    scene2 = img_id.Scene(objects=[
        img_id.SceneObject(proto_id=1, x=5, y=10, scale=10)
    ])

    dist = scene_memory._scene_distance(scene1, scene2)
    print(f"\nDistance between identical scenes: {dist}")
    assert dist == 0.0, "Identical scenes should have distance 0"

    # Different positions
    scene3 = img_id.Scene(objects=[
        img_id.SceneObject(proto_id=1, x=10, y=15, scale=10)
    ])

    dist2 = scene_memory._scene_distance(scene1, scene3)
    print(f"Distance with position diff: {dist2}")
    assert dist2 > 0, "Different positions should have distance > 0"

    # Different proto_ids
    scene4 = img_id.Scene(objects=[
        img_id.SceneObject(proto_id=2, x=5, y=10, scale=10)
    ])

    dist3 = scene_memory._scene_distance(scene1, scene4)
    print(f"Distance with different proto_ids: {dist3}")
    assert dist3 > dist2, "Different proto_ids should have larger distance"

    print("✓ test_scene_distance passed")


def test_empty_scenes():
    """Test handling of empty scenes."""
    scene_memory = img_id.ImgKNN()

    empty_scene = img_id.Scene()
    scene_memory.add_scene(empty_scene)

    results = scene_memory.query(empty_scene, k=1)
    assert len(results) == 1
    assert results[0][1] == 0.0

    print("✓ test_empty_scenes passed")


if __name__ == "__main__":
    print("Running Img-ID tests...\n")
    test_scene_creation()
    test_img_knn()
    test_scene_distance()
    test_empty_scenes()
    print("\n✅ All Img-ID tests passed!")
