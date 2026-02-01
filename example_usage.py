"""
Simple usage examples for the V1 pipeline.
"""

import numpy as np
import sys
sys.path.insert(0, 'v1')

import n2_activation
import edge_filler
import edge_runner
import chain_filter
import features
import obj_knn
import img_id


def example_1_single_image():
    """Example 1: Process a single image."""
    print("Example 1: Process a single image")
    print("-" * 50)

    # Create a simple test image (or load your own)
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[20:44, 20:44, 0] = 255  # Red square

    # Stage 1: Edge detection
    grid = n2_activation.compute_activation_grid(image, tile_size=4, threshold=30.0)
    print(f"Activated tiles: {np.sum(grid.get_activation_map())}")

    # Stage 2: Fill gaps
    filled = edge_filler.edge_filler(grid)

    # Stage 3: Extract chains
    chains = edge_runner.extract_chains_from_grid(filled)
    print(f"Extracted chains: {len(chains)}")

    # Stage 4: Filter chains
    filtered = chain_filter.filter_chains(
        chains,
        grid_height=filled.height,
        grid_width=filled.width,
        min_length=3
    )
    print(f"Filtered chains: {len(filtered)}")

    # Stage 5: Extract features
    objects = features.extract_objects_from_chains(filtered, image, tile_size=4)
    print(f"Detected objects: {len(objects)}")

    for i, obj in enumerate(objects):
        print(f"  Object {i}: centroid={obj['centroid']}, scale={obj['scale']:.2f}")

    print()


def example_2_object_memory():
    """Example 2: Build object memory from multiple detections."""
    print("Example 2: Object memory and recognition")
    print("-" * 50)

    # Create object memory
    memory = obj_knn.ObjKNN(shape_weight=1.0, color_weight=0.1)

    # Simulate detecting objects in different images
    # Object 1: Red square
    obj1 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0], dtype=float)
    proto_id1, is_new1 = memory.get_or_add(obj1, similarity_threshold=0.5)
    print(f"Object 1: proto_id={proto_id1}, is_new={is_new1}")

    # Object 2: Similar red square (should match)
    obj2 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 250, 5, 5], dtype=float)
    proto_id2, is_new2 = memory.get_or_add(obj2, similarity_threshold=0.5)
    print(f"Object 2: proto_id={proto_id2}, is_new={is_new2}")

    # Object 3: Blue circle (different)
    obj3 = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255], dtype=float)
    proto_id3, is_new3 = memory.get_or_add(obj3, similarity_threshold=0.5)
    print(f"Object 3: proto_id={proto_id3}, is_new={is_new3}")

    print(f"\nTotal prototypes in memory: {memory.size()}")

    # Query for similar objects
    results = memory.query(obj1, k=2)
    print(f"\nNearest to Object 1:")
    for proto_id, distance in results:
        print(f"  proto_id={proto_id}, distance={distance:.3f}")

    print()


def example_3_scene_representation():
    """Example 3: Create scene representation."""
    print("Example 3: Scene representation")
    print("-" * 50)

    # Create object memory
    obj_memory = obj_knn.ObjKNN()

    # Simulate extracted objects from an image
    objects = [
        {
            'v_object': np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0], dtype=float),
            'centroid': (10.0, 15.0),
            'scale': 20.5
        },
        {
            'v_object': np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0], dtype=float),
            'centroid': (25.0, 30.0),
            'scale': 15.3
        }
    ]

    # Create scene
    scene = img_id.create_scene_from_objects(objects, obj_memory, similarity_threshold=0.5)

    print(f"Scene with {len(scene)} objects:")
    for obj in scene.objects:
        print(f"  proto_id={obj.proto_id}, pos=({obj.x:.1f}, {obj.y:.1f}), scale={obj.scale:.1f}")

    # Scene as dictionary
    scene_dict = scene.to_dict()
    print(f"\nScene dictionary:")
    print(f"  {scene_dict}")

    print()


def example_4_scene_memory():
    """Example 4: Scene-level memory and similarity."""
    print("Example 4: Scene memory and similarity")
    print("-" * 50)

    # Create scene memory
    scene_memory = img_id.ImgKNN()

    # Create two similar scenes
    scene1 = img_id.Scene(objects=[
        img_id.SceneObject(proto_id=1, x=10, y=15, scale=20),
        img_id.SceneObject(proto_id=2, x=25, y=30, scale=15)
    ])

    scene2 = img_id.Scene(objects=[
        img_id.SceneObject(proto_id=1, x=11, y=16, scale=21),  # Slightly different
        img_id.SceneObject(proto_id=2, x=26, y=31, scale=16)
    ])

    # Create different scene
    scene3 = img_id.Scene(objects=[
        img_id.SceneObject(proto_id=3, x=50, y=50, scale=30)
    ])

    # Add to memory
    id1 = scene_memory.add_scene(scene1)
    id2 = scene_memory.add_scene(scene2)
    id3 = scene_memory.add_scene(scene3)

    print(f"Added {scene_memory.size()} scenes to memory")

    # Query for similar scenes
    results = scene_memory.query(scene1, k=3)
    print(f"\nScenes similar to Scene {id1}:")
    for scene_id, distance in results:
        print(f"  Scene {scene_id}: distance={distance:.2f}")

    print()


def example_5_full_pipeline():
    """Example 5: Complete end-to-end pipeline."""
    print("Example 5: Complete pipeline")
    print("-" * 50)

    # Create test image
    image = np.zeros((128, 128, 3), dtype=np.uint8)
    image[18:46, 18:46, 0] = 255  # Red square
    image[70:98, 18:46, 1] = 255  # Green square

    # Initialize memories
    obj_memory = obj_knn.ObjKNN()
    scene_memory = img_id.ImgKNN()

    # Run pipeline
    print("Running V1 pipeline...")

    # Stages 1-2
    grid = n2_activation.compute_activation_grid(image, tile_size=4, threshold=30.0)
    filled = edge_filler.edge_filler(grid)

    # Stage 3-4
    chains = edge_runner.extract_chains_from_grid(filled)
    filtered = chain_filter.filter_chains(
        chains,
        grid_height=filled.height,
        grid_width=filled.width,
        min_length=3
    )

    # Stage 5
    objects = features.extract_objects_from_chains(filtered, image, tile_size=4)

    # Stage 6-7
    scene = img_id.create_scene_from_objects(objects, obj_memory, similarity_threshold=0.5)
    scene_id = scene_memory.add_scene(scene)

    # Results
    print(f"\nResults:")
    print(f"  Detected {len(objects)} objects")
    print(f"  Created {obj_memory.size()} object prototypes")
    print(f"  Scene {scene_id} with {len(scene)} objects")

    # Display scene
    print(f"\nScene representation:")
    for obj in scene.objects:
        print(f"  proto_id={obj.proto_id}, pos=({obj.x:.1f}, {obj.y:.1f}), scale={obj.scale:.1f}")

    print()


if __name__ == "__main__":
    print("\n" + "=" * 50)
    print("V1 PIPELINE USAGE EXAMPLES")
    print("=" * 50 + "\n")

    example_1_single_image()
    example_2_object_memory()
    example_3_scene_representation()
    example_4_scene_memory()
    example_5_full_pipeline()

    print("=" * 50)
    print("All examples completed successfully!")
    print("=" * 50)
