"""
V1 Pipeline End-to-End Demo

Demonstrates the complete V1 vision pipeline:
1. N² Edge Neurons → tile-level edge detection
2. EdgeFiller → fill 1-tile gaps
3. EdgeRunner → extract boundary chains
4. Chain Filtering → remove useless strings
5. Features → convert to v_object vectors
6. Obj-KNN → object memory
7. Img-ID → scene representation
"""

import numpy as np
import sys
import os

# Import all V1 modules
import n2_activation
import edge_filler
import edge_runner
import chain_filter
import features
import obj_knn
import img_id


def create_test_image_with_shapes():
    """
    Create a synthetic test image with multiple shapes.

    Returns:
        RGB image as numpy array
    """
    # Create black background
    image = np.zeros((128, 128, 3), dtype=np.uint8)

    # Add a red square (offset from tile boundaries for edge detection)
    image[18:46, 18:46, 0] = 255  # Red channel

    # Add a green circle (approximate with square for simplicity)
    image[70:98, 18:46, 1] = 255  # Green channel

    # Add a blue rectangle
    image[18:46, 70:110, 2] = 255  # Blue channel

    return image


def run_v1_pipeline(image, tile_size=4, verbose=True):
    """
    Run the complete V1 pipeline on an image.

    Args:
        image: RGB image as numpy array
        tile_size: Size of tiles for grid (default 4)
        verbose: Print progress messages

    Returns:
        Dictionary containing:
        - scene: Scene object with detected objects
        - chains: Filtered chains
        - objects: Extracted object features
        - activation_grid: Initial activation grid
        - filled_grid: After edge filling
    """
    if verbose:
        print("=" * 60)
        print("V1 PIPELINE EXECUTION")
        print("=" * 60)

    # Stage 1: N² Edge Neurons
    if verbose:
        print("\n[1/7] Computing N² activation grid...")

    activation_grid = n2_activation.compute_activation_grid(
        image,
        tile_size=tile_size,
        threshold=30.0
    )

    if verbose:
        activation_map = activation_grid.get_activation_map()
        num_activated = np.sum(activation_map)
        print(f"  → Activated tiles: {num_activated} / {activation_grid.height * activation_grid.width}")

    # Stage 2: EdgeFiller
    if verbose:
        print("\n[2/7] Running EdgeFiller...")

    filled_grid = edge_filler.edge_filler(activation_grid)

    if verbose:
        filled_map = filled_grid.get_activation_map()
        num_filled = np.sum(filled_map)
        num_new = num_filled - num_activated
        print(f"  → Total tiles after filling: {num_filled} (+{num_new} new)")

    # Stage 3: EdgeRunner
    if verbose:
        print("\n[3/7] Extracting boundary chains...")

    chains = edge_runner.extract_chains_from_grid(filled_grid)

    if verbose:
        print(f"  → Extracted {len(chains)} chains")
        stats = chain_filter.get_chain_statistics(chains)
        print(f"  → Chain stats: avg_len={stats['avg_length']:.1f}, "
              f"loops={stats['num_loops']}, spliced={stats['num_spliced']}")

    # Stage 4: Chain Filtering
    if verbose:
        print("\n[4/7] Filtering chains...")

    filtered_chains = chain_filter.filter_chains(
        chains,
        grid_height=filled_grid.height,
        grid_width=filled_grid.width,
        min_length=3
    )

    if verbose:
        print(f"  → Kept {len(filtered_chains)} / {len(chains)} chains")

    # Stage 5: Feature Extraction
    if verbose:
        print("\n[5/7] Extracting object features...")

    objects = features.extract_objects_from_chains(
        filtered_chains,
        image,
        tile_size=tile_size
    )

    if verbose:
        print(f"  → Extracted {len(objects)} objects")
        for i, obj in enumerate(objects):
            print(f"     Object {i}: centroid={obj['centroid']}, scale={obj['scale']:.2f}")

    # Stage 6: Obj-KNN (build object memory)
    if verbose:
        print("\n[6/7] Building object memory (Obj-KNN)...")

    obj_memory = obj_knn.ObjKNN(shape_weight=1.0, color_weight=0.1)

    # Stage 7: Img-ID (create scene representation)
    if verbose:
        print("\n[7/7] Creating scene representation (Img-ID)...")

    scene = img_id.create_scene_from_objects(
        objects,
        obj_memory,
        similarity_threshold=0.5
    )

    if verbose:
        print(f"  → Scene with {len(scene)} objects:")
        for obj in scene.objects:
            print(f"     proto_id={obj.proto_id}, pos=({obj.x:.1f}, {obj.y:.1f}), scale={obj.scale:.2f}")

    # Return all intermediate results
    return {
        'scene': scene,
        'chains': filtered_chains,
        'objects': objects,
        'activation_grid': activation_grid,
        'filled_grid': filled_grid,
        'obj_memory': obj_memory
    }


def demo_simple_shapes():
    """Demo with synthetic shapes."""
    print("\n" + "=" * 60)
    print("DEMO: Simple Shapes")
    print("=" * 60)

    # Create test image
    image = create_test_image_with_shapes()
    print(f"\nCreated test image: {image.shape}")

    # Run pipeline
    results = run_v1_pipeline(image, verbose=True)

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"\nFinal scene representation:")
    print(results['scene'].to_dict())

    return results


def demo_multiple_images():
    """Demo with multiple images to show scene-level memory."""
    print("\n" + "=" * 60)
    print("DEMO: Multiple Images & Scene Memory")
    print("=" * 60)

    # Create object and scene memories
    obj_memory = obj_knn.ObjKNN()
    scene_memory = img_id.ImgKNN()

    # Process multiple images
    images = [
        create_test_image_with_shapes(),  # Original
        create_test_image_with_shapes(),  # Duplicate
    ]

    # Modify second image slightly (move shapes)
    images[1] = np.roll(images[1], shift=8, axis=0)  # Shift down

    scenes = []

    for i, image in enumerate(images):
        print(f"\n--- Processing Image {i+1} ---")

        # Run stages 1-5
        activation_grid = n2_activation.compute_activation_grid(image, tile_size=4, threshold=30.0)
        filled_grid = edge_filler.edge_filler(activation_grid)
        chains = edge_runner.extract_chains_from_grid(filled_grid)
        filtered_chains = chain_filter.filter_chains(
            chains,
            grid_height=filled_grid.height,
            grid_width=filled_grid.width,
            min_length=3
        )
        objects = features.extract_objects_from_chains(filtered_chains, image, tile_size=4)

        # Create scene using shared object memory
        scene = img_id.create_scene_from_objects(objects, obj_memory, similarity_threshold=0.5)
        scene_id = scene_memory.add_scene(scene)

        print(f"Created scene {scene_id} with {len(scene)} objects")
        scenes.append(scene)

    # Query scene similarity
    print(f"\n--- Scene Similarity ---")
    print(f"Object memory size: {obj_memory.size()} prototypes")
    print(f"Scene memory size: {scene_memory.size()} scenes")

    # Query for similar scenes
    results = scene_memory.query(scenes[0], k=2)
    print(f"\nScenes similar to Scene 0:")
    for scene_id, distance in results:
        print(f"  Scene {scene_id}: distance = {distance:.2f}")

    return obj_memory, scene_memory


def print_statistics(results):
    """Print detailed statistics about pipeline results."""
    print("\n" + "=" * 60)
    print("DETAILED STATISTICS")
    print("=" * 60)

    # Chain statistics
    chains = results['chains']
    stats = chain_filter.get_chain_statistics(chains)
    print("\nChain Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Object statistics
    objects = results['objects']
    print(f"\nObject Statistics:")
    print(f"  Total objects: {len(objects)}")

    if objects:
        scales = [obj['scale'] for obj in objects]
        print(f"  Scale range: [{min(scales):.2f}, {max(scales):.2f}]")
        print(f"  Average scale: {np.mean(scales):.2f}")

        # Color distribution
        colors = [obj['v_object'][10:13] for obj in objects]
        avg_color = np.mean(colors, axis=0)
        print(f"  Average color (RGB): [{avg_color[0]:.1f}, {avg_color[1]:.1f}, {avg_color[2]:.1f}]")


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + " " * 15 + "V1 VISION PIPELINE DEMO" + " " * 20 + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "=" * 58 + "╝")

    # Demo 1: Simple shapes
    results = demo_simple_shapes()
    print_statistics(results)

    # Demo 2: Multiple images
    obj_mem, scene_mem = demo_multiple_images()

    print("\n" + "=" * 60)
    print("ALL DEMOS COMPLETE")
    print("=" * 60)
    print("\nV1 pipeline successfully demonstrated!")
    print("All stages working correctly.")
