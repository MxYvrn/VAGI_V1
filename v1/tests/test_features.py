"""
Tests for Stage 5: Shape + Color Features
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import features
import utils


def test_shape_histogram():
    """Test shape feature computation from direction codes."""
    # Create a chain with known direction codes
    # 4 straight (code 0), 2 small right (code 1), 1 small left (code 2)
    chain = utils.Chain(
        steps=[
            (0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0),  # 4 straight
            (1, 1.414), (1, 1.414),  # 2 small right
            (2, 1.414),  # 1 small left
        ],
        tiles=[(i, 0) for i in range(8)]
    )

    shape_features = features._compute_shape_features(chain)

    print("\nShape features:")
    print(f"  Histogram: {shape_features[:8]}")
    print(f"  Total right: {shape_features[8]}")
    print(f"  Total left: {shape_features[9]}")

    # Check histogram
    assert len(shape_features) == 10
    assert shape_features[0] == 4/7  # 4 out of 7 are code 0
    assert shape_features[1] == 2/7  # 2 out of 7 are code 1
    assert shape_features[2] == 1/7  # 1 out of 7 is code 2

    # Check turn totals
    assert np.isclose(shape_features[8], 2/7), "Total right should be 2/7"
    assert np.isclose(shape_features[9], 1/7), "Total left should be 1/7"

    print("✓ test_shape_histogram passed")


def test_color_extraction():
    """Test color extraction from image."""
    # Create a test image with known colors
    image = np.zeros((32, 32, 3), dtype=np.uint8)

    # Fill region with red
    image[8:24, 8:24, 0] = 255  # Red channel
    image[8:24, 8:24, 1] = 0    # Green channel
    image[8:24, 8:24, 2] = 0    # Blue channel

    # Create a chain around this region (at tile level, 4x4 tiles)
    # Tiles 2-5 in both dimensions (covering pixels 8-24)
    chain = utils.Chain(
        steps=[(0, 1.0)] * 12,  # Dummy steps
        tiles=[
            # Top edge
            (2, 2), (2, 3), (2, 4), (2, 5),
            # Right edge
            (3, 5), (4, 5), (5, 5),
            # Bottom edge
            (5, 4), (5, 3), (5, 2),
            # Left edge
            (4, 2), (3, 2),
        ]
    )

    color_features = features._compute_color_features(chain, image, tile_size=4)

    print("\nColor features:")
    print(f"  R: {color_features[0]}")
    print(f"  G: {color_features[1]}")
    print(f"  B: {color_features[2]}")

    # Should be predominantly red
    assert color_features[0] > 200, "Should have high red value"
    assert color_features[1] < 50, "Should have low green value"
    assert color_features[2] < 50, "Should have low blue value"

    print("✓ test_color_extraction passed")


def test_centroid():
    """Test centroid computation."""
    # Create a chain with known positions
    chain = utils.Chain(
        steps=[],
        tiles=[
            (2, 2), (2, 4), (4, 2), (4, 4)  # Corners of a square
        ]
    )

    centroid = features._compute_centroid(chain)

    print(f"\nCentroid: {centroid}")

    # Centroid should be at (3, 3) - center of the square
    # Remember: centroid is (x, y) where x=column, y=row
    assert np.isclose(centroid[0], 3.0), "X coordinate should be 3"
    assert np.isclose(centroid[1], 3.0), "Y coordinate should be 3"

    print("✓ test_centroid passed")


def test_scale():
    """Test scale (perimeter) computation."""
    # Create a chain with known distances
    chain = utils.Chain(
        steps=[
            (0, 1.0), (0, 1.0), (0, 1.0),  # 3 orthogonal steps
            (1, np.sqrt(2)), (1, np.sqrt(2)),  # 2 diagonal steps
        ],
        tiles=[(i, 0) for i in range(6)]
    )

    scale = features.compute_scale(chain)

    print(f"\nScale: {scale}")

    # Expected: 3*1 + 2*sqrt(2) = 3 + 2.828... ≈ 5.828
    expected = 3 + 2 * np.sqrt(2)
    assert np.isclose(scale, expected), f"Scale should be {expected}"

    print("✓ test_scale passed")


def test_v_object_dimensions():
    """Test that v_object has correct dimensions."""
    # Create test image and chain
    image = np.ones((64, 64, 3), dtype=np.uint8) * 128

    chain = utils.Chain(
        steps=[(0, 1.0), (3, 1.0), (3, 1.0), (3, 1.0)],
        tiles=[(2, 2), (2, 3), (3, 3), (3, 2)]
    )

    v_object, centroid = features.chain_to_v_object(chain, image, tile_size=4)

    print(f"\nv_object shape: {v_object.shape}")
    print(f"v_object: {v_object}")

    assert v_object.shape == (13,), "v_object should be 13-dimensional"
    assert len(centroid) == 2, "Centroid should be 2D"

    # Check that histogram values sum to 1
    hist_sum = np.sum(v_object[:8])
    assert np.isclose(hist_sum, 1.0), "Histogram should sum to 1"

    print("✓ test_v_object_dimensions passed")


def test_extract_objects():
    """Test extracting multiple objects from chains."""
    image = np.ones((64, 64, 3), dtype=np.uint8) * 100

    chains = [
        utils.Chain(
            steps=[(0, 1.0)] * 4,
            tiles=[(2, 2), (2, 3), (2, 4), (2, 5), (2, 6)]
        ),
        utils.Chain(
            steps=[(0, 1.414)] * 4,
            tiles=[(5, 5), (6, 6), (7, 7), (8, 8), (9, 9)]
        ),
    ]

    objects = features.extract_objects_from_chains(chains, image, tile_size=4)

    print(f"\nNumber of objects: {len(objects)}")

    assert len(objects) == 2, "Should extract 2 objects"

    for i, obj in enumerate(objects):
        print(f"\nObject {i}:")
        print(f"  v_object shape: {obj['v_object'].shape}")
        print(f"  centroid: {obj['centroid']}")
        print(f"  scale: {obj['scale']}")

        assert 'v_object' in obj
        assert 'centroid' in obj
        assert 'scale' in obj
        assert 'chain' in obj

    print("✓ test_extract_objects passed")


if __name__ == "__main__":
    print("Running Features tests...\n")
    test_shape_histogram()
    test_color_extraction()
    test_centroid()
    test_scale()
    test_v_object_dimensions()
    test_extract_objects()
    print("\n✅ All Features tests passed!")
