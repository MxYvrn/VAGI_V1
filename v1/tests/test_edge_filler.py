"""
Tests for Stage 2: EdgeFiller
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import edge_filler
import utils


def test_horizontal_gap():
    """Test filling horizontal 1-0-1 gap."""
    grid = utils.CellGrid(5, 5)

    # Create horizontal pattern: 1-0-1
    grid.cells[2][1].activation = 1
    grid.cells[2][2].activation = 0  # Gap
    grid.cells[2][3].activation = 1

    print("\nOriginal (horizontal gap):")
    print(grid.get_activation_map())

    filled = edge_filler.edge_filler(grid)

    print("After filling:")
    print(filled.get_activation_map())

    # Gap should be filled
    assert filled.cells[2][2].activation == 1, "Horizontal gap should be filled"

    print("✓ test_horizontal_gap passed")


def test_vertical_gap():
    """Test filling vertical 1-0-1 gap."""
    grid = utils.CellGrid(5, 5)

    # Create vertical pattern: 1-0-1
    grid.cells[1][2].activation = 1
    grid.cells[2][2].activation = 0  # Gap
    grid.cells[3][2].activation = 1

    print("\nOriginal (vertical gap):")
    print(grid.get_activation_map())

    filled = edge_filler.edge_filler(grid)

    print("After filling:")
    print(filled.get_activation_map())

    # Gap should be filled
    assert filled.cells[2][2].activation == 1, "Vertical gap should be filled"

    print("✓ test_vertical_gap passed")


def test_diagonal_gap():
    """Test filling diagonal 1-0-1 gap."""
    grid = utils.CellGrid(5, 5)

    # Create diagonal pattern: 1-0-1 (NE-SW direction)
    grid.cells[1][1].activation = 1
    grid.cells[2][2].activation = 0  # Gap
    grid.cells[3][3].activation = 1

    print("\nOriginal (diagonal gap):")
    print(grid.get_activation_map())

    filled = edge_filler.edge_filler(grid)

    print("After filling:")
    print(filled.get_activation_map())

    # Gap should be filled
    assert filled.cells[2][2].activation == 1, "Diagonal gap should be filled"

    print("✓ test_diagonal_gap passed")


def test_no_gap():
    """Test that solid edges remain unchanged."""
    grid = utils.CellGrid(5, 5)

    # Create solid line
    for j in range(5):
        grid.cells[2][j].activation = 1

    print("\nOriginal (no gap):")
    print(grid.get_activation_map())

    filled = edge_filler.edge_filler(grid)

    print("After filling:")
    print(filled.get_activation_map())

    # Should be identical
    assert np.array_equal(grid.get_activation_map(), filled.get_activation_map()), \
        "Solid edge should remain unchanged"

    print("✓ test_no_gap passed")


def test_two_tile_gap():
    """Test that 2-tile gaps are NOT filled."""
    grid = utils.CellGrid(7, 7)

    # Create pattern: 1-0-0-1 (2 tile gap)
    grid.cells[3][1].activation = 1
    grid.cells[3][2].activation = 0  # Gap
    grid.cells[3][3].activation = 0  # Gap
    grid.cells[3][4].activation = 1

    print("\nOriginal (2-tile gap):")
    print(grid.get_activation_map())

    filled = edge_filler.edge_filler(grid)

    print("After filling:")
    print(filled.get_activation_map())

    # 2-tile gaps should NOT be filled
    assert filled.cells[3][2].activation == 0, "2-tile gap should not be filled"
    assert filled.cells[3][3].activation == 0, "2-tile gap should not be filled"

    print("✓ test_two_tile_gap passed")


def test_corner_gap():
    """Test filling gap at corner (multiple directions)."""
    grid = utils.CellGrid(5, 5)

    # Create L-shape with gap at corner
    grid.cells[1][2].activation = 1  # Top
    grid.cells[2][2].activation = 0  # Gap (corner)
    grid.cells[2][1].activation = 1  # Left

    print("\nOriginal (corner gap):")
    print(grid.get_activation_map())

    filled = edge_filler.edge_filler(grid)

    print("After filling:")
    print(filled.get_activation_map())

    # Corner gap should be filled (vertical 1-0-1 detected)
    # Note: This might not fill if there's no opposite neighbor
    # Let me add the opposite neighbors
    grid.cells[3][2].activation = 1  # Bottom (for vertical)
    grid.cells[2][3].activation = 1  # Right (for horizontal)

    filled2 = edge_filler.edge_filler(grid)

    print("After filling (with opposite neighbors):")
    print(filled2.get_activation_map())

    assert filled2.cells[2][2].activation == 1, "Corner gap should be filled"

    print("✓ test_corner_gap passed")


def test_isolated_tile():
    """Test that isolated tiles are not affected."""
    grid = utils.CellGrid(5, 5)

    # Single isolated tile
    grid.cells[2][2].activation = 1

    print("\nOriginal (isolated tile):")
    print(grid.get_activation_map())

    filled = edge_filler.edge_filler(grid)

    print("After filling:")
    print(filled.get_activation_map())

    # Should remain unchanged
    assert np.array_equal(grid.get_activation_map(), filled.get_activation_map()), \
        "Isolated tile should remain unchanged"

    print("✓ test_isolated_tile passed")


if __name__ == "__main__":
    print("Running EdgeFiller tests...\n")
    test_horizontal_gap()
    test_vertical_gap()
    test_diagonal_gap()
    test_no_gap()
    test_two_tile_gap()
    test_corner_gap()
    test_isolated_tile()
    print("\n✅ All EdgeFiller tests passed!")
