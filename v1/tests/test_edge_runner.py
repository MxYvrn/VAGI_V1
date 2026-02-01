"""
Tests for Stage 3: Recursive EdgeRunner
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import edge_runner
import utils


def print_chain_info(chain, label="Chain"):
    """Helper to print chain information."""
    print(f"\n{label}:")
    print(f"  Length: {len(chain)} steps")
    print(f"  Tiles: {len(chain.tiles)} positions")
    print(f"  Start: {chain.start_pos}, End: {chain.end_pos}")
    print(f"  Is loop: {chain.is_loop()}")
    print(f"  Spliced: {chain.spliced}")
    print(f"  Steps: {chain.steps[:10]}..." if len(chain.steps) > 10 else f"  Steps: {chain.steps}")


def test_simple_horizontal_line():
    """Test tracing a simple horizontal line."""
    grid = utils.CellGrid(5, 7)

    # Create horizontal line
    for j in range(1, 6):
        grid.cells[2][j].activation = 1

    print("\nGrid:")
    print(grid.get_activation_map())

    chains = edge_runner.extract_chains_from_grid(grid)

    print(f"Number of chains: {len(chains)}")
    for i, chain in enumerate(chains):
        print_chain_info(chain, f"Chain {i}")

    assert len(chains) >= 1, "Should find at least one chain"
    assert len(chains[0]) >= 4, "Chain should have multiple steps"

    print("✓ test_simple_horizontal_line passed")


def test_simple_loop():
    """Test tracing a simple closed loop (square)."""
    grid = utils.CellGrid(10, 10)

    # Create a square loop
    # Top edge
    for j in range(3, 7):
        grid.cells[3][j].activation = 1

    # Bottom edge
    for j in range(3, 7):
        grid.cells[6][j].activation = 1

    # Left edge
    for i in range(3, 7):
        grid.cells[i][3].activation = 1

    # Right edge
    for i in range(3, 7):
        grid.cells[i][6].activation = 1

    print("\nGrid (square loop):")
    print(grid.get_activation_map())

    chains = edge_runner.extract_chains_from_grid(grid)

    print(f"Number of chains: {len(chains)}")
    for i, chain in enumerate(chains):
        print_chain_info(chain, f"Chain {i}")

    assert len(chains) >= 1, "Should find at least one chain"

    # Check if any chain is a loop
    has_loop = any(c.is_loop() for c in chains)
    print(f"Has loop: {has_loop}")

    print("✓ test_simple_loop passed")


def test_branching_y_shape():
    """Test branching at a Y-junction."""
    grid = utils.CellGrid(10, 10)

    # Create Y shape
    # Stem (vertical)
    for i in range(7, 10):
        grid.cells[i][5].activation = 1

    # Junction
    grid.cells[6][5].activation = 1

    # Left branch
    for step in range(4):
        grid.cells[6 - step][5 - step].activation = 1

    # Right branch
    for step in range(4):
        grid.cells[6 - step][5 + step].activation = 1

    print("\nGrid (Y-shape with branching):")
    print(grid.get_activation_map())

    chains = edge_runner.extract_chains_from_grid(grid)

    print(f"Number of chains: {len(chains)}")
    for i, chain in enumerate(chains):
        print_chain_info(chain, f"Chain {i}")

    # Should have multiple chains due to branching
    assert len(chains) >= 2, "Should find multiple chains due to branching"

    print("✓ test_branching_y_shape passed")


def test_cross_shape():
    """Test a cross shape (multiple branches from center)."""
    grid = utils.CellGrid(11, 11)

    # Center point
    center = (5, 5)
    grid.cells[center[0]][center[1]].activation = 1

    # Four arms extending from center
    # North
    for i in range(2, 5):
        grid.cells[i][5].activation = 1

    # South
    for i in range(6, 9):
        grid.cells[i][5].activation = 1

    # West
    for j in range(2, 5):
        grid.cells[5][j].activation = 1

    # East
    for j in range(6, 9):
        grid.cells[5][j].activation = 1

    print("\nGrid (cross shape):")
    print(grid.get_activation_map())

    chains = edge_runner.extract_chains_from_grid(grid)

    print(f"Number of chains: {len(chains)}")
    for i, chain in enumerate(chains):
        print_chain_info(chain, f"Chain {i}")

    # Should have multiple chains (one main + branches)
    assert len(chains) >= 3, "Should find multiple chains from cross junction"

    print("✓ test_cross_shape passed")


def test_isolated_tile():
    """Test single isolated activated tile."""
    grid = utils.CellGrid(5, 5)

    # Single tile
    grid.cells[2][2].activation = 1

    print("\nGrid (isolated tile):")
    print(grid.get_activation_map())

    chains = edge_runner.extract_chains_from_grid(grid)

    print(f"Number of chains: {len(chains)}")

    # Isolated tile should be skipped or produce empty chain
    # Based on current implementation, it should be skipped
    assert len(chains) == 0, "Isolated tile should not produce chain"

    print("✓ test_isolated_tile passed")


def test_two_separate_lines():
    """Test two disconnected line segments."""
    grid = utils.CellGrid(8, 8)

    # First line
    for j in range(1, 4):
        grid.cells[2][j].activation = 1

    # Second line (separate)
    for j in range(5, 8):
        grid.cells[5][j].activation = 1

    print("\nGrid (two separate lines):")
    print(grid.get_activation_map())

    chains = edge_runner.extract_chains_from_grid(grid)

    print(f"Number of chains: {len(chains)}")
    for i, chain in enumerate(chains):
        print_chain_info(chain, f"Chain {i}")

    # Should find two separate chains
    assert len(chains) >= 2, "Should find two separate chains"

    print("✓ test_two_separate_lines passed")


def test_diagonal_line():
    """Test diagonal line tracing."""
    grid = utils.CellGrid(8, 8)

    # Diagonal line
    for i in range(1, 6):
        grid.cells[i][i].activation = 1

    print("\nGrid (diagonal line):")
    print(grid.get_activation_map())

    chains = edge_runner.extract_chains_from_grid(grid)

    print(f"Number of chains: {len(chains)}")
    for i, chain in enumerate(chains):
        print_chain_info(chain, f"Chain {i}")

    assert len(chains) >= 1, "Should find diagonal chain"

    print("✓ test_diagonal_line passed")


if __name__ == "__main__":
    print("Running EdgeRunner tests...\n")
    test_simple_horizontal_line()
    test_simple_loop()
    test_branching_y_shape()
    test_cross_shape()
    test_isolated_tile()
    test_two_separate_lines()
    test_diagonal_line()
    print("\n✅ All EdgeRunner tests passed!")
