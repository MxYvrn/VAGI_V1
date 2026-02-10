"""
Stage 2: EdgeFiller
Fills 1-tile gaps in the pattern 1-0-1 across 8 directions.
"""

import numpy as np

try:
    from .utils import CellGrid
except ImportError:
    from utils import CellGrid


def edge_filler(cells: CellGrid) -> CellGrid:
    """
    Fill 1-tile gaps in activation pattern.

    For each tile (i, j) with activation == 0:
    - For each of 8 directions (dx, dy):
        - Check positions (i - dx, j - dy) and (i + dx, j + dy)
        - If both exist in bounds and both have activation == 1 in ORIGINAL map:
            → set activation = 1 in NEW map

    Args:
        cells: Input CellGrid with initial activations

    Returns:
        New CellGrid with gaps filled
    """
    # Create new grid for output (don't modify in-place)
    filled = CellGrid(cells.height, cells.width)

    # Get original activation map
    original_activation = cells.get_activation_map()

    # 8 directions: N, NE, E, SE, S, SW, W, NW
    directions = [
        (-1, 0),   # N
        (-1, 1),   # NE
        (0, 1),    # E
        (1, 1),    # SE
        (1, 0),    # S
        (1, -1),   # SW
        (0, -1),   # W
        (-1, -1),  # NW
    ]

    # Process each cell
    for i in range(cells.height):
        for j in range(cells.width):
            # Copy original activation
            filled.cells[i][j].activation = original_activation[i, j]

            # If already activated, skip
            if original_activation[i, j] == 1:
                continue

            # Check each direction for 1-0-1 pattern
            should_activate = False
            for dx, dy in directions:
                # Positions on either side
                i_minus = i - dx
                j_minus = j - dy
                i_plus = i + dx
                j_plus = j + dy

                # Check if both are in bounds
                if not cells.in_bounds(i_minus, j_minus):
                    continue
                if not cells.in_bounds(i_plus, j_plus):
                    continue

                # Check if both neighbors are activated
                if (original_activation[i_minus, j_minus] == 1 and
                    original_activation[i_plus, j_plus] == 1):
                    should_activate = True
                    break

            if should_activate:
                filled.cells[i][j].activation = 1

    return filled