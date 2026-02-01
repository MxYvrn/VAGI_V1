"""
Chain Filtering

Removes useless "string" chains and keeps only meaningful boundaries.
"""

from typing import List
from core import Chain


def filter_chains(
    chains: List[Chain],
    grid_height: int,
    grid_width: int,
    min_length: int = 3
) -> List[Chain]:
    """
    Filter chains to keep only meaningful boundaries.

    A chain is KEPT if ANY of:
    1. It forms a loop (start == end)
    2. It touches the border (start or end on edge of grid)
    3. It ended via splicing (connected to existing chain)

    A chain is DROPPED if ALL of:
    - Open (start != end)
    - Both start and end are interior (not on border)
    - Did not splice
    - Length < min_length (noise)

    Args:
        chains: List of Chain objects to filter
        grid_height: Height of the tile grid
        grid_width: Width of the tile grid
        min_length: Minimum chain length to keep

    Returns:
        Filtered list of chains
    """
    filtered = []

    for chain in chains:
        if len(chain) < min_length:
            continue

        if chain.is_loop():
            filtered.append(chain)
            continue

        if chain.spliced:
            filtered.append(chain)
            continue

        if _touches_border(chain.start_pos, grid_height, grid_width):
            filtered.append(chain)
            continue

        if _touches_border(chain.end_pos, grid_height, grid_width):
            filtered.append(chain)
            continue

    return filtered


def _touches_border(
    pos: tuple,
    grid_height: int,
    grid_width: int
) -> bool:
    """Check if a position touches the grid border."""
    if pos is None:
        return False

    i, j = pos

    if i == 0 or i == grid_height - 1:
        return True
    if j == 0 or j == grid_width - 1:
        return True

    return False
