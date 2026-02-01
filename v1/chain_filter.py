"""
Stage 4: Chain Filtering
Removes useless "string" chains and keeps only meaningful boundaries.
"""

from typing import List

try:
    from .utils import Chain
except ImportError:
    from utils import Chain


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
        # Skip chains that are too short
        if len(chain) < min_length:
            continue

        # Keep if it's a loop
        if chain.is_loop():
            filtered.append(chain)
            continue

        # Keep if it was created via splicing
        if chain.spliced:
            filtered.append(chain)
            continue

        # Keep if start or end touches border
        if _touches_border(chain.start_pos, grid_height, grid_width):
            filtered.append(chain)
            continue

        if _touches_border(chain.end_pos, grid_height, grid_width):
            filtered.append(chain)
            continue

        # Otherwise drop this chain (floating string)

    return filtered


def _touches_border(
    pos: tuple,
    grid_height: int,
    grid_width: int
) -> bool:
    """
    Check if a position touches the grid border.

    Args:
        pos: (i, j) tile position
        grid_height: Height of tile grid
        grid_width: Width of tile grid

    Returns:
        True if position is on border
    """
    if pos is None:
        return False

    i, j = pos

    # Check if on any edge
    if i == 0 or i == grid_height - 1:
        return True
    if j == 0 or j == grid_width - 1:
        return True

    return False


def deduplicate_chains(chains: List[Chain]) -> List[Chain]:
    """
    Remove duplicate chains.

    Since branching can create overlapping chains, we might want to
    deduplicate or select the best representatives.

    For V1, we can use a simple heuristic:
    - Keep the longest chain among chains with the same start position
    - Or keep all chains for now and let downstream processing handle it

    Args:
        chains: List of chains to deduplicate

    Returns:
        Deduplicated list
    """
    # For V1, simple approach: keep all chains
    # Later versions can implement smarter deduplication
    return chains


def get_chain_statistics(chains: List[Chain]) -> dict:
    """
    Compute statistics about a list of chains.

    Args:
        chains: List of chains

    Returns:
        Dictionary with statistics
    """
    if not chains:
        return {
            'num_chains': 0,
            'num_loops': 0,
            'num_spliced': 0,
            'total_length': 0,
            'avg_length': 0,
            'max_length': 0,
            'min_length': 0
        }

    lengths = [len(c) for c in chains]

    return {
        'num_chains': len(chains),
        'num_loops': sum(1 for c in chains if c.is_loop()),
        'num_spliced': sum(1 for c in chains if c.spliced),
        'total_length': sum(lengths),
        'avg_length': sum(lengths) / len(chains) if chains else 0,
        'max_length': max(lengths) if lengths else 0,
        'min_length': min(lengths) if lengths else 0
    }
