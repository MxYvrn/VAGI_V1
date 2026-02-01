"""
Stage 3: Recursive EdgeRunner
Traces connected boundaries on the tile grid with branching and splicing support.
"""

from typing import List, Tuple, Optional
import copy

try:
    from .utils import (
        CellGrid, Chain, get_direction, get_distance, compute_turn_code,
        get_neighbors_8, DIRECTION_VECTORS
    )
except ImportError:
    from utils import (
        CellGrid, Chain, get_direction, get_distance, compute_turn_code,
        get_neighbors_8, DIRECTION_VECTORS
    )


def extract_chains_from_grid(cells: CellGrid) -> List[Chain]:
    """
    Extract boundary chains from activation grid using recursive EdgeRunner.

    Args:
        cells: CellGrid with activation values (after EdgeFiller)

    Returns:
        List of Chain objects representing boundaries
    """
    chains = []

    # Find all seed positions (activated, unvisited tiles)
    seeds = []
    for i in range(cells.height):
        for j in range(cells.width):
            if cells[i, j].activation == 1 and cells[i, j].visited == 0:
                seeds.append((i, j))

    # Process each connected component
    for seed_pos in seeds:
        # Skip if already visited (might have been visited by previous component)
        if cells[seed_pos].visited == 1:
            continue

        # Find initial direction from seed
        initial_dir = _find_initial_direction(seed_pos, cells)
        if initial_dir is None:
            # Isolated single tile, mark visited and skip
            cells[seed_pos].visited = 1
            continue

        # Start new chain from seed
        new_chains = _edge_runner_recursive(
            pos=seed_pos,
            direction=initial_dir,
            cells=cells,
            chain_steps=[],
            chain_tiles=[seed_pos],
            parent_chain_id=len(chains)
        )

        # Add all resulting chains
        chains.extend(new_chains)

    return chains


def _find_initial_direction(pos: Tuple[int, int], cells: CellGrid) -> Optional[int]:
    """
    Find initial direction from a seed position.
    Returns the direction to the first active neighbor found.
    """
    neighbors = get_neighbors_8(pos, cells)

    for neighbor_pos in neighbors:
        if cells[neighbor_pos].activation == 1:
            return get_direction(pos, neighbor_pos)

    return None


def _edge_runner_recursive(
    pos: Tuple[int, int],
    direction: int,
    cells: CellGrid,
    chain_steps: List[Tuple[int, float]],
    chain_tiles: List[Tuple[int, int]],
    parent_chain_id: int
) -> List[Chain]:
    """
    Recursively trace a boundary chain with branching and splicing.

    Args:
        pos: Current position (i, j)
        direction: Current direction (0-7)
        cells: CellGrid being traced
        chain_steps: Accumulated steps [(code, distance), ...]
        chain_tiles: Accumulated tile positions [(i, j), ...]
        parent_chain_id: ID to assign to main chain

    Returns:
        List of Chain objects (main chain + any branches)
    """
    # Mark current position as visited
    cells[pos].visited = 1
    cells[pos].chain_id = parent_chain_id
    cells[pos].index_in_chain = len(chain_tiles) - 1

    current_pos = pos
    current_dir = direction
    current_steps = list(chain_steps)
    current_tiles = list(chain_tiles)
    result_chains = []

    while True:
        # Get all active neighbors
        all_neighbors = get_neighbors_8(current_pos, cells)
        active_neighbors = [
            nb for nb in all_neighbors
            if cells[nb].activation == 1
        ]

        # Split into visited and unvisited
        unvisited = [nb for nb in active_neighbors if cells[nb].visited == 0]
        visited = [nb for nb in active_neighbors if cells[nb].visited == 1]

        # CASE A: Unvisited neighbors exist
        if len(unvisited) >= 1:
            if len(unvisited) == 1:
                # Single path: continue tracing
                next_pos = unvisited[0]
                new_dir = get_direction(current_pos, next_pos)
                turn_code = compute_turn_code(current_dir, new_dir)
                dist = get_distance(new_dir)

                # Append step
                current_steps.append((turn_code, dist))
                current_tiles.append(next_pos)

                # Mark as visited
                cells[next_pos].visited = 1
                cells[next_pos].chain_id = parent_chain_id
                cells[next_pos].index_in_chain = len(current_tiles) - 1

                # Update current state
                current_pos = next_pos
                current_dir = new_dir

            else:
                # Branch point: multiple unvisited neighbors
                # Choose "main" direction (minimal turn from current direction)
                main_neighbor, main_turn = _choose_main_direction(
                    current_pos, current_dir, unvisited
                )

                # Create branches for other neighbors
                for nb in unvisited:
                    if nb == main_neighbor:
                        continue

                    # Compute step to this branch
                    branch_dir = get_direction(current_pos, nb)
                    branch_turn = compute_turn_code(current_dir, branch_dir)
                    branch_dist = get_distance(branch_dir)

                    # Create new branch chain
                    branch_steps = current_steps + [(branch_turn, branch_dist)]
                    branch_tiles = current_tiles + [nb]
                    branch_chain_id = parent_chain_id + len(result_chains) + 1

                    # Mark branch start
                    cells[nb].visited = 1
                    cells[nb].chain_id = branch_chain_id
                    cells[nb].index_in_chain = len(branch_tiles) - 1

                    # Recursively trace branch
                    branch_chains = _edge_runner_recursive(
                        pos=nb,
                        direction=branch_dir,
                        cells=cells,
                        chain_steps=branch_steps,
                        chain_tiles=branch_tiles,
                        parent_chain_id=branch_chain_id
                    )

                    result_chains.extend(branch_chains)

                # Continue with main direction
                new_dir = get_direction(current_pos, main_neighbor)
                turn_code = compute_turn_code(current_dir, new_dir)
                dist = get_distance(new_dir)

                current_steps.append((turn_code, dist))
                current_tiles.append(main_neighbor)

                cells[main_neighbor].visited = 1
                cells[main_neighbor].chain_id = parent_chain_id
                cells[main_neighbor].index_in_chain = len(current_tiles) - 1

                current_pos = main_neighbor
                current_dir = new_dir

        # CASE B: No unvisited neighbors, but visited neighbors exist (splicing)
        elif len(visited) > 0:
            # Choose a visited neighbor to splice into
            splice_neighbor = _choose_splice_target(current_pos, current_dir, visited)

            # Compute step to splice point
            splice_dir = get_direction(current_pos, splice_neighbor)
            splice_turn = compute_turn_code(current_dir, splice_dir)
            splice_dist = get_distance(splice_dir)

            current_steps.append((splice_turn, splice_dist))
            current_tiles.append(splice_neighbor)

            # CRITICAL: Check if we've returned to the original seed (loop detection)
            # A loop is when the last tile equals the first tile
            is_loop = (len(current_tiles) >= 3 and
                      current_tiles[-1] == current_tiles[0])

            # Get the suffix of the chain we're splicing into
            # Note: We don't actually splice in this implementation - we just mark
            # that this chain ended at a visited tile
            # The chain filtering stage will use this information

            # Create final chain
            chain = Chain(
                steps=current_steps,
                tiles=current_tiles,
                chain_id=parent_chain_id,
                spliced=(not is_loop)  # If it's a loop, don't mark as spliced
            )
            result_chains.insert(0, chain)  # Main chain goes first

            return result_chains

        # CASE C: Dead end (no neighbors at all, or only current position)
        else:
            # Create final chain
            chain = Chain(
                steps=current_steps,
                tiles=current_tiles,
                chain_id=parent_chain_id,
                spliced=False
            )
            result_chains.insert(0, chain)  # Main chain goes first

            return result_chains


def _choose_main_direction(
    pos: Tuple[int, int],
    current_dir: int,
    candidates: List[Tuple[int, int]]
) -> Tuple[Tuple[int, int], int]:
    """
    Choose the "main" direction from multiple candidates.
    Selects the neighbor with minimal turn from current direction.

    Returns:
        (chosen_neighbor, turn_angle)
    """
    min_turn = float('inf')
    best_neighbor = candidates[0]

    for nb in candidates:
        new_dir = get_direction(pos, nb)
        turn_code = compute_turn_code(current_dir, new_dir)

        # Map turn codes to absolute angles for comparison
        # 0=0°, 1=45°, 2=-45°, 3=90°, 4=-90°, 5=135°, 6=-135°, 7=180°
        turn_angles = {0: 0, 1: 45, 2: 45, 3: 90, 4: 90, 5: 135, 6: 135, 7: 180}
        abs_turn = turn_angles.get(turn_code, 0)

        if abs_turn < min_turn:
            min_turn = abs_turn
            best_neighbor = nb

    return best_neighbor, min_turn


def _choose_splice_target(
    pos: Tuple[int, int],
    current_dir: int,
    candidates: List[Tuple[int, int]]
) -> Tuple[int, int]:
    """
    Choose which visited neighbor to splice into.
    For now, just pick the one with minimal turn (same as main direction).
    """
    neighbor, _ = _choose_main_direction(pos, current_dir, candidates)
    return neighbor
