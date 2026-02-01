"""
Recursive EdgeRunner

Traces connected boundaries on the tile grid with branching and splicing support.
"""

from typing import List, Tuple, Optional
from core import (
    CellGrid, Chain, get_direction, get_distance, compute_turn_code,
    get_neighbors_8
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

    seeds = []
    for i in range(cells.height):
        for j in range(cells.width):
            if cells[i, j].activation == 1 and cells[i, j].visited == 0:
                seeds.append((i, j))

    for seed_pos in seeds:
        if cells[seed_pos].visited == 1:
            continue

        initial_dir = _find_initial_direction(seed_pos, cells)
        if initial_dir is None:
            cells[seed_pos].visited = 1
            continue

        new_chains = _edge_runner_recursive(
            pos=seed_pos,
            direction=initial_dir,
            cells=cells,
            chain_steps=[],
            chain_tiles=[seed_pos],
            parent_chain_id=len(chains)
        )

        chains.extend(new_chains)

    return chains


def _find_initial_direction(pos: Tuple[int, int], cells: CellGrid) -> Optional[int]:
    """Find initial direction from a seed position."""
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
    cells[pos].visited = 1
    cells[pos].chain_id = parent_chain_id
    cells[pos].index_in_chain = len(chain_tiles) - 1

    current_pos = pos
    current_dir = direction
    current_steps = list(chain_steps)
    current_tiles = list(chain_tiles)
    result_chains = []

    while True:
        all_neighbors = get_neighbors_8(current_pos, cells)
        active_neighbors = [
            nb for nb in all_neighbors
            if cells[nb].activation == 1
        ]

        unvisited = [nb for nb in active_neighbors if cells[nb].visited == 0]
        visited = [nb for nb in active_neighbors if cells[nb].visited == 1]

        # CASE A: Unvisited neighbors exist
        if len(unvisited) >= 1:
            if len(unvisited) == 1:
                next_pos = unvisited[0]
                new_dir = get_direction(current_pos, next_pos)
                turn_code = compute_turn_code(current_dir, new_dir)
                dist = get_distance(new_dir)

                current_steps.append((turn_code, dist))
                current_tiles.append(next_pos)

                cells[next_pos].visited = 1
                cells[next_pos].chain_id = parent_chain_id
                cells[next_pos].index_in_chain = len(current_tiles) - 1

                current_pos = next_pos
                current_dir = new_dir

            else:
                # Branch point
                main_neighbor, main_turn = _choose_main_direction(
                    current_pos, current_dir, unvisited
                )

                for nb in unvisited:
                    if nb == main_neighbor:
                        continue

                    branch_dir = get_direction(current_pos, nb)
                    branch_turn = compute_turn_code(current_dir, branch_dir)
                    branch_dist = get_distance(branch_dir)

                    branch_steps = current_steps + [(branch_turn, branch_dist)]
                    branch_tiles = current_tiles + [nb]
                    branch_chain_id = parent_chain_id + len(result_chains) + 1

                    cells[nb].visited = 1
                    cells[nb].chain_id = branch_chain_id
                    cells[nb].index_in_chain = len(branch_tiles) - 1

                    branch_chains = _edge_runner_recursive(
                        pos=nb,
                        direction=branch_dir,
                        cells=cells,
                        chain_steps=branch_steps,
                        chain_tiles=branch_tiles,
                        parent_chain_id=branch_chain_id
                    )

                    result_chains.extend(branch_chains)

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
            splice_neighbor = _choose_splice_target(current_pos, current_dir, visited)

            splice_dir = get_direction(current_pos, splice_neighbor)
            splice_turn = compute_turn_code(current_dir, splice_dir)
            splice_dist = get_distance(splice_dir)

            current_steps.append((splice_turn, splice_dist))
            current_tiles.append(splice_neighbor)

            # Check if we've returned to the original seed (loop detection)
            is_loop = (len(current_tiles) >= 3 and
                      current_tiles[-1] == current_tiles[0])

            chain = Chain(
                steps=current_steps,
                tiles=current_tiles,
                chain_id=parent_chain_id,
                spliced=(not is_loop)
            )
            result_chains.insert(0, chain)

            return result_chains

        # CASE C: Dead end
        else:
            chain = Chain(
                steps=current_steps,
                tiles=current_tiles,
                chain_id=parent_chain_id,
                spliced=False
            )
            result_chains.insert(0, chain)

            return result_chains


def _choose_main_direction(
    pos: Tuple[int, int],
    current_dir: int,
    candidates: List[Tuple[int, int]]
) -> Tuple[Tuple[int, int], int]:
    """Choose the main direction from multiple candidates (minimal turn)."""
    min_turn = float('inf')
    best_neighbor = candidates[0]

    for nb in candidates:
        new_dir = get_direction(pos, nb)
        turn_code = compute_turn_code(current_dir, new_dir)

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
    """Choose which visited neighbor to splice into."""
    neighbor, _ = _choose_main_direction(pos, current_dir, candidates)
    return neighbor
