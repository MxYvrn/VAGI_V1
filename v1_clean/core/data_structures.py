"""
V1 Core Data Structures

Tile grid representation and chain data structures.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class Cell:
    """Single tile in the grid."""
    activation: int = 0
    visited: int = 0
    chain_id: int = -1
    index_in_chain: int = -1


class CellGrid:
    """Grid of tiles with convenient access."""

    def __init__(self, height: int, width: int):
        self.height = height
        self.width = width
        self.cells = [[Cell() for _ in range(width)] for _ in range(height)]

    def __getitem__(self, pos: Tuple[int, int]) -> Cell:
        i, j = pos
        return self.cells[i][j]

    def in_bounds(self, i: int, j: int) -> bool:
        return 0 <= i < self.height and 0 <= j < self.width

    def get_activation_map(self) -> np.ndarray:
        return np.array([[self.cells[i][j].activation
                         for j in range(self.width)]
                        for i in range(self.height)])

    def set_activation_map(self, activation_map: np.ndarray):
        for i in range(self.height):
            for j in range(self.width):
                self.cells[i][j].activation = int(activation_map[i, j])


@dataclass
class Chain:
    """Boundary chain with direction codes and tile positions."""
    steps: List[Tuple[int, float]] = field(default_factory=list)
    tiles: List[Tuple[int, int]] = field(default_factory=list)
    chain_id: int = -1
    spliced: bool = False

    def __len__(self):
        return len(self.tiles)

    def num_steps(self) -> int:
        return len(self.steps)

    def num_tiles(self) -> int:
        return len(self.tiles)

    @property
    def start_pos(self) -> Optional[Tuple[int, int]]:
        return self.tiles[0] if self.tiles else None

    @property
    def end_pos(self) -> Optional[Tuple[int, int]]:
        return self.tiles[-1] if self.tiles else None

    def is_loop(self) -> bool:
        return len(self.tiles) >= 2 and self.start_pos == self.end_pos


# Direction encoding: 8 compass directions
# N=0, NE=1, E=2, SE=3, S=4, SW=5, W=6, NW=7
DIRECTION_VECTORS = [
    (-1, 0),   # 0: N
    (-1, 1),   # 1: NE
    (0, 1),    # 2: E
    (1, 1),    # 3: SE
    (1, 0),    # 4: S
    (1, -1),   # 5: SW
    (0, -1),   # 6: W
    (-1, -1),  # 7: NW
]

ORTHOGONAL_DIRS = {0, 2, 4, 6}
DIAGONAL_DIRS = {1, 3, 5, 7}


def get_direction(from_pos: Tuple[int, int], to_pos: Tuple[int, int]) -> int:
    """Get absolute direction code (0-7) from from_pos to to_pos."""
    di = to_pos[0] - from_pos[0]
    dj = to_pos[1] - from_pos[1]

    for dir_code, (ddi, ddj) in enumerate(DIRECTION_VECTORS):
        if di == ddi and dj == ddj:
            return dir_code

    raise ValueError(f"Positions {from_pos} and {to_pos} are not 8-connected neighbors")


def get_distance(dir_code: int) -> float:
    """Get distance for a direction code."""
    if dir_code in ORTHOGONAL_DIRS:
        return 1.0
    elif dir_code in DIAGONAL_DIRS:
        return np.sqrt(2)
    else:
        raise ValueError(f"Invalid direction code: {dir_code}")


def compute_turn_code(prev_dir: int, new_dir: int) -> int:
    """
    Compute relative turn code (0-7) based on change in direction.

    0: straight (0°)
    1: small right (+45°)
    2: small left (-45°)
    3: medium right (+90°)
    4: medium left (-90°)
    5: big right (+135°)
    6: big left (-135°)
    7: U-turn (±180°)
    """
    diff = (new_dir - prev_dir) % 8

    turn_map = {
        0: 0,  # straight
        1: 1,  # small right (+45°)
        2: 3,  # medium right (+90°)
        3: 5,  # big right (+135°)
        4: 7,  # U-turn (180°)
        5: 6,  # big left (-135°)
        6: 4,  # medium left (-90°)
        7: 2,  # small left (-45°)
    }

    return turn_map[diff]


def get_neighbors_8(pos: Tuple[int, int], grid: CellGrid) -> List[Tuple[int, int]]:
    """Get all 8-connected neighbors that are in bounds."""
    i, j = pos
    neighbors = []

    for di, dj in DIRECTION_VECTORS:
        ni, nj = i + di, j + dj
        if grid.in_bounds(ni, nj):
            neighbors.append((ni, nj))

    return neighbors
