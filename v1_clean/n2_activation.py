"""
N² Edge Neurons

Divides image into 4x4 tiles and computes binary activation based on local pixel variation.
"""

import numpy as np
from core import CellGrid


def compute_activation_grid(
    image: np.ndarray,
    tile_size: int = 4,
    threshold: float = 30.0
) -> CellGrid:
    """
    Compute N² activation grid from raw RGB image.

    Args:
        image: RGB image as numpy array of shape (H, W, 3)
        tile_size: Size of each tile (default 4x4 pixels)
        threshold: Activation threshold for max-min pixel variation

    Returns:
        CellGrid with activation values set (0 or 1)
    """
    H, W = image.shape[:2]

    tile_H = H // tile_size
    tile_W = W // tile_size

    grid = CellGrid(tile_H, tile_W)

    for i in range(tile_H):
        for j in range(tile_W):
            y_start = i * tile_size
            y_end = y_start + tile_size
            x_start = j * tile_size
            x_end = x_start + tile_size

            tile = image[y_start:y_end, x_start:x_end]
            activation = _compute_tile_activation(tile, threshold)
            grid.cells[i][j].activation = activation

    return grid


def _compute_tile_activation(tile: np.ndarray, threshold: float) -> int:
    """
    Compute activation for a single tile based on pixel variation.

    Args:
        tile: Tile pixels as numpy array of shape (tile_size, tile_size, channels)
        threshold: Activation threshold

    Returns:
        1 if activated, 0 otherwise
    """
    if len(tile.shape) == 3 and tile.shape[2] == 3:
        gray = 0.299 * tile[:, :, 0] + 0.587 * tile[:, :, 1] + 0.114 * tile[:, :, 2]
    else:
        gray = tile.squeeze()

    variation = np.max(gray) - np.min(gray)

    return 1 if variation > threshold else 0
