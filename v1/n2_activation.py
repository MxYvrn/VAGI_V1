"""
Stage 1: N² Edge Neurons (Runtime)
Divides image into 4x4 tiles and computes binary activation using trained neuron weights.
"""

import numpy as np

try:
    from .utils import CellGrid
    from .tile_neuron import TileNeuron
except ImportError:
    from utils import CellGrid
    from tile_neuron import TileNeuron


def compute_activation_grid(
    image: np.ndarray,
    weights_path: str = "n2_tile_weights.json",
    tile_size: int = 4
) -> CellGrid:
    """
    Compute N² activation grid from raw RGB image using trained neuron.

    Args:
        image: RGB image as numpy array of shape (H, W, 3), dtype uint8 or float
        weights_path: Path to JSON file containing neuron weights
        tile_size: Size of each tile (default 4x4 pixels)

    Returns:
        CellGrid with activation values set (0 or 1)
    """
    H, W = image.shape[:2]

    # Compute tile grid dimensions (truncate remainder)
    tile_H = H // tile_size
    tile_W = W // tile_size

    # Load trained neuron weights
    neuron = TileNeuron(tile_size=tile_size)
    neuron.load_weights(weights_path)

    # Initialize cell grid
    grid = CellGrid(tile_H, tile_W)

    # Ensure image is float32 in [0, 255] range
    if image.dtype == np.uint8:
        image = image.astype(np.float32)

    # Process each tile
    for i in range(tile_H):
        for j in range(tile_W):
            # Extract tile pixels
            y_start = i * tile_size
            y_end = y_start + tile_size
            x_start = j * tile_size
            x_end = x_start + tile_size

            tile = image[y_start:y_end, x_start:x_end, :]

            # Compute activation using trained neuron
            activation = neuron.predict_label(tile)

            # Initialize cell
            grid.cells[i][j].activation = activation
            grid.cells[i][j].visited = 0
            grid.cells[i][j].chain_id = -1
            grid.cells[i][j].index_in_chain = -1

    return grid


def visualize_activation(grid: CellGrid, tile_size: int = 4) -> np.ndarray:
    """
    Create a visualization of the activation grid.

    Args:
        grid: CellGrid with activation values
        tile_size: Size of each tile for visualization

    Returns:
        Image array with activation visualization (0 or 255)
    """
    H_pixels = grid.height * tile_size
    W_pixels = grid.width * tile_size
    vis = np.zeros((H_pixels, W_pixels), dtype=np.uint8)

    for i in range(grid.height):
        for j in range(grid.width):
            if grid.cells[i][j].activation == 1:
                y_start = i * tile_size
                y_end = y_start + tile_size
                x_start = j * tile_size
                x_end = x_start + tile_size
                vis[y_start:y_end, x_start:x_end] = 255

    return vis
