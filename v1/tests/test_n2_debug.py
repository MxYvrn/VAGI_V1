"""
Debug test for N² activation
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import n2_activation

# Create black image with white square
image = np.zeros((64, 64, 3), dtype=np.uint8)
image[20:44, 20:44, :] = 255  # 24x24 white square

# Check which tiles this affects
tile_size = 4
print("Tile boundaries:")
print(f"White square pixels: rows 20-43, cols 20-43")
print(f"In tiles: row {20//4}-{43//4}, col {20//4}-{43//4}")
print(f"That's tile row 5-10, col 5-10")

# Now compute activation and see what happens
grid = n2_activation.compute_activation_grid(image, tile_size=4, threshold=30.0)

# Let's check specific tiles
for i in range(4, 12):
    for j in range(4, 12):
        y_start = i * tile_size
        y_end = y_start + tile_size
        x_start = j * tile_size
        x_end = x_start + tile_size

        tile = image[y_start:y_end, x_start:x_end, 0]  # Just red channel
        variation = np.max(tile) - np.min(tile)
        activation = grid.cells[i][j].activation

        if variation > 0 or activation > 0:
            print(f"Tile ({i},{j}): pixels [{y_start}:{y_end}, {x_start}:{x_end}], "
                  f"min={np.min(tile)}, max={np.max(tile)}, var={variation}, act={activation}")
