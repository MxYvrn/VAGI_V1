"""
Shape + Color Features

Converts boundary chains into v_object vectors (shape histograms + color).
"""

import numpy as np
from typing import Tuple, List
from core import Chain


def chain_to_v_object(
    chain: Chain,
    image: np.ndarray,
    tile_size: int = 4
) -> Tuple[np.ndarray, Tuple[float, float]]:
    """
    Convert a chain to a v_object feature vector.

    v_object = [
        hist_0, hist_1, ..., hist_7,  # 8 direction histogram values
        total_right_turn,              # sum of right turn histogram bins
        total_left_turn,               # sum of left turn histogram bins
        R_mean, G_mean, B_mean         # average RGB color
    ]

    Args:
        chain: Chain object with steps and tiles
        image: Original RGB image
        tile_size: Tile size used for grid

    Returns:
        (v_object, centroid)
        - v_object: 13-dimensional feature vector
        - centroid: (x, y) in tile coordinates
    """
    shape_features = _compute_shape_features(chain)
    color_features = _compute_color_features(chain, image, tile_size)
    v_object = np.concatenate([shape_features, color_features])
    centroid = _compute_centroid(chain)

    return v_object, centroid


def _compute_shape_features(chain: Chain) -> np.ndarray:
    """
    Compute shape features from chain direction codes.

    Returns:
        10-dimensional array: [hist_0..hist_7, total_right, total_left]
    """
    hist = np.zeros(8, dtype=np.float64)

    for code, dist in chain.steps:
        hist[code] += 1

    if len(chain.steps) > 0:
        hist = hist / len(chain.steps)

    # Right turns: codes 1, 3, 5
    # Left turns: codes 2, 4, 6
    # U-turn: code 7 (split evenly)
    total_right = hist[1] + hist[3] + hist[5] + 0.5 * hist[7]
    total_left = hist[2] + hist[4] + hist[6] + 0.5 * hist[7]

    features = np.concatenate([hist, [total_right, total_left]])

    return features


def _compute_color_features(
    chain: Chain,
    image: np.ndarray,
    tile_size: int
) -> np.ndarray:
    """
    Compute average RGB color inside the boundary via scanline fill.

    Returns:
        3-dimensional array: [R_mean, G_mean, B_mean]
    """
    if not chain.tiles:
        return np.array([0.0, 0.0, 0.0])

    boundary_tiles = set(chain.tiles)

    rows = [t[0] for t in chain.tiles]
    cols = [t[1] for t in chain.tiles]
    min_row = min(rows)
    max_row = max(rows)
    min_col = min(cols)
    max_col = max(cols)

    sum_r = 0.0
    sum_g = 0.0
    sum_b = 0.0
    count = 0

    # Scanline fill: for each row, fill between leftmost and rightmost boundary
    for i in range(min_row, max_row + 1):
        boundary_cols = sorted([tj for (ti, tj) in boundary_tiles if ti == i])

        if len(boundary_cols) < 2:
            for j in boundary_cols:
                y_start = i * tile_size
                y_end = min(y_start + tile_size, image.shape[0])
                x_start = j * tile_size
                x_end = min(x_start + tile_size, image.shape[1])

                tile_pixels = image[y_start:y_end, x_start:x_end]
                if tile_pixels.size > 0:
                    sum_r += np.sum(tile_pixels[:, :, 0])
                    sum_g += np.sum(tile_pixels[:, :, 1])
                    sum_b += np.sum(tile_pixels[:, :, 2])
                    count += tile_pixels.shape[0] * tile_pixels.shape[1]
            continue

        left_col = boundary_cols[0]
        right_col = boundary_cols[-1]

        y_start = i * tile_size
        y_end = min(y_start + tile_size, image.shape[0])
        x_start = left_col * tile_size
        x_end = min((right_col + 1) * tile_size, image.shape[1])

        if y_end > y_start and x_end > x_start:
            span_pixels = image[y_start:y_end, x_start:x_end]
            sum_r += np.sum(span_pixels[:, :, 0])
            sum_g += np.sum(span_pixels[:, :, 1])
            sum_b += np.sum(span_pixels[:, :, 2])
            count += span_pixels.shape[0] * span_pixels.shape[1]

    if count > 0:
        r_mean = sum_r / count
        g_mean = sum_g / count
        b_mean = sum_b / count
    else:
        r_mean = g_mean = b_mean = 0.0

    return np.array([r_mean, g_mean, b_mean])


def _compute_centroid(chain: Chain) -> Tuple[float, float]:
    """
    Compute centroid of chain in tile coordinates.

    Returns:
        (x, y) centroid in tile coordinates
    """
    if not chain.tiles:
        return (0.0, 0.0)

    rows = [t[0] for t in chain.tiles]
    cols = [t[1] for t in chain.tiles]

    centroid_i = np.mean(rows)
    centroid_j = np.mean(cols)

    # Return as (x, y) where x=j (column), y=i (row)
    return (centroid_j, centroid_i)


def compute_scale(chain: Chain) -> float:
    """
    Compute scale of a chain (perimeter).

    Returns:
        Total perimeter (sum of distances)
    """
    if not chain.steps:
        return 0.0

    return sum(dist for code, dist in chain.steps)


def extract_objects_from_chains(
    chains: List[Chain],
    image: np.ndarray,
    tile_size: int = 4
) -> List[dict]:
    """
    Extract object representations from chains.

    Args:
        chains: List of Chain objects
        image: Original RGB image
        tile_size: Tile size used for grid

    Returns:
        List of object dictionaries, each containing:
        - v_object: 13D feature vector
        - centroid: (x, y) position
        - scale: perimeter
        - chain: original Chain object
    """
    objects = []

    for chain in chains:
        v_object, centroid = chain_to_v_object(chain, image, tile_size)
        scale = compute_scale(chain)

        obj = {
            'v_object': v_object,
            'centroid': centroid,
            'scale': scale,
            'chain': chain
        }

        objects.append(obj)

    return objects
