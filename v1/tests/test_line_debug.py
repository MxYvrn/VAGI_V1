"""Debug simple line test."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import edge_runner
from utils import CellGrid

# Horizontal line (3 tiles)
pattern = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
])

grid = CellGrid(3, 5)
grid.set_activation_map(pattern)

print("Input pattern (3-tile horizontal line):")
for i in range(3):
    row = ""
    for j in range(5):
        row += "█" if pattern[i, j] == 1 else "·"
    print(f"  {row}")

print("\nExpected: 3 tiles [(1,1), (1,2), (1,3)]")

chains = edge_runner.extract_chains_from_grid(grid)

print(f"\nExtracted {len(chains)} chains:")
for idx, chain in enumerate(chains):
    print(f"\nChain {idx}:")
    print(f"  Tiles ({len(chain.tiles)}): {chain.tiles}")
    print(f"  Steps ({chain.num_steps()}): {chain.steps}")
    print(f"  Start: {chain.start_pos}")
    print(f"  End: {chain.end_pos}")

# Analysis
if len(chains) > 0:
    main_chain = chains[0]
    print(f"\n{'='*60}")
    print("ANALYSIS:")
    print(f"{'='*60}")
    print(f"Expected tiles: 3")
    print(f"Actual tiles: {len(main_chain.tiles)}")

    if len(main_chain.tiles) == 3:
        print("✓ Tile count correct!")
    elif len(main_chain.tiles) == 4:
        print("✗ One extra tile detected")
        print(f"  Tiles: {main_chain.tiles}")
        print("  Checking for duplicates or border tiles...")

        # Check if there's a duplicate
        if len(set(main_chain.tiles)) < len(main_chain.tiles):
            print("  → Found duplicate tiles (possibly a loop closure)")

        # Check if there's a tile outside expected range
        expected = {(1, 1), (1, 2), (1, 3)}
        actual = set(main_chain.tiles)
        extra = actual - expected
        if extra:
            print(f"  → Extra tiles not in expected set: {extra}")
