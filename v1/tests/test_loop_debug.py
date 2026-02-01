"""Debug loop detection in EdgeRunner."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import edge_runner
from utils import CellGrid

# Simple 3x3 square loop
pattern = np.array([
    [0, 0, 0, 0, 0],
    [0, 1, 1, 1, 0],
    [0, 1, 0, 1, 0],
    [0, 1, 1, 1, 0],
    [0, 0, 0, 0, 0],
])

grid = CellGrid(5, 5)
grid.set_activation_map(pattern)

print("Input pattern:")
for i in range(5):
    row = ""
    for j in range(5):
        row += "█" if pattern[i, j] == 1 else "·"
    print(f"  {row}")

chains = edge_runner.extract_chains_from_grid(grid)

print(f"\nExtracted {len(chains)} chains:")
for idx, chain in enumerate(chains):
    print(f"\nChain {idx}:")
    print(f"  Tiles ({len(chain.tiles)}): {chain.tiles[:10]}{'...' if len(chain.tiles) > 10 else ''}")
    print(f"  Steps ({chain.num_steps()}): {len(chain.steps)}")
    print(f"  Start: {chain.start_pos}")
    print(f"  End: {chain.end_pos}")
    print(f"  Is loop: {chain.is_loop()}")
    print(f"  Spliced: {chain.spliced}")
    print(f"  Chain ID: {chain.chain_id}")

    # Check if end equals start
    if chain.start_pos == chain.end_pos:
        print(f"  ✓ Forms a closed loop!")

print("\n" + "=" * 60)
print("ANALYSIS:")
print("=" * 60)

# Count loops
loops = [c for c in chains if c.is_loop()]
spliced = [c for c in chains if c.spliced]

print(f"Chains that are loops: {len(loops)}")
print(f"Chains that are spliced: {len(spliced)}")

if loops:
    print("\nLoop chains found:")
    for c in loops:
        print(f"  {c.start_pos} -> {c.end_pos} ({len(c.tiles)} tiles)")
else:
    print("\n⚠ NO LOOPS DETECTED")
    print("Checking why...")

    # Check all chains for potential loops
    for idx, chain in enumerate(chains):
        if len(chain.tiles) >= 2:
            unique_tiles = set(chain.tiles)
            if len(unique_tiles) < len(chain.tiles):
                print(f"\nChain {idx} has duplicate tiles (potential loop):")
                print(f"  Total tiles: {len(chain.tiles)}")
                print(f"  Unique tiles: {len(unique_tiles)}")
                print(f"  First tile: {chain.tiles[0]}")
                print(f"  Last tile: {chain.tiles[-1]}")

                # Find duplicates
                from collections import Counter
                tile_counts = Counter(chain.tiles)
                duplicates = {t: count for t, count in tile_counts.items() if count > 1}
                print(f"  Duplicates: {duplicates}")
