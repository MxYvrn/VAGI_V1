"""Test with a simpler loop pattern - a thin ring with no branching."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import edge_runner
from utils import CellGrid

# Create a simple loop: a horizontal line that wraps
# This should force a loop because there's no branching
pattern = np.array([
    [0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 1, 0, 0, 0, 1, 0],
    [0, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 0, 0, 0],
])

grid = CellGrid(5, 7)
grid.set_activation_map(pattern)

print("Input pattern (thin ring):")
for i in range(5):
    row = ""
    for j in range(7):
        row += "█" if pattern[i, j] == 1 else "·"
    print(f"  {row}")

chains = edge_runner.extract_chains_from_grid(grid)

print(f"\nExtracted {len(chains)} chains:")
for idx, chain in enumerate(chains):
    print(f"\nChain {idx}:")
    print(f"  Tiles: {chain.tiles}")
    print(f"  Start: {chain.start_pos}")
    print(f"  End: {chain.end_pos}")
    print(f"  Is loop: {chain.is_loop()}")
    print(f"  Spliced: {chain.spliced}")

loops = [c for c in chains if c.is_loop()]
print(f"\n{'='*60}")
print(f"Total loops detected: {len(loops)}")

if loops:
    print("✓ SUCCESS: Loop detected!")
else:
    print("✗ FAILURE: No loops detected")

# Also test an even simpler case: 4 tiles in a square
print("\n" + "="*60)
print("Test 2: Minimal 2x2 square (4 tiles)")
print("="*60)

pattern2 = np.array([
    [0, 0, 0, 0],
    [0, 1, 1, 0],
    [0, 1, 1, 0],
    [0, 0, 0, 0],
])

grid2 = CellGrid(4, 4)
grid2.set_activation_map(pattern2)

print("\nInput pattern:")
for i in range(4):
    row = ""
    for j in range(4):
        row += "█" if pattern2[i, j] == 1 else "·"
    print(f"  {row}")

chains2 = edge_runner.extract_chains_from_grid(grid2)

print(f"\nExtracted {len(chains2)} chains:")
for idx, chain in enumerate(chains2):
    print(f"\nChain {idx}:")
    print(f"  Tiles: {chain.tiles}")
    print(f"  Start: {chain.start_pos}")
    print(f"  End: {chain.end_pos}")
    print(f"  Is loop: {chain.is_loop()}")
    print(f"  Spliced: {chain.spliced}")

loops2 = [c for c in chains2 if c.is_loop()]
print(f"\nTotal loops detected: {len(loops2)}")

if loops2:
    print("✓ SUCCESS: Loop detected!")
else:
    print("✗ FAILURE: No loops detected")
