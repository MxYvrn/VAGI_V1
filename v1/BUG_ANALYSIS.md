# V1 VALIDATION BUG ANALYSIS

## Test Results Summary
- **Total Tests**: 22
- **Passed**: 17 (77.3%)
- **Failed**: 5 (22.7%)

---

## BUG #1: EdgeRunner - Tile Count Off-by-One

### Symptom
Test "EdgeRunner - Simple Line" expects 3 tiles for a 3-tile horizontal line, but gets 4 tiles.

### Root Cause
In [edge_runner.py:59](edge_runner.py#L59), the initial chain starts with `chain_tiles=[seed_pos]`, which includes the seed tile. However, when continuing the trace, tiles are added without checking if we're revisiting the seed.

### Expected vs Actual
```
Pattern:  [1, 1, 1]  (3 active tiles)
Expected: 3 tiles in chain
Actual:   4 tiles in chain
```

### Analysis
Looking at the code flow:
1. Seed position `(1, 1)` is added to `chain_tiles` at initialization
2. First neighbor `(1, 2)` is found and added
3. Second neighbor `(1, 3)` is found and added
4. When no more neighbors exist, we might be adding an extra tile or the seed is being counted incorrectly

**Actually, this might NOT be a bug!** The chain includes the STEPS between tiles, so a 3-tile line has:
- Tile 0: (1, 1) - seed
- Tile 1: (1, 2) - first step
- Tile 2: (1, 3) - second step

But we're getting 4 tiles, which suggests we might be appending the endpoint twice, OR the test expectation is wrong.

**Verdict**: Need to check if the test is counting tiles vs steps. A 3-tile line should have 2 steps but 3 tiles.

---

## BUG #2: EdgeRunner - Loop Detection Failing

### Symptom
Test "EdgeRunner - Closed Loop" extracts 8 chains but none are detected as loops.

### Root Cause
In [edge_runner.py:204-231](edge_runner.py#L204-L231), when we encounter visited neighbors (splicing case), we create a chain marked as `spliced=True` and return. However, we don't check if the visited neighbor is the **original seed position**, which would indicate a loop.

### Expected Behavior
For a closed loop pattern:
```
·███·
·█·█·
·███·
```

The trace should:
1. Start at seed (e.g., `(1,1)`)
2. Trace around the perimeter
3. Return to `(1,1)` - detect as loop
4. Mark chain with `start_pos == end_pos`

### Actual Behavior
The recursive algorithm with branching is creating **multiple chains** (8 chains) because:
- Each tile with multiple neighbors creates branches
- These branches are marked as "spliced" when they hit visited tiles
- None complete a full loop back to the original seed

### Why This Happens
The square has 4 corners. Each corner has 3 neighbors (orthogonal + diagonal). This creates excessive branching. The algorithm is working as designed for **general boundary tracing**, but not optimized for **simple closed loops**.

**Critical Issue**: The loop detection logic in `Chain.is_loop()` checks `start_pos == end_pos`, but due to splicing, we're ending at intermediate visited tiles, not the seed.

### Fix Needed
When splicing into a visited tile, check if that tile is the **original seed** (`chain_tiles[0]`). If yes, create a loop instead of a splice.

---

## BUG #3: Chain Filter - Min Length Blocking Valid Chains

### Symptom
Test "Chain Filter - Keep Border Touch" fails - a 3-tile border-touching chain is filtered out.
Test "Chain Filter - Keep Spliced" fails - a 3-tile spliced chain is filtered out.

### Root Cause
In [chain_filter.py:46-48](chain_filter.py#L46-L48), the filter checks `len(chain) < min_length` BEFORE checking other keep conditions.

```python
# Skip chains that are too short
if len(chain) < min_length:
    continue
```

But `len(chain)` returns the number of **steps**, not tiles!

For a 3-tile chain:
- Tiles: `[(0, 2), (1, 2), (2, 2)]` - 3 tiles
- Steps: `[(0, 1.0), (0, 1.0)]` - 2 steps
- `len(chain)` = 2 (steps)
- `min_length` = 3

So it gets filtered out despite being valid!

### Fix Needed
Either:
1. Change the length check to `len(chain.tiles) < min_length`, OR
2. Adjust `min_length` to be in terms of steps (e.g., `min_length=2` for 3 tiles)

**Recommendation**: Use `len(chain.tiles)` for clarity, since "length" is more intuitive as tile count.

---

## BUG #4: N² Activation - Horizontal Edge Not Fully Detected

### Symptom
Test creates a horizontal edge (top half white, bottom half black), but boundary tiles are not fully activated.

### Root Cause
The test creates a 16x16 image with:
```
Top half (rows 0-7): white (255)
Bottom half (rows 8-15): black (0)
```

With 4x4 tiles, this creates a 4x4 tile grid.
- Tile row 0: covers pixels 0-3 (all white)
- Tile row 1: covers pixels 4-7 (all white)
- Tile row 2: covers pixels 8-11 (all black)
- Tile row 3: covers pixels 12-15 (all black)

The **edge is at pixel row 8**, which is the **start of tile row 2**.

For tile (1, j): pixels 4-7 are all white → max-min = 0 → no activation
For tile (2, j): pixels 8-11 are all black → max-min = 0 → no activation

**The edge falls BETWEEN tiles, not within a tile!**

### Why This Happens
The N² activation detects variation **within** a tile. If an edge aligns perfectly with tile boundaries, no single tile contains both colors.

### Is This a Bug?
**No, this is correct behavior for V1!** The spec says tiles activate based on **local pixel variation**. If the edge is aligned to tile boundaries, the tiles are uniform.

### Test Fix Needed
The test should create edges that **cross through tiles**, not align with tile boundaries. E.g., put the edge at pixel row 6 instead of row 8.

---

## BUG #5: EdgeRunner - Chain Length Confusion (Steps vs Tiles)

### Related to Bug #1

### Core Issue
The `Chain` class has:
- `steps: List[Tuple[int, float]]` - turn codes and distances
- `tiles: List[Tuple[int, int]]` - tile positions
- `__len__()` returns `len(self.steps)`

For a 3-tile chain:
- Tiles = 3
- Steps = 2 (transitions between tiles)

### Confusion Points
1. `len(chain)` returns step count, not tile count
2. Tests expect tile count
3. Chain filter uses `len(chain)` for min_length check

### Recommendation
Add explicit methods:
- `chain.num_steps()` → `len(self.steps)`
- `chain.num_tiles()` → `len(self.tiles)`
- Change `__len__()` to return tiles (more intuitive)

---

## SUMMARY OF FIXES NEEDED

### Priority 1 (Correctness Bugs)
1. **EdgeRunner Loop Detection**: Check if splice target is original seed → create loop
2. **Chain Filter Length Check**: Use `len(chain.tiles)` instead of `len(chain.steps)`

### Priority 2 (API Clarity)
3. **Chain.__len__()**: Return tile count instead of step count
4. **Add explicit methods**: `num_tiles()` and `num_steps()`

### Priority 3 (Test Fixes)
5. **N² Activation Test**: Adjust test to put edge within tiles, not on boundaries

---

## NEXT STEPS

1. Fix bugs in order of priority
2. Re-run validation suite
3. Add more stress tests for:
   - Complex loop patterns
   - Multi-level branching
   - Diagonal vs orthogonal edge alignment
   - Large objects (100+ tiles)
   - Tiny objects (2-3 tiles)
