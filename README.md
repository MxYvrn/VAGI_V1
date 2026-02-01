# V1 Vision Pipeline

A custom algorithmic (non-neural) vision system that extracts object boundaries and represents scenes via KNN memories.

## Architecture Overview

V1 is **NOT** a neural network or CNN. It is an **algorithmic system** with the following pipeline:

```
Raw RGB Image
  ↓
[1] N² Edge Neurons (4×4 tiles, simple edge detection)
  ↓
[2] EdgeFiller (fix 1–0–1 gaps only, no dilation)
  ↓
[3] Recursive EdgeRunner (branching + splicing, builds boundary chains)
  ↓
[4] Chain Filtering (remove useless "strings")
  ↓
[5] Boundary → v_object (shape histograms + average RGB via scanline fill)
  ↓
[6] Obj-KNN (object memory)
  ↓
[7] Img-ID (scene = {proto_id, x, y, scale})
  ↓
[8] Img-KNN (scene memory)
```

## Key Features

- **Tile-based processing**: Works on 4×4 pixel tiles, not individual pixels
- **Explicit edge tracing**: Recursive algorithm with branching and splicing
- **Shape encoding**: 8-direction histogram representation
- **Color extraction**: Scanline fill for average RGB inside boundaries
- **KNN memories**: Separate memories for objects and scenes
- **No deep learning**: Pure algorithmic implementation

## Installation

```bash
cd VAGI_V1/v1
# No external dependencies beyond NumPy
```

## Quick Start

```python
import numpy as np
from v1 import demo

# Run the demo
results = demo.demo_simple_shapes()

# Or process your own image
from v1.demo import run_v1_pipeline

image = np.array(...)  # Your RGB image
results = run_v1_pipeline(image, tile_size=4, verbose=True)

# Access results
scene = results['scene']
objects = results['objects']
chains = results['chains']
```

## Module Structure

```
v1/
├── __init__.py              # Package initialization
├── utils.py                 # Common data structures (Cell, Chain, direction encoding)
├── n2_activation.py         # Stage 1: Tile-level edge detection
├── edge_filler.py           # Stage 2: Gap filling
├── edge_runner.py           # Stage 3: Boundary chain extraction
├── chain_filter.py          # Stage 4: Chain filtering
├── features.py              # Stage 5: Shape + color feature extraction
├── obj_knn.py               # Stage 6: Object memory
├── img_id.py                # Stage 7: Scene representation & memory
├── demo.py                  # End-to-end demonstrations
└── tests/                   # Unit tests for each stage
    ├── test_n2.py
    ├── test_edge_filler.py
    ├── test_edge_runner.py
    ├── test_chain_filter.py
    ├── test_features.py
    ├── test_obj_knn.py
    └── test_img_id.py
```

## Stage Details

### Stage 1: N² Edge Neurons

- Divides image into 4×4 pixel tiles
- Computes activation (0 or 1) based on local pixel variation
- Uses simple heuristic: `max(pixels) - min(pixels) > threshold`

**Parameters:**
- `tile_size`: Default 4
- `threshold`: Default 30.0 (for 0-255 range)

### Stage 2: EdgeFiller

- Fills 1-tile gaps in pattern 1–0–1
- Checks all 8 directions (orthogonal + diagonal)
- Uses separate output grid (no in-place modification)

### Stage 3: Recursive EdgeRunner

- Traces connected boundaries with 8-connectivity
- **Branching**: Creates new chains at junctions
- **Splicing**: Connects to already-visited tiles
- Encodes direction changes as relative turn codes (0-7)

**Turn codes:**
- 0: straight (0°)
- 1: small right (+45°)
- 2: small left (-45°)
- 3: medium right (+90°)
- 4: medium left (-90°)
- 5: big right (+135°)
- 6: big left (-135°)
- 7: U-turn (±180°)

### Stage 4: Chain Filtering

Keeps chains that are:
1. Closed loops (start == end)
2. Touch image border
3. Created via splicing
4. Length ≥ `min_length` (default 3)

Drops: floating interior strings

### Stage 5: Features (v_object)

Converts each chain to a **13-dimensional vector**:

```
v_object = [
    hist_0, hist_1, ..., hist_7,  # 8 bins: normalized direction histogram
    total_right_turn,              # Sum of right turn bins
    total_left_turn,               # Sum of left turn bins
    R_mean, G_mean, B_mean         # Average RGB from scanline fill
]
```

Also computes:
- **Centroid**: (x, y) in tile coordinates
- **Scale**: Perimeter (sum of distances)

### Stage 6: Obj-KNN

Object memory with weighted distance metric:

```
distance = sqrt(shape_dist² + color_dist²)
  where:
    shape_dist = L2_norm(shape1 - shape2) * shape_weight
    color_dist = L2_norm(color1 - color2) * color_weight
```

**Default weights:**
- `shape_weight = 1.0`
- `color_weight = 0.1`

**Methods:**
- `add_object(v_object)` → proto_id
- `query(v_object, k=1)` → nearest neighbors
- `get_or_add(v_object, threshold)` → finds or creates prototype

### Stage 7: Img-ID & Img-KNN

**Scene representation:**
```python
Scene = [
    SceneObject(proto_id, x, y, scale),
    SceneObject(proto_id, x, y, scale),
    ...
]
```

**Scene memory** supports:
- Scene-to-scene similarity queries
- Simple heuristic: count mismatches + position differences

## Running Tests

```bash
cd v1/tests

# Test individual stages
python3 test_n2.py
python3 test_edge_filler.py
python3 test_edge_runner.py
python3 test_chain_filter.py
python3 test_features.py
python3 test_obj_knn.py
python3 test_img_id.py

# Run complete demo
cd ..
python3 demo.py
```

## Example Output

The demo produces output like:

```
============================================================
V1 PIPELINE EXECUTION
============================================================

[1/7] Computing N² activation grid...
  → Activated tiles: 56 / 1024

[2/7] Running EdgeFiller...
  → Total tiles after filling: 64 (+8 new)

[3/7] Extracting boundary chains...
  → Extracted 50 chains
  → Chain stats: avg_len=13.5, loops=2, spliced=50

[4/7] Filtering chains...
  → Kept 46 / 50 chains

[5/7] Extracting object features...
  → Extracted 46 objects

[6/7] Building object memory (Obj-KNN)...

[7/7] Creating scene representation (Img-ID)...
  → Scene with 46 objects
```

## Design Principles

1. **Correctness over speed**: Clear, tested implementations
2. **Clarity over cleverness**: Readable, well-commented code
3. **Faithful to spec**: No "optimizations" that change the architecture
4. **Modular**: Each stage independently testable
5. **Algorithmic**: No neural networks, no pre-trained models

## Tuning Parameters

### N² Activation Threshold

```python
grid = n2_activation.compute_activation_grid(image, threshold=30.0)
```

- Lower threshold → more activated tiles (more sensitive to edges)
- Higher threshold → fewer activated tiles (only strong edges)
- Typical range: 20-50 for 8-bit images

### Chain Filtering Min Length

```python
filtered = chain_filter.filter_chains(chains, min_length=3)
```

- Smaller → keep more detail (but more noise)
- Larger → cleaner but might miss small objects
- Typical range: 3-10

### Obj-KNN Similarity Threshold

```python
proto_id, is_new = obj_knn.get_or_add(v_object, similarity_threshold=0.5)
```

- Lower → more strict matching (more prototypes)
- Higher → more lenient (fewer prototypes, more grouping)
- Typical range: 0.3-2.0

### Obj-KNN Weight Balance

```python
memory = obj_knn.ObjKNN(shape_weight=1.0, color_weight=0.1)
```

- Increase `color_weight` to make color more important
- Decrease `color_weight` to focus on shape
- Typical: shape_weight 10× color_weight

## Limitations & Future Work

### Current Limitations

1. **Multiple objects produce many chains**: Branching creates redundant chains
2. **Simple scanline fill**: May over-fill concave shapes
3. **Scene distance is basic**: Simple count + position heuristic
4. **No scale invariance**: Objects at different scales seen as different
5. **Tile alignment sensitivity**: Edges aligned to tile grid may not activate

### Potential Improvements

1. **Chain deduplication**: Better merging of overlapping chains
2. **Advanced fill**: Proper inside/outside determination
3. **Hungarian matching**: Optimal object pairing in scene distance
4. **Scale normalization**: Make features scale-invariant
5. **Multi-resolution**: Process at multiple tile sizes
6. **Adaptive thresholds**: Per-region threshold adaptation

## Technical Notes

### Direction Encoding

Absolute directions (N=0, NE=1, E=2, SE=3, S=4, SW=5, W=6, NW=7) are converted to **relative turn codes** during chain building.

### Scanline Fill Algorithm

For each row in the bounding box:
1. Find leftmost and rightmost boundary tiles
2. Fill all pixels between them
3. Accumulate RGB values
4. Compute average at the end

This is a simplified version; real flood fill would be more accurate but slower.

### Branching vs Splicing

- **Branching**: Multiple unvisited neighbors → spawn recursive calls
- **Splicing**: Only visited neighbors → connect to existing chain

Both are essential for handling complex boundary topology.

## License

This is a reference implementation for educational and research purposes.

## Citation

If you use this implementation, please cite as:

```
V1 Vision Pipeline (VAGI_V1)
An algorithmic vision system with explicit boundary extraction and KNN memories
2025
```
