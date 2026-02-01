# V1 Vision Pipeline

Clean, production-ready implementation of the V1 boundary-based vision system.

## Architecture

```
Raw Image
  ↓
N² Edge Neurons (4×4 tiles)
  ↓
EdgeFiller (1-0-1 gap repair)
  ↓
Recursive EdgeRunner (branching + splicing)
  ↓
Chain Filtering (loops, borders, spliced)
  ↓
Features (shape histograms + color)
  ↓
Obj-KNN (object memory)
  ↓
Img-ID (scene representation)
  ↓
Img-KNN (scene memory)
```

## Modules

### `core/`
Core data structures:
- `Cell`: Tile in the grid
- `CellGrid`: 2D grid of tiles
- `Chain`: Boundary chain with direction codes
- Direction encoding utilities (8-way compass)
- Turn code computation (0-7)

### `n2_activation.py`
N² Edge Neurons - divides image into 4×4 tiles, computes activation based on local pixel variation.

### `edge_filler.py`
EdgeFiller - fills 1-tile gaps in pattern 1-0-1 across 8 directions.

### `edge_runner.py`
Recursive EdgeRunner - traces boundaries with branching and splicing:
- Handles multiple unvisited neighbors (branching)
- Splices when hitting visited tiles
- Detects closed loops

### `chain_filter.py`
Chain Filtering - keeps meaningful boundaries:
- Closed loops
- Border-touching chains
- Spliced chains
- Drops internal floating strings

### `features.py`
Feature extraction:
- Shape: 8-bin direction histogram + turn statistics
- Color: RGB average via scanline fill
- Centroid and scale computation
- 13D v_object vector

### `obj_knn.py`
Object Memory (Obj-KNN):
- Weighted distance metric (shape + color)
- Prototype storage
- KNN queries
- get_or_add for automatic proto_id assignment

### `img_id.py`
Scene representation (Img-ID) and Scene Memory (Img-KNN):
- Scene object storage (proto_id, position, scale)
- Scene-level similarity queries

## Usage

```python
import numpy as np
from PIL import Image

# Import V1 modules
from v1_clean import (
    compute_activation_grid,
    edge_filler,
    extract_chains_from_grid,
    filter_chains,
    extract_objects_from_chains,
    ObjectMemoryKNN,
    build_scene,
    SceneMemoryKNN,
)

# Load image
image = np.array(Image.open('image.jpg').convert('RGB'))

# Run pipeline
grid = compute_activation_grid(image, tile_size=4, threshold=30.0)
filled = edge_filler(grid)
chains = extract_chains_from_grid(filled)
filtered = filter_chains(chains, grid.height, grid.width, min_length=3)
objects = extract_objects_from_chains(filtered, image, tile_size=4)

# Build scene
obj_memory = ObjectMemoryKNN()
scene = build_scene(objects, obj_memory, similarity_threshold=0.5)

# Store in scene memory
scene_memory = SceneMemoryKNN()
scene_id = scene_memory.add(scene)
```

## Direction Encoding

Absolute directions (0-7):
- 0: N
- 1: NE
- 2: E
- 3: SE
- 4: S
- 5: SW
- 6: W
- 7: NW

Turn codes (0-7):
- 0: straight (0°)
- 1: small right (+45°)
- 2: small left (-45°)
- 3: medium right (+90°)
- 4: medium left (-90°)
- 5: big right (+135°)
- 6: big left (-135°)
- 7: U-turn (±180°)

## v_object Structure

13-dimensional feature vector:
```
[0:8]   - Direction histogram (8 bins, normalized)
[8]     - Total right turns
[9]     - Total left turns
[10:13] - RGB color (average)
```

## Design Principles

- No preprocessing
- No CNNs or deep learning
- Exact specification compliance
- Modular, clean architecture
- Production-ready code
