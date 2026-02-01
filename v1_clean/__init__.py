"""
V1 Vision Pipeline

Raw Image → N² Edge Neurons → EdgeFiller → Recursive EdgeRunner →
Chain Filtering → Features → Obj-KNN → Img-ID → Img-KNN
"""

from core import (
    Cell,
    CellGrid,
    Chain,
    get_direction,
    get_distance,
    compute_turn_code,
    get_neighbors_8,
)

from n2_activation import compute_activation_grid
from edge_filler import edge_filler
from edge_runner import extract_chains_from_grid
from chain_filter import filter_chains
from features import (
    chain_to_v_object,
    compute_scale,
    extract_objects_from_chains,
)
from obj_knn import ObjectMemoryKNN
from img_id import (
    SceneObject,
    Scene,
    build_scene,
    SceneMemoryKNN,
)

__all__ = [
    # Core structures
    'Cell',
    'CellGrid',
    'Chain',
    'get_direction',
    'get_distance',
    'compute_turn_code',
    'get_neighbors_8',

    # Pipeline stages
    'compute_activation_grid',
    'edge_filler',
    'extract_chains_from_grid',
    'filter_chains',
    'chain_to_v_object',
    'compute_scale',
    'extract_objects_from_chains',

    # Memory systems
    'ObjectMemoryKNN',
    'SceneObject',
    'Scene',
    'build_scene',
    'SceneMemoryKNN',
]
