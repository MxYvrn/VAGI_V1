"""
Tests for Stage 6: Obj-KNN (Object Memory)
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import obj_knn


def test_add_and_query():
    """Test adding objects and querying."""
    memory = obj_knn.ObjKNN()

    # Create some test objects
    obj1 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0], dtype=float)  # Red square
    obj2 = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0], dtype=float)  # Green circle
    obj3 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 250, 5, 5], dtype=float)  # Similar to obj1

    # Add objects
    id1 = memory.add_object(obj1)
    id2 = memory.add_object(obj2)
    id3 = memory.add_object(obj3)

    print(f"\nAdded objects with IDs: {id1}, {id2}, {id3}")
    print(f"Memory size: {memory.size()}")

    # Query for nearest to obj1
    query_result = memory.query(obj1, k=1)
    print(f"Nearest to obj1: {query_result}")

    assert query_result[0][0] == id1, "Should find itself as nearest"
    assert query_result[0][1] == 0.0, "Distance to itself should be 0"

    # Query for nearest to obj3
    query_result = memory.query(obj3, k=2)
    print(f"Top 2 nearest to obj3: {query_result}")

    # obj3 should be nearest to itself, then obj1 (similar shape and color)
    assert query_result[0][0] == id3, "Should find itself first"

    print("✓ test_add_and_query passed")


def test_weighted_distance():
    """Test that color and shape are weighted appropriately."""
    memory = obj_knn.ObjKNN(shape_weight=1.0, color_weight=0.1)

    # Object with specific shape
    obj_shape = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 100, 100, 100], dtype=float)

    # Object with same shape, different color
    obj_diff_color = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 200, 200, 200], dtype=float)

    # Object with different shape, same color
    obj_diff_shape = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 100, 100, 100], dtype=float)

    memory.add_object(obj_diff_color, proto_id=1)
    memory.add_object(obj_diff_shape, proto_id=2)

    # Query with obj_shape
    results = memory.query(obj_shape, k=2)
    print(f"\nQuery results: {results}")
    print(f"  Proto 1 (same shape, diff color) distance: {results[1][1]:.2f}")
    print(f"  Proto 2 (diff shape, same color) distance: {results[0][1]:.2f}")

    # The shape difference is weighted heavily, but the color difference is large
    # Shape diff: [1,0...] vs [0,1...] has L2 norm = sqrt(2)
    # Color diff: [100,100,100] vs [200,200,200] has L2 norm = sqrt(3*100^2) = 173
    # Weighted: shape_dist = sqrt(2)*1.0, color_dist = 173*0.1 = 17.3
    # So actually the same-color object is closer!
    assert results[0][0] == 2, "Same color should be closer due to large color difference"
    assert results[1][0] == 1, "Different color should be farther"

    print("✓ test_weighted_distance passed")


def test_get_or_add():
    """Test get_or_add functionality."""
    memory = obj_knn.ObjKNN()

    obj1 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 255, 0, 0], dtype=float)
    obj2 = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 250, 5, 5], dtype=float)  # Similar
    obj3 = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 255], dtype=float)  # Different

    # Add first object
    id1, is_new1 = memory.get_or_add(obj1, similarity_threshold=1.0)
    print(f"\nFirst add: id={id1}, is_new={is_new1}")
    assert is_new1, "First object should be new"

    # Add similar object - should return existing
    id2, is_new2 = memory.get_or_add(obj2, similarity_threshold=1.0)
    print(f"Similar object: id={id2}, is_new={is_new2}")
    assert not is_new2, "Similar object should not create new prototype"
    assert id2 == id1, "Should return same proto_id"

    # Add different object - should create new
    id3, is_new3 = memory.get_or_add(obj3, similarity_threshold=1.0)
    print(f"Different object: id={id3}, is_new={is_new3}")
    assert is_new3, "Different object should create new prototype"
    assert id3 != id1, "Should get different proto_id"

    print(f"Final memory size: {memory.size()}")

    print("✓ test_get_or_add passed")


def test_get_prototype():
    """Test retrieving prototypes by ID."""
    memory = obj_knn.ObjKNN()

    obj1 = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13], dtype=float)
    id1 = memory.add_object(obj1)

    retrieved = memory.get_prototype(id1)

    print(f"\nOriginal: {obj1}")
    print(f"Retrieved: {retrieved}")

    assert np.array_equal(obj1, retrieved), "Retrieved should match original"

    # Test non-existent ID
    none_result = memory.get_prototype(999)
    assert none_result is None, "Non-existent ID should return None"

    print("✓ test_get_prototype passed")


def test_clear():
    """Test clearing memory."""
    memory = obj_knn.ObjKNN()

    # Add some objects
    for i in range(5):
        obj = np.random.rand(13)
        memory.add_object(obj)

    print(f"\nSize before clear: {memory.size()}")
    assert memory.size() == 5

    memory.clear()

    print(f"Size after clear: {memory.size()}")
    assert memory.size() == 0

    print("✓ test_clear passed")


def test_k_nearest():
    """Test k-nearest neighbors query."""
    memory = obj_knn.ObjKNN()

    # Add several objects
    for i in range(10):
        obj = np.random.rand(13)
        memory.add_object(obj)

    query_obj = np.random.rand(13)

    # Query for top 3
    results = memory.query(query_obj, k=3)

    print(f"\nTop 3 results: {results}")
    assert len(results) == 3, "Should return 3 results"

    # Verify sorted by distance
    assert results[0][1] <= results[1][1], "Should be sorted by distance"
    assert results[1][1] <= results[2][1], "Should be sorted by distance"

    print("✓ test_k_nearest passed")


def test_distance_threshold():
    """Test distance threshold filtering."""
    memory = obj_knn.ObjKNN()

    obj1 = np.zeros(13)
    obj2 = np.ones(13) * 10  # Far away

    memory.add_object(obj1, proto_id=1)
    memory.add_object(obj2, proto_id=2)

    # Query with strict threshold
    results = memory.query(np.zeros(13), k=5, distance_threshold=1.0)

    print(f"\nResults with threshold=1.0: {results}")

    # Should only return obj1 (distance 0), not obj2 (distance ~10)
    assert len(results) == 1, "Should filter out distant objects"
    assert results[0][0] == 1, "Should only return close object"

    print("✓ test_distance_threshold passed")


if __name__ == "__main__":
    print("Running Obj-KNN tests...\n")
    test_add_and_query()
    test_weighted_distance()
    test_get_or_add()
    test_get_prototype()
    test_clear()
    test_k_nearest()
    test_distance_threshold()
    print("\n✅ All Obj-KNN tests passed!")
