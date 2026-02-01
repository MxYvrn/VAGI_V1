"""
Tests for Stage 4: Chain Filtering
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chain_filter
import utils


def test_filter_loops():
    """Test that loops are always kept."""
    # Create a loop chain
    loop_chain = utils.Chain(
        steps=[(0, 1.0), (3, 1.0), (3, 1.0), (3, 1.0)],
        tiles=[(2, 2), (2, 3), (3, 3), (3, 2), (2, 2)],  # Forms a loop
        spliced=False
    )

    # Create a non-loop chain
    line_chain = utils.Chain(
        steps=[(0, 1.0), (0, 1.0)],
        tiles=[(2, 2), (2, 3), (2, 4)],
        spliced=False
    )

    chains = [loop_chain, line_chain]

    filtered = chain_filter.filter_chains(chains, grid_height=10, grid_width=10, min_length=2)

    print(f"\nOriginal chains: {len(chains)}")
    print(f"Filtered chains: {len(filtered)}")
    print(f"Loop chain kept: {loop_chain in filtered}")
    print(f"Line chain kept: {line_chain in filtered}")

    # Loop should be kept
    assert loop_chain in filtered, "Loop chain should be kept"

    print("✓ test_filter_loops passed")


def test_filter_border_touching():
    """Test that chains touching borders are kept."""
    # Chain starting at border
    border_chain = utils.Chain(
        steps=[(0, 1.0), (0, 1.0)],
        tiles=[(0, 5), (0, 6), (0, 7)],  # Starts at row 0 (border)
        spliced=False
    )

    # Interior chain (floating string)
    interior_chain = utils.Chain(
        steps=[(0, 1.0), (0, 1.0)],
        tiles=[(5, 5), (5, 6), (5, 7)],
        spliced=False
    )

    chains = [border_chain, interior_chain]

    filtered = chain_filter.filter_chains(chains, grid_height=10, grid_width=10, min_length=2)

    print(f"\nBorder chain kept: {border_chain in filtered}")
    print(f"Interior chain kept: {interior_chain in filtered}")

    # Border chain should be kept, interior should be dropped
    assert border_chain in filtered, "Border-touching chain should be kept"
    assert interior_chain not in filtered, "Interior floating string should be dropped"

    print("✓ test_filter_border_touching passed")


def test_filter_spliced():
    """Test that spliced chains are kept."""
    # Spliced chain
    spliced_chain = utils.Chain(
        steps=[(0, 1.0), (0, 1.0)],
        tiles=[(5, 5), (5, 6), (5, 7)],
        spliced=True  # Mark as spliced
    )

    # Non-spliced interior chain
    non_spliced_chain = utils.Chain(
        steps=[(0, 1.0), (0, 1.0)],
        tiles=[(5, 5), (5, 6), (5, 7)],
        spliced=False
    )

    chains = [spliced_chain, non_spliced_chain]

    filtered = chain_filter.filter_chains(chains, grid_height=10, grid_width=10, min_length=2)

    print(f"\nSpliced chain kept: {spliced_chain in filtered}")
    print(f"Non-spliced chain kept: {non_spliced_chain in filtered}")

    # Spliced should be kept, non-spliced interior should be dropped
    assert spliced_chain in filtered, "Spliced chain should be kept"
    assert non_spliced_chain not in filtered, "Non-spliced interior chain should be dropped"

    print("✓ test_filter_spliced passed")


def test_filter_min_length():
    """Test that chains shorter than min_length are dropped."""
    # Short chain
    short_chain = utils.Chain(
        steps=[(0, 1.0)],
        tiles=[(0, 0), (0, 1)],
        spliced=False
    )

    # Long chain
    long_chain = utils.Chain(
        steps=[(0, 1.0), (0, 1.0), (0, 1.0), (0, 1.0)],
        tiles=[(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
        spliced=False
    )

    chains = [short_chain, long_chain]

    filtered = chain_filter.filter_chains(chains, grid_height=10, grid_width=10, min_length=3)

    print(f"\nShort chain kept: {short_chain in filtered}")
    print(f"Long chain kept: {long_chain in filtered}")

    # Short should be dropped due to min_length
    assert short_chain not in filtered, "Short chain should be dropped"
    assert long_chain in filtered, "Long chain should be kept"

    print("✓ test_filter_min_length passed")


def test_chain_statistics():
    """Test chain statistics computation."""
    chains = [
        utils.Chain(
            steps=[(0, 1.0)] * 5,
            tiles=[(i, 0) for i in range(6)],
            spliced=False
        ),
        utils.Chain(
            steps=[(0, 1.0)] * 3,
            tiles=[(0, j) for j in range(4)],
            spliced=True
        ),
        utils.Chain(
            steps=[(0, 1.0)] * 4,
            tiles=[(2, 2), (2, 3), (3, 3), (3, 2), (2, 2)],  # Loop
            spliced=False
        ),
    ]

    stats = chain_filter.get_chain_statistics(chains)

    print("\nChain statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    assert stats['num_chains'] == 3
    assert stats['num_loops'] == 1
    assert stats['num_spliced'] == 1
    assert stats['min_length'] == 3
    assert stats['max_length'] == 5

    print("✓ test_chain_statistics passed")


def test_empty_chains():
    """Test filtering with empty chain list."""
    filtered = chain_filter.filter_chains([], grid_height=10, grid_width=10)

    assert len(filtered) == 0, "Empty input should return empty output"

    stats = chain_filter.get_chain_statistics([])
    assert stats['num_chains'] == 0

    print("✓ test_empty_chains passed")


if __name__ == "__main__":
    print("Running Chain Filtering tests...\n")
    test_filter_loops()
    test_filter_border_touching()
    test_filter_spliced()
    test_filter_min_length()
    test_chain_statistics()
    test_empty_chains()
    print("\n✅ All Chain Filtering tests passed!")
