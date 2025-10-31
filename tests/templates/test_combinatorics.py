"""Test combinatorial utilities."""

from __future__ import annotations

from bead.templates.combinatorics import (
    cartesian_product,
    count_combinations,
    stratified_sample,
)


def test_cartesian_product_two_lists() -> None:
    """Test Cartesian product with two lists."""
    result = list(cartesian_product([1, 2], ["a", "b"]))

    assert len(result) == 4
    assert (1, "a") in result
    assert (1, "b") in result
    assert (2, "a") in result
    assert (2, "b") in result


def test_cartesian_product_three_lists() -> None:
    """Test Cartesian product with three lists."""
    result = list(cartesian_product([1, 2], ["a"], [True, False]))

    assert len(result) == 4
    assert (1, "a", True) in result
    assert (2, "a", False) in result


def test_cartesian_product_empty_list() -> None:
    """Test Cartesian product with empty list."""
    result = list(cartesian_product([1, 2], []))

    assert len(result) == 0


def test_cartesian_product_single_list() -> None:
    """Test Cartesian product with single list."""
    result = list(cartesian_product([1, 2, 3]))

    assert len(result) == 3
    assert (1,) in result
    assert (2,) in result
    assert (3,) in result


def test_count_combinations_basic() -> None:
    """Test counting combinations."""
    count = count_combinations([1, 2], ["a", "b", "c"])

    assert count == 6


def test_count_combinations_three_lists() -> None:
    """Test counting with three lists."""
    count = count_combinations([1, 2], ["a", "b"], [True, False])

    assert count == 8


def test_count_combinations_empty() -> None:
    """Test counting with empty list."""
    count = count_combinations([1, 2], [])

    assert count == 0


def test_count_combinations_single_list() -> None:
    """Test counting with single list."""
    count = count_combinations([1, 2, 3, 4, 5])

    assert count == 5


def test_stratified_sample_balanced() -> None:
    """Test stratified sampling balances groups."""
    groups = {
        "group_a": [1, 2, 3, 4, 5],
        "group_b": [6, 7, 8, 9, 10],
    }

    sample = stratified_sample(groups, n_per_group=2, seed=42)

    assert len(sample) == 4


def test_stratified_sample_deterministic() -> None:
    """Test stratified sampling is deterministic with seed."""
    groups = {
        "group_a": [1, 2, 3, 4, 5],
        "group_b": [6, 7, 8, 9, 10],
    }

    sample1 = stratified_sample(groups, n_per_group=3, seed=42)
    sample2 = stratified_sample(groups, n_per_group=3, seed=42)

    assert sample1 == sample2


def test_stratified_sample_more_than_available() -> None:
    """Test stratified sampling when n_per_group > group size."""
    groups = {
        "small_group": [1, 2],
    }

    sample = stratified_sample(groups, n_per_group=10, seed=42)

    # Should sample all available items
    assert len(sample) == 2


def test_stratified_sample_no_seed() -> None:
    """Test stratified sampling without seed."""
    groups = {
        "group_a": [1, 2, 3, 4, 5],
        "group_b": [6, 7, 8, 9, 10],
    }

    sample = stratified_sample(groups, n_per_group=2)

    # Should still produce a sample (but non-deterministic)
    assert len(sample) == 4


def test_stratified_sample_empty_groups() -> None:
    """Test stratified sampling with empty groups."""
    groups: dict[str, list[int]] = {}

    sample = stratified_sample(groups, n_per_group=5, seed=42)

    assert len(sample) == 0
