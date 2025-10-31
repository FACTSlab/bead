"""Combinatorial utilities for template filling."""

from __future__ import annotations

import itertools
import random
from collections.abc import Iterator
from typing import Any, TypeVar

T = TypeVar("T")


def cartesian_product(*iterables: list[Any]) -> Iterator[tuple[Any, ...]]:
    """Generate Cartesian product of iterables.

    Equivalent to itertools.product but with explicit type hints
    and documentation for template filling use case.

    Parameters
    ----------
    *iterables : list[Any]
        Variable number of iterables to combine.

    Yields
    ------
    tuple[Any, ...]
        Each combination from the Cartesian product.

    Examples
    --------
    >>> list(cartesian_product([1, 2], ['a', 'b']))
    [(1, 'a'), (1, 'b'), (2, 'a'), (2, 'b')]
    """
    return itertools.product(*iterables)


def count_combinations(*iterables: list[Any]) -> int:
    """Count total combinations without generating them.

    Calculate the size of the Cartesian product space efficiently
    without actually generating combinations.

    Parameters
    ----------
    *iterables : list[Any]
        Variable number of iterables.

    Returns
    -------
    int
        Total number of combinations.

    Examples
    --------
    >>> count_combinations([1, 2], ['a', 'b'], [True, False])
    8
    """
    count = 1
    for iterable in iterables:
        count *= len(iterable)
    return count


def stratified_sample[T](
    groups: dict[str, list[T]],
    n_per_group: int,
    seed: int | None = None,
) -> list[T]:
    """Sample items from groups with balanced representation.

    Ensure each group is represented approximately equally in the sample.

    Parameters
    ----------
    groups : dict[str, list[Any]]
        Dictionary mapping group names to lists of items.
    n_per_group : int
        Number of items to sample from each group.
    seed : int | None
        Random seed for reproducibility.

    Returns
    -------
    list[Any]
        Sampled items, balanced across groups.

    Examples
    --------
    >>> groups = {"verbs": [v1, v2, v3], "nouns": [n1, n2, n3]}
    >>> sample = stratified_sample(groups, n_per_group=2, seed=42)
    >>> len(sample)
    4
    """
    if seed is not None:
        random.seed(seed)

    sampled: list[T] = []
    for group_items in groups.values():
        # Sample with replacement if n_per_group > len(group_items)
        k = min(n_per_group, len(group_items))
        sampled.extend(random.sample(group_items, k))

    return sampled
