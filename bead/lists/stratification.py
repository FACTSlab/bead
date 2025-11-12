"""Stratification utilities for quantile-based item assignment.

This module provides utilities for assigning items to quantile bins based
on numeric properties, with optional stratification by grouping variables.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from typing import Any, TypeVar
from uuid import UUID

import numpy as np

T = TypeVar("T")


def assign_quantiles(
    items: list[T],
    property_getter: Callable[[T], float],
    n_quantiles: int = 10,
    stratify_by: Callable[[T], Any] | None = None,
) -> dict[T, int]:
    """Assign quantile bins to items based on numeric property.

    Divides items into n_quantiles bins based on the distribution of
    a numeric property extracted via property_getter. Optionally stratifies
    by a grouping variable, computing separate quantiles for each group.

    Parameters
    ----------
    items : list[T]
        List of items to assign to quantile bins.
    property_getter : Callable[[T], float]
        Function that extracts a numeric value from each item.
        This value is used to compute quantiles.
    n_quantiles : int
        Number of quantile bins (default: 10 for deciles).
        Must be >= 2.
    stratify_by : Callable[[T], Any] | None
        Optional function that extracts a grouping variable from each item.
        If provided, quantiles are computed separately for each group.
        Groups can be any hashable type (str, int, UUID, etc.).

    Returns
    -------
    dict[T, int]
        Dictionary mapping each item to its quantile bin (0 to n_quantiles-1).

    Raises
    ------
    ValueError
        If n_quantiles < 2 or items list is empty.

    Examples
    --------
    Basic usage with simple numeric values:
    >>> items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> result = assign_quantiles(
    ...     items,
    ...     property_getter=lambda x: x,
    ...     n_quantiles=4
    ... )
    >>> result[1]  # First item in lowest quartile
    0
    >>> result[10]  # Last item in highest quartile
    3

    With Item objects and stratification:
    >>> from bead.items.item import Item
    >>> from uuid import uuid4
    >>> items = [
    ...     Item(item_template_id=uuid4(), item_metadata={"score": 10.5, "group": "A"}),
    ...     Item(item_template_id=uuid4(), item_metadata={"score": 5.2, "group": "A"}),
    ...     Item(item_template_id=uuid4(), item_metadata={"score": 8.1, "group": "B"}),
    ...     Item(item_template_id=uuid4(), item_metadata={"score": 3.3, "group": "B"}),
    ... ]
    >>> result = assign_quantiles(
    ...     items,
    ...     property_getter=lambda x: x.item_metadata["score"],
    ...     n_quantiles=2,
    ...     stratify_by=lambda x: x.item_metadata["group"]
    ... )  # doctest: +SKIP

    With UUID keys (common pattern):
    >>> from uuid import UUID
    >>> item_uuids = [uuid4() for _ in range(100)]
    >>> item_scores = {uid: float(i) for i, uid in enumerate(item_uuids)}
    >>> result = assign_quantiles(
    ...     item_uuids,
    ...     property_getter=lambda uid: item_scores[uid],
    ...     n_quantiles=10
    ... )  # doctest: +SKIP
    """
    if not items:
        raise ValueError("items list cannot be empty")

    if n_quantiles < 2:
        raise ValueError(f"n_quantiles must be >= 2, got {n_quantiles}")

    # If no stratification, compute quantiles for all items
    if stratify_by is None:
        return _assign_quantiles_single_group(items, property_getter, n_quantiles)

    # Stratified: compute quantiles separately for each group
    groups: dict[Any, list[T]] = defaultdict(list)
    for item in items:
        group_key = stratify_by(item)
        groups[group_key].append(item)

    # Compute quantiles for each group
    result: dict[T, int] = {}
    for group_items in groups.values():
        group_result = _assign_quantiles_single_group(
            group_items, property_getter, n_quantiles
        )
        result.update(group_result)

    return result


def _assign_quantiles_single_group(
    items: list[T],
    property_getter: Callable[[T], float],
    n_quantiles: int,
) -> dict[T, int]:
    """Assign quantiles to items within a single group.

    Parameters
    ----------
    items : list[T]
        List of items in this group.
    property_getter : Callable[[T], float]
        Function to extract numeric value.
    n_quantiles : int
        Number of quantile bins.

    Returns
    -------
    dict[T, int]
        Mapping from item to quantile bin (0 to n_quantiles-1).
    """
    if not items:
        return {}

    # Extract scores
    scores: np.ndarray[Any, np.dtype[np.floating[Any]]] = np.array(
        [property_getter(item) for item in items]
    )

    # Compute quantile edges
    # linspace(0, 1, n+1) gives [0, 1/n, 2/n, ..., 1]
    quantile_edges: np.ndarray[Any, np.dtype[np.floating[Any]]] = np.quantile(
        scores, np.linspace(0, 1, n_quantiles + 1)
    )

    # Assign each item to a quantile bin
    result: dict[T, int] = {}
    for item, score in zip(items, scores.tolist(), strict=True):
        # searchsorted finds the index where score would be inserted
        # We use quantile_edges[1:] to exclude the 0th edge
        # This maps scores to bins [0, n_quantiles-1]
        quantile_idx = int(np.searchsorted(quantile_edges[1:], float(score)))
        result[item] = quantile_idx

    return result


def assign_quantiles_by_uuid(
    item_ids: list[UUID],
    item_metadata: dict[UUID, dict[str, Any]],
    property_key: str,
    n_quantiles: int = 10,
    stratify_by_key: str | None = None,
) -> dict[UUID, int]:
    """Assign quantile bins to items by UUID with metadata lookup.

    Convenience function for the common pattern of working with UUIDs
    and metadata dictionaries (stand-off annotation pattern).

    Parameters
    ----------
    item_ids : list[UUID]
        List of item UUIDs.
    item_metadata : dict[UUID, dict[str, Any]]
        Metadata dictionary mapping UUIDs to their metadata dicts.
    property_key : str
        Key in item_metadata[uuid] dict to use for quantile computation.
    n_quantiles : int
        Number of quantile bins (default: 10).
    stratify_by_key : str | None
        Optional key in metadata dict to use for stratification.

    Returns
    -------
    dict[UUID, int]
        Dictionary mapping each UUID to its quantile bin (0 to n_quantiles-1).

    Raises
    ------
    ValueError
        If property_key missing from any item's metadata.
    KeyError
        If any UUID not found in item_metadata.

    Examples
    --------
    >>> from uuid import uuid4
    >>> uuids = [uuid4() for _ in range(100)]
    >>> metadata = {
    ...     uid: {"score": float(i), "group": "A" if i < 50 else "B"}
    ...     for i, uid in enumerate(uuids)
    ... }
    >>> result = assign_quantiles_by_uuid(
    ...     uuids,
    ...     metadata,
    ...     property_key="score",
    ...     n_quantiles=4,
    ...     stratify_by_key="group"
    ... )  # doctest: +SKIP
    """
    # Validate that all items have the property
    for uid in item_ids:
        if uid not in item_metadata:
            raise KeyError(f"UUID {uid} not found in item_metadata")
        if property_key not in item_metadata[uid]:
            raise ValueError(
                f"Property '{property_key}' not found in metadata for UUID {uid}"
            )

    # Create property getter
    def property_getter(uid: UUID) -> float:
        value = item_metadata[uid][property_key]
        return float(value)

    # Create stratification getter if needed
    stratify_func: Callable[[UUID], int | float | str | bool] | None
    if stratify_by_key:
        if any(stratify_by_key not in item_metadata[uid] for uid in item_ids):
            raise ValueError(
                f"Stratification key '{stratify_by_key}' not found in all items"
            )

        def stratify_getter(uid: UUID) -> int | float | str | bool:
            return item_metadata[uid][stratify_by_key]  # type: ignore[return-value]

        stratify_func = stratify_getter
    else:
        stratify_func = None

    return assign_quantiles(item_ids, property_getter, n_quantiles, stratify_func)
