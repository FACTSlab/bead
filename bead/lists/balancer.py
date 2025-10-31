"""Quantile balancing for experimental list partitioning.

This module provides the QuantileBalancer class for ensuring uniform distribution
of items across quantiles of a numeric property. Uses NumPy for efficient
quantile computation and maintains stand-off annotation pattern (works with UUIDs).
"""

from __future__ import annotations

from collections.abc import Callable
from uuid import UUID

import numpy as np


class QuantileBalancer:
    """Ensures uniform distribution of items across quantiles.

    Used by stratified partitioning strategy to create balanced distribution
    of numeric properties (e.g., LM probabilities, word frequencies).

    Works with UUIDs only (stand-off annotation). Requires value_func callable
    to extract numeric values from items via their UUIDs.

    Parameters
    ----------
    n_quantiles : int, default=5
        Number of quantiles to create (must be >= 2).
    random_seed : int | None, default=None
        Random seed for reproducibility. If None, uses non-deterministic RNG.

    Attributes
    ----------
    n_quantiles : int
        Number of quantiles to create.
    random_seed : int | None
        Random seed for reproducibility.

    Examples
    --------
    >>> from uuid import uuid4
    >>> import numpy as np
    >>> balancer = QuantileBalancer(n_quantiles=5, random_seed=42)
    >>> # Create items with known values
    >>> items = [uuid4() for _ in range(100)]
    >>> values = {item: float(i) for i, item in enumerate(items)}
    >>> value_func = lambda uid: values[uid]
    >>> # Balance across 4 lists, 5 items per quantile per list
    >>> lists = balancer.balance(items, value_func, n_lists=4,
    ...                          items_per_quantile_per_list=5)
    >>> len(lists)
    4
    """

    def __init__(self, n_quantiles: int = 5, random_seed: int | None = None) -> None:
        if n_quantiles < 2:
            raise ValueError(f"n_quantiles must be >= 2, got {n_quantiles}")

        self.n_quantiles = n_quantiles
        self.random_seed = random_seed
        self._rng = np.random.default_rng(random_seed)

    def balance(
        self,
        item_ids: list[UUID],
        value_func: Callable[[UUID], float],
        n_lists: int,
        items_per_quantile_per_list: int,
    ) -> list[list[UUID]]:
        """Balance items across lists and quantiles.

        Distributes items uniformly across quantiles and lists to ensure
        balanced representation of the numeric property across all lists.

        Parameters
        ----------
        item_ids : list[UUID]
            UUIDs of items to balance.
        value_func : Callable[[UUID], float]
            Function to extract numeric value from item UUID.
        n_lists : int
            Number of lists to create.
        items_per_quantile_per_list : int
            Target number of items per quantile per list.

        Returns
        -------
        list[list[UUID]]
            Balanced lists of item UUIDs.

        Raises
        ------
        ValueError
            If n_lists < 1 or items_per_quantile_per_list < 1.

        Examples
        --------
        >>> from uuid import uuid4
        >>> balancer = QuantileBalancer(n_quantiles=5, random_seed=42)
        >>> items = [uuid4() for _ in range(100)]
        >>> values = {item: float(i) for i, item in enumerate(items)}
        >>> lists = balancer.balance(items, lambda uid: values[uid], 4, 5)
        >>> all(len(lst) == 25 for lst in lists)  # 5 quantiles * 5 items
        True

        Notes
        -----
        - Items are assigned to quantiles using np.percentile and np.digitize
        - Within each quantile, items are shuffled before distribution
        - If insufficient items exist in a quantile, fewer items are assigned
        """
        if n_lists < 1:
            raise ValueError(f"n_lists must be >= 1, got {n_lists}")
        if items_per_quantile_per_list < 1:
            raise ValueError(
                f"items_per_quantile_per_list must be >= 1, "
                f"got {items_per_quantile_per_list}"
            )

        # Create quantile-based strata
        strata = self._create_strata(item_ids, value_func)

        # Initialize lists
        lists: list[list[UUID]] = [[] for _ in range(n_lists)]

        # Distribute items from each quantile across lists
        for q in range(self.n_quantiles):
            q_items = strata[q]

            # Shuffle items in this quantile
            q_items_array = np.array(q_items)
            self._rng.shuffle(q_items_array)

            # Distribute to lists
            for list_idx in range(n_lists):
                # Take items for this list
                start_idx = list_idx * items_per_quantile_per_list
                end_idx = start_idx + items_per_quantile_per_list
                list_items = q_items_array[start_idx:end_idx].tolist()

                lists[list_idx].extend(list_items)

        return lists

    def compute_balance_score(
        self, item_ids: list[UUID], value_func: Callable[[UUID], float]
    ) -> float:
        """Compute balance score for items.

        Score is 1.0 for perfect balance (uniform distribution across quantiles),
        lower for imbalanced distributions. Score is based on deviation from
        expected uniform distribution.

        Parameters
        ----------
        item_ids : list[UUID]
            UUIDs of items to score.
        value_func : Callable[[UUID], float]
            Function to extract numeric values.

        Returns
        -------
        float
            Balance score (0.0-1.0, higher is better).

        Examples
        --------
        >>> from uuid import uuid4
        >>> balancer = QuantileBalancer(n_quantiles=5)
        >>> # Uniformly distributed values
        >>> items = [uuid4() for _ in range(100)]
        >>> values = {item: float(i) for i, item in enumerate(items)}
        >>> score = balancer.compute_balance_score(items, lambda uid: values[uid])
        >>> score > 0.9  # Should be close to 1.0
        True

        Notes
        -----
        - Returns 0.0 for empty item lists
        - Uses mean absolute deviation from expected uniform count
        """
        if not item_ids:
            return 0.0

        # Compute values
        values = np.array([value_func(item_id) for item_id in item_ids])

        # Create expected quantile bins
        expected_quantiles = np.linspace(0, 100, self.n_quantiles + 1)
        expected_bins = np.percentile(values, expected_quantiles)

        # Count items in each quantile
        quantile_assignments = np.digitize(values, expected_bins) - 1
        quantile_assignments = np.clip(quantile_assignments, 0, self.n_quantiles - 1)

        quantile_counts = np.bincount(quantile_assignments, minlength=self.n_quantiles)

        # Compute uniformity score
        expected_count = len(item_ids) / self.n_quantiles
        deviations = np.abs(quantile_counts - expected_count)
        score = 1.0 - (np.mean(deviations) / expected_count)

        return float(max(0.0, score))

    def _create_strata(
        self, item_ids: list[UUID], value_func: Callable[[UUID], float]
    ) -> dict[int, list[UUID]]:
        """Create quantile-based strata from items.

        Parameters
        ----------
        item_ids : list[UUID]
            UUIDs of items to stratify.
        value_func : Callable[[UUID], float]
            Function to extract numeric values.

        Returns
        -------
        dict[int, list[UUID]]
            Dictionary mapping quantile index (0 to n_quantiles-1) to list
            of item UUIDs in that quantile.

        Notes
        -----
        - Uses np.percentile to compute quantile boundaries
        - Uses np.digitize to assign items to quantiles
        - Edge cases are handled by clipping to valid quantile range
        """
        # Extract values
        values = np.array([value_func(item_id) for item_id in item_ids])

        # Compute quantile bins
        quantiles = np.linspace(0, 100, self.n_quantiles + 1)
        bins = np.percentile(values, quantiles)

        # Assign items to quantiles
        quantile_assignments = np.digitize(values, bins) - 1
        quantile_assignments = np.clip(quantile_assignments, 0, self.n_quantiles - 1)

        # Group items by quantile
        strata: dict[int, list[UUID]] = {q: [] for q in range(self.n_quantiles)}
        for item_id, q in zip(item_ids, quantile_assignments, strict=False):
            strata[q].append(item_id)

        return strata
