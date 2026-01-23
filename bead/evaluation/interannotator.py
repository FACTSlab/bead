"""Inter-annotator agreement metrics.

This module provides inter-annotator agreement metrics for assessing
reliability and consistency across multiple human annotators.
Uses sklearn.metrics for Cohen's kappa, statsmodels for Fleiss' kappa,
and krippendorff package for Krippendorff's alpha.
"""

from __future__ import annotations

from itertools import combinations
from typing import Literal

import numpy as np
from krippendorff import alpha as krippendorff_alpha
from sklearn.metrics import cohen_kappa_score
from statsmodels.stats.inter_rater import fleiss_kappa as statsmodels_fleiss_kappa

# Type alias for krippendorff metric levels
type KrippendorffMetric = Literal["nominal", "ordinal", "interval", "ratio"]

# Type alias for rating values (categorical, ordinal, interval, or ratio)
type Label = int | str | float


class InterAnnotatorMetrics:
    """Inter-annotator agreement metrics for reliability assessment.

    Provides static methods for computing various agreement metrics:
    - Percentage agreement (simple)
    - Cohen's kappa (2 raters, categorical)
    - Fleiss' kappa (multiple raters, categorical)
    - Krippendorff's alpha (general, multiple data types)
    - Pairwise agreement (all pairs of raters)

    Examples
    --------
    >>> # Cohen's kappa for 2 raters
    >>> rater1 = [0, 1, 0, 1, 1]
    >>> rater2 = [0, 1, 1, 1, 1]
    >>> InterAnnotatorMetrics.cohens_kappa(rater1, rater2)
    0.6
    >>> # Percentage agreement
    >>> InterAnnotatorMetrics.percentage_agreement(rater1, rater2)
    0.8
    """

    @staticmethod
    def percentage_agreement(rater1: list[Label], rater2: list[Label]) -> float:
        """Compute simple percentage agreement between two raters.

        Parameters
        ----------
        rater1 : list[Label]
            Ratings from first rater.
        rater2 : list[Label]
            Ratings from second rater.

        Returns
        -------
        float
            Percentage agreement (0.0 to 1.0).

        Raises
        ------
        ValueError
            If rater lists have different lengths.

        Examples
        --------
        >>> rater1 = [1, 2, 3, 1, 2]
        >>> rater2 = [1, 2, 2, 1, 2]
        >>> InterAnnotatorMetrics.percentage_agreement(rater1, rater2)
        0.8
        """
        if len(rater1) != len(rater2):
            raise ValueError(
                f"Rater lists must have same length: {len(rater1)} != {len(rater2)}"
            )

        if not rater1:
            return 1.0

        agreements = sum(r1 == r2 for r1, r2 in zip(rater1, rater2, strict=True))
        return agreements / len(rater1)

    @staticmethod
    def cohens_kappa(rater1: list[Label], rater2: list[Label]) -> float:
        """Compute Cohen's kappa for two raters.

        Cohen's kappa measures agreement between two raters beyond chance.
        Values range from -1 (complete disagreement) to 1 (perfect agreement),
        with 0 indicating chance-level agreement.

        Parameters
        ----------
        rater1 : list[Label]
            Ratings from first rater.
        rater2 : list[Label]
            Ratings from second rater.

        Returns
        -------
        float
            Cohen's kappa coefficient.

        Raises
        ------
        ValueError
            If rater lists have different lengths or are empty.

        Examples
        --------
        >>> # Perfect agreement
        >>> rater1 = [0, 1, 0, 1]
        >>> rater2 = [0, 1, 0, 1]
        >>> InterAnnotatorMetrics.cohens_kappa(rater1, rater2)
        1.0
        >>> # No agreement beyond chance
        >>> rater1 = [0, 0, 1, 1]
        >>> rater2 = [1, 1, 0, 0]
        >>> kappa = InterAnnotatorMetrics.cohens_kappa(rater1, rater2)
        >>> abs(kappa - (-1.0)) < 0.01
        True
        """
        if len(rater1) != len(rater2):
            raise ValueError(
                f"Rater lists must have same length: {len(rater1)} != {len(rater2)}"
            )

        if not rater1:
            raise ValueError("Rater lists cannot be empty")

        # Check for single category case (sklearn returns NaN)
        unique_values = set(rater1) | set(rater2)
        if len(unique_values) == 1:
            return 1.0  # Perfect agreement by definition

        result = cohen_kappa_score(rater1, rater2)
        # Handle NaN case (can happen with extreme distributions)
        if np.isnan(result):
            return 1.0
        return float(result)

    @staticmethod
    def fleiss_kappa(ratings_matrix: np.ndarray[int, np.dtype[np.int_]]) -> float:  # type: ignore[type-arg]
        """Compute Fleiss' kappa for multiple raters.

        Fleiss' kappa generalizes Cohen's kappa to multiple raters. It measures
        agreement beyond chance when multiple raters assign categorical ratings
        to a set of items.

        Parameters
        ----------
        ratings_matrix : np.ndarray
            Matrix of shape (n_items, n_categories) where element [i, j]
            contains the number of raters who assigned item i to category j.

        Returns
        -------
        float
            Fleiss' kappa coefficient.

        Raises
        ------
        ValueError
            If matrix is empty or has wrong shape.
        ImportError
            If statsmodels is not installed.

        Examples
        --------
        >>> # 4 items, 3 categories, 5 raters each
        >>> # Item 1: 3 raters chose cat 0, 2 chose cat 1, 0 chose cat 2
        >>> ratings = np.array([
        ...     [3, 2, 0],  # Item 1
        ...     [0, 0, 5],  # Item 2
        ...     [2, 3, 0],  # Item 3
        ...     [1, 1, 3],  # Item 4
        ... ])
        >>> kappa = InterAnnotatorMetrics.fleiss_kappa(ratings)
        >>> 0.0 <= kappa <= 1.0
        True
        """
        if statsmodels_fleiss_kappa is None:
            msg = "statsmodels required for Fleiss' kappa. pip install statsmodels"
            raise ImportError(msg)

        if ratings_matrix.size == 0:
            raise ValueError("Ratings matrix cannot be empty")

        n_items, n_categories = ratings_matrix.shape

        if n_items == 0 or n_categories == 0:
            raise ValueError(f"Invalid matrix shape: ({n_items}, {n_categories})")

        # Check that all items have the same number of raters
        rater_counts = ratings_matrix.sum(axis=1)
        if not np.allclose(rater_counts, rater_counts[0]):
            raise ValueError(
                "All items must have same number of raters. "
                f"Got counts: {rater_counts.tolist()}"
            )

        return float(statsmodels_fleiss_kappa(ratings_matrix))

    @staticmethod
    def krippendorff_alpha(
        reliability_data: dict[str, list[Label | None]],
        metric: str = "nominal",
    ) -> float:
        """Compute Krippendorff's alpha for multiple raters.

        Krippendorff's alpha is the most general inter-rater reliability
        measure. It handles:
        - Any number of raters
        - Missing data
        - Different data types (nominal, ordinal, interval, ratio)

        Parameters
        ----------
        reliability_data : dict[str, list[Label | None]]
            Dictionary mapping rater IDs to their ratings. Each rater's
            ratings list must have same length (use None for missing values).
        metric : str, default="nominal"
            Distance metric to use:
            - "nominal": for categorical data (default)
            - "ordinal": for ordered categories
            - "interval": for interval-scaled data
            - "ratio": for ratio-scaled data

        Returns
        -------
        float
            Krippendorff's alpha coefficient (1.0 = perfect agreement,
            0.0 = chance agreement, < 0.0 = systematic disagreement).

        Raises
        ------
        ValueError
            If reliability_data is empty or rater lists have different lengths.

        Examples
        --------
        >>> # 3 raters, 5 items (with one missing value)
        >>> data = {
        ...     'rater1': [1, 2, 3, 4, 5],
        ...     'rater2': [1, 2, 3, 4, 5],
        ...     'rater3': [1, 2, None, 4, 5]
        ... }
        >>> alpha = InterAnnotatorMetrics.krippendorff_alpha(data)
        >>> alpha > 0.8  # High agreement
        True
        """
        if not reliability_data:
            raise ValueError("reliability_data cannot be empty")

        # Convert to reliability matrix (items × raters)
        rater_ids = list(reliability_data.keys())
        n_items = len(reliability_data[rater_ids[0]])

        # Check all raters have same number of items
        for rater_id, ratings in reliability_data.items():
            if len(ratings) != n_items:
                raise ValueError(
                    f"All raters must rate same number of items: "
                    f"{rater_id} has {len(ratings)}, expected {n_items}"
                )

        # Convert to format expected by krippendorff package
        # Format: rows are coders/raters, columns are units/items
        # Missing values should be np.nan
        reliability_matrix: list[list[float]] = []
        all_values: list[Label] = []
        for rater_id in rater_ids:
            rater_ratings: list[float] = []
            for rating in reliability_data[rater_id]:
                if rating is None:
                    rater_ratings.append(np.nan)
                else:
                    is_numeric = isinstance(rating, (int, float))
                    val = float(rating) if is_numeric else hash(rating)
                    rater_ratings.append(val)
                    all_values.append(rating)
            reliability_matrix.append(rater_ratings)

        # Handle edge cases that krippendorff package doesn't handle
        if len(all_values) == 0:
            # All missing data
            return 0.0

        # Check if there are any pairwise comparisons possible
        # (at least one item must have ratings from at least 2 raters)
        comparisons_possible = False
        for item_idx in range(n_items):
            n_raters_for_item = sum(
                1
                for rater_id in rater_ids
                if reliability_data[rater_id][item_idx] is not None
            )
            if n_raters_for_item >= 2:
                comparisons_possible = True
                break

        if not comparisons_possible:
            # No pairwise comparisons possible
            return 0.0

        unique_values = set(all_values)
        if len(unique_values) <= 1:
            # All same value - perfect agreement by definition
            return 1.0

        # Map metric names to krippendorff package names
        metric_map: dict[str, KrippendorffMetric] = {
            "nominal": "nominal",
            "ordinal": "ordinal",
            "interval": "interval",
            "ratio": "ratio",
        }

        if metric not in metric_map:
            raise ValueError(
                f"Unknown metric: {metric}. Must be one of: "
                "'nominal', 'ordinal', 'interval', 'ratio'"
            )

        return float(
            krippendorff_alpha(
                reliability_matrix,
                level_of_measurement=metric_map[metric],
            )
        )

    @staticmethod
    def pairwise_agreement(
        ratings: dict[str, list[Label]],
    ) -> dict[str, dict[str, float]]:
        """Compute pairwise agreement metrics for all rater pairs.

        Parameters
        ----------
        ratings : dict[str, list[Label]]
            Dictionary mapping rater IDs to their ratings.

        Returns
        -------
        dict[str, dict[str, float]]
            Nested dictionary with structure:
            {
                'percentage_agreement': {('rater1', 'rater2'): 0.85, ...},
                'cohens_kappa': {('rater1', 'rater2'): 0.75, ...}
            }

        Examples
        --------
        >>> ratings = {
        ...     'rater1': [1, 2, 3],
        ...     'rater2': [1, 2, 3],
        ...     'rater3': [1, 2, 2]
        ... }
        >>> result = InterAnnotatorMetrics.pairwise_agreement(ratings)
        >>> result['percentage_agreement'][('rater1', 'rater2')]
        1.0
        >>> result['cohens_kappa'][('rater1', 'rater2')]
        1.0
        """
        rater_ids = list(ratings.keys())

        if len(rater_ids) < 2:
            return {
                "percentage_agreement": {},
                "cohens_kappa": {},
            }

        percentage_agreements = {}
        kappas = {}

        # Compute for all pairs
        for rater1_id, rater2_id in combinations(rater_ids, 2):
            pair = (rater1_id, rater2_id)

            # Percentage agreement
            perc_agr = InterAnnotatorMetrics.percentage_agreement(
                ratings[rater1_id], ratings[rater2_id]
            )
            percentage_agreements[pair] = perc_agr

            # Cohen's kappa
            kappa = InterAnnotatorMetrics.cohens_kappa(
                ratings[rater1_id], ratings[rater2_id]
            )
            kappas[pair] = kappa

        return {
            "percentage_agreement": percentage_agreements,
            "cohens_kappa": kappas,
        }
