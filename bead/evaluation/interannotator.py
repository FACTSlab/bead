"""Inter-annotator agreement metrics.

This module provides comprehensive inter-annotator agreement metrics for
assessing reliability and consistency across multiple human annotators.
Supports Cohen's kappa (2 raters), Fleiss' kappa (multiple raters), and
Krippendorff's alpha (most general).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

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
        rater1 : list[Any]
            Ratings from first rater.
        rater2 : list[Any]
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
        rater1 : list[Any]
            Ratings from first rater.
        rater2 : list[Any]
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

        # Convert to numpy arrays
        r1 = np.array(rater1)
        r2 = np.array(rater2)

        # Get all categories
        categories = sorted(set(r1) | set(r2))
        n_categories = len(categories)
        n = len(r1)

        # Create confusion matrix
        confusion = np.zeros((n_categories, n_categories))

        for i, cat1 in enumerate(categories):
            for j, cat2 in enumerate(categories):
                confusion[i, j] = np.sum((r1 == cat1) & (r2 == cat2))  # type: ignore[call-overload]

        # Observed agreement
        p_o = np.trace(confusion) / n

        # Expected agreement by chance
        p_e = 0.0
        for i in range(n_categories):
            p_e += (confusion[i, :].sum() / n) * (  # type: ignore[operator]
                confusion[:, i].sum() / n  # type: ignore[operator]
            )

        # Cohen's kappa
        if p_e == 1.0:
            # All items in one category
            return 1.0

        kappa = (p_o - p_e) / (1.0 - p_e)
        return float(kappa)

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
        if ratings_matrix.size == 0:
            raise ValueError("Ratings matrix cannot be empty")

        n_items, n_categories = ratings_matrix.shape

        if n_items == 0 or n_categories == 0:
            raise ValueError(f"Invalid matrix shape: ({n_items}, {n_categories})")

        # Number of raters per item (assuming constant)
        n_raters = ratings_matrix.sum(axis=1)[0]  # type: ignore[index, operator]

        # Check that all items have same number of raters
        if not np.allclose(ratings_matrix.sum(axis=1), n_raters):  # type: ignore[operator]
            raise ValueError("All items must have same number of raters")

        # Proportion of all assignments to each category
        p_j = ratings_matrix.sum(axis=0) / (n_items * n_raters)  # type: ignore[operator]

        # Observed agreement for each item
        p_i = (
            np.sum(ratings_matrix * (ratings_matrix - 1), axis=1)  # type: ignore[call-overload]
            / (n_raters * (n_raters - 1))
        )

        # Mean observed agreement
        p_bar = p_i.mean()

        # Expected agreement by chance
        p_e = np.sum(p_j**2)  # type: ignore[call-overload]

        # Fleiss' kappa
        if p_e == 1.0:
            return 1.0

        kappa = (p_bar - p_e) / (1.0 - p_e)
        return float(kappa)

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
        reliability_data : dict[str, list[Any]]
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

        # Build reliability matrix
        reliability_matrix_list: list[list[Label | None]] = []
        for rater_id in rater_ids:
            reliability_matrix_list.append(reliability_data[rater_id])

        reliability_matrix = np.array(reliability_matrix_list, dtype=object)  # type: ignore[assignment]

        # Compute distance function based on metric
        if metric == "nominal":

            def distance_fn(a: Label, b: Label) -> float:
                return 0.0 if a == b else 1.0

        elif metric == "ordinal":
            # For ordinal, use rank-based distance
            def ordinal_distance(a: Label, b: Label) -> float:
                # Get unique values and rank them
                values = [v for row in reliability_matrix for v in row if v is not None]
                unique_values = sorted(set(values))
                value_to_rank = {v: i for i, v in enumerate(unique_values)}

                rank_a = value_to_rank[a]
                rank_b = value_to_rank[b]
                return (rank_a - rank_b) ** 2

            distance_fn = ordinal_distance
        elif metric in ("interval", "ratio"):

            def distance_fn(a: Label, b: Label) -> float:
                return (float(a) - float(b)) ** 2

        else:
            raise ValueError(
                f"Unknown metric: {metric}. Must be one of: "
                "'nominal', 'ordinal', 'interval', 'ratio'"
            )

        # Compute observed disagreement
        observed_disagreement = 0.0
        n_comparisons = 0

        for item_idx in range(n_items):
            # Get all non-missing ratings for this item
            item_ratings: list[Label] = [
                reliability_matrix[rater_idx, item_idx]  # type: ignore[misc, index]
                for rater_idx in range(len(rater_ids))
                if reliability_matrix[rater_idx, item_idx] is not None  # type: ignore[index]
            ]

            # Count pairwise disagreements for this item
            for val1, val2 in combinations(item_ratings, 2):
                observed_disagreement += distance_fn(val1, val2)  # type: ignore[arg-type]
                n_comparisons += 1

        if n_comparisons == 0:
            # No comparisons possible (all missing data)
            return 0.0

        observed_disagreement /= n_comparisons

        # Compute expected disagreement (chance)
        all_values: list[Label] = []
        for rater_idx in range(len(rater_ids)):
            for item_idx in range(n_items):
                val = reliability_matrix[rater_idx, item_idx]  # type: ignore[misc, index]
                if val is not None:
                    all_values.append(val)  # type: ignore[arg-type]

        if len(all_values) < 2:
            return 0.0

        # Expected disagreement from all possible pairs
        expected_disagreement = 0.0
        n_possible = 0

        for val1, val2 in combinations(all_values, 2):
            expected_disagreement += distance_fn(val1, val2)  # type: ignore[arg-type]
            n_possible += 1

        if n_possible == 0:
            return 0.0

        expected_disagreement /= n_possible

        # Krippendorff's alpha
        if expected_disagreement == 0.0:
            # No disagreement possible (all values identical)
            return 1.0

        alpha = 1.0 - (observed_disagreement / expected_disagreement)
        return float(alpha)

    @staticmethod
    def pairwise_agreement(
        ratings: dict[str, list[Label]],
    ) -> dict[str, dict[str, float]]:
        """Compute pairwise agreement metrics for all rater pairs.

        Parameters
        ----------
        ratings : dict[str, list[Any]]
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
