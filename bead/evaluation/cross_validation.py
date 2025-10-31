"""Cross-validation utilities for model evaluation.

This module provides K-fold cross-validation with support for stratified
sampling, ensuring representative splits for model evaluation.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import TypeVar

import numpy as np

from bead.active_learning.trainers.base import BaseTrainer
from bead.items.models import Item

T = TypeVar("T")

# Type alias for stratification values (typically labels)
type StratificationValue = int | str | float | bool

# Recursive type for values during property path traversal
type TraversalValue = StratificationValue | dict[str, TraversalValue]


class CrossValidator:
    """K-fold cross-validation for model evaluation.

    Supports standard and stratified K-fold cross-validation for evaluating
    model performance. Stratification ensures balanced distribution of a
    categorical variable across folds.

    Parameters
    ----------
    k : int, default=5
        Number of folds for cross-validation (must be >= 2).
    shuffle : bool, default=True
        Whether to shuffle data before splitting into folds.
    random_seed : int | None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    k : int
        Number of folds.
    shuffle : bool
        Whether to shuffle data.
    random_seed : int | None
        Random seed.

    Examples
    --------
    >>> from uuid import uuid4
    >>> cv = CrossValidator(k=5, shuffle=True, random_seed=42)
    >>> items = [
    ...     Item(item_template_id=uuid4(), rendered_elements={})
    ...     for _ in range(100)
    ... ]
    >>> folds = cv.k_fold_split(items)
    >>> len(folds)
    5
    >>> # Each fold is (train_items, test_items)
    >>> train, test = folds[0]
    >>> len(train) + len(test) == len(items)
    True
    """

    def __init__(
        self,
        k: int = 5,
        shuffle: bool = True,
        random_seed: int | None = None,
    ) -> None:
        """Initialize cross-validator.

        Parameters
        ----------
        k : int
            Number of folds (must be >= 2).
        shuffle : bool
            Whether to shuffle data before splitting.
        random_seed : int | None
            Random seed for reproducibility.

        Raises
        ------
        ValueError
            If k < 2.
        """
        if k < 2:
            raise ValueError(f"k must be >= 2, got {k}")

        self.k = k
        self.shuffle = shuffle
        self.random_seed = random_seed

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def k_fold_split(
        self,
        items: list[T],
        stratify_by: str | None = None,
    ) -> list[tuple[list[T], list[T]]]:
        """Split items into K folds for cross-validation.

        Parameters
        ----------
        items : list[T]
            Items to split (typically list[Item]).
        stratify_by : str | None
            Property path for stratification (e.g., "metadata.label").
            If provided, ensures balanced distribution of this property
            across folds. Uses DSL-style dot notation.

        Returns
        -------
        list[tuple[list[T], list[T]]]
            List of (train_items, test_items) for each fold.
            Length equals k.

        Examples
        --------
        >>> cv = CrossValidator(k=3, random_seed=42)
        >>> items = list(range(10))
        >>> folds = cv.k_fold_split(items)
        >>> len(folds)
        3
        >>> # Verify all items used exactly once as test
        >>> all_test = [item for _, test in folds for item in test]
        >>> set(all_test) == set(items)
        True
        """
        if not items:
            return []

        if stratify_by is not None:
            return self._stratified_k_fold_split(items, stratify_by)
        else:
            return self._standard_k_fold_split(items)

    def _standard_k_fold_split(self, items: list[T]) -> list[tuple[list[T], list[T]]]:
        """Perform standard K-fold split without stratification.

        Parameters
        ----------
        items : list[T]
            Items to split.

        Returns
        -------
        list[tuple[list[T], list[T]]]
            K folds of (train, test) splits.
        """
        # Shuffle if requested
        if self.shuffle:
            items = items.copy()
            random.shuffle(items)

        # Compute fold sizes
        n = len(items)
        fold_sizes = [n // self.k] * self.k
        for i in range(n % self.k):
            fold_sizes[i] += 1

        # Create folds
        folds: list[tuple[list[T], list[T]]] = []
        start = 0

        for fold_size in fold_sizes:
            # Test set for this fold
            test = items[start : start + fold_size]

            # Train set is everything else
            train = items[:start] + items[start + fold_size :]

            folds.append((train, test))
            start += fold_size

        return folds

    def _stratified_k_fold_split(
        self, items: list[T], stratify_by: str
    ) -> list[tuple[list[T], list[T]]]:
        """Stratified K-fold split maintaining class distribution.

        Parameters
        ----------
        items : list[T]
            Items to split.
        stratify_by : str
            Property path for stratification.

        Returns
        -------
        list[tuple[list[T], list[T]]]
            K folds with stratified splits.
        """
        # Group items by stratification variable
        groups: dict[StratificationValue, list[T]] = defaultdict(list)

        for item in items:
            # Extract stratification value
            value = self._get_property_value(item, stratify_by)
            groups[value].append(item)

        # Shuffle within each group
        if self.shuffle:
            for group in groups.values():
                random.shuffle(group)

        # Split each group into k parts
        group_folds: dict[StratificationValue, list[list[T]]] = {}

        for label, group_items in groups.items():
            n_group = len(group_items)
            fold_sizes = [n_group // self.k] * self.k
            for i in range(n_group % self.k):
                fold_sizes[i] += 1

            # Create fold splits for this group
            group_folds[label] = []
            start = 0
            for fold_size in fold_sizes:
                group_folds[label].append(group_items[start : start + fold_size])
                start += fold_size

        # Combine folds across groups
        folds: list[tuple[list[T], list[T]]] = []

        for fold_idx in range(self.k):
            # Test set: fold_idx items from each group
            test: list[T] = []
            train: list[T] = []

            for label in groups:
                # Test items for this fold
                test.extend(group_folds[label][fold_idx])

                # Train items from all other folds
                for other_fold_idx in range(self.k):
                    if other_fold_idx != fold_idx:
                        train.extend(group_folds[label][other_fold_idx])

            folds.append((train, test))

        return folds

    def _get_property_value(
        self, item: object, property_path: str
    ) -> StratificationValue:
        """Extract property value from item using dot notation.

        Parameters
        ----------
        item : object
            Item to extract property from (typically an Item object).
        property_path : str
            Dot-separated property path (e.g., "metadata.label").

        Returns
        -------
        StratificationValue
            Property value (must be a hashable primitive for stratification).

        Raises
        ------
        AttributeError
            If property path is invalid.
        TypeError
            If extracted value is not a valid stratification type.
        """
        parts = property_path.split(".")
        current: object = item

        for part in parts:
            if hasattr(current, part):  # type: ignore[arg-type]
                current = getattr(current, part)  # type: ignore[arg-type]
            elif isinstance(current, dict) and part in current:
                current = current[part]  # type: ignore[index]
            else:
                raise AttributeError(
                    f"Cannot access '{part}' in property path '{property_path}'"
                )

        # Validate and narrow type: ensure the final value is a stratification value
        if isinstance(current, bool):
            return current
        elif isinstance(current, int):
            return current
        elif isinstance(current, str):
            return current
        elif isinstance(current, float):
            return current
        else:
            raise TypeError(
                f"Property '{property_path}' must be int, str, float, or bool, "
                f"got {type(current).__name__}"  # type: ignore[arg-type]
            )

    def evaluate_fold(
        self,
        trainer: BaseTrainer,
        train_items: list[Item],
        test_items: list[Item],
        metrics: list[str] | None = None,
    ) -> dict[str, float]:
        """Evaluate a single fold.

        Parameters
        ----------
        trainer : BaseTrainer
            Trainer to use for training the model.
        train_items : list[Item]
            Training items for this fold.
        test_items : list[Item]
            Test items for this fold.
        metrics : list[str] | None
            List of metric names to compute. If None, uses trainer's default.

        Returns
        -------
        dict[str, float]
            Computed metrics for this fold.

        Examples
        --------
        >>> # Requires a real trainer and items
        >>> # cv = CrossValidator()
        >>> # fold_metrics = cv.evaluate_fold(trainer, train, test)
        >>> # fold_metrics['accuracy']  # doctest: +SKIP
        0.85
        """
        # Train model on train_items
        metadata = trainer.train(train_items, eval_data=test_items)

        # Return evaluation metrics
        return metadata.metrics

    def cross_validate(
        self,
        trainer: BaseTrainer,
        items: list[Item],
        stratify_by: str | None = None,
        metrics: list[str] | None = None,
    ) -> dict[str, dict[str, float] | list[dict[str, float]] | int]:
        """Run K-fold cross-validation.

        Parameters
        ----------
        trainer : BaseTrainer
            Trainer to use for each fold.
        items : list[Item]
            All items to evaluate with cross-validation.
        stratify_by : str | None
            Property for stratified sampling.
        metrics : list[str] | None
            Metrics to compute.

        Returns
        -------
        dict[str, dict[str, float] | list[dict[str, float]] | int]
            Aggregated results with mean, std, and individual fold metrics.
            Keys include: 'mean', 'std', 'fold_results', 'n_folds'.

        Examples
        --------
        >>> # Requires a real trainer and items
        >>> # cv = CrossValidator(k=5)
        >>> # results = cv.cross_validate(trainer, items)
        >>> # results['mean']['accuracy']  # doctest: +SKIP
        0.85
        >>> # results['std']['accuracy']  # doctest: +SKIP
        0.03
        """
        # Create folds
        folds = self.k_fold_split(items, stratify_by=stratify_by)

        # Evaluate each fold
        fold_results: list[dict[str, float]] = []
        for train_items, test_items in folds:
            fold_metrics = self.evaluate_fold(trainer, train_items, test_items, metrics)
            fold_results.append(fold_metrics)  # type: ignore[arg-type]

        # Aggregate results
        return self.aggregate_results(fold_results)  # type: ignore[arg-type]

    @staticmethod
    def aggregate_results(
        fold_results: list[dict[str, float]],
    ) -> dict[str, dict[str, float] | list[dict[str, float]] | int]:
        """Aggregate metrics across folds.

        Computes mean, standard deviation, and confidence intervals for
        each metric across all folds.

        Parameters
        ----------
        fold_results : list[dict[str, float]]
            List of metric dictionaries, one per fold.

        Returns
        -------
        dict[str, dict[str, float] | list[dict[str, float]] | int]
            Aggregated results with keys:
            - 'mean': dict[str, float] of mean values per metric
            - 'std': dict[str, float] of standard deviations per metric
            - 'fold_results': list[dict[str, float]] of original fold results
            - 'n_folds': int number of folds

        Examples
        --------
        >>> fold_results = [
        ...     {'accuracy': 0.85, 'f1': 0.82},
        ...     {'accuracy': 0.87, 'f1': 0.84},
        ...     {'accuracy': 0.83, 'f1': 0.80}
        ... ]
        >>> agg = CrossValidator.aggregate_results(fold_results)
        >>> agg['mean']['accuracy']
        0.85
        >>> agg['n_folds']
        3
        """
        if not fold_results:
            return {
                "mean": {},
                "std": {},
                "fold_results": [],
                "n_folds": 0,
            }

        # Extract all metric names
        metric_names = list(fold_results[0].keys())

        # Compute mean and std for each metric
        mean_metrics = {}
        std_metrics = {}

        for metric_name in metric_names:
            values = [fold[metric_name] for fold in fold_results]
            mean_metrics[metric_name] = float(np.mean(values))
            std_metrics[metric_name] = float(np.std(values))

        return {
            "mean": mean_metrics,
            "std": std_metrics,
            "fold_results": fold_results,
            "n_folds": len(fold_results),
        }
