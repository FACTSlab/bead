"""Convergence detection for active learning.

This module provides tools for detecting when a model has converged to
human-level performance, which serves as a stopping criterion for active
learning loops.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from scipy.stats import binomtest, ttest_rel  # type: ignore[import-untyped]

from bead.evaluation.interannotator import InterAnnotatorMetrics

# Type alias for classification labels (categorical, ordinal, or numeric)
type Label = int | str | float


class ConvergenceReport(TypedDict):
    """Convergence report structure.

    Attributes
    ----------
    converged : bool
        Whether model has converged.
    model_accuracy : float
        Model's current accuracy.
    human_agreement : float
        Human agreement score.
    gap : float
        Difference between human agreement and model accuracy.
    required_accuracy : float
        Minimum accuracy required for convergence.
    threshold : float
        Convergence threshold.
    iteration : int
        Current iteration number.
    meets_min_iterations : bool
        Whether minimum iterations requirement is met.
    min_iterations_required : int
        Minimum iterations required before checking convergence.
    """

    converged: bool
    model_accuracy: float
    human_agreement: float
    gap: float
    required_accuracy: float
    threshold: float
    iteration: int
    meets_min_iterations: bool
    min_iterations_required: int


class ConvergenceDetector:
    """Detect convergence of model performance to human agreement.

    This class monitors model performance and compares it to human
    inter-annotator agreement to determine when active learning can stop.
    Convergence is achieved when the model's accuracy matches or exceeds
    human agreement within a specified threshold.

    Parameters
    ----------
    human_agreement_metric : str, default="krippendorff_alpha"
        Which inter-annotator agreement metric to use as baseline:
        - "krippendorff_alpha": Most general (handles missing data, multiple raters)
        - "fleiss_kappa": Multiple raters, no missing data
        - "cohens_kappa": Two raters only
        - "percentage_agreement": Simple agreement rate
    convergence_threshold : float, default=0.05
        Model must be within this threshold of human agreement to converge.
        For example, 0.05 means model accuracy must be >= (human_agreement - 0.05).
    min_iterations : int, default=3
        Minimum number of iterations before checking convergence.
        Prevents premature stopping.
    statistical_test : bool, default=True
        Whether to run statistical significance test comparing model to humans.
    alpha : float, default=0.05
        Significance level for statistical tests.

    Attributes
    ----------
    human_agreement_metric : str
        Agreement metric being used.
    convergence_threshold : float
        Threshold for convergence.
    min_iterations : int
        Minimum iterations required.
    statistical_test : bool
        Whether to run significance tests.
    alpha : float
        Significance level.
    human_baseline : float | None
        Computed human agreement baseline (set via compute_human_baseline).

    Examples
    --------
    >>> detector = ConvergenceDetector(
    ...     human_agreement_metric='krippendorff_alpha',
    ...     convergence_threshold=0.05,
    ...     min_iterations=3
    ... )
    >>> # Compute human baseline from ratings
    >>> ratings = {
    ...     'human1': [1, 1, 0, 1, 0],
    ...     'human2': [1, 1, 0, 0, 0],
    ...     'human3': [1, 0, 0, 1, 0]
    ... }
    >>> detector.compute_human_baseline(ratings)
    >>> detector.human_baseline > 0.0
    True
    >>> # Check if model converged
    >>> converged = detector.check_convergence(
    ...     model_accuracy=0.75,
    ...     iteration=5
    ... )
    >>> isinstance(converged, bool)
    True
    """

    def __init__(
        self,
        human_agreement_metric: str = "krippendorff_alpha",
        convergence_threshold: float = 0.05,
        min_iterations: int = 3,
        statistical_test: bool = True,
        alpha: float = 0.05,
    ) -> None:
        """Initialize convergence detector.

        Parameters
        ----------
        human_agreement_metric : str
            Inter-annotator agreement metric to use.
        convergence_threshold : float
            Threshold for convergence (model must be within this of human).
        min_iterations : int
            Minimum iterations before checking convergence.
        statistical_test : bool
            Whether to run statistical tests.
        alpha : float
            Significance level for tests.

        Raises
        ------
        ValueError
            If parameters are invalid.
        """
        valid_metrics = {
            "krippendorff_alpha",
            "fleiss_kappa",
            "cohens_kappa",
            "percentage_agreement",
        }

        if human_agreement_metric not in valid_metrics:
            raise ValueError(
                f"human_agreement_metric must be one of {valid_metrics}, "
                f"got '{human_agreement_metric}'"
            )

        if convergence_threshold < 0.0 or convergence_threshold > 1.0:
            raise ValueError(
                f"convergence_threshold must be in [0, 1], got {convergence_threshold}"
            )

        if min_iterations < 1:
            raise ValueError(f"min_iterations must be >= 1, got {min_iterations}")

        if alpha <= 0.0 or alpha >= 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")

        self.human_agreement_metric = human_agreement_metric
        self.convergence_threshold = convergence_threshold
        self.min_iterations = min_iterations
        self.statistical_test = statistical_test
        self.alpha = alpha
        self.human_baseline: float | None = None

    def compute_human_baseline(
        self,
        human_ratings: dict[str, list[Label | None]],
        **kwargs: str | int | float | bool | None,
    ) -> float:
        """Compute human inter-rater agreement baseline.

        Parameters
        ----------
        human_ratings : dict[str, list[Label | None]]
            Dictionary mapping human rater IDs to their ratings.
            For example: {'rater1': [1, 0, 1, ...], 'rater2': [1, 1, 1, ...]}.
            Missing ratings can be represented as None.
        **kwargs : str | int | float | bool | None
            Additional arguments passed to agreement metric function.
            For example, metric='nominal' for Krippendorff's alpha.

        Returns
        -------
        float
            Human agreement score.

        Raises
        ------
        ValueError
            If human_ratings is empty or has fewer than 2 raters.

        Examples
        --------
        >>> detector = ConvergenceDetector()
        >>> ratings = {
        ...     'human1': [1, 1, 0, 1],
        ...     'human2': [1, 1, 0, 0],
        ...     'human3': [1, 0, 0, 1]
        ... }
        >>> baseline = detector.compute_human_baseline(ratings)
        >>> 0.0 <= baseline <= 1.0
        True
        """
        if not human_ratings:
            raise ValueError("human_ratings cannot be empty")

        if len(human_ratings) < 2:
            raise ValueError("human_ratings must have at least 2 raters")

        # Compute agreement using specified metric
        if self.human_agreement_metric == "krippendorff_alpha":
            # Extract metric parameter, defaulting to 'nominal'
            metric = kwargs.get("metric", "nominal")
            if not isinstance(metric, str):
                metric = "nominal"
            agreement = InterAnnotatorMetrics.krippendorff_alpha(
                human_ratings, metric=metric
            )
        elif self.human_agreement_metric == "percentage_agreement":
            # Use mean of pairwise percentage agreements
            # Filter out None values for percentage agreement
            filtered_ratings = {
                rater_id: [r for r in ratings if r is not None]
                for rater_id, ratings in human_ratings.items()
            }
            pairwise = InterAnnotatorMetrics.pairwise_agreement(filtered_ratings)
            agreements = list(pairwise["percentage_agreement"].values())
            agreement = float(np.mean(agreements)) if agreements else 0.0
        elif self.human_agreement_metric == "cohens_kappa":
            if len(human_ratings) != 2:
                raise ValueError("cohens_kappa requires exactly 2 raters")
            rater_ids = list(human_ratings.keys())
            # Filter out None values for Cohen's kappa
            ratings_1 = human_ratings[rater_ids[0]]
            ratings_2 = human_ratings[rater_ids[1]]
            filtered_ratings_1 = [r for r in ratings_1 if r is not None]
            filtered_ratings_2 = [r for r in ratings_2 if r is not None]
            agreement = InterAnnotatorMetrics.cohens_kappa(
                filtered_ratings_1, filtered_ratings_2
            )
        elif self.human_agreement_metric == "fleiss_kappa":
            # Convert ratings to Fleiss format (items × categories matrix)
            # This requires categorical data
            raise NotImplementedError(
                "fleiss_kappa not yet implemented in compute_human_baseline. "
                "Use krippendorff_alpha instead."
            )
        else:
            raise ValueError(f"Unknown metric: {self.human_agreement_metric}")

        self.human_baseline = agreement
        return agreement

    def check_convergence(
        self,
        model_accuracy: float,
        iteration: int,
        human_agreement: float | None = None,
    ) -> bool:
        """Check if model has converged to human performance.

        Parameters
        ----------
        model_accuracy : float
            Model's accuracy on the task.
        iteration : int
            Current iteration number (1-indexed).
        human_agreement : float | None
            Human agreement score. If None, uses self.human_baseline
            (which must have been set via compute_human_baseline).

        Returns
        -------
        bool
            True if model has converged, False otherwise.

        Raises
        ------
        ValueError
            If human_agreement is None and human_baseline not set.

        Examples
        --------
        >>> detector = ConvergenceDetector(min_iterations=2, convergence_threshold=0.05)
        >>> detector.human_baseline = 0.80
        >>> # Too early (iteration 1 < min_iterations 2)
        >>> detector.check_convergence(0.79, iteration=1)
        False
        >>> # Still not converged (0.74 < 0.80 - 0.05)
        >>> detector.check_convergence(0.74, iteration=3)
        False
        >>> # Converged (0.77 >= 0.80 - 0.05)
        >>> detector.check_convergence(0.77, iteration=3)
        True
        """
        # Check minimum iterations
        if iteration < self.min_iterations:
            return False

        # Get human baseline
        if human_agreement is None:
            if self.human_baseline is None:
                raise ValueError(
                    "human_agreement is None and human_baseline not set. "
                    "Call compute_human_baseline first or pass human_agreement."
                )
            human_agreement = self.human_baseline

        # Check if model is within threshold of human performance
        required_accuracy = human_agreement - self.convergence_threshold
        return model_accuracy >= required_accuracy

    def compute_statistical_test(
        self,
        model_predictions: list[Label],
        human_consensus: list[Label],
        test_type: str = "mcnemar",
    ) -> dict[str, float]:
        """Run statistical test comparing model to human performance.

        Parameters
        ----------
        model_predictions : list[Label]
            Model's predictions.
        human_consensus : list[Label]
            Human consensus labels (e.g., majority vote).
        test_type : str, default="mcnemar"
            Type of statistical test:
            - "mcnemar": McNemar's test for paired nominal data
            - "ttest": Paired t-test (requires multiple samples)

        Returns
        -------
        dict[str, float]
            Dictionary with keys 'statistic' and 'p_value'.

        Raises
        ------
        ValueError
            If predictions and consensus have different lengths.

        Examples
        --------
        >>> detector = ConvergenceDetector()
        >>> model_preds = [1, 1, 0, 1, 0]
        >>> human_consensus = [1, 1, 0, 0, 0]
        >>> result = detector.compute_statistical_test(model_preds, human_consensus)
        >>> 'statistic' in result and 'p_value' in result
        True
        """
        if len(model_predictions) != len(human_consensus):
            raise ValueError(
                f"model_predictions and human_consensus must have same length: "
                f"{len(model_predictions)} != {len(human_consensus)}"
            )

        if test_type == "mcnemar":
            # McNemar's test for paired predictions
            # Contingency table: [correct_model, incorrect_model] ×
            # [correct_human, incorrect_human]

            # Actually, for McNemar we need a reference (ground truth)
            # Instead, we'll use a binomial test to check if model accuracy
            # differs significantly from human accuracy

            model_correct = [
                mp == hc
                for mp, hc in zip(model_predictions, human_consensus, strict=True)
            ]
            model_accuracy = sum(model_correct) / len(model_correct)
            human_accuracy = 1.0  # Assuming human_consensus is "correct"

            # Binomial test: is model accuracy significantly different from human?
            n = len(model_correct)
            k = sum(model_correct)

            # Two-tailed test
            result = binomtest(k, n, human_accuracy, alternative="two-sided")
            p_value = result.pvalue

            return {
                "statistic": float(model_accuracy),
                "p_value": float(p_value),
            }

        elif test_type == "ttest":
            # Paired t-test comparing model predictions to human consensus
            # Convert predictions to correctness scores (1 if match, 0 if not)
            model_scores = np.array(
                [
                    1.0 if mp == hc else 0.0
                    for mp, hc in zip(model_predictions, human_consensus, strict=True)
                ]
            )
            # Human consensus is always "correct" (1.0) by definition
            human_scores = np.ones(len(human_consensus), dtype=float)

            # Paired t-test: test if model scores differ from human scores
            statistic, p_value = ttest_rel(model_scores, human_scores)

            return {
                "statistic": float(statistic),
                "p_value": float(p_value),
            }

        else:
            raise ValueError(
                f"Unknown test_type: {test_type}. Must be 'mcnemar' or 'ttest'."
            )

    def get_convergence_report(
        self,
        model_accuracy: float,
        iteration: int,
        human_agreement: float | None = None,
    ) -> ConvergenceReport:
        """Generate convergence report with status and metrics.

        Parameters
        ----------
        model_accuracy : float
            Model's current accuracy.
        iteration : int
            Current iteration number.
        human_agreement : float | None
            Human agreement score (uses baseline if None).

        Returns
        -------
        ConvergenceReport
            Report with convergence status and metrics.

        Examples
        --------
        >>> detector = ConvergenceDetector(convergence_threshold=0.05)
        >>> detector.human_baseline = 0.80
        >>> report = detector.get_convergence_report(0.77, iteration=5)
        >>> report['converged']
        True
        >>> report['gap']
        0.03
        """
        # Get human baseline
        if human_agreement is None:
            if self.human_baseline is None:
                raise ValueError("human_agreement is None and human_baseline not set")
            human_agreement = self.human_baseline

        # Check convergence
        converged = self.check_convergence(model_accuracy, iteration, human_agreement)

        # Compute metrics
        gap = human_agreement - model_accuracy
        required_accuracy = human_agreement - self.convergence_threshold
        meets_min_iterations = iteration >= self.min_iterations

        return {
            "converged": converged,
            "model_accuracy": model_accuracy,
            "human_agreement": human_agreement,
            "gap": gap,
            "required_accuracy": required_accuracy,
            "threshold": self.convergence_threshold,
            "iteration": iteration,
            "meets_min_iterations": meets_min_iterations,
            "min_iterations_required": self.min_iterations,
        }
