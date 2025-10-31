"""Sampling strategies for active learning.

This module implements various uncertainty quantification methods for
active learning item selection, including entropy, margin, and least
confidence sampling, plus a random baseline.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Literal

import numpy as np


class SamplingStrategy(ABC):
    """Base class for active learning sampling strategies.

    All sampling strategies must implement compute_scores to quantify
    uncertainty or informativeness of predictions, and select_top_k
    to select the most informative items.

    Examples
    --------
    >>> import numpy as np
    >>> class MyStrategy(SamplingStrategy):
    ...     def compute_scores(self, probabilities):
    ...         return np.max(probabilities, axis=1)
    >>> strategy = MyStrategy()
    >>> probs = np.array([[0.7, 0.2, 0.1], [0.4, 0.4, 0.2]])
    >>> scores = strategy.compute_scores(probs)
    >>> indices = strategy.select_top_k(scores, k=1)
    >>> len(indices)
    1
    """

    @abstractmethod
    def compute_scores(self, probabilities: np.ndarray) -> np.ndarray:
        """Compute uncertainty scores from prediction probabilities.

        Parameters
        ----------
        probabilities : np.ndarray
            Prediction probabilities with shape (n_samples, n_classes).
            Each row should sum to 1.0.

        Returns
        -------
        np.ndarray
            Uncertainty scores with shape (n_samples,).
            Higher scores indicate more informative/uncertain items.

        Examples
        --------
        >>> strategy = UncertaintySampling()  # doctest: +SKIP
        >>> probs = np.array([[0.5, 0.5], [0.9, 0.1]])  # doctest: +SKIP
        >>> scores = strategy.compute_scores(probs)  # doctest: +SKIP
        >>> scores[0] > scores[1]  # First is more uncertain  # doctest: +SKIP
        True
        """
        pass

    def select_top_k(self, scores: np.ndarray, k: int) -> np.ndarray:
        """Select top k items by score.

        Parameters
        ----------
        scores : np.ndarray
            Uncertainty scores with shape (n_samples,).
        k : int
            Number of items to select.

        Returns
        -------
        np.ndarray
            Indices of top k items with shape (k,).
            If k > len(scores), returns all indices.
            If k <= 0, returns empty array.

        Examples
        --------
        >>> strategy = UncertaintySampling()
        >>> scores = np.array([0.5, 0.9, 0.3, 0.7])
        >>> indices = strategy.select_top_k(scores, k=2)
        >>> list(indices)
        [1, 3]
        """
        if k <= 0:
            return np.array([], dtype=int)

        if k >= len(scores):
            return np.arange(len(scores))

        # Get indices of top k scores (descending order)
        return np.argsort(scores)[-k:][::-1]


class UncertaintySampling(SamplingStrategy):
    """Entropy-based uncertainty sampling.

    Selects items where the model's prediction entropy is highest,
    indicating maximum uncertainty across all classes.

    Mathematical definition:
        H(p) = -∑(p_i * log(p_i))

    where p is the probability distribution over classes.

    Examples
    --------
    >>> import numpy as np
    >>> strategy = UncertaintySampling()
    >>> # Uniform distribution (high entropy)
    >>> probs = np.array([[0.33, 0.33, 0.34]])
    >>> score = strategy.compute_scores(probs)
    >>> score[0] > 1.0  # High uncertainty
    True
    >>> # Confident prediction (low entropy)
    >>> probs = np.array([[0.9, 0.05, 0.05]])
    >>> score = strategy.compute_scores(probs)
    >>> score[0] < 0.5  # Low uncertainty
    True
    """

    def compute_scores(self, probabilities: np.ndarray) -> np.ndarray:
        """Compute entropy for each prediction.

        Parameters
        ----------
        probabilities : np.ndarray
            Prediction probabilities with shape (n_samples, n_classes).

        Returns
        -------
        np.ndarray
            Entropy scores with shape (n_samples,).
            Higher entropy indicates more uncertainty.

        Examples
        --------
        >>> strategy = UncertaintySampling()
        >>> probs = np.array([[0.5, 0.5], [0.9, 0.1]])
        >>> scores = strategy.compute_scores(probs)
        >>> scores[0] > scores[1]  # Uniform is more uncertain
        True
        """
        # Add small epsilon to avoid log(0)
        epsilon = 1e-10
        probs_safe = np.clip(probabilities, epsilon, 1.0)

        # Compute entropy: -∑(p * log(p))
        entropy = -np.sum(probs_safe * np.log(probs_safe), axis=1)

        return entropy


class MarginSampling(SamplingStrategy):
    """Margin-based uncertainty sampling.

    Selects items where the margin between the top two predicted classes
    is smallest, indicating uncertainty between the two most likely options.

    Mathematical definition:
        margin(p) = 1 - (p₁ - p₂)

    where p₁ and p₂ are the highest and second-highest probabilities.

    Examples
    --------
    >>> import numpy as np
    >>> strategy = MarginSampling()
    >>> # Small margin (uncertain)
    >>> probs = np.array([[0.51, 0.49, 0.0]])
    >>> score = strategy.compute_scores(probs)
    >>> score[0] > 0.95  # High uncertainty
    True
    >>> # Large margin (confident)
    >>> probs = np.array([[0.9, 0.05, 0.05]])
    >>> score = strategy.compute_scores(probs)
    >>> score[0] < 0.2  # Low uncertainty
    True
    """

    def compute_scores(self, probabilities: np.ndarray) -> np.ndarray:
        """Compute margin scores for each prediction.

        Parameters
        ----------
        probabilities : np.ndarray
            Prediction probabilities with shape (n_samples, n_classes).

        Returns
        -------
        np.ndarray
            Margin scores with shape (n_samples,).
            Higher scores indicate smaller margin (more uncertainty).

        Examples
        --------
        >>> strategy = MarginSampling()
        >>> probs = np.array([[0.6, 0.3, 0.1], [0.8, 0.15, 0.05]])
        >>> scores = strategy.compute_scores(probs)
        >>> scores[0] > scores[1]  # First has smaller margin
        True
        """
        # Sort probabilities in descending order
        sorted_probs = np.sort(probabilities, axis=1)

        # Get top 2 probabilities
        top1 = sorted_probs[:, -1]
        top2 = sorted_probs[:, -2]

        # Compute margin: 1 - (p1 - p2)
        margin = 1.0 - (top1 - top2)

        return margin


class LeastConfidenceSampling(SamplingStrategy):
    """Least confidence sampling.

    Selects items where the model is least confident, measured as
    1 minus the maximum predicted probability.

    Mathematical definition:
        lc(p) = 1 - max(p)

    where p is the probability distribution over classes.

    Examples
    --------
    >>> import numpy as np
    >>> strategy = LeastConfidenceSampling()
    >>> # Low confidence
    >>> probs = np.array([[0.4, 0.3, 0.3]])
    >>> score = strategy.compute_scores(probs)
    >>> score[0] == 0.6  # 1 - 0.4
    True
    >>> # High confidence
    >>> probs = np.array([[0.95, 0.03, 0.02]])
    >>> score = strategy.compute_scores(probs)
    >>> score[0] == 0.05  # 1 - 0.95
    True
    """

    def compute_scores(self, probabilities: np.ndarray) -> np.ndarray:
        """Compute least confidence scores for each prediction.

        Parameters
        ----------
        probabilities : np.ndarray
            Prediction probabilities with shape (n_samples, n_classes).

        Returns
        -------
        np.ndarray
            Least confidence scores with shape (n_samples,).
            Higher scores indicate lower confidence (more uncertainty).

        Examples
        --------
        >>> strategy = LeastConfidenceSampling()
        >>> probs = np.array([[0.5, 0.5], [0.9, 0.1]])
        >>> scores = strategy.compute_scores(probs)
        >>> scores[0] > scores[1]  # First is less confident
        True
        """
        # Get maximum probability for each sample
        max_probs = np.max(probabilities, axis=1)

        # Compute least confidence: 1 - max(p)
        least_confidence = 1.0 - max_probs

        return least_confidence


class RandomSampling(SamplingStrategy):
    """Random sampling baseline.

    Selects items randomly, serving as a baseline for comparison
    with uncertainty-based methods. Uses seeded random number generation
    for reproducibility.

    Parameters
    ----------
    seed : int | None
        Random seed for reproducibility. If None, uses non-deterministic seed.

    Attributes
    ----------
    rng : np.random.Generator
        Random number generator.

    Examples
    --------
    >>> import numpy as np
    >>> strategy = RandomSampling(seed=42)
    >>> probs = np.array([[0.9, 0.1], [0.5, 0.5]])
    >>> scores = strategy.compute_scores(probs)
    >>> len(scores) == 2
    True
    >>> # Scores are random, not based on probabilities
    >>> strategy2 = RandomSampling(seed=42)
    >>> scores2 = strategy2.compute_scores(probs)
    >>> np.allclose(scores, scores2)  # Same seed gives same results
    True
    """

    def __init__(self, seed: int | None = None) -> None:
        """Initialize random sampling strategy.

        Parameters
        ----------
        seed : int | None
            Random seed for reproducibility.
        """
        self.rng = np.random.default_rng(seed)

    def compute_scores(self, probabilities: np.ndarray) -> np.ndarray:
        """Generate random scores for each item.

        Parameters
        ----------
        probabilities : np.ndarray
            Prediction probabilities with shape (n_samples, n_classes).
            Not used in random sampling, but required by interface.

        Returns
        -------
        np.ndarray
            Random scores with shape (n_samples,).

        Examples
        --------
        >>> strategy = RandomSampling(seed=123)
        >>> probs = np.array([[0.9, 0.1], [0.1, 0.9]])
        >>> scores = strategy.compute_scores(probs)
        >>> len(scores) == 2
        True
        >>> 0.0 <= scores[0] <= 1.0
        True
        """
        n_samples = probabilities.shape[0]
        return self.rng.random(n_samples)


# Type alias for strategy methods
StrategyMethod = Literal["entropy", "margin", "least_confidence", "random"]


def create_strategy(
    method: StrategyMethod, seed: int | None = None
) -> SamplingStrategy:
    """Create a sampling strategy instance.

    Parameters
    ----------
    method : StrategyMethod
        Strategy method name ("entropy", "margin", "least_confidence", "random").
    seed : int | None
        Random seed for random strategy. Ignored for other strategies.

    Returns
    -------
    SamplingStrategy
        Instantiated sampling strategy.

    Raises
    ------
    ValueError
        If method is not recognized.

    Examples
    --------
    >>> strategy = create_strategy("entropy")
    >>> isinstance(strategy, UncertaintySampling)
    True
    >>> strategy = create_strategy("margin")
    >>> isinstance(strategy, MarginSampling)
    True
    >>> strategy = create_strategy("least_confidence")
    >>> isinstance(strategy, LeastConfidenceSampling)
    True
    >>> strategy = create_strategy("random", seed=42)
    >>> isinstance(strategy, RandomSampling)
    True
    """
    match method:
        case "entropy":
            return UncertaintySampling()
        case "margin":
            return MarginSampling()
        case "least_confidence":
            return LeastConfidenceSampling()
        case "random":
            return RandomSampling(seed=seed)
        case _:
            raise ValueError(f"Unknown sampling method: {method}")
