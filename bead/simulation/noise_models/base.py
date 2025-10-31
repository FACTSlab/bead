"""Base class for noise models."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class NoiseModel(ABC):
    """Abstract base for noise models.

    Noise models add human-like variability to simulated responses.
    They can:
    - Scale probabilities by temperature
    - Add systematic biases (length, frequency, position)
    - Inject random noise
    """

    @abstractmethod
    def apply(
        self,
        value: str | int | float | list[str],
        context: dict[str, object],
        rng: np.random.RandomState,
    ) -> str | int | float | list[str]:
        """Apply noise to value.

        Parameters
        ----------
        value : str | int | float | list[str]
            Original value (probability, score, choice, etc.).
        context : dict[str, object]
            Additional context (item, template, strategy, etc.).
        rng : np.random.RandomState
            Random number generator.

        Returns
        -------
        str | int | float | list[str]
            Value with noise applied.
        """
