"""Random noise injection model."""

from __future__ import annotations

import numpy as np

from bead.simulation.noise_models.base import NoiseModel


class RandomNoiseModel(NoiseModel):
    """Random noise injection model.

    Adds random noise to responses:
    - Gaussian noise for numeric values
    - Uniform noise for numeric values
    - Random flipping for choice tasks

    Examples
    --------
    >>> noise_model = RandomNoiseModel(noise_type="gaussian", strength=0.5)
    >>> # Adds gaussian noise with stddev=0.5 to numeric responses
    """

    def __init__(self, noise_type: str = "gaussian", strength: float = 1.0) -> None:
        """Initialize random noise model.

        Parameters
        ----------
        noise_type : str
            Type of noise ("gaussian" or "uniform").
        strength : float
            Noise strength (stddev for gaussian, range for uniform).
        """
        self.noise_type = noise_type
        self.strength = strength

    def apply(
        self,
        value: str | int | float | bool | list[str],
        context: dict[str, str | int | float | bool | list[str]],
        rng: np.random.RandomState,
    ) -> str | int | float | bool | list[str]:
        """Apply random noise.

        Parameters
        ----------
        value
            Original value.
        context : dict
            Context with item, template, strategy.
        rng : np.random.RandomState
            Random number generator.

        Returns
        -------
        str | int | float | bool | list[str]
            Value with noise applied.
        """
        if self.strength == 0.0:
            return value

        # Apply noise based on value type
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return self._add_numeric_noise(value, rng)
        else:
            # For non-numeric, return as-is
            return value

    def _add_numeric_noise(
        self, value: int | float, rng: np.random.RandomState
    ) -> int | float:
        """Add noise to numeric value."""
        if self.noise_type == "gaussian":
            noisy_value = value + rng.normal(0, self.strength)
        elif self.noise_type == "uniform":
            noisy_value = value + rng.uniform(-self.strength, self.strength)
        else:
            noisy_value = value

        # Preserve type
        if isinstance(value, int):
            return int(round(noisy_value))
        else:
            return float(noisy_value)
