"""Temperature-based noise model."""

from __future__ import annotations

import numpy as np

from bead.simulation.noise_models.base import NoiseModel


class TemperatureNoiseModel(NoiseModel):
    """Temperature scaling for probability distributions.

    Scales logits or probabilities by temperature before sampling:
        - temperature < 1.0: More deterministic (sharper distribution)
        - temperature = 1.0: No change
        - temperature > 1.0: More random (flatter distribution)

    For forced choice, modifies the softmax:
        P_i = exp(score_i / T) / sum(exp(score_j / T))

    Examples
    --------
    >>> noise_model = TemperatureNoiseModel(temperature=2.0)
    >>> # More random decisions
    """

    def __init__(self, temperature: float = 1.0) -> None:
        """Initialize temperature noise model.

        Parameters
        ----------
        temperature : float
            Temperature scaling factor (> 0).

        Raises
        ------
        ValueError
            If temperature <= 0.
        """
        if temperature <= 0:
            msg = "Temperature must be positive"
            raise ValueError(msg)
        self.temperature = temperature

    def apply(
        self,
        value: str | int | float | list[str],
        context: dict[str, object],
        rng: np.random.RandomState,
    ) -> str | int | float | list[str]:
        """Apply temperature scaling.

        For forced_choice, re-samples with scaled probabilities.
        For ordinal_scale, adds scaled noise to value.

        Parameters
        ----------
        value : str | int | float | list[str]
            Original value (choice, rating, etc.).
        context : dict[str, object]
            Context with item, template, strategy.
        rng : np.random.RandomState
            Random number generator.

        Returns
        -------
        str | int | float | list[str]
            Value with temperature applied.
        """
        strategy = context.get("strategy")

        if strategy and hasattr(strategy, "supported_task_type"):
            task_type = strategy.supported_task_type

            if task_type == "forced_choice":
                # For forced choice, temperature is already handled in strategy
                # by applying it to the softmax computation
                # Here we just return the value as-is
                return value

            elif task_type == "ordinal_scale":
                # For ordinal, add temperature-scaled gaussian noise
                if isinstance(value, (int, float)):
                    noise = rng.normal(0, self.temperature * 0.5)
                    return value + noise
                return value

        # Default: no modification
        return value
