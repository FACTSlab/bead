"""Magnitude estimation simulation strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bead.simulation.strategies.base import SimulationStrategy

if TYPE_CHECKING:
    from bead.items.models import Item, ItemTemplate


class MagnitudeStrategy(SimulationStrategy):
    """Strategy for magnitude estimation tasks.

    Handles unbounded numeric magnitude estimation. Converts model outputs
    (typically LM scores) to positive magnitude values.

    For LM scores (typically negative log probabilities):
        magnitude = exp(-score / scale_factor)

    This maps:
        - Better scores (less negative) -> larger magnitudes
        - Worse scores (more negative) -> smaller magnitudes

    Examples
    --------
    >>> strategy = MagnitudeStrategy()
    >>> strategy.supported_task_type
    'magnitude'
    """

    def __init__(self, scale_factor: float = 10.0) -> None:
        """Initialize magnitude strategy.

        Parameters
        ----------
        scale_factor : float
            Scaling factor for converting scores to magnitudes.
            Higher values produce more variation. Default: 10.0.
        """
        self.scale_factor = scale_factor

    @property
    def supported_task_type(self) -> str:
        """Return 'magnitude'."""
        return "magnitude"

    def validate_item(self, item: Item, item_template: ItemTemplate) -> None:
        """Validate item for magnitude estimation.

        Checks:
        - task_type is 'magnitude'
        - Item has model outputs OR can fall back

        Parameters
        ----------
        item : Item
            Item to validate.
        item_template : ItemTemplate
            Template defining task.

        Raises
        ------
        ValueError
            If validation fails.
        """
        if item_template.task_type != "magnitude":
            raise ValueError(
                f"Expected task_type 'magnitude', got '{item_template.task_type}'"
            )

    def simulate_response(
        self,
        item: Item,
        item_template: ItemTemplate,
        model_output_key: str,
        rng: np.random.RandomState,
    ) -> float:
        """Generate magnitude estimation response.

        Parameters
        ----------
        item : Item
            Item to respond to.
        item_template : ItemTemplate
            Template defining task.
        model_output_key : str
            Key for model outputs (e.g., "lm_score").
        rng : np.random.RandomState
            Random number generator.

        Returns
        -------
        float
            Estimated magnitude (positive value).
        """
        # Extract model output (expect single value)
        scores = self.extract_model_outputs(item, model_output_key, required_count=1)

        if scores is None:
            # Fallback to random positive value (log-normal)
            return float(rng.lognormal(mean=0, sigma=1))

        score = scores[0]

        # Convert score to magnitude
        # For LM scores (negative), exp(-score/scale) gives positive magnitude
        # For positive scores, use exp(score/scale)
        if score < 0:
            magnitude = np.exp(-score / self.scale_factor)
        else:
            magnitude = np.exp(score / self.scale_factor)

        return float(magnitude)
