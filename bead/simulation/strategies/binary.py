"""Binary choice simulation strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bead.simulation.strategies.base import SimulationStrategy

if TYPE_CHECKING:
    from bead.items.models import Item, ItemTemplate


class BinaryStrategy(SimulationStrategy):
    """Strategy for binary tasks (yes/no, true/false).

    Uses model outputs to compute probability of "yes" response,
    then samples from Bernoulli distribution.

    For binary tasks with LM score:
        P(yes) = sigmoid(score / temperature)

    Examples
    --------
    >>> strategy = BinaryStrategy()
    >>> strategy.supported_task_type
    'binary'
    """

    @property
    def supported_task_type(self) -> str:
        """Return 'binary'.

        Returns
        -------
        str
            Task type identifier.
        """
        return "binary"

    def validate_item(self, item: Item, item_template: ItemTemplate) -> None:
        """Validate item for binary choice.

        Checks:
        - task_type is 'binary'
        - Item has appropriate model outputs OR can fall back

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
        if item_template.task_type != "binary":
            msg = f"Expected task_type 'binary', got '{item_template.task_type}'"
            raise ValueError(msg)

    def simulate_response(
        self,
        item: Item,
        item_template: ItemTemplate,
        model_output_key: str,
        rng: np.random.RandomState,
    ) -> bool:
        """Generate binary response.

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
        bool
            Binary response (True/False).
        """
        # Extract model output (expecting single score)
        scores = self.extract_model_outputs(item, model_output_key, required_count=1)

        if scores is None:
            # Fallback to uniform random (50/50)
            return bool(rng.rand() > 0.5)

        # Convert score to probability using sigmoid
        score = scores[0]
        # Sigmoid: 1 / (1 + exp(-x))
        prob_yes = 1.0 / (1.0 + np.exp(-score))

        # Sample from Bernoulli
        return bool(rng.rand() < prob_yes)
