"""Forced choice simulation strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bead.simulation.strategies.base import SimulationStrategy

if TYPE_CHECKING:
    from bead.items.models import Item, ItemTemplate


class ForcedChoiceStrategy(SimulationStrategy):
    """Strategy for forced_choice tasks (n-AFC).

    Handles 2AFC, 3AFC, 4AFC, etc. Uses model outputs to compute
    preference probabilities, then samples categorically.

    For 2AFC with LM scores:
        P(choose A) = sigmoid((score_A - score_B) / temperature)

    For n-AFC with LM scores:
        P(choose i) = softmax([score_1, ..., score_n] / temperature)[i]

    Examples
    --------
    >>> strategy = ForcedChoiceStrategy()
    >>> strategy.supported_task_type
    'forced_choice'
    """

    @property
    def supported_task_type(self) -> str:
        """Return 'forced_choice'.

        Returns
        -------
        str
            Task type identifier.
        """
        return "forced_choice"

    def validate_item(self, item: Item, item_template: ItemTemplate) -> None:
        """Validate item for forced choice.

        Checks:
        - task_type is 'forced_choice'
        - task_spec.options is defined
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
        if item_template.task_type != "forced_choice":
            msg = f"Expected task_type 'forced_choice', got '{item_template.task_type}'"
            raise ValueError(msg)

        if not item_template.task_spec.options:
            msg = "task_spec.options must be defined for forced_choice"
            raise ValueError(msg)

        if len(item_template.task_spec.options) < 2:
            msg = "forced_choice requires at least 2 options"
            raise ValueError(msg)

    def simulate_response(
        self,
        item: Item,
        item_template: ItemTemplate,
        model_output_key: str,
        rng: np.random.RandomState,
    ) -> str:
        """Generate forced choice response.

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
        str
            Chosen option name.
        """
        options = item_template.task_spec.options
        if options is None:
            msg = "task_spec.options must be defined"
            raise ValueError(msg)

        n_options = len(options)

        # Extract model outputs for each option
        scores = self.extract_model_outputs(item, model_output_key, n_options)

        if scores is None:
            # Fallback to uniform random
            choice_idx = rng.randint(0, n_options)
            return options[choice_idx]

        # Convert scores to probabilities using softmax
        # (will be scaled by noise model later)
        scores_array = np.array(scores)
        exp_scores = np.exp(scores_array - np.max(scores_array))  # Numerical stability
        probs = exp_scores / np.sum(exp_scores)

        # Sample from distribution
        choice_idx = rng.choice(n_options, p=probs)

        return options[choice_idx]
