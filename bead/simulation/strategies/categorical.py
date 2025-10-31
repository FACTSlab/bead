"""Categorical choice simulation strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bead.simulation.strategies.base import SimulationStrategy

if TYPE_CHECKING:
    from bead.items.models import Item, ItemTemplate


class CategoricalStrategy(SimulationStrategy):
    """Strategy for categorical tasks (unordered multi-class).

    Similar to forced_choice but for unordered categories (e.g., NLI labels,
    sentiment classes). Uses softmax over model outputs.

    For categorical with LM scores:
        P(category_i) = softmax([score_1, ..., score_n] / temperature)[i]

    Examples
    --------
    >>> strategy = CategoricalStrategy()
    >>> strategy.supported_task_type
    'categorical'
    """

    @property
    def supported_task_type(self) -> str:
        """Return 'categorical'.

        Returns
        -------
        str
            Task type identifier.
        """
        return "categorical"

    def validate_item(self, item: Item, item_template: ItemTemplate) -> None:
        """Validate item for categorical choice.

        Checks:
        - task_type is 'categorical'
        - task_spec.options is defined
        - At least 2 options available

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
        if item_template.task_type != "categorical":
            msg = f"Expected task_type 'categorical', got '{item_template.task_type}'"
            raise ValueError(msg)

        if not item_template.task_spec.options:
            msg = "task_spec.options must be defined for categorical"
            raise ValueError(msg)

        if len(item_template.task_spec.options) < 2:
            msg = "categorical requires at least 2 options"
            raise ValueError(msg)

    def simulate_response(
        self,
        item: Item,
        item_template: ItemTemplate,
        model_output_key: str,
        rng: np.random.RandomState,
    ) -> str:
        """Generate categorical response.

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
            Chosen category name.
        """
        options = item_template.task_spec.options
        if options is None:
            msg = "task_spec.options must be defined"
            raise ValueError(msg)

        n_options = len(options)

        # Extract model outputs for each category
        scores = self.extract_model_outputs(item, model_output_key, n_options)

        if scores is None:
            # Fallback to uniform random
            choice_idx = rng.randint(0, n_options)
            return options[choice_idx]

        # Convert scores to probabilities using softmax
        scores_array = np.array(scores)
        exp_scores = np.exp(scores_array - np.max(scores_array))  # Numerical stability
        probs = exp_scores / np.sum(exp_scores)

        # Sample from distribution
        choice_idx = rng.choice(n_options, p=probs)

        return options[choice_idx]
