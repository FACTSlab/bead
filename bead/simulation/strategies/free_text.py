"""Free text simulation strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bead.simulation.strategies.base import SimulationStrategy

if TYPE_CHECKING:
    from bead.items.item import Item
    from bead.items.item_template import ItemTemplate


class FreeTextStrategy(SimulationStrategy):
    """Strategy for free_text tasks.

    Handles free text generation using rule-based approaches.
    For simulations, this typically:
    - Extracts text from rendered_elements
    - Uses templates if provided
    - Falls back to simple defaults

    Note: This is a simplified implementation for simulation purposes.
    For realistic free text generation, consider using LLMs.

    Examples
    --------
    >>> strategy = FreeTextStrategy()
    >>> strategy.supported_task_type
    'free_text'
    """

    @property
    def supported_task_type(self) -> str:
        """Return 'free_text'."""
        return "free_text"

    def validate_item(self, item: Item, item_template: ItemTemplate) -> None:
        """Validate item for free text.

        Checks:
        - task_type is 'free_text'

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
        if item_template.task_type != "free_text":
            raise ValueError(
                f"Expected task_type 'free_text', got '{item_template.task_type}'"
            )

    def simulate_response(
        self,
        item: Item,
        item_template: ItemTemplate,
        model_output_key: str,
        rng: np.random.RandomState,
    ) -> str:
        """Generate free text response.

        Parameters
        ----------
        item : Item
            Item to respond to.
        item_template : ItemTemplate
            Template defining task.
        model_output_key : str
            Key for model outputs (unused for free text).
        rng : np.random.RandomState
            Random number generator.

        Returns
        -------
        str
            Generated text response.
        """
        # Check if there's a ground truth response we can use
        if hasattr(item, "item_metadata") and "response" in item.item_metadata:
            return str(item.item_metadata["response"])

        # Check for text template
        if hasattr(item, "item_metadata") and "response_template" in item.item_metadata:
            return str(item.item_metadata["response_template"])

        # Try to extract from rendered elements
        if hasattr(item, "rendered_elements") and item.rendered_elements:
            # Get first text element as fallback
            for value in item.rendered_elements.values():
                if len(value) > 0:
                    # Return first non-empty string
                    return value

        # Final fallback: generic response
        return "No response"
