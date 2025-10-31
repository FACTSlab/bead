"""Base class for simulation strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from bead.items.models import Item, ItemTemplate


class SimulationStrategy(ABC):
    """Abstract base for task-specific simulation strategies.

    Each strategy handles one task type (forced_choice, ordinal_scale, etc.)
    and converts model outputs into appropriate responses.

    Strategies should:
    1. Validate item compatibility with task type
    2. Extract relevant model outputs
    3. Generate response in correct format for task
    4. Handle missing model outputs gracefully
    """

    @property
    @abstractmethod
    def supported_task_type(self) -> str:
        """Return supported task type (e.g., 'forced_choice').

        Returns
        -------
        str
            Task type identifier.
        """

    @abstractmethod
    def validate_item(self, item: Item, item_template: ItemTemplate) -> None:
        """Validate item is compatible with this strategy.

        Parameters
        ----------
        item : Item
            Item to validate.
        item_template : ItemTemplate
            Template defining task structure.

        Raises
        ------
        ValueError
            If item incompatible with this strategy.
        """

    @abstractmethod
    def simulate_response(
        self,
        item: Item,
        item_template: ItemTemplate,
        model_output_key: str,
        rng: np.random.RandomState,
    ) -> str | int | float | list[str]:
        """Generate simulated response for item.

        Parameters
        ----------
        item : Item
            Item to respond to.
        item_template : ItemTemplate
            Template defining task structure.
        model_output_key : str
            Key to extract from model outputs.
        rng : np.random.RandomState
            Random number generator.

        Returns
        -------
        str | int | float | list[str]
            Simulated response (format depends on task type).
        """

    def extract_model_outputs(
        self, item: Item, key: str, required_count: int | None = None
    ) -> list[float] | None:
        """Extract model outputs from item.

        Parameters
        ----------
        item : Item
            Item to extract from.
        key : str
            Key to look for.
        required_count : int | None
            Expected number of outputs.

        Returns
        -------
        list[float] | None
            Extracted values or None if missing.
        """
        # Try model_outputs first
        values: list[float] = []
        if hasattr(item, "model_outputs"):
            for output in item.model_outputs:
                if output.operation == key:
                    values.append(float(output.output))

        # Try item_metadata as fallback
        if not values and hasattr(item, "item_metadata"):
            # Look for keys like "lm_score1", "lm_score2", etc.
            for i in range(1, (required_count or 10) + 1):
                key_with_num = f"{key}{i}"
                if key_with_num in item.item_metadata:
                    values.append(float(item.item_metadata[key_with_num]))

        if not values:
            return None

        if required_count and len(values) != required_count:
            return None

        return values
