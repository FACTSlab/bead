"""Base class for simulated annotators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from bead.config.models import SimulatedAnnotatorConfig
    from bead.items.models import Item, ItemTemplate
    from bead.simulation.noise_models.base import NoiseModel
    from bead.simulation.strategies.base import SimulationStrategy


class SimulatedAnnotator(ABC):
    """Abstract base for simulated annotators.

    An annotator combines:
    - Task-specific strategy (how to respond to each task type)
    - Noise model (how to add human-like variability)
    - Configuration (model output keys, random seed, etc.)

    The annotator orchestrates the simulation process and provides
    a unified interface for generating judgments.
    """

    def __init__(
        self, config: SimulatedAnnotatorConfig, random_state: int | None = None
    ) -> None:
        """Initialize annotator.

        Parameters
        ----------
        config : SimulatedAnnotatorConfig
            Configuration for annotator.
        random_state : int | None
            Random seed (overrides config if provided).
        """
        self.config = config
        self.random_state = random_state or config.random_state
        self.rng = np.random.RandomState(self.random_state)

        # Will be set by subclasses
        self.strategies: dict[str, SimulationStrategy] = {}
        self.noise_model: NoiseModel | None = None

    @classmethod
    def from_config(cls, config: SimulatedAnnotatorConfig) -> SimulatedAnnotator:
        """Create annotator from configuration.

        Parameters
        ----------
        config : SimulatedAnnotatorConfig
            Configuration specifying annotator type and parameters.

        Returns
        -------
        SimulatedAnnotator
            Configured annotator instance.

        Raises
        ------
        ValueError
            If strategy is unknown.

        Examples
        --------
        >>> from bead.config.models import SimulatedAnnotatorConfig
        >>> config = SimulatedAnnotatorConfig(strategy="lm_score")
        >>> annotator = SimulatedAnnotator.from_config(config)
        """
        # Import here to avoid circular dependency
        from bead.simulation.annotators.distance_based import (  # noqa: PLC0415
            DistanceBasedAnnotator,
        )
        from bead.simulation.annotators.lm_based import (  # noqa: PLC0415
            LMBasedAnnotator,
        )
        from bead.simulation.annotators.oracle import OracleAnnotator  # noqa: PLC0415
        from bead.simulation.annotators.random import RandomAnnotator  # noqa: PLC0415

        if config.strategy == "lm_score":
            return LMBasedAnnotator(config)
        elif config.strategy == "random":
            return RandomAnnotator(config)
        elif config.strategy == "oracle":
            return OracleAnnotator(config)
        elif config.strategy == "distance":
            return DistanceBasedAnnotator(config)
        else:
            msg = f"Unknown strategy: {config.strategy}"
            raise ValueError(msg)

    @abstractmethod
    def annotate(
        self, item: Item, item_template: ItemTemplate
    ) -> str | int | float | list[str]:
        """Generate annotation for single item.

        Parameters
        ----------
        item : Item
            Item to annotate.
        item_template : ItemTemplate
            Template defining task structure.

        Returns
        -------
        str | int | float | list[str]
            Annotation (format depends on task type).
        """

    def annotate_batch(
        self,
        items: list[Item],
        item_templates: list[ItemTemplate] | ItemTemplate,
    ) -> dict[str, str | int | float | list[str]]:
        """Generate annotations for batch of items.

        Parameters
        ----------
        items : list[Item]
            Items to annotate.
        item_templates : list[ItemTemplate] | ItemTemplate
            Templates (one per item or single template for all).

        Returns
        -------
        dict[str, str | int | float | list[str]]
            Mapping from item ID to annotation.

        Examples
        --------
        >>> annotations = annotator.annotate_batch(items, template)
        >>> annotations[str(items[0].id)]
        'option_a'
        """
        # Handle single template
        templates_list: list[ItemTemplate]
        if not isinstance(item_templates, list):
            templates_list = [item_templates] * len(items)
        else:
            templates_list = item_templates

        # Annotate each item
        annotations: dict[str, str | int | float | list[str]] = {}
        for item, template in zip(items, templates_list, strict=True):
            annotation = self.annotate(item, template)
            annotations[str(item.id)] = annotation

        return annotations

    def get_strategy(self, task_type: str) -> SimulationStrategy:
        """Get strategy for task type.

        Parameters
        ----------
        task_type : str
            Task type (e.g., "forced_choice").

        Returns
        -------
        SimulationStrategy
            Strategy for this task type.

        Raises
        ------
        ValueError
            If task type not supported.
        """
        if task_type not in self.strategies:
            msg = f"Task type '{task_type}' not supported by {self.__class__.__name__}"
            raise ValueError(msg)
        return self.strategies[task_type]
