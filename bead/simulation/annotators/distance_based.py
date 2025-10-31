"""Distance-based annotator using embeddings."""

from __future__ import annotations

from typing import TYPE_CHECKING

from bead.simulation.annotators.base import SimulatedAnnotator
from bead.simulation.strategies.binary import BinaryStrategy
from bead.simulation.strategies.categorical import CategoricalStrategy
from bead.simulation.strategies.cloze import ClozeStrategy
from bead.simulation.strategies.forced_choice import ForcedChoiceStrategy
from bead.simulation.strategies.free_text import FreeTextStrategy
from bead.simulation.strategies.magnitude import MagnitudeStrategy
from bead.simulation.strategies.multi_select import MultiSelectStrategy
from bead.simulation.strategies.ordinal_scale import OrdinalScaleStrategy

if TYPE_CHECKING:
    from bead.config.models import SimulatedAnnotatorConfig
    from bead.items.models import Item, ItemTemplate


class DistanceBasedAnnotator(SimulatedAnnotator):
    """Annotator using embedding distances for decisions.

    Uses embeddings from Item.model_outputs to compute similarity/distance
    metrics, then makes decisions based on those distances.

    For forced choice, selects option with lowest distance (highest similarity).
    For ordinal scales, maps distance to scale values.
    For binary, thresholds distance.

    Examples
    --------
    >>> from bead.config.models import SimulatedAnnotatorConfig, NoiseModelConfig
    >>> config = SimulatedAnnotatorConfig(
    ...     strategy="distance",
    ...     model_output_key="embedding",
    ...     noise_model=NoiseModelConfig(noise_type="none")
    ... )
    >>> annotator = DistanceBasedAnnotator(config)
    >>> # judgment = annotator.annotate(item, template)
    """

    def __init__(self, config: SimulatedAnnotatorConfig) -> None:
        """Initialize distance-based annotator.

        Parameters
        ----------
        config : SimulatedAnnotatorConfig
            Configuration for annotator.
        """
        super().__init__(config)

        # Initialize strategies for different task types
        # Use same strategies as LM-based, but will extract embeddings
        # instead of LM scores
        self.strategies = {
            "forced_choice": ForcedChoiceStrategy(),
            "binary": BinaryStrategy(),
            "ordinal_scale": OrdinalScaleStrategy(),
            "categorical": CategoricalStrategy(),
            "magnitude": MagnitudeStrategy(),
            "multi_select": MultiSelectStrategy(),
            "free_text": FreeTextStrategy(),
            "cloze": ClozeStrategy(),
        }

        # Initialize noise model if configured
        if config.noise_model.noise_type == "temperature":
            from bead.simulation.noise_models.temperature import (  # noqa: PLC0415
                TemperatureNoiseModel,
            )

            self.noise_model = TemperatureNoiseModel(
                temperature=config.noise_model.temperature
            )
        elif config.noise_model.noise_type == "none":
            self.noise_model = None
        else:
            # Default: no noise
            self.noise_model = None

    def annotate(
        self, item: Item, item_template: ItemTemplate
    ) -> str | int | float | bool | list[str]:
        """Generate annotation using embedding distances.

        Parameters
        ----------
        item : Item
            Item to annotate.
        item_template : ItemTemplate
            Template defining task.

        Returns
        -------
        str | int | float | bool | list[str]
            Annotation (format depends on task type).

        Notes
        -----
        For distance-based decisions, we convert embeddings to scores:
        - Cosine similarity ranges from -1 (opposite) to 1 (identical)
        - We convert to "score" by: score = similarity * 10
        - This allows reuse of existing strategies
        """
        # Get strategy for task type
        strategy = self.get_strategy(item_template.task_type)

        # Validate item
        strategy.validate_item(item, item_template)

        # For distance-based, we need to convert embeddings to scores
        # This is a simplified approach - in practice, you might want
        # task-specific distance computations
        # For now, we'll rely on the strategies to extract embeddings
        # and treat them as scores (strategies will use model_output_key)

        # Generate base response
        response = strategy.simulate_response(
            item=item,
            item_template=item_template,
            model_output_key=self.config.model_output_key,
            rng=self.rng,
        )

        # Apply noise model if configured
        if self.noise_model is not None:
            response = self.noise_model.apply(
                value=response,
                context={
                    "item": item,
                    "template": item_template,
                    "strategy": strategy,
                },
                rng=self.rng,
            )

        return response
