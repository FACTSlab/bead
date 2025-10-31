"""LM score-based annotator."""

from __future__ import annotations

from typing import TYPE_CHECKING

from bead.simulation.annotators.base import SimulatedAnnotator
from bead.simulation.noise_models.temperature import TemperatureNoiseModel
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


class LMBasedAnnotator(SimulatedAnnotator):
    """Annotator using language model scores for decisions.

    Uses LM log probabilities or scores from Item.model_outputs
    to make informed decisions. Applies noise model for variability.

    Supports all task types via pluggable strategies.

    Examples
    --------
    >>> from bead.config.models import SimulatedAnnotatorConfig, NoiseModelConfig
    >>> config = SimulatedAnnotatorConfig(
    ...     strategy="lm_score",
    ...     model_output_key="lm_score",
    ...     noise_model=NoiseModelConfig(noise_type="temperature", temperature=1.5)
    ... )
    >>> annotator = LMBasedAnnotator(config)
    >>> # judgment = annotator.annotate(item, template)
    """

    def __init__(self, config: SimulatedAnnotatorConfig) -> None:
        """Initialize LM-based annotator.

        Parameters
        ----------
        config : SimulatedAnnotatorConfig
            Configuration for annotator.
        """
        super().__init__(config)

        # Initialize strategies for different task types
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

        # Initialize noise model
        if config.noise_model.noise_type == "temperature":
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
    ) -> str | int | float | list[str]:
        """Generate annotation using LM scores.

        Parameters
        ----------
        item : Item
            Item to annotate.
        item_template : ItemTemplate
            Template defining task.

        Returns
        -------
        str | int | float | list[str]
            Annotation (format depends on task type).
        """
        # Get strategy for task type
        strategy = self.get_strategy(item_template.task_type)

        # Validate item
        strategy.validate_item(item, item_template)

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
                context={"item": item, "template": item_template, "strategy": strategy},
                rng=self.rng,
            )

        return response
