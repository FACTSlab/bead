"""Simulation framework for generating synthetic human judgments.

This module provides comprehensive simulation infrastructure for testing
active learning pipelines without requiring real human data. The framework:

- Supports all task types (forced_choice, ordinal_scale, magnitude, etc.)
- Respects task specifications from ItemTemplate
- Uses model outputs (LM scores, embeddings) for informed decisions
- Provides configurable noise models
- Extends the existing DSL for custom simulation logic

Examples
--------
>>> from bead.simulation import SimulatedAnnotator
>>> from bead.config.models import SimulatedAnnotatorConfig, NoiseModelConfig
>>>
>>> # Create annotator with configuration
>>> config = SimulatedAnnotatorConfig(
...     strategy="lm_score",
...     noise_model=NoiseModelConfig(noise_type="temperature", temperature=1.0)
... )
>>> annotator = SimulatedAnnotator.from_config(config)
>>>
>>> # Annotate items (automatically handles any task type)
>>> judgments = annotator.annotate_batch(items)
"""

from bead.simulation.annotators.base import SimulatedAnnotator
from bead.simulation.annotators.lm_based import LMBasedAnnotator
from bead.simulation.noise_models.base import NoiseModel
from bead.simulation.noise_models.temperature import TemperatureNoiseModel
from bead.simulation.runner import SimulationRunner
from bead.simulation.strategies.base import SimulationStrategy
from bead.simulation.strategies.binary import BinaryStrategy
from bead.simulation.strategies.categorical import CategoricalStrategy
from bead.simulation.strategies.forced_choice import ForcedChoiceStrategy
from bead.simulation.strategies.ordinal_scale import OrdinalScaleStrategy

__all__ = [
    "SimulatedAnnotator",
    "LMBasedAnnotator",
    "NoiseModel",
    "TemperatureNoiseModel",
    "SimulationRunner",
    "SimulationStrategy",
    "BinaryStrategy",
    "CategoricalStrategy",
    "ForcedChoiceStrategy",
    "OrdinalScaleStrategy",
]
