"""Simulation framework for generating synthetic human judgments.

Provides annotators, noise models, and strategies for testing active
learning pipelines without real human data.
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
