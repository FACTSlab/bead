"""Task-specific simulation strategies."""

from bead.simulation.strategies.base import SimulationStrategy
from bead.simulation.strategies.binary import BinaryStrategy
from bead.simulation.strategies.categorical import CategoricalStrategy
from bead.simulation.strategies.cloze import ClozeStrategy
from bead.simulation.strategies.forced_choice import ForcedChoiceStrategy
from bead.simulation.strategies.free_text import FreeTextStrategy
from bead.simulation.strategies.magnitude import MagnitudeStrategy
from bead.simulation.strategies.multi_select import MultiSelectStrategy
from bead.simulation.strategies.ordinal_scale import OrdinalScaleStrategy

__all__ = [
    "SimulationStrategy",
    "BinaryStrategy",
    "CategoricalStrategy",
    "ClozeStrategy",
    "ForcedChoiceStrategy",
    "FreeTextStrategy",
    "MagnitudeStrategy",
    "MultiSelectStrategy",
    "OrdinalScaleStrategy",
]
