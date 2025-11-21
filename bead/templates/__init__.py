"""Template filling functionality.

Provides template filling strategies (exhaustive, random, stratified) and
constraint resolution for generating experimental stimuli.
"""

from __future__ import annotations

from bead.templates.filler import CSPFiller, FilledTemplate, TemplateFiller
from bead.templates.resolver import ConstraintResolver
from bead.templates.strategies import (
    ExhaustiveStrategy,
    RandomStrategy,
    StrategyFiller,
    StratifiedStrategy,
)

__all__ = [
    "TemplateFiller",  # ABC
    "CSPFiller",
    "StrategyFiller",
    "FilledTemplate",
    "ConstraintResolver",
    "ExhaustiveStrategy",
    "RandomStrategy",
    "StratifiedStrategy",
]
