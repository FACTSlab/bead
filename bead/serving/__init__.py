"""Structural interfaces for externally trained, read-only predictors."""

from bead.serving.conformance import (
    ConformanceError,
    ConformanceReport,
    NLICase,
    PredictionCase,
    run_predictor_conformance,
)
from bead.serving.protocols import (
    ModelIdentity,
    NLICapability,
    NLIProbabilities,
    PersistablePredictor,
    ProbabilisticPredictor,
    ServingPredictor,
    TaskCompatibility,
)
from bead.serving.registry import PredictorFactory, PredictorRegistry

__all__ = [
    "ConformanceError",
    "ConformanceReport",
    "ModelIdentity",
    "NLICapability",
    "NLICase",
    "NLIProbabilities",
    "PersistablePredictor",
    "PredictionCase",
    "PredictorFactory",
    "PredictorRegistry",
    "ProbabilisticPredictor",
    "ServingPredictor",
    "TaskCompatibility",
    "run_predictor_conformance",
]
