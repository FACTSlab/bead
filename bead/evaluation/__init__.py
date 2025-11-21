"""Evaluation module for model and human performance assessment.

Provides cross-validation, inter-annotator agreement metrics, model
performance metrics, and convergence detection for active learning.
"""

from bead.evaluation.convergence import ConvergenceDetector
from bead.evaluation.cross_validation import CrossValidator
from bead.evaluation.interannotator import InterAnnotatorMetrics
from bead.evaluation.model_metrics import ModelMetrics

__all__ = [
    "CrossValidator",
    "InterAnnotatorMetrics",
    "ModelMetrics",
    "ConvergenceDetector",
]
