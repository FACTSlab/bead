"""Evaluation module for model and human performance assessment.

Provides cross-validation, inter-annotator agreement metrics, model
performance metrics, and convergence detection for active learning.
"""

from bead.evaluation.convergence import ConvergenceDetector
from bead.evaluation.interannotator import InterAnnotatorMetrics

__all__ = [
    "InterAnnotatorMetrics",
    "ConvergenceDetector",
]
