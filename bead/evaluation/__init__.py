"""Evaluation module for model and human performance assessment.

This module provides comprehensive evaluation tools for:
- Cross-validation (K-fold, stratified sampling)
- Inter-annotator agreement metrics (Cohen's kappa, Fleiss' kappa, Krippendorff's alpha)
- Model performance metrics (accuracy, precision, recall, F1)
- Convergence detection (comparing model to human performance)

The evaluation infrastructure supports active learning workflows where models
are iteratively trained and evaluated until they reach human-level agreement.
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
