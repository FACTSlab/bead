"""Training framework adapters for bead.

This module provides trainer implementations for different ML frameworks:
- HuggingFace Transformers
- PyTorch Lightning

All trainers implement the BaseTrainer interface and return ModelMetadata
for tracking training results.
"""

from __future__ import annotations

from bead.active_learning.trainers.base import BaseTrainer, ModelMetadata
from bead.active_learning.trainers.huggingface import HuggingFaceTrainer
from bead.active_learning.trainers.lightning import PyTorchLightningTrainer
from bead.active_learning.trainers.registry import (
    get_trainer,
    list_trainers,
    register_trainer,
)

__all__ = [
    "BaseTrainer",
    "HuggingFaceTrainer",
    "ModelMetadata",
    "PyTorchLightningTrainer",
    "get_trainer",
    "list_trainers",
    "register_trainer",
]
