"""Training framework adapters.

Provides trainer implementations for HuggingFace Transformers and PyTorch
Lightning. All trainers implement the BaseTrainer interface.
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
