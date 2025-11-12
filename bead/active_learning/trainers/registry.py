"""Trainer registry for framework selection.

This module provides a registry for managing different trainer implementations,
allowing users to select trainers by name.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from bead.active_learning.trainers.huggingface import HuggingFaceTrainer
from bead.active_learning.trainers.lightning import PyTorchLightningTrainer

if TYPE_CHECKING:
    from bead.active_learning.trainers.base import BaseTrainer

_TRAINERS: dict[str, type[BaseTrainer]] = {}


def register_trainer(name: str, trainer_class: type[BaseTrainer]) -> None:
    """Register a trainer class.

    Parameters
    ----------
    name : str
        Trainer name (e.g., "huggingface", "pytorch_lightning").
    trainer_class : type[BaseTrainer]
        Trainer class to register.

    Examples
    --------
    >>> from bead.active_learning.trainers.base import BaseTrainer
    >>> class MyTrainer(BaseTrainer):  # doctest: +SKIP
    ...     def train(self, train_data, eval_data=None):
    ...         pass
    ...     def save_model(self, output_dir, metadata):
    ...         pass
    ...     def load_model(self, model_dir):
    ...         pass
    >>> register_trainer("my_trainer", MyTrainer)  # doctest: +SKIP
    >>> "my_trainer" in list_trainers()  # doctest: +SKIP
    True
    """
    _TRAINERS[name] = trainer_class


def get_trainer(name: str) -> type[BaseTrainer]:
    """Get trainer class by name.

    Parameters
    ----------
    name : str
        Trainer name.

    Returns
    -------
    type[BaseTrainer]
        Trainer class.

    Raises
    ------
    ValueError
        If trainer name is not registered.

    Examples
    --------
    >>> trainer_class = get_trainer("huggingface")
    >>> trainer_class.__name__
    'HuggingFaceTrainer'
    >>> get_trainer("unknown")  # doctest: +SKIP
    Traceback (most recent call last):
        ...
    ValueError: Unknown trainer: unknown. Available trainers: huggingface,
    pytorch_lightning
    """
    if name not in _TRAINERS:
        available = ", ".join(list_trainers())
        msg = f"Unknown trainer: {name}. Available trainers: {available}"
        raise ValueError(msg)
    return _TRAINERS[name]


def list_trainers() -> list[str]:
    """List available trainers.

    Returns
    -------
    list[str]
        List of registered trainer names.

    Examples
    --------
    >>> trainers = list_trainers()
    >>> "huggingface" in trainers
    True
    >>> "pytorch_lightning" in trainers
    True
    """
    return list(_TRAINERS.keys())


# Register built-in trainers
register_trainer("huggingface", HuggingFaceTrainer)
register_trainer("pytorch_lightning", PyTorchLightningTrainer)
