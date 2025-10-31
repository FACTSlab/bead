"""Base interfaces for active learning models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import numpy as np

from bead.data.base import BeadBaseModel
from bead.items.models import Item

if TYPE_CHECKING:
    from bead.items.models import ItemTemplate, TaskType

__all__ = ["ActiveLearningModel", "ModelPrediction"]


class ModelPrediction(BeadBaseModel):
    """Prediction output for a single item.

    Attributes
    ----------
    item_id : str
        Unique identifier for the item.
    probabilities : dict[str, float]
        Predicted probabilities for each class/option.
        Keys are option names (e.g., "option_a", "option_b") or class labels.
    predicted_class : str
        The predicted class/option with highest probability.
    confidence : float
        Confidence score (max probability).

    Examples
    --------
    >>> prediction = ModelPrediction(
    ...     item_id="abc123",
    ...     probabilities={"option_a": 0.7, "option_b": 0.3},
    ...     predicted_class="option_a",
    ...     confidence=0.7
    ... )
    >>> prediction.predicted_class
    'option_a'
    """

    item_id: str
    probabilities: dict[str, float]
    predicted_class: str
    confidence: float


class ActiveLearningModel(ABC):
    """Base class for all active learning models.

    All models must implement task compatibility validation and standard
    training/prediction interfaces. Models are specific to task types
    and must validate items match their supported tasks.

    Attributes
    ----------
    supported_task_types : list[TaskType]
        List of task types this model can handle.

    Examples
    --------
    >>> class MyModel(ActiveLearningModel):
    ...     @property
    ...     def supported_task_types(self):
    ...         return ["forced_choice"]
    ...     def validate_item_compatibility(self, item, item_template):
    ...         pass
    ...     def train(
    ...         self, items, labels, validation_items=None, validation_labels=None
    ...     ):
    ...         return {}
    ...     def predict(self, items):
    ...         return []
    ...     def predict_proba(self, items):
    ...         return np.array([])
    ...     def save(self, path):
    ...         pass
    ...     def load(self, path):
    ...         pass
    """

    @property
    @abstractmethod
    def supported_task_types(self) -> list[TaskType]:
        """Get list of task types this model supports.

        Returns
        -------
        list[TaskType]
            List of supported TaskType literals from items.models.

        Examples
        --------
        >>> model.supported_task_types
        ['forced_choice']
        """
        pass

    @abstractmethod
    def validate_item_compatibility(
        self, item: Item, item_template: ItemTemplate
    ) -> None:
        """Validate that an item is compatible with this model.

        Parameters
        ----------
        item : Item
            Item to validate.
        item_template : ItemTemplate
            Template the item was constructed from.

        Raises
        ------
        ValueError
            If item's task_type is not in supported_task_types.
        ValueError
            If item is missing required elements.
        ValueError
            If item structure is incompatible with model.

        Examples
        --------
        >>> model.validate_item_compatibility(item, template)  # doctest: +SKIP
        """
        pass

    @abstractmethod
    def train(
        self,
        items: list[Item],
        labels: list[str],
        validation_items: list[Item] | None = None,
        validation_labels: list[str] | None = None,
    ) -> dict[str, float]:
        """Train model on labeled items.

        Parameters
        ----------
        items : list[Item]
            Training items.
        labels : list[str]
            Training labels (format depends on task type).
        validation_items : list[Item] | None
            Optional validation items.
        validation_labels : list[str] | None
            Optional validation labels.

        Returns
        -------
        dict[str, float]
            Training metrics with keys like "train_accuracy", "train_loss",
            "val_accuracy", etc.

        Raises
        ------
        ValueError
            If items and labels have different lengths.
        ValueError
            If validation data is incomplete.
        ValueError
            If labels are invalid for this task type.

        Examples
        --------
        >>> metrics = model.train(train_items, train_labels)  # doctest: +SKIP
        >>> print(f"Accuracy: {metrics['train_accuracy']:.2f}")  # doctest: +SKIP
        """
        pass

    @abstractmethod
    def predict(self, items: list[Item]) -> list[ModelPrediction]:
        """Predict class labels for items.

        Parameters
        ----------
        items : list[Item]
            Items to predict.

        Returns
        -------
        list[ModelPrediction]
            Predictions with probabilities and predicted class for each item.

        Raises
        ------
        ValueError
            If model has not been trained.
        ValueError
            If items are incompatible with model.

        Examples
        --------
        >>> predictions = model.predict(test_items)  # doctest: +SKIP
        >>> for pred in predictions:  # doctest: +SKIP
        ...     print(f"{pred.predicted_class}: {pred.confidence:.2f}")
        """
        pass

    @abstractmethod
    def predict_proba(self, items: list[Item]) -> np.ndarray:
        """Predict class probabilities for items.

        Parameters
        ----------
        items : list[Item]
            Items to predict.

        Returns
        -------
        np.ndarray
            Array of shape (n_items, n_classes) with probabilities.
            Each row sums to 1.0 for classification tasks.

        Raises
        ------
        ValueError
            If model has not been trained.
        ValueError
            If items are incompatible with model.

        Examples
        --------
        >>> proba = model.predict_proba(test_items)  # doctest: +SKIP
        >>> proba.shape  # doctest: +SKIP
        (10, 2)
        """
        pass

    @abstractmethod
    def save(self, path: str) -> None:
        """Save model to disk.

        Parameters
        ----------
        path : str
            File or directory path to save the model.

        Raises
        ------
        ValueError
            If model has not been trained.

        Examples
        --------
        >>> model.save("/path/to/model")  # doctest: +SKIP
        """
        pass

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk.

        Parameters
        ----------
        path : str
            File or directory path to load the model from.

        Raises
        ------
        FileNotFoundError
            If model file/directory does not exist.

        Examples
        --------
        >>> model.load("/path/to/model")  # doctest: +SKIP
        """
        pass
