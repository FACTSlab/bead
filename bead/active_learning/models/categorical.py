"""Categorical model for unordered category selection.

Expected architecture: Multi-class classification with softmax output.
Predicts one category from a set of unordered categories.
"""

from __future__ import annotations

import numpy as np

from bead.active_learning.models.base import ActiveLearningModel, ModelPrediction
from bead.items.models import Item, ItemTemplate, TaskType

__all__ = ["CategoricalModel"]


class CategoricalModel(ActiveLearningModel):
    """Model for categorical tasks.

    Categorical tasks select one category from unordered options.
    Uses multi-class classification with softmax activation.
    """

    @property
    def supported_task_types(self) -> list[TaskType]:
        """Get supported task types.

        Returns
        -------
        list[TaskType]
            List containing "categorical".
        """
        return ["categorical"]

    def validate_item_compatibility(
        self, item: Item, item_template: ItemTemplate
    ) -> None:
        """Validate item is compatible with categorical model.

        Parameters
        ----------
        item : Item
            Item to validate.
        item_template : ItemTemplate
            Template the item was constructed from.

        Raises
        ------
        ValueError
            If task_type is not "categorical".
        """
        if item_template.task_type != "categorical":
            raise ValueError(
                f"Expected task_type 'categorical', got '{item_template.task_type}'"
            )

    def train(
        self,
        items: list[Item],
        labels: list[str],
        validation_items: list[Item] | None = None,
        validation_labels: list[str] | None = None,
    ) -> dict[str, float]:
        """Train model on categorical data.

        Parameters
        ----------
        items : list[Item]
            Training items.
        labels : list[str]
            Training labels.
        validation_items : list[Item] | None
            Optional validation items.
        validation_labels : list[str] | None
            Optional validation labels.

        Returns
        -------
        dict[str, float]
            Training metrics.

        Raises
        ------
        NotImplementedError
            This model is not yet implemented.
        """
        raise NotImplementedError("CategoricalModel training not yet implemented")

    def predict(self, items: list[Item]) -> list[ModelPrediction]:
        """Predict for categorical items.

        Parameters
        ----------
        items : list[Item]
            Items to predict.

        Returns
        -------
        list[ModelPrediction]
            Predictions.

        Raises
        ------
        NotImplementedError
            This model is not yet implemented.
        """
        raise NotImplementedError("CategoricalModel prediction not yet implemented")

    def predict_proba(self, items: list[Item]) -> np.ndarray:
        """Predict probabilities for categorical items.

        Parameters
        ----------
        items : list[Item]
            Items to predict.

        Returns
        -------
        np.ndarray
            Probability array.

        Raises
        ------
        NotImplementedError
            This model is not yet implemented.
        """
        raise NotImplementedError("CategoricalModel predict_proba not yet implemented")

    def save(self, path: str) -> None:
        """Save model to disk.

        Parameters
        ----------
        path : str
            Path to save model.

        Raises
        ------
        NotImplementedError
            This model is not yet implemented.
        """
        raise NotImplementedError("CategoricalModel save not yet implemented")

    def load(self, path: str) -> None:
        """Load model from disk.

        Parameters
        ----------
        path : str
            Path to load model from.

        Raises
        ------
        NotImplementedError
            This model is not yet implemented.
        """
        raise NotImplementedError("CategoricalModel load not yet implemented")
