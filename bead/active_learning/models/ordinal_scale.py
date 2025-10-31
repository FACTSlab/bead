"""Ordinal scale model for ordered rating scales.

Expected architecture: Ordinal regression with ordered classes (e.g., 1-7 Likert scale).
Uses cumulative link model or ordinal loss functions to preserve ordering.
"""

from __future__ import annotations

import numpy as np

from bead.active_learning.models.base import ActiveLearningModel, ModelPrediction
from bead.items.models import Item, ItemTemplate, TaskType

__all__ = ["OrdinalScaleModel"]


class OrdinalScaleModel(ActiveLearningModel):
    """Model for ordinal_scale tasks.

    Ordinal scale tasks collect ratings on ordered discrete scales
    (e.g., Likert scales). Uses ordinal regression to preserve the
    ordered nature of scale points.
    """

    @property
    def supported_task_types(self) -> list[TaskType]:
        """Get supported task types.

        Returns
        -------
        list[TaskType]
            List containing "ordinal_scale".
        """
        return ["ordinal_scale"]

    def validate_item_compatibility(
        self, item: Item, item_template: ItemTemplate
    ) -> None:
        """Validate item is compatible with ordinal scale model.

        Parameters
        ----------
        item : Item
            Item to validate.
        item_template : ItemTemplate
            Template the item was constructed from.

        Raises
        ------
        ValueError
            If task_type is not "ordinal_scale".
        """
        if item_template.task_type != "ordinal_scale":
            raise ValueError(
                f"Expected task_type 'ordinal_scale', got '{item_template.task_type}'"
            )

    def train(
        self,
        items: list[Item],
        labels: list[str],
        validation_items: list[Item] | None = None,
        validation_labels: list[str] | None = None,
    ) -> dict[str, float]:
        """Train model on ordinal scale data.

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
        raise NotImplementedError("OrdinalScaleModel training not yet implemented")

    def predict(self, items: list[Item]) -> list[ModelPrediction]:
        """Predict for ordinal scale items.

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
        raise NotImplementedError("OrdinalScaleModel prediction not yet implemented")

    def predict_proba(self, items: list[Item]) -> np.ndarray:
        """Predict probabilities for ordinal scale items.

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
        raise NotImplementedError("OrdinalScaleModel predict_proba not yet implemented")

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
        raise NotImplementedError("OrdinalScaleModel save not yet implemented")

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
        raise NotImplementedError("OrdinalScaleModel load not yet implemented")
