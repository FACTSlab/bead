"""Magnitude model for unbounded numeric judgments.

Expected architecture: Regression with single continuous output value.
Uses MSE loss to predict unbounded numeric ratings or measurements.
"""

from __future__ import annotations

import numpy as np

from bead.active_learning.models.base import ActiveLearningModel, ModelPrediction
from bead.items.models import Item, ItemTemplate, TaskType

__all__ = ["MagnitudeModel"]


class MagnitudeModel(ActiveLearningModel):
    """Model for magnitude tasks.

    Magnitude tasks collect unbounded numeric values (e.g., reaction time,
    plausibility scores). Uses regression to predict continuous outputs.
    """

    @property
    def supported_task_types(self) -> list[TaskType]:
        """Get supported task types.

        Returns
        -------
        list[TaskType]
            List containing "magnitude".
        """
        return ["magnitude"]

    def validate_item_compatibility(
        self, item: Item, item_template: ItemTemplate
    ) -> None:
        """Validate item is compatible with magnitude model.

        Parameters
        ----------
        item : Item
            Item to validate.
        item_template : ItemTemplate
            Template the item was constructed from.

        Raises
        ------
        ValueError
            If task_type is not "magnitude".
        """
        if item_template.task_type != "magnitude":
            raise ValueError(
                f"Expected task_type 'magnitude', got '{item_template.task_type}'"
            )

    def train(
        self,
        items: list[Item],
        labels: list[str],
        validation_items: list[Item] | None = None,
        validation_labels: list[str] | None = None,
    ) -> dict[str, float]:
        """Train model on magnitude data.

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
        raise NotImplementedError("MagnitudeModel training not yet implemented")

    def predict(self, items: list[Item]) -> list[ModelPrediction]:
        """Predict for magnitude items.

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
        raise NotImplementedError("MagnitudeModel prediction not yet implemented")

    def predict_proba(self, items: list[Item]) -> np.ndarray:
        """Predict probabilities for magnitude items.

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
        raise NotImplementedError("MagnitudeModel predict_proba not yet implemented")

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
        raise NotImplementedError("MagnitudeModel save not yet implemented")

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
        raise NotImplementedError("MagnitudeModel load not yet implemented")
