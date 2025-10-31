"""Cloze model for fill-in-the-blank tasks.

Expected architecture: Masked token prediction using language model.
Predicts tokens for unfilled slots in cloze items.
"""

from __future__ import annotations

import numpy as np

from bead.active_learning.models.base import ActiveLearningModel, ModelPrediction
from bead.items.models import Item, ItemTemplate, TaskType

__all__ = ["ClozeModel"]


class ClozeModel(ActiveLearningModel):
    """Model for cloze tasks.

    Cloze tasks involve filling in blanks in partially complete items.
    Uses masked language modeling for token prediction.
    """

    @property
    def supported_task_types(self) -> list[TaskType]:
        """Get supported task types.

        Returns
        -------
        list[TaskType]
            List containing "cloze".
        """
        return ["cloze"]

    def validate_item_compatibility(
        self, item: Item, item_template: ItemTemplate
    ) -> None:
        """Validate item is compatible with cloze model.

        Parameters
        ----------
        item : Item
            Item to validate.
        item_template : ItemTemplate
            Template the item was constructed from.

        Raises
        ------
        ValueError
            If task_type is not "cloze".
        """
        if item_template.task_type != "cloze":
            raise ValueError(
                f"Expected task_type 'cloze', got '{item_template.task_type}'"
            )

    def train(
        self,
        items: list[Item],
        labels: list[str],
        validation_items: list[Item] | None = None,
        validation_labels: list[str] | None = None,
    ) -> dict[str, float]:
        """Train model on cloze data.

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
        raise NotImplementedError("ClozeModel training not yet implemented")

    def predict(self, items: list[Item]) -> list[ModelPrediction]:
        """Predict for cloze items.

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
        raise NotImplementedError("ClozeModel prediction not yet implemented")

    def predict_proba(self, items: list[Item]) -> np.ndarray:
        """Predict probabilities for cloze items.

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
        raise NotImplementedError("ClozeModel predict_proba not yet implemented")

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
        raise NotImplementedError("ClozeModel save not yet implemented")

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
        raise NotImplementedError("ClozeModel load not yet implemented")
