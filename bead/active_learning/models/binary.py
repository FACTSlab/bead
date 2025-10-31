"""Binary model for yes/no or true/false judgments.

Expected architecture: Binary classification with 2-class output.
Different from 2AFC in semantics - represents absolute judgment rather than choice.
"""

from __future__ import annotations

import numpy as np

from bead.active_learning.models.base import ActiveLearningModel, ModelPrediction
from bead.items.models import Item, ItemTemplate, TaskType

__all__ = ["BinaryModel"]


class BinaryModel(ActiveLearningModel):
    """Model for binary tasks.

    Binary tasks collect yes/no or true/false judgments.
    Uses binary classification (2-class) output.
    """

    @property
    def supported_task_types(self) -> list[TaskType]:
        """Get supported task types.

        Returns
        -------
        list[TaskType]
            List containing "binary".
        """
        return ["binary"]

    def validate_item_compatibility(
        self, item: Item, item_template: ItemTemplate
    ) -> None:
        """Validate item is compatible with binary model.

        Parameters
        ----------
        item : Item
            Item to validate.
        item_template : ItemTemplate
            Template the item was constructed from.

        Raises
        ------
        ValueError
            If task_type is not "binary".
        """
        if item_template.task_type != "binary":
            raise ValueError(
                f"Expected task_type 'binary', got '{item_template.task_type}'"
            )

    def train(
        self,
        items: list[Item],
        labels: list[str],
        validation_items: list[Item] | None = None,
        validation_labels: list[str] | None = None,
    ) -> dict[str, float]:
        """Train model on binary data.

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
        raise NotImplementedError("BinaryModel training not yet implemented")

    def predict(self, items: list[Item]) -> list[ModelPrediction]:
        """Predict for binary items.

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
        raise NotImplementedError("BinaryModel prediction not yet implemented")

    def predict_proba(self, items: list[Item]) -> np.ndarray:
        """Predict probabilities for binary items.

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
        raise NotImplementedError("BinaryModel predict_proba not yet implemented")

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
        raise NotImplementedError("BinaryModel save not yet implemented")

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
        raise NotImplementedError("BinaryModel load not yet implemented")
