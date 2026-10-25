"""Structural capabilities for read-only model serving.

The protocols in this module describe independent serving capabilities. A
consumer can request only the operations it uses, while a provider can
implement the protocols structurally without inheriting from a bead base
class. In particular, probabilistic predictors do not need to implement
training, mixed effects, language-model scoring, or embeddings.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Protocol, Self, runtime_checkable

import numpy as np

from bead.items.item import Item
from bead.items.item_template import ItemTemplate, TaskType
from bead.protocol.encoding import ResponseEncoding

__all__ = [
    "ModelIdentity",
    "NLICapability",
    "NLIProbabilities",
    "PersistablePredictor",
    "ProbabilisticPredictor",
    "ServingPredictor",
    "TaskCompatibility",
]


type NLIProbabilities = Mapping[str, float]
"""Three-way natural language inference probabilities.

The conformance suite requires exactly three keys: ``"entailment"``,
``"neutral"``, and ``"contradiction"``. Values must be finite probabilities
that sum to one.
"""


@runtime_checkable
class ModelIdentity(Protocol):
    """Identity metadata published by a serving model.

    Attributes
    ----------
    model_name
        Stable name of the model family or artifact.
    model_version
        Version of the model or artifact.
    """

    @property
    def model_name(self) -> str:
        """Return the stable model name.

        Returns
        -------
        str
            Stable model name.
        """
        ...

    @property
    def model_version(self) -> str:
        """Return the model or artifact version.

        Returns
        -------
        str
            Model or artifact version.
        """
        ...


@runtime_checkable
class TaskCompatibility(Protocol):
    """Task types and item structures accepted by a predictor.

    Attributes
    ----------
    supported_task_types
        Collection of task types accepted by the predictor.
    """

    @property
    def supported_task_types(self) -> Collection[TaskType]:
        """Return the task types accepted by the predictor.

        Returns
        -------
        Collection[TaskType]
            Accepted task types.
        """
        ...

    def validate_item_compatibility(
        self, item: Item, item_template: ItemTemplate
    ) -> None:
        """Validate that an item can be scored under its template.

        Parameters
        ----------
        item
            Item to validate.
        item_template
            Template that defines the item's task and structure.

        Raises
        ------
        ValueError
            If the item cannot be scored by the predictor.
        """
        ...


@runtime_checkable
class ProbabilisticPredictor(Protocol):
    """Read-only probability prediction with declared column semantics.

    Column ``i`` of the returned matrix corresponds to
    ``response_encoding.labels[i]``. Implementations return one row per item,
    with finite probabilities in ``[0, 1]`` that sum to one.
    """

    @property
    def response_encoding(self) -> ResponseEncoding:
        """Return the response encoding that names probability columns.

        Returns
        -------
        ResponseEncoding
            Ordered labels and scale type for the probability matrix.
        """
        ...

    def predict_proba(
        self,
        items: list[Item],
        participant_ids: list[str] | None = None,
    ) -> np.ndarray:
        """Return a probability matrix with shape ``(items, levels)``.

        Parameters
        ----------
        items
            Items to score.
        participant_ids
            Optional participant identifier for each item.

        Returns
        -------
        numpy.ndarray
            Probability matrix with one row per item and one column per
            response-encoding label.
        """
        ...


@runtime_checkable
class PersistablePredictor(Protocol):
    """Native-artifact persistence for a serving model."""

    def save(self, path: str | Path) -> None:
        """Save the predictor's native artifact.

        Parameters
        ----------
        path
            File or directory path for the provider's native artifact.
        """
        ...

    @classmethod
    def load(cls, path: str | Path) -> Self:
        """Load a predictor from a native artifact.

        Parameters
        ----------
        path
            File or directory path containing the provider's native artifact.

        Returns
        -------
        Self
            Restored predictor.
        """
        ...


@runtime_checkable
class NLICapability(Protocol):
    """Three-way natural language inference scoring."""

    def compute_nli(self, premise: str, hypothesis: str) -> NLIProbabilities:
        """Return entailment, neutral, and contradiction probabilities.

        Parameters
        ----------
        premise
            Premise text.
        hypothesis
            Hypothesis text.

        Returns
        -------
        NLIProbabilities
            Mapping with exactly the keys ``"entailment"``, ``"neutral"``,
            and ``"contradiction"``.
        """
        ...


@runtime_checkable
class ServingPredictor(
    ModelIdentity,
    TaskCompatibility,
    ProbabilisticPredictor,
    PersistablePredictor,
    NLICapability,
    Protocol,
):
    """Composition of all read-only serving capabilities.

    This convenience protocol is used by the full conformance suite. Consumers
    that need fewer operations should type against the independent protocols
    above.
    """
