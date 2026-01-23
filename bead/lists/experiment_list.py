"""Experiment list data model for organizing experimental items.

This module provides the ExperimentList model for organizing experimental items
into lists for presentation to participants. Lists use stand-off annotation with
UUID references to items rather than embedding full item objects.

The model supports:
- Item assignment tracking via UUIDs
- Presentation order specification
- Constraint satisfaction tracking
- Balance metrics computation
"""

from __future__ import annotations

import random
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bead.items.item_template import MetadataValue
else:
    # Recursive type for metadata values
    type MetadataValue = (
        str | int | float | bool | None | dict[str, MetadataValue] | list[MetadataValue]
    )
from uuid import UUID

from pydantic import Field, field_validator, model_validator

from bead.data.base import BeadBaseModel
from bead.lists.constraints import ListConstraint


# Factory functions for default values
def _empty_uuid_list() -> list[UUID]:
    """Return empty UUID list."""
    return []


def _empty_constraint_list() -> list[ListConstraint]:
    """Return empty ListConstraint list."""
    return []


def _empty_uuid_bool_dict() -> dict[UUID, bool]:
    """Return empty UUID-to-bool dict."""
    return {}


def _empty_metadata_dict() -> dict[str, MetadataValue]:
    """Return empty metadata dictionary."""
    return {}


class ExperimentList(BeadBaseModel):
    """A list of experimental items for participant presentation.

    Uses stand-off annotation - stores only item UUIDs, not full items.
    Items can be looked up by UUID from an ItemCollection or Repository.

    Attributes
    ----------
    name : str
        Name of this list (e.g., "list_0", "practice_list").
    list_number : int
        Numeric identifier for this list (must be >= 0).
    item_refs : list[UUID]
        UUIDs of items in this list (stand-off annotation).
    list_constraints : list[ListConstraint]
        Constraints this list must satisfy.
    constraint_satisfaction : dict[UUID, bool]
        Map of constraint UUIDs to satisfaction status.
    presentation_order : list[UUID] | None
        Explicit presentation order (if None, use item_refs order).
        Must contain exactly the same UUIDs as item_refs.
    list_metadata : dict[str, Any]
        Metadata for this list.
    balance_metrics : dict[str, Any]
        Metrics about list balance (e.g., distribution statistics).

    Examples
    --------
    >>> from uuid import uuid4
    >>> exp_list = ExperimentList(
    ...     name="list_0",
    ...     list_number=0
    ... )
    >>> item_id = uuid4()
    >>> exp_list.add_item(item_id)
    >>> len(exp_list.item_refs)
    1
    >>> exp_list.shuffle_order(seed=42)
    >>> exp_list.get_presentation_order()[0] == item_id
    True
    """

    name: str = Field(..., description="List name")
    list_number: int = Field(..., ge=0, description="Numeric list identifier")
    item_refs: list[UUID] = Field(
        default_factory=_empty_uuid_list, description="Item UUIDs (stand-off)"
    )
    list_constraints: list[ListConstraint] = Field(
        default_factory=_empty_constraint_list, description="List constraints"
    )
    constraint_satisfaction: dict[UUID, bool] = Field(
        default_factory=_empty_uuid_bool_dict,
        description="Constraint satisfaction status",
    )
    presentation_order: list[UUID] | None = Field(
        default=None, description="Explicit presentation order"
    )
    list_metadata: dict[str, MetadataValue] = Field(
        default_factory=_empty_metadata_dict, description="List metadata"
    )
    balance_metrics: dict[str, MetadataValue] = Field(
        default_factory=_empty_metadata_dict, description="Balance metrics"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty.

        Parameters
        ----------
        v : str
            Name to validate.

        Returns
        -------
        str
            Validated name (whitespace stripped).

        Raises
        ------
        ValueError
            If name is empty or contains only whitespace.
        """
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v.strip()

    @model_validator(mode="after")
    def validate_presentation_order(self) -> ExperimentList:
        """Validate presentation_order matches item_refs.

        If presentation_order is set, it must contain exactly the same UUIDs
        as item_refs (no more, no less, no duplicates).

        Returns
        -------
        ExperimentList
            Validated list.

        Raises
        ------
        ValueError
            If presentation_order doesn't match item_refs.
        """
        if self.presentation_order is None:
            return self

        # Check for duplicates in presentation_order
        if len(self.presentation_order) != len(set(self.presentation_order)):
            raise ValueError("presentation_order contains duplicate UUIDs")

        # Check that sets match
        item_set = set(self.item_refs)
        order_set = set(self.presentation_order)

        if order_set != item_set:
            extra = order_set - item_set
            missing = item_set - order_set

            error_parts: list[str] = []
            if extra:
                error_parts.append(f"extra UUIDs: {extra}")
            if missing:
                error_parts.append(f"missing UUIDs: {missing}")

            raise ValueError(
                f"presentation_order must contain exactly same UUIDs "
                f"as item_refs ({', '.join(error_parts)})"
            )

        return self

    def add_item(self, item_id: UUID) -> None:
        """Add an item to this list.

        Parameters
        ----------
        item_id : UUID
            UUID of item to add.

        Examples
        --------
        >>> from uuid import uuid4
        >>> exp_list = ExperimentList(name="test", list_number=0)
        >>> item_id = uuid4()
        >>> exp_list.add_item(item_id)
        >>> item_id in exp_list.item_refs
        True
        """
        self.item_refs.append(item_id)
        self.update_modified_time()

    def remove_item(self, item_id: UUID) -> None:
        """Remove an item from this list.

        Parameters
        ----------
        item_id : UUID
            UUID of item to remove.

        Raises
        ------
        ValueError
            If item_id is not in the list.

        Examples
        --------
        >>> from uuid import uuid4
        >>> exp_list = ExperimentList(name="test", list_number=0)
        >>> item_id = uuid4()
        >>> exp_list.add_item(item_id)
        >>> exp_list.remove_item(item_id)
        >>> item_id in exp_list.item_refs
        False
        """
        if item_id not in self.item_refs:
            raise ValueError(f"Item {item_id} not found in list")
        self.item_refs.remove(item_id)

        # Also remove from presentation_order if present
        if self.presentation_order is not None and item_id in self.presentation_order:
            self.presentation_order.remove(item_id)

        self.update_modified_time()

    def shuffle_order(self, seed: int | None = None) -> None:
        """Shuffle presentation order.

        Creates a randomized presentation order from item_refs.
        Uses random.Random(seed) for reproducible shuffling.

        Parameters
        ----------
        seed : int | None
            Random seed for reproducibility.

        Examples
        --------
        >>> from uuid import uuid4
        >>> exp_list = ExperimentList(name="test", list_number=0)
        >>> exp_list.add_item(uuid4())
        >>> exp_list.add_item(uuid4())
        >>> exp_list.shuffle_order(seed=42)
        >>> exp_list.presentation_order is not None
        True
        """
        rng = random.Random(seed)
        self.presentation_order = self.item_refs.copy()
        rng.shuffle(self.presentation_order)
        self.update_modified_time()

    def get_presentation_order(self) -> list[UUID]:
        """Get the presentation order.

        Returns presentation_order if set, otherwise returns item_refs.

        Returns
        -------
        list[UUID]
            UUIDs in presentation order.

        Examples
        --------
        >>> from uuid import uuid4
        >>> exp_list = ExperimentList(name="test", list_number=0)
        >>> item_id = uuid4()
        >>> exp_list.add_item(item_id)
        >>> exp_list.get_presentation_order()[0] == item_id
        True
        """
        return self.presentation_order if self.presentation_order else self.item_refs
