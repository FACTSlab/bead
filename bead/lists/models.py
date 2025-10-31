"""Data models for experimental lists and list collections.

This module provides data models for organizing experimental items into lists
for presentation to participants. Lists use stand-off annotation with UUID
references to items rather than embedding full item objects.

The models support:
- Item assignment tracking via UUIDs
- Presentation order specification
- Constraint satisfaction tracking
- Balance metrics computation
- Partitioning metadata
"""

from __future__ import annotations

import random
from typing import Any
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


def _empty_any_dict() -> dict[str, Any]:
    """Return empty string-to-Any dict."""
    return {}


def _empty_experiment_list_list() -> list[ExperimentList]:
    """Return empty ExperimentList list."""
    return []


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
    list_metadata: dict[str, Any] = Field(
        default_factory=_empty_any_dict, description="List metadata"
    )
    balance_metrics: dict[str, Any] = Field(
        default_factory=_empty_any_dict, description="Balance metrics"
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


class ListCollection(BeadBaseModel):
    """A collection of experimental lists.

    Contains multiple ExperimentList instances along with metadata about
    the partitioning process that created them.

    Attributes
    ----------
    name : str
        Name of this collection.
    source_items_id : UUID
        UUID of source ItemCollection.
    lists : list[ExperimentList]
        The experimental lists.
    partitioning_strategy : str
        Strategy used for partitioning (e.g., "balanced", "random", "stratified").
    partitioning_config : dict[str, Any]
        Configuration for partitioning.
    partitioning_stats : dict[str, Any]
        Statistics about the partitioning process.

    Examples
    --------
    >>> from uuid import uuid4
    >>> collection = ListCollection(
    ...     name="my_lists",
    ...     source_items_id=uuid4(),
    ...     partitioning_strategy="balanced"
    ... )
    >>> exp_list = ExperimentList(name="list_0", list_number=0)
    >>> collection.add_list(exp_list)
    >>> len(collection.lists)
    1
    """

    name: str = Field(..., description="Collection name")
    source_items_id: UUID = Field(..., description="Source ItemCollection UUID")
    lists: list[ExperimentList] = Field(
        default_factory=_empty_experiment_list_list, description="Experimental lists"
    )
    partitioning_strategy: str = Field(..., description="Partitioning strategy used")
    partitioning_config: dict[str, Any] = Field(
        default_factory=_empty_any_dict, description="Partitioning configuration"
    )
    partitioning_stats: dict[str, Any] = Field(
        default_factory=_empty_any_dict, description="Partitioning statistics"
    )

    @field_validator("name", "partitioning_strategy")
    @classmethod
    def validate_non_empty_string(cls, v: str) -> str:
        """Validate string fields are non-empty.

        Parameters
        ----------
        v : str
            String to validate.

        Returns
        -------
        str
            Validated string (whitespace stripped).

        Raises
        ------
        ValueError
            If string is empty or contains only whitespace.
        """
        if not v or not v.strip():
            raise ValueError("Field must be non-empty")
        return v.strip()

    @field_validator("lists")
    @classmethod
    def validate_unique_list_numbers(
        cls, v: list[ExperimentList]
    ) -> list[ExperimentList]:
        """Validate all list_numbers are unique.

        Parameters
        ----------
        v : list[ExperimentList]
            Lists to validate.

        Returns
        -------
        list[ExperimentList]
            Validated lists.

        Raises
        ------
        ValueError
            If duplicate list_numbers found.
        """
        if not v:
            return v

        list_numbers = [exp_list.list_number for exp_list in v]
        if len(list_numbers) != len(set(list_numbers)):
            duplicates = [num for num in list_numbers if list_numbers.count(num) > 1]
            raise ValueError(f"Duplicate list_numbers found: {set(duplicates)}")

        return v

    def add_list(self, exp_list: ExperimentList) -> None:
        """Add a list to the collection.

        Parameters
        ----------
        exp_list : ExperimentList
            List to add.

        Examples
        --------
        >>> from uuid import uuid4
        >>> collection = ListCollection(
        ...     name="test",
        ...     source_items_id=uuid4(),
        ...     partitioning_strategy="balanced"
        ... )
        >>> exp_list = ExperimentList(name="list_0", list_number=0)
        >>> collection.add_list(exp_list)
        >>> len(collection.lists)
        1
        """
        self.lists.append(exp_list)
        self.update_modified_time()

    def get_list_by_number(self, list_number: int) -> ExperimentList | None:
        """Get a list by its number.

        Parameters
        ----------
        list_number : int
            List number to search for.

        Returns
        -------
        ExperimentList | None
            List with matching number, or None if not found.

        Examples
        --------
        >>> from uuid import uuid4
        >>> collection = ListCollection(
        ...     name="test",
        ...     source_items_id=uuid4(),
        ...     partitioning_strategy="balanced"
        ... )
        >>> exp_list = ExperimentList(name="list_0", list_number=0)
        >>> collection.add_list(exp_list)
        >>> found = collection.get_list_by_number(0)
        >>> found is not None
        True
        """
        for exp_list in self.lists:
            if exp_list.list_number == list_number:
                return exp_list
        return None

    def get_all_item_refs(self) -> list[UUID]:
        """Return all unique item UUIDs across all lists.

        Returns
        -------
        list[UUID]
            All unique item UUIDs.

        Examples
        --------
        >>> from uuid import uuid4
        >>> collection = ListCollection(
        ...     name="test",
        ...     source_items_id=uuid4(),
        ...     partitioning_strategy="balanced"
        ... )
        >>> exp_list = ExperimentList(name="list_0", list_number=0)
        >>> item_id = uuid4()
        >>> exp_list.add_item(item_id)
        >>> collection.add_list(exp_list)
        >>> item_id in collection.get_all_item_refs()
        True
        """
        all_refs: set[UUID] = set()
        for exp_list in self.lists:
            all_refs.update(exp_list.item_refs)
        return list(all_refs)

    def validate_coverage(self, all_item_ids: set[UUID]) -> dict[str, Any]:
        """Check that all items are assigned exactly once.

        Validates that:
        - All items in all_item_ids are assigned to at least one list
        - No item appears in multiple lists (items assigned exactly once)

        Parameters
        ----------
        all_item_ids : set[UUID]
            Set of all item UUIDs that should be assigned.

        Returns
        -------
        dict[str, Any]
            Validation report with keys:
            - "valid": bool - Whether validation passed
            - "missing_items": list[UUID] - Items not assigned to any list
            - "duplicate_items": list[UUID] - Items assigned to multiple lists
            - "total_assigned": int - Total assignments across all lists

        Examples
        --------
        >>> from uuid import uuid4
        >>> collection = ListCollection(
        ...     name="test",
        ...     source_items_id=uuid4(),
        ...     partitioning_strategy="balanced"
        ... )
        >>> item_id = uuid4()
        >>> exp_list = ExperimentList(name="list_0", list_number=0)
        >>> exp_list.add_item(item_id)
        >>> collection.add_list(exp_list)
        >>> result = collection.validate_coverage({item_id})
        >>> result["valid"]
        True
        """
        # Count assignments for each item
        item_counts: dict[UUID, int] = {}
        for exp_list in self.lists:
            for item_id in exp_list.item_refs:
                item_counts[item_id] = item_counts.get(item_id, 0) + 1

        # Find missing items (in all_item_ids but not assigned)
        assigned_items = set(item_counts.keys())
        missing_items = list(all_item_ids - assigned_items)

        # Find duplicate items (assigned more than once)
        duplicate_items = [
            item_id for item_id, count in item_counts.items() if count > 1
        ]

        # Validation passes if no missing and no duplicates
        valid = len(missing_items) == 0 and len(duplicate_items) == 0

        return {
            "valid": valid,
            "missing_items": missing_items,
            "duplicate_items": duplicate_items,
            "total_assigned": sum(item_counts.values()),
        }
