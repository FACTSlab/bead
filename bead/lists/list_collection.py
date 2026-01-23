"""List collection data model for managing multiple experimental lists.

This module provides the ListCollection model for managing multiple ExperimentList
instances along with metadata about the partitioning process that created them.

The model supports:
- Multiple experimental lists
- Partitioning metadata tracking
- Coverage validation (ensuring all items are assigned exactly once)
- List lookup by number
"""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict
from uuid import UUID

from pydantic import Field, field_validator

from bead.data.base import BeadBaseModel
from bead.lists.experiment_list import ExperimentList

if TYPE_CHECKING:
    from bead.items.item_template import MetadataValue
else:
    # Recursive type for metadata values
    type MetadataValue = (
        str | int | float | bool | None | dict[str, MetadataValue] | list[MetadataValue]
    )


class CoverageValidationResult(TypedDict):
    """Result of coverage validation."""

    valid: bool
    missing_items: list[UUID]
    duplicate_items: list[UUID]
    total_assigned: int


# Factory functions for default values
def _empty_experiment_list_list() -> list[ExperimentList]:
    """Return empty ExperimentList list."""
    return []


def _empty_metadata_dict() -> dict[str, MetadataValue]:
    """Return empty metadata dictionary."""
    return {}


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
    partitioning_config: dict[str, MetadataValue] = Field(
        default_factory=_empty_metadata_dict, description="Partitioning configuration"
    )
    partitioning_stats: dict[str, MetadataValue] = Field(
        default_factory=_empty_metadata_dict, description="Partitioning statistics"
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

    def validate_coverage(self, all_item_ids: set[UUID]) -> CoverageValidationResult:
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
        CoverageValidationResult
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
