"""Metadata tracking models for provenance and processing history.

This module provides models for tracking provenance chains and processing history
for all bead objects. This enables full traceability of data transformations.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import Field

from bead.data.base import BeadBaseModel
from bead.data.timestamps import now_iso8601


def _empty_provenance_list() -> list[ProvenanceRecord]:
    """Create empty provenance list."""
    return []


def _empty_processing_list() -> list[ProcessingRecord]:
    """Create empty processing list."""
    return []


class ProvenanceRecord(BeadBaseModel):
    """Record of a provenance relationship between objects.

    Tracks a single parent-child relationship in the provenance chain, including
    what the parent was, its type, and the nature of the relationship.

    Attributes
    ----------
    parent_id : UUID
        UUID of the parent object in the provenance chain
    parent_type : str
        Type name of the parent object (e.g., "LexicalItem", "Template")
    relationship : str
        Type of relationship (e.g., "derived_from", "filled_from", "generated_from")
    timestamp : datetime
        When this relationship was established (UTC with timezone)

    Examples
    --------
    >>> from uuid import uuid4
    >>> parent_id = uuid4()
    >>> record = ProvenanceRecord(
    ...     parent_id=parent_id,
    ...     parent_type="Template",
    ...     relationship="filled_from"
    ... )
    >>> record.parent_type
    'Template'
    >>> record.timestamp is not None
    True
    """

    parent_id: UUID
    parent_type: str
    relationship: str
    timestamp: datetime = Field(default_factory=now_iso8601)


class ProcessingRecord(BeadBaseModel):
    """Record of a processing operation applied to an object.

    Tracks a single operation in the processing history, including the operation
    name, parameters used, when it was performed, and who/what performed it.

    Attributes
    ----------
    operation : str
        Name of the operation (e.g., "fill_template", "apply_constraint", "filter")
    parameters : dict[str, Any]
        Parameters passed to the operation (default: empty dict)
    timestamp : datetime
        When the operation was performed (UTC with timezone)
    operator : str | None
        Who/what performed the operation (e.g., "TemplateFiller-v1.0", user ID)
        (default: None)

    Examples
    --------
    >>> record = ProcessingRecord(
    ...     operation="fill_template",
    ...     parameters={"strategy": "exhaustive", "max_items": 100},
    ...     operator="TemplateFiller-v1.0"
    ... )
    >>> record.operation
    'fill_template'
    >>> record.parameters["strategy"]
    'exhaustive'
    >>> record.timestamp is not None
    True
    """

    operation: str
    parameters: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=now_iso8601)
    operator: str | None = None


class MetadataTracker(BeadBaseModel):
    """Metadata tracking for provenance and processing history.

    Tracks both provenance (where data came from) and processing history
    (what operations were applied) for complete data lineage.

    Attributes
    ----------
    provenance : list[ProvenanceRecord]
        Chain of provenance relationships (default: empty list)
    processing_history : list[ProcessingRecord]
        History of processing operations (default: empty list)
    custom_metadata : dict[str, Any]
        Custom metadata fields (default: empty dict)

    Examples
    --------
    >>> from uuid import uuid4
    >>> tracker = MetadataTracker()
    >>> parent_id = uuid4()
    >>> tracker.add_provenance(parent_id, "Template", "filled_from")
    >>> tracker.add_processing("fill_template", {"strategy": "exhaustive"})
    >>> len(tracker.provenance)
    1
    >>> len(tracker.processing_history)
    1
    >>> chain = tracker.get_provenance_chain()
    >>> len(chain)
    1
    """

    provenance: list[ProvenanceRecord] = Field(default_factory=_empty_provenance_list)
    processing_history: list[ProcessingRecord] = Field(
        default_factory=_empty_processing_list
    )
    custom_metadata: dict[str, Any] = Field(default_factory=dict)

    def add_provenance(
        self, parent_id: UUID, parent_type: str, relationship: str
    ) -> None:
        """Add a provenance record to the chain.

        Creates a new provenance record and adds it to the provenance list.
        The timestamp is automatically set to the current time.

        Parameters
        ----------
        parent_id : UUID
            UUID of the parent object
        parent_type : str
            Type name of the parent object (e.g., "Template", "LexicalItem")
        relationship : str
            Type of relationship (e.g., "derived_from", "filled_from")

        Examples
        --------
        >>> from uuid import uuid4
        >>> tracker = MetadataTracker()
        >>> parent_id = uuid4()
        >>> tracker.add_provenance(parent_id, "Template", "filled_from")
        >>> len(tracker.provenance)
        1
        >>> tracker.provenance[0].parent_type
        'Template'
        """
        record = ProvenanceRecord(
            parent_id=parent_id, parent_type=parent_type, relationship=relationship
        )
        self.provenance.append(record)

    def add_processing(
        self,
        operation: str,
        parameters: dict[str, Any] | None = None,
        operator: str | None = None,
    ) -> None:
        """Add a processing record to the history.

        Creates a new processing record and adds it to the processing history.
        The timestamp is automatically set to the current time.

        Parameters
        ----------
        operation : str
            Name of the operation performed
        parameters : dict[str, Any] | None, optional
            Parameters passed to the operation (default: None, which creates empty dict)
        operator : str | None, optional
            Who/what performed the operation (default: None)

        Examples
        --------
        >>> tracker = MetadataTracker()
        >>> tracker.add_processing("fill_template", {"strategy": "exhaustive"})
        >>> len(tracker.processing_history)
        1
        >>> tracker.processing_history[0].operation
        'fill_template'
        >>> tracker.add_processing("filter", operator="FilterSystem-v2.0")
        >>> tracker.processing_history[1].operator
        'FilterSystem-v2.0'
        """
        if parameters is None:
            parameters = {}
        record = ProcessingRecord(
            operation=operation, parameters=parameters, operator=operator
        )
        self.processing_history.append(record)

    def get_provenance_chain(self) -> list[UUID]:
        """Get the full provenance chain as a list of parent UUIDs.

        Returns the parent UUIDs in the order they were added to the provenance list.

        Returns
        -------
        list[UUID]
            List of parent UUIDs in chronological order

        Examples
        --------
        >>> from uuid import uuid4
        >>> tracker = MetadataTracker()
        >>> parent1 = uuid4()
        >>> parent2 = uuid4()
        >>> tracker.add_provenance(parent1, "Template", "filled_from")
        >>> tracker.add_provenance(parent2, "LexicalItem", "derived_from")
        >>> chain = tracker.get_provenance_chain()
        >>> len(chain)
        2
        >>> chain[0] == parent1
        True
        """
        return [record.parent_id for record in self.provenance]

    def get_recent_processing(self, n: int = 5) -> list[ProcessingRecord]:
        """Get the N most recent processing records.

        Returns the most recent processing records, up to N records. If there
        are fewer than N records, returns all available records.

        Parameters
        ----------
        n : int, optional
            Number of recent records to return (default: 5)

        Returns
        -------
        list[ProcessingRecord]
            List of up to N most recent processing records, newest first

        Examples
        --------
        >>> tracker = MetadataTracker()
        >>> tracker.add_processing("operation1")
        >>> tracker.add_processing("operation2")
        >>> tracker.add_processing("operation3")
        >>> recent = tracker.get_recent_processing(n=2)
        >>> len(recent)
        2
        >>> recent[0].operation
        'operation3'
        >>> recent[1].operation
        'operation2'
        """
        return list(reversed(self.processing_history[-n:]))
