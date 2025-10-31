"""Base Pydantic model for all bead objects.

This module provides BeadBaseModel, the foundational Pydantic v2 model that all
bead data models should inherit from. It provides automatic ID generation,
timestamp tracking, and versioning.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from bead.data.identifiers import generate_uuid
from bead.data.timestamps import now_iso8601


class BeadBaseModel(BaseModel):
    """Base Pydantic model for all bead objects.

    This model provides foundational fields and configuration that all bead
    data models inherit. It includes automatic ID generation using UUIDv7,
    timestamp tracking for creation and modification, versioning, and metadata.

    Attributes
    ----------
    id : UUID
        Unique identifier (UUIDv7) automatically generated on creation
    created_at : datetime
        UTC timestamp when object was created
    modified_at : datetime
        UTC timestamp when object was last modified
    version : str
        Version string for schema versioning (default: "1.0.0")
    metadata : dict[str, Any]
        Optional metadata dictionary for arbitrary key-value pairs

    Examples
    --------
    >>> class MyModel(BeadBaseModel):
    ...     name: str
    ...     value: int
    >>> obj = MyModel(name="test", value=42)
    >>> obj.id  # doctest: +SKIP
    UUID('...')
    >>> obj.version
    '1.0.0'
    >>> obj.update_modified_time()
    >>> obj.modified_at > obj.created_at
    True
    """

    model_config = ConfigDict(
        extra="forbid",  # Disallow extra fields not defined in model
        frozen=False,  # Allow modification after creation
        validate_assignment=True,  # Validate when assigning to fields
    )

    id: UUID = Field(default_factory=generate_uuid)
    created_at: datetime = Field(default_factory=now_iso8601)
    modified_at: datetime = Field(default_factory=now_iso8601)
    version: str = Field(default="1.0.0")
    metadata: dict[str, Any] = Field(default_factory=dict)

    def update_modified_time(self) -> None:
        """Update the modified_at timestamp to current UTC time.

        This method should be called whenever the object is modified to
        maintain accurate modification tracking.

        Examples
        --------
        >>> obj = BeadBaseModel()
        >>> original_time = obj.modified_at
        >>> import time
        >>> time.sleep(0.01)  # Small delay to ensure different timestamp
        >>> obj.update_modified_time()
        >>> obj.modified_at > original_time
        True
        """
        self.modified_at = now_iso8601()
