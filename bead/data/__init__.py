"""Data infrastructure for bead package.

This module provides core data models, identifiers, timestamps,
serialization utilities, metadata tracking, repository pattern,
and validation utilities.
"""

from __future__ import annotations

from bead.data.base import BeadBaseModel
from bead.data.identifiers import extract_timestamp, generate_uuid, is_valid_uuid7
from bead.data.metadata import (
    MetadataTracker,
    ProcessingRecord,
    ProvenanceRecord,
)
from bead.data.repository import Repository
from bead.data.serialization import (
    DeserializationError,
    SerializationError,
    append_jsonlines,
    read_jsonlines,
    stream_jsonlines,
    write_jsonlines,
)
from bead.data.timestamps import format_iso8601, now_iso8601, parse_iso8601
from bead.data.validation import (
    ValidationReport,
    validate_jsonlines_file,
    validate_provenance_chain,
    validate_uuid_references,
)

__all__ = [
    # Base model
    "BeadBaseModel",
    # Identifiers
    "generate_uuid",
    "extract_timestamp",
    "is_valid_uuid7",
    # Timestamps
    "now_iso8601",
    "parse_iso8601",
    "format_iso8601",
    # Serialization
    "write_jsonlines",
    "read_jsonlines",
    "stream_jsonlines",
    "append_jsonlines",
    "SerializationError",
    "DeserializationError",
    # Metadata
    "MetadataTracker",
    "ProvenanceRecord",
    "ProcessingRecord",
    # Repository
    "Repository",
    # Validation
    "ValidationReport",
    "validate_jsonlines_file",
    "validate_uuid_references",
    "validate_provenance_chain",
]
