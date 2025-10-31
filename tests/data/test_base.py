"""Tests for BeadBaseModel."""

from __future__ import annotations

import time

import pytest
from pydantic import ValidationError

from bead.data.base import BeadBaseModel
from bead.data.identifiers import is_valid_uuid7


class SampleModel(BeadBaseModel):
    """Sample model for base model tests."""

    name: str
    value: int


def test_beadbasemodel_creates_uuid() -> None:
    """Test that BeadBaseModel automatically generates UUID."""
    obj = SampleModel(name="test", value=42)
    assert obj.id is not None
    assert is_valid_uuid7(obj.id)


def test_beadbasemodel_creates_timestamps() -> None:
    """Test that BeadBaseModel automatically creates timestamps."""
    obj = SampleModel(name="test", value=42)
    assert obj.created_at is not None
    assert obj.modified_at is not None
    assert obj.created_at.tzinfo is not None
    assert obj.modified_at.tzinfo is not None


def test_beadbasemodel_default_version() -> None:
    """Test that BeadBaseModel has default version."""
    obj = SampleModel(name="test", value=42)
    assert obj.version == "1.0.0"


def test_beadbasemodel_default_metadata() -> None:
    """Test that BeadBaseModel has default empty metadata."""
    obj = SampleModel(name="test", value=42)
    assert obj.metadata == {}


def test_beadbasemodel_update_modified_time() -> None:
    """Test that update_modified_time updates timestamp."""
    obj = SampleModel(name="test", value=42)
    original_modified = obj.modified_at

    time.sleep(0.01)  # Small delay to ensure different timestamp
    obj.update_modified_time()

    assert obj.modified_at > original_modified


def test_beadbasemodel_forbids_extra_fields() -> None:
    """Test that BeadBaseModel forbids extra fields."""
    with pytest.raises(ValidationError):
        SampleModel(name="test", value=42, extra_field="not allowed")  # type: ignore[call-arg]


def test_beadbasemodel_validates_on_assignment() -> None:
    """Test that BeadBaseModel validates on assignment."""
    obj = SampleModel(name="test", value=42)

    with pytest.raises(ValidationError):
        obj.value = "not an int"  # type: ignore[assignment]


def test_beadbasemodel_timestamps_ordered() -> None:
    """Test that created_at <= modified_at."""
    obj = SampleModel(name="test", value=42)
    assert obj.created_at <= obj.modified_at


def test_beadbasemodel_custom_metadata() -> None:
    """Test that custom metadata can be provided."""
    metadata = {"key": "value", "number": 42}
    obj = SampleModel(name="test", value=42, metadata=metadata)
    assert obj.metadata == metadata


def test_beadbasemodel_custom_version() -> None:
    """Test that custom version can be provided."""
    obj = SampleModel(name="test", value=42, version="2.0.0")
    assert obj.version == "2.0.0"
