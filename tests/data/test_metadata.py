"""Tests for metadata tracking models."""

from __future__ import annotations

import time
from uuid import uuid4

from bead.data.metadata import MetadataTracker, ProcessingRecord, ProvenanceRecord


# ProvenanceRecord tests
def test_provenance_record_creation() -> None:
    """Test creating a provenance record with all fields."""
    parent_id = uuid4()
    record = ProvenanceRecord(
        parent_id=parent_id, parent_type="Template", relationship="filled_from"
    )

    assert record.parent_id == parent_id
    assert record.parent_type == "Template"
    assert record.relationship == "filled_from"
    assert record.id is not None  # BeadBaseModel field
    assert record.created_at is not None  # BeadBaseModel field


def test_provenance_record_has_timestamp() -> None:
    """Test that provenance record has automatic timestamp."""
    parent_id = uuid4()
    record = ProvenanceRecord(
        parent_id=parent_id, parent_type="LexicalItem", relationship="derived_from"
    )

    assert record.timestamp is not None
    assert record.timestamp.tzinfo is not None  # Has timezone info


def test_provenance_record_serialization() -> None:
    """Test serializing and deserializing a provenance record."""
    parent_id = uuid4()
    record = ProvenanceRecord(
        parent_id=parent_id, parent_type="Template", relationship="filled_from"
    )

    # Serialize
    data = record.model_dump()
    assert data["parent_id"] == parent_id  # UUID object, not string
    assert data["parent_type"] == "Template"
    assert data["relationship"] == "filled_from"

    # Deserialize
    restored = ProvenanceRecord.model_validate(data)
    assert restored.parent_id == parent_id
    assert restored.parent_type == record.parent_type
    assert restored.relationship == record.relationship


# ProcessingRecord tests
def test_processing_record_creation() -> None:
    """Test creating a processing record with all fields."""
    record = ProcessingRecord(
        operation="fill_template",
        parameters={"strategy": "exhaustive", "max_items": 100},
        operator="TemplateFiller-v1.0",
    )

    assert record.operation == "fill_template"
    assert record.parameters["strategy"] == "exhaustive"
    assert record.parameters["max_items"] == 100
    assert record.operator == "TemplateFiller-v1.0"
    assert record.timestamp is not None


def test_processing_record_default_parameters() -> None:
    """Test that processing record has empty dict as default parameters."""
    record = ProcessingRecord(operation="test_operation")

    assert record.parameters == {}
    assert isinstance(record.parameters, dict)


def test_processing_record_optional_operator() -> None:
    """Test that operator field can be None."""
    record = ProcessingRecord(operation="test_operation", parameters={"key": "value"})

    assert record.operator is None


# MetadataTracker tests
def test_metadata_tracker_creation() -> None:
    """Test creating an empty metadata tracker."""
    tracker = MetadataTracker()

    assert tracker.provenance == []
    assert tracker.processing_history == []
    assert tracker.custom_metadata == {}
    assert tracker.id is not None  # BeadBaseModel field


def test_add_provenance() -> None:
    """Test adding a provenance record."""
    tracker = MetadataTracker()
    parent_id = uuid4()

    tracker.add_provenance(parent_id, "Template", "filled_from")

    assert len(tracker.provenance) == 1
    assert tracker.provenance[0].parent_id == parent_id
    assert tracker.provenance[0].parent_type == "Template"
    assert tracker.provenance[0].relationship == "filled_from"


def test_add_provenance_sets_timestamp() -> None:
    """Test that add_provenance automatically sets timestamp."""
    tracker = MetadataTracker()
    parent_id = uuid4()

    tracker.add_provenance(parent_id, "Template", "filled_from")

    record = tracker.provenance[0]
    assert record.timestamp is not None
    assert record.timestamp.tzinfo is not None


def test_add_multiple_provenance() -> None:
    """Test adding multiple provenance records preserves order."""
    tracker = MetadataTracker()
    parent1 = uuid4()
    parent2 = uuid4()
    parent3 = uuid4()

    tracker.add_provenance(parent1, "Template", "filled_from")
    time.sleep(0.01)  # Small delay to ensure different timestamps
    tracker.add_provenance(parent2, "LexicalItem", "derived_from")
    time.sleep(0.01)
    tracker.add_provenance(parent3, "Constraint", "filtered_by")

    assert len(tracker.provenance) == 3
    assert tracker.provenance[0].parent_id == parent1
    assert tracker.provenance[1].parent_id == parent2
    assert tracker.provenance[2].parent_id == parent3


def test_add_processing() -> None:
    """Test adding a processing record."""
    tracker = MetadataTracker()

    tracker.add_processing("fill_template", {"strategy": "exhaustive"})

    assert len(tracker.processing_history) == 1
    assert tracker.processing_history[0].operation == "fill_template"
    assert tracker.processing_history[0].parameters["strategy"] == "exhaustive"


def test_add_processing_with_parameters() -> None:
    """Test adding processing record with parameters stores them correctly."""
    tracker = MetadataTracker()
    params = {"strategy": "exhaustive", "max_items": 100, "timeout": 30}

    tracker.add_processing("fill_template", params, "TemplateFiller-v1.0")

    record = tracker.processing_history[0]
    assert record.parameters == params
    assert record.operator == "TemplateFiller-v1.0"


def test_add_processing_without_operator() -> None:
    """Test adding processing record without operator sets it to None."""
    tracker = MetadataTracker()

    tracker.add_processing("test_operation", {"key": "value"})

    record = tracker.processing_history[0]
    assert record.operator is None


def test_get_provenance_chain() -> None:
    """Test getting provenance chain as list of UUIDs."""
    tracker = MetadataTracker()
    parent1 = uuid4()
    parent2 = uuid4()
    parent3 = uuid4()

    tracker.add_provenance(parent1, "Template", "filled_from")
    tracker.add_provenance(parent2, "LexicalItem", "derived_from")
    tracker.add_provenance(parent3, "Constraint", "filtered_by")

    chain = tracker.get_provenance_chain()

    assert len(chain) == 3
    assert chain[0] == parent1
    assert chain[1] == parent2
    assert chain[2] == parent3


def test_get_recent_processing() -> None:
    """Test getting N most recent processing records."""
    tracker = MetadataTracker()

    # Add 5 processing records
    for i in range(5):
        tracker.add_processing(f"operation{i}", {"index": i})
        time.sleep(0.01)  # Small delay to ensure different timestamps

    # Get 3 most recent
    recent = tracker.get_recent_processing(n=3)

    assert len(recent) == 3
    # Should be in reverse order (newest first)
    assert recent[0].operation == "operation4"
    assert recent[1].operation == "operation3"
    assert recent[2].operation == "operation2"


def test_get_recent_processing_fewer_than_n() -> None:
    """Test getting recent processing when there are fewer than N records."""
    tracker = MetadataTracker()

    tracker.add_processing("operation1")
    tracker.add_processing("operation2")

    recent = tracker.get_recent_processing(n=5)

    assert len(recent) == 2
    assert recent[0].operation == "operation2"
    assert recent[1].operation == "operation1"


def test_custom_metadata() -> None:
    """Test adding custom metadata to tracker."""
    tracker = MetadataTracker()

    tracker.custom_metadata["author"] = "Alice"
    tracker.custom_metadata["project"] = "bead"
    tracker.custom_metadata["version"] = "1.0"

    assert tracker.custom_metadata["author"] == "Alice"
    assert tracker.custom_metadata["project"] == "bead"
    assert tracker.custom_metadata["version"] == "1.0"


def test_metadata_serialization_roundtrip() -> None:
    """Test serializing and deserializing full metadata tracker."""
    tracker = MetadataTracker()
    parent_id = uuid4()

    # Add data
    tracker.add_provenance(parent_id, "Template", "filled_from")
    tracker.add_processing("fill_template", {"strategy": "exhaustive"})
    tracker.custom_metadata["test"] = "value"

    # Serialize
    data = tracker.model_dump()
    assert len(data["provenance"]) == 1
    assert len(data["processing_history"]) == 1
    assert data["custom_metadata"]["test"] == "value"

    # Deserialize
    restored = MetadataTracker.model_validate(data)
    assert len(restored.provenance) == 1
    assert restored.provenance[0].parent_id == parent_id
    assert len(restored.processing_history) == 1
    assert restored.processing_history[0].operation == "fill_template"
    assert restored.custom_metadata["test"] == "value"
