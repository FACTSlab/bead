"""Tests for validation utilities."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

from bead.data.base import BeadBaseModel
from bead.data.metadata import MetadataTracker
from bead.data.serialization import write_jsonlines
from bead.data.validation import (
    ValidationReport,
    validate_jsonlines_file,
    validate_provenance_chain,
    validate_uuid_references,
)


# Test models
class SimpleItem(BeadBaseModel):
    """Simple model with UUID reference."""

    name: str
    parent_id: UUID | None = None


class ItemWithList(BeadBaseModel):
    """Model with list of UUID references."""

    name: str
    parent_ids: list[UUID] = []


class Template(BeadBaseModel):
    """Template model."""

    name: str


# ValidationReport tests
def test_validation_report_creation() -> None:
    """Test creating a validation report with defaults."""
    report = ValidationReport(valid=True)

    assert report.valid is True
    assert report.errors == []
    assert report.warnings == []
    assert report.object_count == 0


def test_validation_report_add_error() -> None:
    """Test adding an error sets valid to False."""
    report = ValidationReport(valid=True)

    report.add_error("Something went wrong")

    assert report.valid is False
    assert len(report.errors) == 1
    assert "Something went wrong" in report.errors


def test_validation_report_add_warning() -> None:
    """Test adding a warning does not change valid status."""
    report = ValidationReport(valid=True)

    report.add_warning("This might be an issue")

    assert report.valid is True  # Still valid
    assert len(report.warnings) == 1
    assert "This might be an issue" in report.warnings


def test_validation_report_has_errors() -> None:
    """Test has_errors method."""
    report = ValidationReport(valid=True)

    assert not report.has_errors()

    report.add_error("error")
    assert report.has_errors()


def test_validation_report_has_warnings() -> None:
    """Test has_warnings method."""
    report = ValidationReport(valid=True)

    assert not report.has_warnings()

    report.add_warning("warning")
    assert report.has_warnings()


# File validation tests
def test_validate_jsonlines_file_valid(tmp_path: Path) -> None:
    """Test validating a valid JSONLines file."""
    file_path = tmp_path / "valid.jsonl"

    # Create valid file
    objects = [SimpleItem(name="test1"), SimpleItem(name="test2")]
    write_jsonlines(objects, file_path)

    # Validate
    report = validate_jsonlines_file(file_path, SimpleItem)

    assert report.valid is True
    assert len(report.errors) == 0
    assert report.object_count == 2


def test_validate_jsonlines_file_invalid(tmp_path: Path) -> None:
    """Test validating a file with errors in non-strict mode."""
    file_path = tmp_path / "invalid.jsonl"

    # Create file with valid and invalid lines
    file_path.write_text(
        '{"name": "valid", "id": "01234567-89ab-cdef-0123-456789abcdef", '
        '"created_at": "2025-01-01T00:00:00+00:00", '
        '"modified_at": "2025-01-01T00:00:00+00:00", '
        '"version": "1.0.0", "metadata": {}}\n'
        '{"invalid": "no name field"}\n'
        '{"name": "valid2", "id": "01234567-89ab-cdef-0123-456789abcde0", '
        '"created_at": "2025-01-01T00:00:00+00:00", '
        '"modified_at": "2025-01-01T00:00:00+00:00", '
        '"version": "1.0.0", "metadata": {}}\n'
    )

    # Validate in non-strict mode
    report = validate_jsonlines_file(file_path, SimpleItem, strict=False)

    assert report.valid is False
    assert len(report.errors) > 0
    assert report.object_count == 2  # Only valid objects counted


def test_validate_jsonlines_file_strict_mode(tmp_path: Path) -> None:
    """Test strict mode stops at first error."""
    file_path = tmp_path / "invalid.jsonl"

    # Create file with multiple errors
    file_path.write_text('{"invalid1": "error"}\n{"invalid2": "error"}\n')

    # Validate in strict mode
    report = validate_jsonlines_file(file_path, SimpleItem, strict=True)

    assert report.valid is False
    assert len(report.errors) == 1  # Only first error


def test_validate_jsonlines_file_nonstrict_mode(tmp_path: Path) -> None:
    """Test non-strict mode collects all errors."""
    file_path = tmp_path / "invalid.jsonl"

    # Create file with multiple errors
    file_path.write_text('{"invalid1": "error"}\n{"invalid2": "error"}\n')

    # Validate in non-strict mode
    report = validate_jsonlines_file(file_path, SimpleItem, strict=False)

    assert report.valid is False
    assert len(report.errors) == 2  # All errors collected


def test_validate_jsonlines_file_missing_file(tmp_path: Path) -> None:
    """Test validating a nonexistent file."""
    file_path = tmp_path / "nonexistent.jsonl"

    report = validate_jsonlines_file(file_path, SimpleItem)

    assert report.valid is False
    assert len(report.errors) == 1
    assert "File not found" in report.errors[0]


def test_validate_jsonlines_file_empty_lines(tmp_path: Path) -> None:
    """Test that empty lines are skipped."""
    file_path = tmp_path / "with_empty_lines.jsonl"

    # Create file with empty lines
    obj = SimpleItem(name="test")
    file_path.write_text(
        "\n" + obj.model_dump_json() + "\n\n" + obj.model_dump_json() + "\n\n"
    )

    report = validate_jsonlines_file(file_path, SimpleItem)

    assert report.valid is True
    assert report.object_count == 2


def test_validate_jsonlines_file_counts_objects(tmp_path: Path) -> None:
    """Test that object_count is correct."""
    file_path = tmp_path / "count.jsonl"

    objects = [SimpleItem(name=f"test{i}") for i in range(10)]
    write_jsonlines(objects, file_path)

    report = validate_jsonlines_file(file_path, SimpleItem)

    assert report.valid is True
    assert report.object_count == 10


# Reference validation tests
def test_validate_uuid_references_valid() -> None:
    """Test validating objects with valid references."""
    parent = SimpleItem(name="parent")
    child = SimpleItem(name="child", parent_id=parent.id)

    objects = [child]
    reference_pool = {parent.id: parent}

    report = validate_uuid_references(objects, reference_pool)

    assert report.valid is True
    assert len(report.errors) == 0


def test_validate_uuid_references_missing() -> None:
    """Test validating objects with missing references."""
    child = SimpleItem(name="child", parent_id=uuid4())

    objects = [child]
    reference_pool = {}  # Empty pool

    report = validate_uuid_references(objects, reference_pool)

    assert report.valid is False
    assert len(report.errors) == 1
    assert "missing UUID" in report.errors[0]


def test_validate_uuid_references_list_of_uuids() -> None:
    """Test validating list[UUID] fields."""
    parent1 = SimpleItem(name="parent1")
    parent2 = SimpleItem(name="parent2")
    child = ItemWithList(name="child", parent_ids=[parent1.id, parent2.id])

    objects = [child]
    reference_pool = {parent1.id: parent1, parent2.id: parent2}

    report = validate_uuid_references(objects, reference_pool)

    assert report.valid is True
    assert len(report.errors) == 0


def test_validate_uuid_references_mixed_fields() -> None:
    """Test object with both single and list UUID fields."""
    parent1 = SimpleItem(name="parent1")
    parent2 = SimpleItem(name="parent2")
    parent3 = SimpleItem(name="parent3")

    # Create object with both types
    child = ItemWithList(name="child", parent_ids=[parent1.id, parent2.id])

    objects = [child]
    reference_pool = {parent1.id: parent1, parent2.id: parent2, parent3.id: parent3}

    report = validate_uuid_references(objects, reference_pool)

    assert report.valid is True
    assert len(report.errors) == 0


def test_validate_uuid_references_no_uuid_fields() -> None:
    """Test object with no UUID fields."""

    class SimpleModel(BeadBaseModel):
        name: str
        value: int

    obj = SimpleModel(name="test", value=42)
    objects = [obj]
    reference_pool = {}

    report = validate_uuid_references(objects, reference_pool)

    assert report.valid is True
    assert len(report.errors) == 0


# Provenance validation tests
def test_validate_provenance_chain_valid() -> None:
    """Test validating a valid provenance chain."""
    template = Template(name="template")
    metadata = MetadataTracker()
    metadata.add_provenance(template.id, "Template", "filled_from")

    repository = {template.id: template}

    report = validate_provenance_chain(metadata, repository)

    assert report.valid is True
    assert len(report.errors) == 0


def test_validate_provenance_chain_missing_parent() -> None:
    """Test validating chain with missing parent."""
    metadata = MetadataTracker()
    metadata.add_provenance(uuid4(), "Template", "filled_from")

    repository = {}  # Empty repository

    report = validate_provenance_chain(metadata, repository)

    assert report.valid is False
    assert len(report.errors) == 1
    assert "missing parent" in report.errors[0]


def test_validate_provenance_chain_type_mismatch() -> None:
    """Test validating chain with wrong parent type."""
    template = Template(name="template")
    metadata = MetadataTracker()
    # Wrong type name
    metadata.add_provenance(template.id, "WrongType", "filled_from")

    repository = {template.id: template}

    report = validate_provenance_chain(metadata, repository)

    assert report.valid is False
    assert len(report.errors) == 1
    assert "Expected type" in report.errors[0]


def test_validate_provenance_chain_empty() -> None:
    """Test validating empty provenance chain."""
    metadata = MetadataTracker()
    repository = {}

    report = validate_provenance_chain(metadata, repository)

    assert report.valid is True
    assert len(report.errors) == 0
    assert report.object_count == 0


def test_validate_provenance_chain_multiple_parents() -> None:
    """Test validating chain with multiple parents."""
    template1 = Template(name="template1")
    template2 = Template(name="template2")

    metadata = MetadataTracker()
    metadata.add_provenance(template1.id, "Template", "filled_from")
    metadata.add_provenance(template2.id, "Template", "derived_from")

    repository = {template1.id: template1, template2.id: template2}

    report = validate_provenance_chain(metadata, repository)

    assert report.valid is True
    assert len(report.errors) == 0
    assert report.object_count == 2
