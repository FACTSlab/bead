"""Tests for JSONLines serialization."""

from __future__ import annotations

from pathlib import Path

import pytest

from bead.data.serialization import (
    DeserializationError,
    SerializationError,
    append_jsonlines,
    read_jsonlines,
    stream_jsonlines,
    write_jsonlines,
)
from tests.data.data_helpers import SimpleTestModel


def test_write_jsonlines_creates_file(
    tmp_path: Path, sample_test_objects: list[SimpleTestModel]
) -> None:
    """Test that write_jsonlines creates file."""
    output_path = tmp_path / "output.jsonl"
    write_jsonlines(sample_test_objects, output_path)
    assert output_path.exists()


def test_write_jsonlines_correct_format(
    tmp_path: Path, sample_test_objects: list[SimpleTestModel]
) -> None:
    """Test that write_jsonlines creates correct format."""
    output_path = tmp_path / "output.jsonl"
    write_jsonlines(sample_test_objects, output_path)

    # Read as text and verify format
    lines = output_path.read_text().strip().split("\n")
    assert len(lines) == len(sample_test_objects)

    # Each line should be valid JSON
    for line in lines:
        assert line.startswith("{")
        assert line.endswith("}")


def test_write_jsonlines_append_mode(
    tmp_path: Path, sample_test_objects: list[SimpleTestModel]
) -> None:
    """Test that write_jsonlines append mode works."""
    output_path = tmp_path / "output.jsonl"

    # Write first set
    write_jsonlines(sample_test_objects[:2], output_path)

    # Append second set
    write_jsonlines(sample_test_objects[2:], output_path, append=True)

    # Read back and verify all objects present
    loaded = read_jsonlines(output_path, SimpleTestModel)
    assert len(loaded) == len(sample_test_objects)


def test_write_jsonlines_overwrites_by_default(
    tmp_path: Path, sample_test_objects: list[SimpleTestModel]
) -> None:
    """Test that write_jsonlines overwrites by default."""
    output_path = tmp_path / "output.jsonl"

    # Write first set
    write_jsonlines(sample_test_objects, output_path)

    # Write second set (should overwrite)
    write_jsonlines(sample_test_objects[:1], output_path)

    # Read back and verify only second set present
    loaded = read_jsonlines(output_path, SimpleTestModel)
    assert len(loaded) == 1


def test_write_jsonlines_invalid_path(
    sample_test_objects: list[SimpleTestModel],
) -> None:
    """Test that write_jsonlines raises error for invalid path."""
    # Try to write to non-existent directory
    invalid_path = Path("/nonexistent/directory/output.jsonl")

    with pytest.raises(SerializationError):
        write_jsonlines(sample_test_objects, invalid_path)


def test_read_jsonlines_roundtrip(
    tmp_path: Path, sample_test_objects: list[SimpleTestModel]
) -> None:
    """Test write and read roundtrip."""
    output_path = tmp_path / "output.jsonl"

    write_jsonlines(sample_test_objects, output_path)
    loaded = read_jsonlines(output_path, SimpleTestModel)

    assert len(loaded) == len(sample_test_objects)

    # Compare objects (use model_dump for comparison)
    for orig, loaded_obj in zip(sample_test_objects, loaded, strict=False):
        assert loaded_obj.name == orig.name
        assert loaded_obj.value == orig.value
        assert loaded_obj.id == orig.id


def test_read_jsonlines_empty_file(tmp_path: Path) -> None:
    """Test reading empty file."""
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("")

    loaded = read_jsonlines(empty_file, SimpleTestModel)
    assert loaded == []


def test_read_jsonlines_skips_empty_lines(
    tmp_path: Path, sample_test_objects: list[SimpleTestModel]
) -> None:
    """Test that read_jsonlines skips empty lines."""
    output_path = tmp_path / "output.jsonl"
    write_jsonlines(sample_test_objects, output_path)

    # Add empty lines
    with output_path.open("a") as f:
        f.write("\n\n")

    loaded = read_jsonlines(output_path, SimpleTestModel)
    assert len(loaded) == len(sample_test_objects)


def test_read_jsonlines_skip_errors(tmp_path: Path) -> None:
    """Test read_jsonlines with skip_errors=True."""
    file_path = tmp_path / "corrupted.jsonl"

    # Create file with valid and invalid JSON
    file_path.write_text(
        '{"name": "test1", "value": 1}\n'
        "invalid json line\n"
        '{"name": "test2", "value": 2}\n'
    )

    # Should skip invalid line
    loaded = read_jsonlines(file_path, SimpleTestModel, skip_errors=True)
    assert len(loaded) == 2


def test_read_jsonlines_raises_on_error(tmp_path: Path) -> None:
    """Test read_jsonlines raises error on invalid JSON."""
    file_path = tmp_path / "corrupted.jsonl"

    # Create file with invalid JSON
    file_path.write_text(
        '{"name": "test1", "value": 1}\n'
        "invalid json line\n"
        '{"name": "test2", "value": 2}\n'
    )

    with pytest.raises(DeserializationError):
        read_jsonlines(file_path, SimpleTestModel)


def test_read_jsonlines_nonexistent_file() -> None:
    """Test read_jsonlines with non-existent file."""
    nonexistent = Path("/nonexistent/file.jsonl")

    with pytest.raises(DeserializationError):
        read_jsonlines(nonexistent, SimpleTestModel)


def test_stream_jsonlines_yields_objects(
    tmp_path: Path, sample_test_objects: list[SimpleTestModel]
) -> None:
    """Test that stream_jsonlines yields objects."""
    output_path = tmp_path / "output.jsonl"
    write_jsonlines(sample_test_objects, output_path)

    # Stream should be a generator
    stream = stream_jsonlines(output_path, SimpleTestModel)
    assert hasattr(stream, "__iter__")
    assert hasattr(stream, "__next__")


def test_stream_jsonlines_memory_efficient(
    tmp_path: Path, sample_test_objects: list[SimpleTestModel]
) -> None:
    """Test that stream_jsonlines yields one at a time."""
    output_path = tmp_path / "output.jsonl"
    write_jsonlines(sample_test_objects, output_path)

    # Should be able to iterate one at a time
    stream = stream_jsonlines(output_path, SimpleTestModel)
    first = next(stream)
    assert first.name == "test1"


def test_stream_jsonlines_correct_count(
    tmp_path: Path, sample_test_objects: list[SimpleTestModel]
) -> None:
    """Test that stream_jsonlines yields correct number."""
    output_path = tmp_path / "output.jsonl"
    write_jsonlines(sample_test_objects, output_path)

    # Consume stream and count
    count = sum(1 for _ in stream_jsonlines(output_path, SimpleTestModel))
    assert count == len(sample_test_objects)


def test_append_jsonlines(
    tmp_path: Path, sample_test_objects: list[SimpleTestModel]
) -> None:
    """Test append_jsonlines function."""
    output_path = tmp_path / "output.jsonl"

    # Write initial set
    write_jsonlines(sample_test_objects[:2], output_path)

    # Append more
    append_jsonlines(sample_test_objects[2:], output_path)

    # Verify all present
    loaded = read_jsonlines(output_path, SimpleTestModel)
    assert len(loaded) == len(sample_test_objects)


def test_stream_jsonlines_skips_empty_lines(
    tmp_path: Path, sample_test_objects: list[SimpleTestModel]
) -> None:
    """Test that stream_jsonlines skips empty lines."""
    output_path = tmp_path / "output.jsonl"
    write_jsonlines(sample_test_objects, output_path)

    # Add empty lines
    with output_path.open("a") as f:
        f.write("\n\n")

    # Should still yield correct count
    count = sum(1 for _ in stream_jsonlines(output_path, SimpleTestModel))
    assert count == len(sample_test_objects)


def test_stream_jsonlines_raises_on_error(tmp_path: Path) -> None:
    """Test stream_jsonlines raises error on invalid JSON."""
    file_path = tmp_path / "corrupted.jsonl"

    # Create file with invalid JSON
    file_path.write_text(
        '{"name": "test1", "value": 1}\n'
        "invalid json line\n"
        '{"name": "test2", "value": 2}\n'
    )

    stream = stream_jsonlines(file_path, SimpleTestModel)
    next(stream)  # First one is valid

    # Second line should raise error
    with pytest.raises(DeserializationError):
        next(stream)
