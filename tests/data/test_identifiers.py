"""Tests for UUIDv7 generation and utilities."""

from __future__ import annotations

import time
from uuid import uuid4

from bead.data.identifiers import extract_timestamp, generate_uuid, is_valid_uuid7


def test_generate_uuid_returns_valid_uuid7() -> None:
    """Test that generated UUID is valid UUIDv7."""
    uuid = generate_uuid()
    assert is_valid_uuid7(uuid)


def test_generate_uuid_is_unique() -> None:
    """Test that multiple UUIDs are unique."""
    uuids = [generate_uuid() for _ in range(100)]
    assert len(set(uuids)) == 100


def test_generate_uuid_is_time_ordered() -> None:
    """Test that UUIDs are time-ordered."""
    uuid1 = generate_uuid()
    time.sleep(0.001)  # Small delay to ensure different timestamps
    uuid2 = generate_uuid()
    time.sleep(0.001)
    uuid3 = generate_uuid()

    # UUIDs should be sortable by time
    assert uuid1 < uuid2 < uuid3


def test_extract_timestamp() -> None:
    """Test extracting timestamp from UUIDv7."""
    uuid = generate_uuid()
    timestamp = extract_timestamp(uuid)

    # Timestamp should be in milliseconds since Unix epoch
    current_time = int(time.time() * 1000)

    # Should be within 1 second of current time
    assert abs(timestamp - current_time) < 1000


def test_is_valid_uuid7_with_valid_uuid() -> None:
    """Test validation with valid UUIDv7."""
    uuid = generate_uuid()
    assert is_valid_uuid7(uuid)


def test_is_valid_uuid7_with_invalid_uuid() -> None:
    """Test validation with non-UUIDv7."""
    uuid = uuid4()  # This is version 4
    assert not is_valid_uuid7(uuid)


def test_extract_timestamp_ordering() -> None:
    """Test that extracted timestamps are ordered."""
    uuid1 = generate_uuid()
    time.sleep(0.001)
    uuid2 = generate_uuid()
    time.sleep(0.001)
    uuid3 = generate_uuid()

    ts1 = extract_timestamp(uuid1)
    ts2 = extract_timestamp(uuid2)
    ts3 = extract_timestamp(uuid3)

    assert ts1 < ts2 < ts3
