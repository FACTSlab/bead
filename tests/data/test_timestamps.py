"""Tests for ISO 8601 timestamp utilities."""

from __future__ import annotations

import time
from datetime import UTC, datetime

from bead.data.timestamps import format_iso8601, now_iso8601, parse_iso8601


def test_now_iso8601_returns_utc() -> None:
    """Test that now_iso8601 returns UTC timezone."""
    dt = now_iso8601()
    assert dt.tzinfo == UTC


def test_now_iso8601_has_timezone() -> None:
    """Test that now_iso8601 includes timezone info."""
    dt = now_iso8601()
    assert dt.tzinfo is not None


def test_parse_iso8601_roundtrip() -> None:
    """Test parsing and formatting roundtrip."""
    dt = now_iso8601()
    formatted = format_iso8601(dt)
    parsed = parse_iso8601(formatted)

    # Should be equal (within microsecond precision)
    assert abs((parsed - dt).total_seconds()) < 0.000001


def test_format_iso8601_roundtrip() -> None:
    """Test formatting and parsing roundtrip."""
    iso_str = "2025-10-17T14:23:45.123456+00:00"
    dt = parse_iso8601(iso_str)
    formatted = format_iso8601(dt)

    # Should produce same result
    assert parse_iso8601(formatted) == dt


def test_parse_iso8601_with_timezone() -> None:
    """Test parsing ISO string with timezone."""
    iso_str = "2025-10-17T14:23:45.123456+00:00"
    dt = parse_iso8601(iso_str)

    assert dt.year == 2025
    assert dt.month == 10
    assert dt.day == 17
    assert dt.hour == 14
    assert dt.minute == 23
    assert dt.second == 45
    assert dt.tzinfo is not None


def test_format_iso8601_includes_timezone() -> None:
    """Test that formatted string includes timezone."""
    dt = now_iso8601()
    formatted = format_iso8601(dt)

    # Should include timezone info
    assert "+00:00" in formatted or "Z" in formatted


def test_timestamp_ordering() -> None:
    """Test that timestamps are ordered correctly."""
    dt1 = now_iso8601()
    time.sleep(0.001)  # Small delay
    dt2 = now_iso8601()

    assert dt2 > dt1


def test_format_iso8601_naive_datetime() -> None:
    """Test formatting naive datetime assumes UTC."""
    naive_dt = datetime(2025, 10, 17, 14, 23, 45)
    formatted = format_iso8601(naive_dt)

    # Should add UTC timezone
    parsed = parse_iso8601(formatted)
    assert parsed.tzinfo is not None
