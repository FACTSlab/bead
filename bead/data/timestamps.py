"""ISO 8601 timestamp utilities for bead package.

This module provides functions for creating, parsing, and formatting ISO 8601
timestamps with timezone information. All timestamps use UTC timezone.
"""

from __future__ import annotations

from datetime import UTC, datetime


def now_iso8601() -> datetime:
    """Get current UTC datetime with timezone information.

    Returns the current time in UTC with timezone info attached. This is
    preferred over datetime.utcnow() which is deprecated and doesn't include
    timezone information.

    Returns
    -------
    datetime
        Current UTC datetime with timezone information

    Examples
    --------
    >>> dt = now_iso8601()
    >>> dt.tzinfo is not None
    True
    >>> dt.tzinfo == timezone.utc
    True
    """
    return datetime.now(UTC)


def parse_iso8601(timestamp: str) -> datetime:
    """Parse ISO 8601 timestamp string to datetime.

    Parses an ISO 8601 formatted string into a datetime object. The string
    should include timezone information.

    Parameters
    ----------
    timestamp : str
        ISO 8601 formatted timestamp string (e.g., "2025-10-17T14:23:45.123456+00:00")

    Returns
    -------
    datetime
        Parsed datetime with timezone information

    Examples
    --------
    >>> dt_str = "2025-10-17T14:23:45.123456+00:00"
    >>> dt = parse_iso8601(dt_str)
    >>> dt.year
    2025
    >>> dt.month
    10
    """
    return datetime.fromisoformat(timestamp)


def format_iso8601(dt: datetime) -> str:
    """Format datetime as ISO 8601 string.

    Converts a datetime object to an ISO 8601 formatted string. If the datetime
    doesn't have timezone information, it will be assumed to be UTC.

    Parameters
    ----------
    dt : datetime
        Datetime to format

    Returns
    -------
    str
        ISO 8601 formatted string

    Examples
    --------
    >>> dt = now_iso8601()
    >>> formatted = format_iso8601(dt)
    >>> "+00:00" in formatted or "Z" in formatted
    True
    """
    # If datetime is naive (no timezone), assume UTC
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.isoformat()
