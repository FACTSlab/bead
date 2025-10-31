"""UUIDv7 generation and utilities for bead package.

This module provides functions for generating time-ordered UUIDv7 identifiers,
extracting timestamps from them, and validating UUID versions.
"""

from __future__ import annotations

from uuid import UUID

import uuid_utils


def generate_uuid() -> UUID:
    """Generate a time-ordered UUIDv7.

    UUIDv7 is a time-ordered UUID format that embeds a timestamp in the first
    48 bits, making UUIDs sortable by creation time. This is useful for
    maintaining chronological ordering of database records.

    Returns
    -------
    UUID
        A newly generated UUIDv7 with embedded timestamp

    Examples
    --------
    >>> uuid1 = generate_uuid()
    >>> uuid2 = generate_uuid()
    >>> uuid1 < uuid2  # UUIDs are time-ordered
    True
    """
    # Convert uuid_utils.UUID to standard Python UUID for Pydantic compatibility
    uuid7 = uuid_utils.uuid7()
    return UUID(str(uuid7))


def extract_timestamp(uuid: UUID) -> int:
    """Extract timestamp in milliseconds from a UUIDv7.

    The timestamp is stored in the first 48 bits of the UUID and represents
    milliseconds since Unix epoch (January 1, 1970 00:00:00 UTC).

    Parameters
    ----------
    uuid : UUID
        The UUIDv7 to extract timestamp from

    Returns
    -------
    int
        Timestamp in milliseconds since Unix epoch

    Examples
    --------
    >>> import time
    >>> uuid = generate_uuid()
    >>> timestamp = extract_timestamp(uuid)
    >>> current_time = int(time.time() * 1000)
    >>> abs(timestamp - current_time) < 1000  # Within 1 second
    True
    """
    # UUIDv7 stores timestamp in first 48 bits (6 bytes)
    # UUID.bytes gives us the UUID as 16 bytes
    # Extract first 6 bytes and convert to milliseconds
    timestamp_bytes = uuid.bytes[:6]
    timestamp_ms = int.from_bytes(timestamp_bytes, byteorder="big")
    return timestamp_ms


def is_valid_uuid7(uuid: UUID) -> bool:
    """Check if a UUID is a valid UUIDv7.

    Validates that the UUID has version 7 by checking the version bits
    (bits 48-51) which should be 0111 (7).

    Parameters
    ----------
    uuid : UUID
        The UUID to validate

    Returns
    -------
    bool
        True if the UUID is version 7, False otherwise

    Examples
    --------
    >>> uuid7 = generate_uuid()
    >>> is_valid_uuid7(uuid7)
    True
    >>> from uuid import uuid4
    >>> uuid4_val = uuid4()
    >>> is_valid_uuid7(uuid4_val)
    False
    """
    return uuid.version == 7
