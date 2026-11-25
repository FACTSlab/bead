"""Normalization of append-only JATOS result records."""

from __future__ import annotations

import json
from dataclasses import dataclass

from bead.data.base import JsonValue

_RECORD_TYPES = ("session", "plan", "trial", "event", "end")


@dataclass(frozen=True)
class JATOSResultStream:
    """Raw arrival order and a sequence-ordered view of a complete or partial run.

    Missing ranges include the first and last missing sequence numbers. A
    terminal record indicates that the browser attempted to close the stream.
    Its status and failed sequence numbers are recorded in the record itself.
    """

    records: tuple[dict[str, JsonValue], ...]
    ordered_records: tuple[dict[str, JsonValue], ...]
    missing_seq_ranges: tuple[tuple[int, int], ...]
    malformed_line_numbers: tuple[int, ...]
    session: dict[str, JsonValue] | None
    terminal: dict[str, JsonValue] | None

    @property
    def trials(self) -> tuple[dict[str, JsonValue], ...]:
        """Return one trial per delivered sequence, in trial order."""
        return tuple(r for r in self.ordered_records if r.get("record_type") == "trial")


def parse_jatos_result_stream(
    payload: str | list[JsonValue] | tuple[JsonValue, ...],
) -> JATOSResultStream | None:
    """Parse JSONL text or a sequence of records emitted by a streaming study.

    Parameters
    ----------
    payload
        Raw JSONL text or a parsed sequence of result records.

    Returns
    -------
    JATOSResultStream | None
        The normalized stream, or None for an ordinary aggregate payload.
        For repeated sequence numbers, the first arrival is used in the
        sequence view and every delivered record is kept in the raw view.
    """
    malformed_lines: list[int] = []
    parsed: list[JsonValue]
    if isinstance(payload, str):
        parsed = []
        for index, line in enumerate(payload.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                record: JsonValue = json.loads(line)
            except json.JSONDecodeError:
                malformed_lines.append(index)
            else:
                parsed.append(record)
    else:
        parsed = list(payload)

    records = tuple(record for record in parsed if isinstance(record, dict))
    if not any(r.get("record_type") in _RECORD_TYPES for r in records):
        return None

    indexed: dict[int, dict[str, JsonValue]] = {}
    for record in records:
        seq = record.get("seq")
        if type(seq) is int and seq >= 0:
            indexed.setdefault(seq, record)
    missing: list[tuple[int, int]] = []
    ordered: list[dict[str, JsonValue]] = []
    previous = -1
    for seq, record in sorted(indexed.items()):
        if seq > previous + 1:
            missing.append((previous + 1, seq - 1))
        previous = seq
        ordered.append(record)
    return JATOSResultStream(
        records=records,
        ordered_records=tuple(ordered),
        missing_seq_ranges=tuple(missing),
        malformed_line_numbers=tuple(malformed_lines),
        session=next(
            (record for record in ordered if record.get("record_type") == "session"),
            None,
        ),
        terminal=next(
            (
                record
                for record in reversed(ordered)
                if record.get("record_type") == "end"
            ),
            None,
        ),
    )
