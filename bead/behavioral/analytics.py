"""Behavioral analytics models for bead.

This module provides data models for per-judgment behavioral metrics
and participant-level summaries, linking slopit behavioral data to
bead's item-based experimental structure.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Literal
from uuid import UUID

import pandas as pd
import polars as pl
from pydantic import ConfigDict, Field, computed_field
from slopit.schemas import (
    AnalysisFlag,
    FocusMetrics,
    KeystrokeMetrics,
    Severity,
    TimingMetrics,
)

from bead.data.base import BeadBaseModel, JsonValue
from bead.data.serialization import read_jsonlines, write_jsonlines

# Type alias for supported DataFrame types
DataFrame = pd.DataFrame | pl.DataFrame


def _empty_flag_list() -> list[AnalysisFlag]:
    """Return empty flag list."""
    return []


def _empty_analytics_list() -> list[JudgmentAnalytics]:
    """Return empty analytics list."""
    return []


def _empty_flag_counts() -> dict[str, int]:
    """Return empty flag counts dict."""
    return {}


class JudgmentAnalytics(BeadBaseModel):
    """Behavioral analytics for a single judgment.

    Links slopit behavioral data to a specific item judgment, enabling
    analysis of participant behavior during individual responses.

    Attributes
    ----------
    item_id : UUID
        Reference to the Item being judged.
    participant_id : str
        Participant identifier (from slopit session).
    trial_index : int
        Zero-indexed position in the experiment session.
    session_id : str
        Slopit session identifier.
    response_value : JsonValue
        The participant's response value.
    response_time_ms : int
        Response time in milliseconds.
    keystroke_metrics : KeystrokeMetrics | None
        Keystroke-derived behavioral metrics.
    focus_metrics : FocusMetrics | None
        Focus and visibility metrics.
    timing_metrics : TimingMetrics | None
        Timing-related metrics.
    paste_event_count : int
        Number of paste events during this judgment.
    flags : list[AnalysisFlag]
        Behavioral analysis flags from slopit analyzers.
    max_severity : Severity | None
        Maximum severity among all flags.

    Examples
    --------
    >>> from uuid import uuid4
    >>> analytics = JudgmentAnalytics(
    ...     item_id=uuid4(),
    ...     participant_id="participant_001",
    ...     trial_index=0,
    ...     session_id="session_001",
    ...     response_value=5,
    ...     response_time_ms=2500,
    ... )
    >>> analytics.is_flagged
    False
    """

    # Override to ignore computed_field values during deserialization
    model_config = ConfigDict(extra="ignore")

    # Linkage to judgment
    item_id: UUID = Field(..., description="Item UUID being judged")
    participant_id: str = Field(..., description="Participant identifier")
    trial_index: int = Field(..., ge=0, description="Trial position in session")
    session_id: str = Field(..., description="Slopit session identifier")

    # Response data
    response_value: JsonValue = Field(
        default=None, description="Judgment response value"
    )
    response_time_ms: int = Field(
        ..., ge=0, description="Response time in milliseconds"
    )

    # Behavioral metrics (from slopit)
    keystroke_metrics: KeystrokeMetrics | None = Field(
        default=None, description="Keystroke dynamics metrics"
    )
    focus_metrics: FocusMetrics | None = Field(
        default=None, description="Focus and visibility metrics"
    )
    timing_metrics: TimingMetrics | None = Field(
        default=None, description="Timing metrics"
    )

    # Paste tracking
    paste_event_count: int = Field(
        default=0, ge=0, description="Number of paste events"
    )

    # Flags (from slopit analyzers)
    flags: list[AnalysisFlag] = Field(
        default_factory=_empty_flag_list, description="Analysis flags"
    )
    max_severity: Severity | None = Field(
        default=None, description="Maximum flag severity"
    )

    @computed_field
    @property
    def has_paste_events(self) -> bool:
        """Check if this judgment had any paste events.

        Returns
        -------
        bool
            True if paste events occurred during this judgment.
        """
        return self.paste_event_count > 0

    @computed_field
    @property
    def is_flagged(self) -> bool:
        """Check if this judgment has any behavioral flags.

        Returns
        -------
        bool
            True if any analysis flags are present.
        """
        return len(self.flags) > 0

    def get_flag_types(self) -> list[str]:
        """Get list of flag types for this judgment.

        Returns
        -------
        list[str]
            List of flag type identifiers.
        """
        return [f.type for f in self.flags]


class ParticipantBehavioralSummary(BeadBaseModel):
    """Aggregated behavioral metrics for a participant across all judgments.

    Provides summary statistics useful for identifying participants
    who may require exclusion from analysis.

    Attributes
    ----------
    participant_id : str
        Participant identifier.
    session_id : str
        Slopit session identifier.
    total_judgments : int
        Total number of judgments analyzed.
    flagged_judgments : int
        Number of judgments with behavioral flags.
    mean_response_time_ms : float
        Mean response time across all judgments.
    mean_iki : float | None
        Mean inter-keystroke interval (averaged across judgments).
    total_keystrokes : int
        Total keystrokes across all judgments.
    total_paste_events : int
        Total paste events across all judgments.
    total_blur_events : int
        Total window blur events.
    total_blur_duration_ms : float
        Total time with window blurred in milliseconds.
    flag_counts : dict[str, int]
        Count of each flag type.
    max_severity : Severity | None
        Maximum flag severity across all judgments.

    Examples
    --------
    >>> summary = ParticipantBehavioralSummary(
    ...     participant_id="participant_001",
    ...     session_id="session_001",
    ...     total_judgments=50,
    ...     flagged_judgments=3,
    ...     mean_response_time_ms=2500.0,
    ... )
    >>> summary.flag_rate
    0.06
    """

    # Override to ignore computed_field values during deserialization
    model_config = ConfigDict(extra="ignore")

    participant_id: str = Field(..., description="Participant identifier")
    session_id: str = Field(..., description="Session identifier")

    # Aggregated counts
    total_judgments: int = Field(..., ge=0, description="Total judgments")
    flagged_judgments: int = Field(
        default=0, ge=0, description="Flagged judgment count"
    )
    mean_response_time_ms: float = Field(..., ge=0.0, description="Mean response time")

    # Keystroke aggregates
    mean_iki: float | None = Field(
        default=None, description="Mean inter-keystroke interval"
    )
    total_keystrokes: int = Field(default=0, ge=0, description="Total keystrokes")
    total_paste_events: int = Field(default=0, ge=0, description="Total paste events")

    # Focus aggregates
    total_blur_events: int = Field(default=0, ge=0, description="Total blur events")
    total_blur_duration_ms: float = Field(
        default=0.0, ge=0.0, description="Total blur duration"
    )

    # Flag summary
    flag_counts: dict[str, int] = Field(
        default_factory=_empty_flag_counts, description="Flag type counts"
    )
    max_severity: Severity | None = Field(
        default=None, description="Maximum severity across judgments"
    )

    @computed_field
    @property
    def flag_rate(self) -> float:
        """Calculate proportion of judgments that were flagged.

        Returns
        -------
        float
            Proportion of flagged judgments (0.0 to 1.0).
        """
        if self.total_judgments == 0:
            return 0.0
        return self.flagged_judgments / self.total_judgments

    @computed_field
    @property
    def has_paste_events(self) -> bool:
        """Check if participant had any paste events.

        Returns
        -------
        bool
            True if any paste events occurred.
        """
        return self.total_paste_events > 0


class AnalyticsCollection(BeadBaseModel):
    """Collection of judgment analytics with I/O and filtering support.

    Provides methods for persisting analytics to JSONL files,
    converting to DataFrames, and filtering by behavioral flags.

    Attributes
    ----------
    name : str
        Collection name (e.g., study identifier).
    analytics : list[JudgmentAnalytics]
        List of per-judgment analytics records.

    Examples
    --------
    >>> from uuid import uuid4
    >>> collection = AnalyticsCollection(name="study_001")
    >>> analytics = JudgmentAnalytics(
    ...     item_id=uuid4(),
    ...     participant_id="p001",
    ...     trial_index=0,
    ...     session_id="s001",
    ...     response_time_ms=2000,
    ... )
    >>> collection.add_analytics(analytics)
    >>> len(collection)
    1
    """

    name: str = Field(..., description="Collection name")
    analytics: list[JudgmentAnalytics] = Field(
        default_factory=_empty_analytics_list, description="Analytics records"
    )

    def __len__(self) -> int:
        """Return number of analytics records.

        Returns
        -------
        int
            Number of analytics records in the collection.
        """
        return len(self.analytics)

    def add_analytics(self, analytics: JudgmentAnalytics) -> None:
        """Add a single analytics record to the collection.

        Parameters
        ----------
        analytics : JudgmentAnalytics
            Analytics record to add.
        """
        self.analytics.append(analytics)
        self.update_modified_time()

    def add_many(self, analytics_list: list[JudgmentAnalytics]) -> None:
        """Add multiple analytics records to the collection.

        Parameters
        ----------
        analytics_list : list[JudgmentAnalytics]
            List of analytics records to add.
        """
        self.analytics.extend(analytics_list)
        self.update_modified_time()

    def get_by_participant(self, participant_id: str) -> list[JudgmentAnalytics]:
        """Get all analytics for a specific participant.

        Parameters
        ----------
        participant_id : str
            Participant identifier to filter by.

        Returns
        -------
        list[JudgmentAnalytics]
            Analytics records for the participant.
        """
        return [a for a in self.analytics if a.participant_id == participant_id]

    def get_by_item(self, item_id: UUID) -> list[JudgmentAnalytics]:
        """Get all analytics for a specific item.

        Parameters
        ----------
        item_id : UUID
            Item UUID to filter by.

        Returns
        -------
        list[JudgmentAnalytics]
            Analytics records for the item.
        """
        return [a for a in self.analytics if a.item_id == item_id]

    def filter_flagged(
        self,
        min_severity: Severity | None = None,
        exclude_flagged: bool = False,
    ) -> AnalyticsCollection:
        """Filter analytics by flag status.

        Parameters
        ----------
        min_severity : Severity | None
            If provided, only include analytics with flags at or above this severity.
            Severity order: info < low < medium < high.
        exclude_flagged : bool
            If True, exclude flagged records. If False, include only flagged records.
            Default is False (return flagged records).

        Returns
        -------
        AnalyticsCollection
            New collection with filtered analytics.
        """
        severity_order: dict[str, int] = {"info": 0, "low": 1, "medium": 2, "high": 3}

        def meets_criteria(a: JudgmentAnalytics) -> bool:
            has_flags = a.is_flagged

            if exclude_flagged:
                return not has_flags

            if not has_flags:
                return False

            if min_severity is None:
                return True

            # Check if any flag meets minimum severity
            min_level = severity_order.get(min_severity, 0)
            for flag in a.flags:
                flag_level = severity_order.get(flag.severity, 0)
                if flag_level >= min_level:
                    return True
            return False

        filtered = [a for a in self.analytics if meets_criteria(a)]
        return AnalyticsCollection(name=f"{self.name}_filtered", analytics=filtered)

    def get_participant_ids(self) -> list[str]:
        """Get unique participant IDs in the collection.

        Returns
        -------
        list[str]
            List of unique participant IDs.
        """
        return list({a.participant_id for a in self.analytics})

    def get_participant_summaries(self) -> list[ParticipantBehavioralSummary]:
        """Generate behavioral summaries for all participants.

        Returns
        -------
        list[ParticipantBehavioralSummary]
            Summary for each participant in the collection.
        """
        severity_order: dict[str, int] = {"info": 0, "low": 1, "medium": 2, "high": 3}

        # Group by participant
        by_participant: dict[str, list[JudgmentAnalytics]] = defaultdict(list)
        for a in self.analytics:
            by_participant[a.participant_id].append(a)

        summaries: list[ParticipantBehavioralSummary] = []

        for participant_id, records in by_participant.items():
            # Aggregated metrics
            total = len(records)
            flagged = sum(1 for r in records if r.is_flagged)
            response_times = [r.response_time_ms for r in records]
            mean_rt = sum(response_times) / total if total > 0 else 0.0

            # Keystroke metrics
            ikis: list[float] = []
            total_keystrokes = 0
            total_pastes = 0

            for r in records:
                total_pastes += r.paste_event_count
                if r.keystroke_metrics is not None:
                    total_keystrokes += r.keystroke_metrics.total_keystrokes
                    if r.keystroke_metrics.mean_iki > 0:
                        ikis.append(r.keystroke_metrics.mean_iki)

            mean_iki = sum(ikis) / len(ikis) if ikis else None

            # Focus metrics
            blur_events = 0
            blur_duration = 0.0

            for r in records:
                if r.focus_metrics is not None:
                    blur_events += r.focus_metrics.blur_count
                    blur_duration += r.focus_metrics.total_blur_duration

            # Flag aggregation
            flag_counts: dict[str, int] = defaultdict(int)
            max_severity_level = -1
            max_severity: Severity | None = None

            for r in records:
                for flag in r.flags:
                    flag_counts[flag.type] += 1
                    level = severity_order.get(flag.severity, 0)
                    if level > max_severity_level:
                        max_severity_level = level
                        max_severity = flag.severity

            # Get session_id from first record
            session_id = records[0].session_id if records else ""

            summaries.append(
                ParticipantBehavioralSummary(
                    participant_id=participant_id,
                    session_id=session_id,
                    total_judgments=total,
                    flagged_judgments=flagged,
                    mean_response_time_ms=mean_rt,
                    mean_iki=mean_iki,
                    total_keystrokes=total_keystrokes,
                    total_paste_events=total_pastes,
                    total_blur_events=blur_events,
                    total_blur_duration_ms=blur_duration,
                    flag_counts=dict(flag_counts),
                    max_severity=max_severity,
                )
            )

        return summaries

    # JSONL I/O

    def to_jsonl(self, path: Path | str) -> None:
        """Write analytics to JSONL file.

        Parameters
        ----------
        path : Path | str
            Path to output file.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonlines(self.analytics, path)

    @classmethod
    def from_jsonl(
        cls,
        path: Path | str,
        name: str = "loaded_analytics",
    ) -> AnalyticsCollection:
        """Load analytics from JSONL file.

        Parameters
        ----------
        path : Path | str
            Path to JSONL file.
        name : str
            Name for the collection.

        Returns
        -------
        AnalyticsCollection
            Collection with loaded analytics.
        """
        analytics = read_jsonlines(Path(path), JudgmentAnalytics)
        return cls(name=name, analytics=analytics)

    # DataFrame conversion

    def to_dataframe(
        self,
        backend: Literal["pandas", "polars"] = "pandas",
        include_metrics: bool = True,
        include_flags: bool = True,
    ) -> DataFrame:
        """Convert to pandas or polars DataFrame.

        Parameters
        ----------
        backend : Literal["pandas", "polars"]
            DataFrame backend to use (default: "pandas").
        include_metrics : bool
            If True, flatten behavioral metrics into columns.
        include_flags : bool
            If True, include flag-related columns.

        Returns
        -------
        DataFrame
            pandas or polars DataFrame with analytics data.
        """
        if not self.analytics:
            columns = [
                "item_id",
                "participant_id",
                "trial_index",
                "session_id",
                "response_value",
                "response_time_ms",
            ]
            if backend == "pandas":
                return pd.DataFrame(columns=columns)
            else:
                schema: dict[str, type[pl.Utf8]] = dict.fromkeys(columns, pl.Utf8)
                return pl.DataFrame(schema=schema)

        records: list[dict[str, JsonValue]] = []

        for a in self.analytics:
            record: dict[str, JsonValue] = {
                "item_id": str(a.item_id),
                "participant_id": a.participant_id,
                "trial_index": a.trial_index,
                "session_id": a.session_id,
                "response_value": a.response_value,
                "response_time_ms": a.response_time_ms,
                "paste_event_count": a.paste_event_count,
            }

            if include_metrics:
                # Keystroke metrics
                if a.keystroke_metrics is not None:
                    record["keystroke_total"] = a.keystroke_metrics.total_keystrokes
                    record["keystroke_mean_iki"] = a.keystroke_metrics.mean_iki
                    record["keystroke_std_iki"] = a.keystroke_metrics.std_iki
                    record["keystroke_deletions"] = a.keystroke_metrics.deletions
                else:
                    record["keystroke_total"] = None
                    record["keystroke_mean_iki"] = None
                    record["keystroke_std_iki"] = None
                    record["keystroke_deletions"] = None

                # Focus metrics
                if a.focus_metrics is not None:
                    record["focus_blur_count"] = a.focus_metrics.blur_count
                    record["focus_blur_duration"] = a.focus_metrics.total_blur_duration
                else:
                    record["focus_blur_count"] = None
                    record["focus_blur_duration"] = None

                # Timing metrics
                if a.timing_metrics is not None:
                    record["timing_first_keystroke"] = (
                        a.timing_metrics.first_keystroke_latency
                    )
                    record["timing_total_response"] = (
                        a.timing_metrics.total_response_time
                    )
                else:
                    record["timing_first_keystroke"] = None
                    record["timing_total_response"] = None

            if include_flags:
                record["is_flagged"] = a.is_flagged
                record["flag_count"] = len(a.flags)
                record["max_severity"] = a.max_severity
                record["flag_types"] = ",".join(a.get_flag_types()) if a.flags else None

            records.append(record)

        if backend == "pandas":
            return pd.DataFrame(records)
        else:
            return pl.DataFrame(records)
