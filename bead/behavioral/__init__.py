"""Behavioral analytics for bead.

This module provides tools for extracting, storing, and analyzing
behavioral data captured during experiments via slopit integration.

The main components are:

- **Analytics Models**: `JudgmentAnalytics`, `ParticipantBehavioralSummary`,
  `AnalyticsCollection` for storing per-judgment behavioral metrics
- **Extraction**: Functions to extract analytics from slopit sessions
- **Merging**: Utilities to merge analytics with judgment DataFrames

Examples
--------
Extract analytics from JATOS export:

>>> from bead.behavioral import extract_with_analysis, AnalyticsCollection
>>> collection = extract_with_analysis("data/jatos_export/")
>>> print(f"Extracted {len(collection)} analytics records")

Get participant summaries:

>>> summaries = collection.get_participant_summaries()
>>> for s in summaries:
...     if s.flag_rate > 0.1:
...         print(f"Participant {s.participant_id}: {s.flag_rate:.1%} flagged")

Merge with judgment data:

>>> from bead.behavioral import merge_behavioral_analytics
>>> merged_df = merge_behavioral_analytics(judgments_df, collection)

Filter out flagged judgments:

>>> from bead.behavioral import filter_flagged_judgments
>>> clean_df = filter_flagged_judgments(judgments_df, collection, exclude_flagged=True)

Save and load analytics:

>>> collection.to_jsonl("analytics.jsonl")
>>> loaded = AnalyticsCollection.from_jsonl("analytics.jsonl", name="study_001")
"""

# Re-export key slopit types for convenience
from slopit.behavioral import (
    Analyzer,
    FocusAnalyzer,
    KeystrokeAnalyzer,
    PasteAnalyzer,
    TimingAnalyzer,
)
from slopit.schemas import (
    AnalysisFlag,
    BehavioralData,
    BehavioralMetrics,
    FocusMetrics,
    KeystrokeMetrics,
    Severity,
    SlopitSession,
    SlopitTrial,
    TimingMetrics,
)

from bead.behavioral.analytics import (
    AnalyticsCollection,
    JudgmentAnalytics,
    ParticipantBehavioralSummary,
)
from bead.behavioral.extraction import (
    analyze_sessions,
    extract_from_directory,
    extract_from_file,
    extract_from_session,
    extract_from_trial,
    extract_with_analysis,
)
from bead.behavioral.merging import (
    create_analysis_dataframe_with_behavior,
    filter_flagged_judgments,
    get_exclusion_list,
    merge_behavioral_analytics,
)

__all__ = [
    # Bead models
    "JudgmentAnalytics",
    "ParticipantBehavioralSummary",
    "AnalyticsCollection",
    # Extraction
    "extract_from_trial",
    "extract_from_session",
    "extract_from_file",
    "extract_from_directory",
    "extract_with_analysis",
    "analyze_sessions",
    # Merging
    "merge_behavioral_analytics",
    "filter_flagged_judgments",
    "create_analysis_dataframe_with_behavior",
    "get_exclusion_list",
    # Slopit re-exports
    "SlopitSession",
    "SlopitTrial",
    "BehavioralData",
    "BehavioralMetrics",
    "KeystrokeMetrics",
    "FocusMetrics",
    "TimingMetrics",
    "AnalysisFlag",
    "Severity",
    "Analyzer",
    "KeystrokeAnalyzer",
    "FocusAnalyzer",
    "PasteAnalyzer",
    "TimingAnalyzer",
]
