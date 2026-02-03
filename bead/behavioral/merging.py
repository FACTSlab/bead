"""Utilities for merging behavioral analytics with judgment data.

This module provides functions for joining behavioral analytics with
judgment DataFrames for analysis. All functions support both pandas
and polars DataFrames, preserving the input type.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import pandas as pd
import polars as pl

from bead.behavioral.analytics import AnalyticsCollection

if TYPE_CHECKING:
    from slopit.schemas import Severity

    from bead.participants.collection import IDMappingCollection, ParticipantCollection

# Type alias for supported DataFrame types
DataFrame = pd.DataFrame | pl.DataFrame


def merge_behavioral_analytics(
    judgments_df: DataFrame,
    analytics: AnalyticsCollection,
    item_id_column: str = "item_id",
    participant_id_column: str = "participant_id",
    include_metrics: bool = True,
    include_flags: bool = True,
    how: str = "left",
) -> DataFrame:
    """Merge behavioral analytics into a judgments DataFrame.

    Preserves input DataFrame type (pandas in -> pandas out,
    polars in -> polars out).

    Parameters
    ----------
    judgments_df : DataFrame
        DataFrame containing judgment data.
    analytics : AnalyticsCollection
        Collection of behavioral analytics.
    item_id_column : str
        Column in judgments_df containing item IDs (default: "item_id").
    participant_id_column : str
        Column in judgments_df containing participant IDs.
    include_metrics : bool
        If True, include flattened behavioral metrics columns.
    include_flags : bool
        If True, include flag-related columns.
    how : str
        Merge type: "left", "inner", "outer" (default: "left").

    Returns
    -------
    DataFrame
        Merged DataFrame with behavioral analytics columns added.

    Examples
    --------
    >>> import pandas as pd
    >>> judgments = pd.DataFrame({
    ...     "item_id": ["uuid1", "uuid2"],
    ...     "participant_id": ["p1", "p1"],
    ...     "response": [5, 3],
    ... })
    >>> # merged = merge_behavioral_analytics(judgments, analytics_collection)
    """
    is_polars = isinstance(judgments_df, pl.DataFrame)

    # Convert analytics to DataFrame with same backend
    backend: Literal["pandas", "polars"] = "polars" if is_polars else "pandas"
    analytics_df = analytics.to_dataframe(
        backend=backend,
        include_metrics=include_metrics,
        include_flags=include_flags,
    )

    if is_polars:
        assert isinstance(judgments_df, pl.DataFrame)
        assert isinstance(analytics_df, pl.DataFrame)

        # Polars join on both item_id and participant_id
        return judgments_df.join(
            analytics_df,
            left_on=[item_id_column, participant_id_column],
            right_on=["item_id", "participant_id"],
            how=how,  # type: ignore[arg-type]
            suffix="_behavioral",
        )
    else:
        assert isinstance(judgments_df, pd.DataFrame)
        assert isinstance(analytics_df, pd.DataFrame)

        # Pandas merge
        merged = pd.merge(
            judgments_df,
            analytics_df,
            left_on=[item_id_column, participant_id_column],
            right_on=["item_id", "participant_id"],
            how=how,  # type: ignore[arg-type]
            suffixes=("", "_behavioral"),
        )

        # Remove duplicate columns if created
        for col in ["item_id_behavioral", "participant_id_behavioral"]:
            if col in merged.columns:
                merged = merged.drop(columns=[col])

        return merged


def filter_flagged_judgments(
    judgments_df: DataFrame,
    analytics: AnalyticsCollection,
    item_id_column: str = "item_id",
    participant_id_column: str = "participant_id",
    min_severity: Severity | None = None,
    exclude_flagged: bool = True,
) -> DataFrame:
    """Filter judgments based on behavioral flags.

    Preserves input DataFrame type.

    Parameters
    ----------
    judgments_df : DataFrame
        DataFrame containing judgment data.
    analytics : AnalyticsCollection
        Collection of behavioral analytics.
    item_id_column : str
        Column containing item IDs.
    participant_id_column : str
        Column containing participant IDs.
    min_severity : Severity | None
        Minimum severity level for filtering. If None, any flag counts.
    exclude_flagged : bool
        If True, exclude flagged judgments (default).
        If False, keep only flagged judgments.

    Returns
    -------
    DataFrame
        Filtered DataFrame.

    Examples
    --------
    >>> # Keep only unflagged judgments
    >>> clean_df = filter_flagged_judgments(judgments, analytics, exclude_flagged=True)
    >>> # Keep only high-severity flagged judgments for review
    >>> flagged_df = filter_flagged_judgments(
    ...     judgments, analytics, min_severity="high", exclude_flagged=False
    ... )
    """
    is_polars = isinstance(judgments_df, pl.DataFrame)

    # Get filtered analytics
    filtered_analytics = analytics.filter_flagged(
        min_severity=min_severity,
        exclude_flagged=False,  # Get flagged records
    )

    # Build set of flagged (item_id, participant_id) pairs
    flagged_pairs: set[tuple[str, str]] = {
        (str(a.item_id), a.participant_id) for a in filtered_analytics.analytics
    }

    if is_polars:
        assert isinstance(judgments_df, pl.DataFrame)

        def check_flagged(item_id: str, participant_id: str) -> bool:
            return (str(item_id), str(participant_id)) in flagged_pairs

        # Create mask column
        df_with_flag = judgments_df.with_columns(
            pl.struct([item_id_column, participant_id_column])
            .map_elements(
                lambda row: (str(row[item_id_column]), str(row[participant_id_column]))
                in flagged_pairs,
                return_dtype=pl.Boolean,
            )
            .alias("_is_flagged")
        )

        if exclude_flagged:
            result = df_with_flag.filter(~pl.col("_is_flagged"))
        else:
            result = df_with_flag.filter(pl.col("_is_flagged"))

        return result.drop("_is_flagged")

    else:
        assert isinstance(judgments_df, pd.DataFrame)

        # Create mask
        mask = judgments_df.apply(
            lambda row: (str(row[item_id_column]), str(row[participant_id_column]))
            in flagged_pairs,
            axis=1,
        )

        if exclude_flagged:
            return judgments_df[~mask].copy()
        else:
            return judgments_df[mask].copy()


def create_analysis_dataframe_with_behavior(
    judgments_df: DataFrame,
    participants: ParticipantCollection,
    analytics: AnalyticsCollection,
    id_mappings: IDMappingCollection | None = None,
    external_id_column: str | None = None,
    participant_id_column: str = "participant_id",
    item_id_column: str = "item_id",
    metadata_columns: list[str] | None = None,
    include_metrics: bool = True,
    include_flags: bool = True,
) -> DataFrame:
    """Create analysis-ready DataFrame with metadata and behavioral analytics.

    Combines both participant and behavioral merging in one step.
    Preserves input DataFrame type.

    Parameters
    ----------
    judgments_df : DataFrame
        Raw judgment data.
    participants : ParticipantCollection
        Participant collection with metadata.
    analytics : AnalyticsCollection
        Behavioral analytics collection.
    id_mappings : IDMappingCollection | None
        ID mappings (required if external_id_column is provided).
    external_id_column : str | None
        Column with external IDs to resolve.
    participant_id_column : str
        Column with participant IDs (after resolution).
    item_id_column : str
        Column with item IDs.
    metadata_columns : list[str] | None
        Participant metadata columns to include.
    include_metrics : bool
        If True, include behavioral metrics columns.
    include_flags : bool
        If True, include flag columns.

    Returns
    -------
    DataFrame
        Analysis-ready DataFrame with both metadata and behavioral data.

    Examples
    --------
    >>> analysis_df = create_analysis_dataframe_with_behavior(
    ...     judgments,
    ...     participants,
    ...     analytics,
    ...     id_mappings=mappings,
    ...     external_id_column="PROLIFIC_PID",
    ... )
    """
    # Import here to avoid circular imports
    from bead.participants.merging import (  # noqa: PLC0415
        merge_participant_metadata,
        resolve_external_ids,
    )

    df = judgments_df

    # Step 1: Resolve external IDs if needed
    if external_id_column is not None and id_mappings is not None:
        df = resolve_external_ids(
            df,
            id_mappings,
            external_id_column=external_id_column,
            output_column=participant_id_column,
        )

    # Step 2: Merge participant metadata
    df = merge_participant_metadata(
        df,
        participants,
        id_column=participant_id_column,
        metadata_columns=metadata_columns,
    )

    # Step 3: Merge behavioral analytics
    df = merge_behavioral_analytics(
        df,
        analytics,
        item_id_column=item_id_column,
        participant_id_column=participant_id_column,
        include_metrics=include_metrics,
        include_flags=include_flags,
    )

    return df


def get_exclusion_list(
    analytics: AnalyticsCollection,
    min_flag_rate: float = 0.1,
    min_severity: Severity | None = None,
) -> list[str]:
    """Get list of participant IDs that should be excluded based on flags.

    Identifies participants with flag rates above the threshold.

    Parameters
    ----------
    analytics : AnalyticsCollection
        Behavioral analytics collection.
    min_flag_rate : float
        Minimum proportion of flagged judgments for exclusion (default: 0.1).
    min_severity : Severity | None
        Only count flags at or above this severity.

    Returns
    -------
    list[str]
        Participant IDs recommended for exclusion.

    Examples
    --------
    >>> exclude = get_exclusion_list(analytics, min_flag_rate=0.2)
    >>> clean_df = judgments_df[~judgments_df["participant_id"].isin(exclude)]
    """
    # Apply severity filter if specified
    if min_severity is not None:
        filtered = analytics.filter_flagged(
            min_severity=min_severity, exclude_flagged=False
        )
    else:
        filtered = analytics

    summaries = filtered.get_participant_summaries()

    return [s.participant_id for s in summaries if s.flag_rate >= min_flag_rate]
