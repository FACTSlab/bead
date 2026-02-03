"""Utilities for merging participant metadata with judgment data.

This module provides functions for joining participant metadata with
judgment DataFrames for analysis. All functions support both pandas
and polars DataFrames, preserving the input type.
"""

from __future__ import annotations

import pandas as pd
import polars as pl

from bead.participants.collection import IDMappingCollection, ParticipantCollection

# Type alias for supported DataFrame types
DataFrame = pd.DataFrame | pl.DataFrame


def merge_participant_metadata(
    judgments_df: DataFrame,
    participants: ParticipantCollection,
    id_column: str = "participant_id",
    metadata_columns: list[str] | None = None,
    how: str = "left",
) -> DataFrame:
    """Merge participant metadata into a judgments DataFrame.

    Preserves input DataFrame type (pandas in -> pandas out,
    polars in -> polars out).

    Parameters
    ----------
    judgments_df : DataFrame
        DataFrame containing judgment data with participant IDs.
    participants : ParticipantCollection
        Collection of participants with metadata.
    id_column : str
        Column in judgments_df containing participant IDs (default: "participant_id").
    metadata_columns : list[str] | None
        Specific metadata columns to include. If None, includes all.
    how : str
        Merge type: "left", "inner", "outer" (default: "left").

    Returns
    -------
    DataFrame
        Merged DataFrame with participant metadata columns added.

    Examples
    --------
    >>> import pandas as pd
    >>> from bead.participants.collection import ParticipantCollection
    >>> from bead.participants.models import Participant
    >>> judgments = pd.DataFrame({
    ...     "participant_id": ["uuid1", "uuid2"],
    ...     "response": [5, 3],
    ... })
    >>> collection = ParticipantCollection(name="test")
    >>> # ... add participants ...
    >>> # merged = merge_participant_metadata(judgments, collection)
    """
    is_polars = isinstance(judgments_df, pl.DataFrame)

    # Convert participants to DataFrame with same backend
    backend = "polars" if is_polars else "pandas"
    participant_df = participants.to_dataframe(
        backend=backend,  # type: ignore[arg-type]
        include_fields=metadata_columns,
        flatten_metadata=True,
    )

    if is_polars:
        assert isinstance(judgments_df, pl.DataFrame)
        assert isinstance(participant_df, pl.DataFrame)

        # Polars join
        return judgments_df.join(
            participant_df,
            left_on=id_column,
            right_on="participant_id",
            how=how,  # type: ignore[arg-type]
            suffix="_participant",
        )
    else:
        assert isinstance(judgments_df, pd.DataFrame)
        assert isinstance(participant_df, pd.DataFrame)

        # Pandas merge
        merged = pd.merge(
            judgments_df,
            participant_df,
            left_on=id_column,
            right_on="participant_id",
            how=how,  # type: ignore[arg-type]
            suffixes=("", "_participant"),
        )

        # Remove duplicate participant_id column if created
        if "participant_id_participant" in merged.columns:
            merged = merged.drop(columns=["participant_id_participant"])

        return merged


def resolve_external_ids(
    df: DataFrame,
    id_mappings: IDMappingCollection,
    external_id_column: str = "PROLIFIC_PID",
    output_column: str = "participant_id",
    drop_unresolved: bool = False,
) -> DataFrame:
    """Resolve external IDs to internal participant UUIDs.

    Preserves input DataFrame type.

    Parameters
    ----------
    df : DataFrame
        DataFrame with external participant IDs.
    id_mappings : IDMappingCollection
        Collection of ID mappings.
    external_id_column : str
        Column containing external IDs (default: "PROLIFIC_PID").
    output_column : str
        Column name for resolved UUIDs (default: "participant_id").
    drop_unresolved : bool
        If True, drop rows with unresolved IDs (default: False).

    Returns
    -------
    DataFrame
        DataFrame with resolved participant UUIDs.

    Examples
    --------
    >>> import pandas as pd
    >>> from uuid import uuid4
    >>> from bead.participants.collection import IDMappingCollection
    >>> raw_data = pd.DataFrame({
    ...     "PROLIFIC_PID": ["ABC123", "DEF456"],
    ...     "response": [5, 3],
    ... })
    >>> mappings = IDMappingCollection(name="test", source="prolific")
    >>> pid = uuid4()
    >>> mappings.add_mapping("ABC123", pid)
    >>> resolved = resolve_external_ids(raw_data, mappings)
    >>> output_column in resolved.columns
    True
    """
    is_polars = isinstance(df, pl.DataFrame)

    # Create lookup dict
    lookup: dict[str, str] = {
        m.external_id: str(m.participant_id)
        for m in id_mappings.mappings
        if m.is_active
    }

    if is_polars:
        assert isinstance(df, pl.DataFrame)

        # Polars: use map_elements or replace
        result = df.with_columns(
            pl.col(external_id_column)
            .map_elements(lambda x: lookup.get(x), return_dtype=pl.Utf8)
            .alias(output_column)
        )

        if drop_unresolved:
            result = result.filter(pl.col(output_column).is_not_null())

        return result
    else:
        assert isinstance(df, pd.DataFrame)

        # Pandas: use map
        result = df.copy()
        result[output_column] = result[external_id_column].map(lookup)

        if drop_unresolved:
            result = result.dropna(subset=[output_column])

        return result


def create_analysis_dataframe(
    judgments_df: DataFrame,
    participants: ParticipantCollection,
    id_mappings: IDMappingCollection | None = None,
    external_id_column: str | None = None,
    participant_id_column: str = "participant_id",
    metadata_columns: list[str] | None = None,
) -> DataFrame:
    """Create analysis-ready DataFrame with resolved IDs and metadata.

    Convenience function that:
    1. Resolves external IDs to internal UUIDs (if id_mappings provided)
    2. Merges participant metadata
    3. Returns a clean DataFrame ready for analysis

    Preserves input DataFrame type.

    Parameters
    ----------
    judgments_df : DataFrame
        Raw judgment data.
    participants : ParticipantCollection
        Participant collection with metadata.
    id_mappings : IDMappingCollection | None
        ID mappings (required if external_id_column is provided).
    external_id_column : str | None
        Column with external IDs to resolve.
    participant_id_column : str
        Column with participant IDs (after resolution).
    metadata_columns : list[str] | None
        Metadata columns to include.

    Returns
    -------
    DataFrame
        Analysis-ready DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> from bead.participants.collection import (
    ...     ParticipantCollection, IDMappingCollection
    ... )
    >>> raw_judgments = pd.DataFrame({
    ...     "PROLIFIC_PID": ["ABC123"],
    ...     "response": [5],
    ... })
    >>> participants = ParticipantCollection(name="test")
    >>> mappings = IDMappingCollection(name="test", source="prolific")
    >>> # analysis_df = create_analysis_dataframe(
    >>> #     raw_judgments,
    >>> #     participants,
    >>> #     id_mappings=mappings,
    >>> #     external_id_column="PROLIFIC_PID",
    >>> # )
    """
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

    return df


def build_participant_lookup(
    participants: ParticipantCollection,
    key_field: str | None = None,
) -> dict[str, dict[str, str | int | float | bool | None]]:
    """Build a lookup dictionary from participant collection.

    Useful for manual merging or custom processing.

    Parameters
    ----------
    participants : ParticipantCollection
        Collection of participants.
    key_field : str | None
        If provided, use this metadata field as the key instead of UUID.

    Returns
    -------
    dict[str, dict[str, str | int | float | bool | None]]
        Lookup from participant ID (or key_field) to metadata dict.

    Examples
    --------
    >>> from bead.participants.collection import ParticipantCollection
    >>> from bead.participants.models import Participant
    >>> collection = ParticipantCollection(name="test")
    >>> p = Participant(participant_metadata={"age": 25})
    >>> collection.add_participant(p)
    >>> lookup = build_participant_lookup(collection)
    >>> str(p.id) in lookup
    True
    """
    result: dict[str, dict[str, str | int | float | bool | None]] = {}

    for p in participants.participants:
        # Determine key
        if key_field is not None:
            key = str(p.participant_metadata.get(key_field, ""))
        else:
            key = str(p.id)

        # Extract simple metadata values
        metadata: dict[str, str | int | float | bool | None] = {}
        for k, v in p.participant_metadata.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                metadata[k] = v

        result[key] = metadata

    return result
