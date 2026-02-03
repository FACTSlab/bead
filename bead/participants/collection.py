"""Participant collection with JSONL I/O and DataFrame support.

This module provides ParticipantCollection and IDMappingCollection for
managing multiple participants with JSONL serialization and pandas/polars
DataFrame conversion for analysis.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal
from uuid import UUID

import pandas as pd
import polars as pl
from pydantic import Field, field_validator

from bead.data.base import BeadBaseModel, JsonValue
from bead.data.serialization import read_jsonlines, write_jsonlines
from bead.participants.models import Participant, ParticipantIDMapping

if TYPE_CHECKING:
    from bead.participants.metadata_spec import ParticipantMetadataSpec

# Type alias for supported DataFrame types (same pattern as bead/resources/lexicon.py)
DataFrame = pd.DataFrame | pl.DataFrame


def _empty_participant_list() -> list[Participant]:
    """Return empty participant list."""
    return []


def _empty_mapping_list() -> list[ParticipantIDMapping]:
    """Return empty mapping list."""
    return []


class ParticipantCollection(BeadBaseModel):
    """Collection of participants with JSONL I/O and DataFrame support.

    Provides methods for managing multiple participants, saving/loading
    from JSONL files, and converting to pandas/polars DataFrames for analysis.

    Attributes
    ----------
    name : str
        Name of this collection.
    participants : list[Participant]
        List of participants.
    metadata_spec_name : str | None
        Name of the metadata spec used (for documentation).

    Examples
    --------
    >>> collection = ParticipantCollection(name="study_001_participants")
    >>> participant = Participant(
    ...     participant_metadata={"age": 25, "education": "bachelors"}
    ... )
    >>> collection.add_participant(participant)
    >>> len(collection.participants)
    1
    >>> collection.to_jsonl("participants.jsonl")  # doctest: +SKIP
    """

    name: str = Field(..., description="Collection name")
    participants: list[Participant] = Field(
        default_factory=_empty_participant_list, description="Participants"
    )
    metadata_spec_name: str | None = Field(
        default=None, description="Metadata spec used"
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate name is non-empty.

        Parameters
        ----------
        v : str
            Collection name to validate.

        Returns
        -------
        str
            Validated collection name.

        Raises
        ------
        ValueError
            If name is empty or whitespace only.
        """
        if not v or not v.strip():
            raise ValueError("Collection name cannot be empty")
        return v.strip()

    def __len__(self) -> int:
        """Return number of participants.

        Returns
        -------
        int
            Number of participants in the collection.
        """
        return len(self.participants)

    def add_participant(self, participant: Participant) -> None:
        """Add a participant to the collection.

        Parameters
        ----------
        participant : Participant
            Participant to add.

        Examples
        --------
        >>> collection = ParticipantCollection(name="test")
        >>> p = Participant(participant_metadata={"age": 25})
        >>> collection.add_participant(p)
        >>> len(collection)
        1
        """
        self.participants.append(participant)
        self.update_modified_time()

    def add_participants(self, participants: list[Participant]) -> None:
        """Add multiple participants to the collection.

        Parameters
        ----------
        participants : list[Participant]
            Participants to add.

        Examples
        --------
        >>> collection = ParticipantCollection(name="test")
        >>> ps = [Participant(), Participant()]
        >>> collection.add_participants(ps)
        >>> len(collection)
        2
        """
        self.participants.extend(participants)
        self.update_modified_time()

    def get_by_id(self, participant_id: UUID) -> Participant | None:
        """Get participant by UUID.

        Parameters
        ----------
        participant_id : UUID
            Participant UUID to find.

        Returns
        -------
        Participant | None
            Participant if found, None otherwise.

        Examples
        --------
        >>> collection = ParticipantCollection(name="test")
        >>> p = Participant()
        >>> collection.add_participant(p)
        >>> found = collection.get_by_id(p.id)
        >>> found is not None
        True
        """
        for p in self.participants:
            if p.id == participant_id:
                return p
        return None

    def get_by_attribute(self, key: str, value: JsonValue) -> list[Participant]:
        """Get participants by metadata attribute value.

        Parameters
        ----------
        key : str
            Attribute name.
        value : JsonValue
            Value to match.

        Returns
        -------
        list[Participant]
            Participants with matching attribute.

        Examples
        --------
        >>> collection = ParticipantCollection(name="test")
        >>> p1 = Participant(participant_metadata={"age": 25})
        >>> p2 = Participant(participant_metadata={"age": 30})
        >>> collection.add_participants([p1, p2])
        >>> matches = collection.get_by_attribute("age", 25)
        >>> len(matches)
        1
        """
        return [
            p for p in self.participants if p.participant_metadata.get(key) == value
        ]

    def validate_all(self, spec: ParticipantMetadataSpec) -> dict[UUID, list[str]]:
        """Validate all participants against a specification.

        Parameters
        ----------
        spec : ParticipantMetadataSpec
            Specification to validate against.

        Returns
        -------
        dict[UUID, list[str]]
            Mapping from participant ID to list of validation errors.
            Empty dict if all valid.

        Examples
        --------
        >>> from bead.participants.metadata_spec import (
        ...     FieldSpec, ParticipantMetadataSpec
        ... )
        >>> spec = ParticipantMetadataSpec(
        ...     name="test",
        ...     fields=[FieldSpec(name="age", field_type="int", required=True)]
        ... )
        >>> collection = ParticipantCollection(name="test")
        >>> p = Participant(participant_metadata={"age": 25})
        >>> collection.add_participant(p)
        >>> errors = collection.validate_all(spec)
        >>> len(errors)
        0
        """
        errors: dict[UUID, list[str]] = {}
        for p in self.participants:
            is_valid, error_list = p.validate_against_spec(spec)
            if not is_valid:
                errors[p.id] = error_list
        return errors

    # JSONL I/O

    def to_jsonl(self, path: Path | str) -> None:
        """Write participants to JSONL file.

        Parameters
        ----------
        path : Path | str
            Path to output file.

        Examples
        --------
        >>> collection = ParticipantCollection(name="test")
        >>> collection.add_participant(Participant())
        >>> collection.to_jsonl("/tmp/participants.jsonl")  # doctest: +SKIP
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonlines(self.participants, path)

    @classmethod
    def from_jsonl(
        cls,
        path: Path | str,
        name: str = "loaded_participants",
    ) -> ParticipantCollection:
        """Load participants from JSONL file.

        Parameters
        ----------
        path : Path | str
            Path to JSONL file.
        name : str
            Name for the collection.

        Returns
        -------
        ParticipantCollection
            Collection with loaded participants.

        Examples
        --------
        >>> collection = ParticipantCollection.from_jsonl(
        ...     "participants.jsonl"
        ... )  # doctest: +SKIP
        """
        participants = read_jsonlines(Path(path), Participant)
        return cls(name=name, participants=participants)

    # DataFrame conversion

    def to_dataframe(
        self,
        backend: Literal["pandas", "polars"] = "pandas",
        include_fields: list[str] | None = None,
        exclude_fields: list[str] | None = None,
        flatten_metadata: bool = True,
    ) -> DataFrame:
        """Convert to pandas or polars DataFrame.

        Parameters
        ----------
        backend : Literal["pandas", "polars"]
            DataFrame backend to use (default: "pandas").
        include_fields : list[str] | None
            If provided, only include these metadata fields.
        exclude_fields : list[str] | None
            If provided, exclude these metadata fields.
        flatten_metadata : bool
            If True, flatten participant_metadata into top-level columns.

        Returns
        -------
        DataFrame
            pandas or polars DataFrame with participant data.
            Always includes 'participant_id' column (as string).

        Examples
        --------
        >>> collection = ParticipantCollection(name="test")
        >>> p = Participant(participant_metadata={"age": 25})
        >>> collection.add_participant(p)
        >>> df = collection.to_dataframe()
        >>> "participant_id" in df.columns
        True
        >>> "age" in df.columns
        True
        """
        if not self.participants:
            # Return empty DataFrame with expected columns
            columns = ["participant_id", "created_at", "study_id"]
            if backend == "pandas":
                return pd.DataFrame(columns=columns)
            else:
                schema: dict[str, type[pl.Utf8]] = dict.fromkeys(columns, pl.Utf8)
                return pl.DataFrame(schema=schema)

        records: list[dict[str, JsonValue]] = []

        for p in self.participants:
            record: dict[str, JsonValue] = {
                "participant_id": str(p.id),
                "created_at": p.created_at.isoformat(),
                "study_id": p.study_id,
            }

            if flatten_metadata:
                for key, value in p.participant_metadata.items():
                    # Apply include/exclude filters
                    if include_fields is not None and key not in include_fields:
                        continue
                    if exclude_fields is not None and key in exclude_fields:
                        continue
                    record[key] = value
            else:
                record["participant_metadata"] = p.participant_metadata

            records.append(record)

        if backend == "pandas":
            return pd.DataFrame(records)
        else:
            return pl.DataFrame(records)

    @classmethod
    def from_dataframe(
        cls,
        df: DataFrame,
        name: str,
        id_column: str = "participant_id",
        metadata_columns: list[str] | None = None,
    ) -> ParticipantCollection:
        """Create collection from pandas or polars DataFrame.

        Parameters
        ----------
        df : DataFrame
            pandas or polars DataFrame with participant data.
        name : str
            Name for the collection.
        id_column : str
            Column containing participant IDs (default: "participant_id").
            If column exists, uses those UUIDs; otherwise generates new ones.
        metadata_columns : list[str] | None
            Columns to include in participant_metadata.
            If None, includes all columns except id_column.

        Returns
        -------
        ParticipantCollection
            Collection with participants from DataFrame.

        Examples
        --------
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     "age": [25, 30],
        ...     "education": ["bachelors", "masters"]
        ... })
        >>> collection = ParticipantCollection.from_dataframe(df, "test")
        >>> len(collection)
        2
        """
        # Check if it's a polars DataFrame
        is_polars = isinstance(df, pl.DataFrame)

        # Get columns, handling both pandas and polars
        if is_polars:
            assert isinstance(df, pl.DataFrame)
            columns_list: list[str] = df.columns
        else:
            assert isinstance(df, pd.DataFrame)
            columns_list = list(df.columns)

        # Convert to dict format for iteration
        rows: list[dict[str, JsonValue]]
        if is_polars:
            assert isinstance(df, pl.DataFrame)
            rows = df.to_dicts()  # type: ignore[assignment]
        else:
            assert isinstance(df, pd.DataFrame)
            rows = df.to_dict("records")  # type: ignore[assignment]

        participants: list[Participant] = []

        for row in rows:
            # Handle participant ID
            pid: UUID | None = None
            if id_column in columns_list:
                try:
                    pid = UUID(str(row[id_column]))
                except (ValueError, TypeError):
                    pid = None  # Will use auto-generated UUID

            # Build metadata dict
            metadata: dict[str, JsonValue] = {}
            columns = metadata_columns or [c for c in columns_list if c != id_column]
            for col in columns:
                if col in row and row[col] is not None:
                    # Handle pandas NaN
                    value = row[col]
                    if is_polars:
                        # Polars uses None for nulls
                        metadata[col] = value
                    else:
                        # Pandas uses NaN - check with pd.notna
                        if pd.notna(value):
                            metadata[col] = value

            # Create participant
            if pid is not None:
                participant = Participant(
                    id=pid,
                    participant_metadata=metadata,
                )
            else:
                participant = Participant(participant_metadata=metadata)

            participants.append(participant)

        return cls(name=name, participants=participants)


class IDMappingCollection(BeadBaseModel):
    """Collection of ID mappings (stored separately for privacy).

    This collection should be stored in a SEPARATE file from participant
    data for IRB/privacy compliance.

    Attributes
    ----------
    name : str
        Name of this mapping collection.
    mappings : list[ParticipantIDMapping]
        List of ID mappings.
    source : str
        Primary source of external IDs (e.g., "prolific").

    Examples
    --------
    >>> from uuid import uuid4
    >>> collection = IDMappingCollection(name="study_001", source="prolific")
    >>> mapping = collection.add_mapping("PROLIFIC_ABC123", uuid4())
    >>> collection.get_participant_id("PROLIFIC_ABC123") is not None
    True
    """

    name: str = Field(..., description="Collection name")
    mappings: list[ParticipantIDMapping] = Field(
        default_factory=_empty_mapping_list, description="ID mappings"
    )
    source: str = Field(..., description="Primary external ID source")

    @field_validator("name", "source")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        """Validate string fields are non-empty.

        Parameters
        ----------
        v : str
            String to validate.

        Returns
        -------
        str
            Validated string.

        Raises
        ------
        ValueError
            If string is empty or whitespace only.
        """
        if not v or not v.strip():
            raise ValueError("Field cannot be empty")
        return v.strip()

    def __len__(self) -> int:
        """Return number of mappings.

        Returns
        -------
        int
            Number of mappings in the collection.
        """
        return len(self.mappings)

    def add_mapping(
        self,
        external_id: str,
        participant_id: UUID,
        external_source: str | None = None,
    ) -> ParticipantIDMapping:
        """Create and add a new ID mapping.

        Parameters
        ----------
        external_id : str
            External participant ID.
        participant_id : UUID
            Internal participant UUID.
        external_source : str | None
            Source of external ID (defaults to collection's source).

        Returns
        -------
        ParticipantIDMapping
            The created mapping.

        Examples
        --------
        >>> from uuid import uuid4
        >>> collection = IDMappingCollection(name="test", source="prolific")
        >>> mapping = collection.add_mapping("ABC123", uuid4())
        >>> mapping.external_source
        'prolific'
        """
        mapping = ParticipantIDMapping(
            external_id=external_id,
            external_source=external_source or self.source,
            participant_id=participant_id,
        )
        self.mappings.append(mapping)
        self.update_modified_time()
        return mapping

    def get_participant_id(self, external_id: str) -> UUID | None:
        """Look up internal participant ID from external ID.

        Parameters
        ----------
        external_id : str
            External ID to look up.

        Returns
        -------
        UUID | None
            Internal participant ID if found, None otherwise.

        Examples
        --------
        >>> from uuid import uuid4
        >>> collection = IDMappingCollection(name="test", source="prolific")
        >>> pid = uuid4()
        >>> collection.add_mapping("ABC123", pid)
        >>> collection.get_participant_id("ABC123") == pid
        True
        >>> collection.get_participant_id("UNKNOWN") is None
        True
        """
        for m in self.mappings:
            if m.external_id == external_id and m.is_active:
                return m.participant_id
        return None

    def get_external_id(self, participant_id: UUID) -> str | None:
        """Look up external ID from internal participant ID.

        Parameters
        ----------
        participant_id : UUID
            Internal participant ID to look up.

        Returns
        -------
        str | None
            External ID if found, None otherwise.

        Examples
        --------
        >>> from uuid import uuid4
        >>> collection = IDMappingCollection(name="test", source="prolific")
        >>> pid = uuid4()
        >>> collection.add_mapping("ABC123", pid)
        >>> collection.get_external_id(pid)
        'ABC123'
        """
        for m in self.mappings:
            if m.participant_id == participant_id and m.is_active:
                return m.external_id
        return None

    def deactivate_all(self) -> int:
        """Deactivate all mappings (for bulk privacy removal).

        Returns
        -------
        int
            Number of mappings deactivated.

        Examples
        --------
        >>> from uuid import uuid4
        >>> collection = IDMappingCollection(name="test", source="prolific")
        >>> collection.add_mapping("ABC123", uuid4())
        >>> collection.add_mapping("DEF456", uuid4())
        >>> count = collection.deactivate_all()
        >>> count
        2
        """
        count = 0
        for m in self.mappings:
            if m.is_active:
                m.deactivate()
                count += 1
        self.update_modified_time()
        return count

    # JSONL I/O

    def to_jsonl(self, path: Path | str) -> None:
        """Write mappings to JSONL file.

        Parameters
        ----------
        path : Path | str
            Path to output file.

        Examples
        --------
        >>> from uuid import uuid4
        >>> collection = IDMappingCollection(name="test", source="prolific")
        >>> collection.add_mapping("ABC123", uuid4())
        >>> collection.to_jsonl("/tmp/mappings.jsonl")  # doctest: +SKIP
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonlines(self.mappings, path)

    @classmethod
    def from_jsonl(
        cls,
        path: Path | str,
        name: str = "loaded_mappings",
        source: str = "unknown",
    ) -> IDMappingCollection:
        """Load mappings from JSONL file.

        Parameters
        ----------
        path : Path | str
            Path to JSONL file.
        name : str
            Name for the collection.
        source : str
            External ID source.

        Returns
        -------
        IDMappingCollection
            Collection with loaded mappings.

        Examples
        --------
        >>> collection = IDMappingCollection.from_jsonl(
        ...     "mappings.jsonl", source="prolific"
        ... )  # doctest: +SKIP
        """
        mappings = read_jsonlines(Path(path), ParticipantIDMapping)
        return cls(name=name, mappings=mappings, source=source)
