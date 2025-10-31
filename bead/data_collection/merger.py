"""Data merger for JATOS and Prolific data.

This module provides the DataMerger class for merging experimental results
from JATOS with participant metadata from Prolific. The merger matches
records based on participant IDs and handles unmatched records gracefully.
"""

from __future__ import annotations

from typing import Any


class DataMerger:
    """Merges JATOS results with Prolific metadata.

    This class merges experimental data from JATOS with participant
    demographics and metadata from Prolific based on participant IDs.

    Parameters
    ----------
    merge_key : str
        Key to merge on (e.g., "PROLIFIC_PID"). Default is "PROLIFIC_PID".

    Attributes
    ----------
    merge_key : str
        Key to merge on (e.g., "PROLIFIC_PID").

    Examples
    --------
    Create a merger with custom key::

        merger = DataMerger(merge_key="PROLIFIC_PID")
        merged_data = merger.merge(jatos_results, prolific_submissions)
    """

    def __init__(self, merge_key: str = "PROLIFIC_PID") -> None:
        self.merge_key = merge_key

    def merge(
        self,
        jatos_results: list[dict[str, Any]],
        prolific_submissions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """Merge JATOS and Prolific data.

        Merges experimental results from JATOS with participant submissions
        from Prolific by matching on participant IDs. Returns merged records
        with both JATOS data and Prolific metadata.

        Parameters
        ----------
        jatos_results : list[dict[str, Any]]
            JATOS results from JATOSDataCollector.
        prolific_submissions : list[dict[str, Any]]
            Prolific submissions from ProlificDataCollector.

        Returns
        -------
        list[dict[str, Any]]
            Merged data with structure:
            {
                "jatos_data": {...},
                "prolific_metadata": {...} | None,
                "merged": bool
            }

        Examples
        --------
        ::

            jatos_results = [
                {"data": {"PROLIFIC_PID": "abc123"}, "metadata": {}}
            ]
            prolific_submissions = [
                {"participant_id": "abc123", "status": "APPROVED"}
            ]
            merged = merger.merge(jatos_results, prolific_submissions)
            assert merged[0]["merged"] is True
        """
        # Create lookup by Prolific participant ID
        prolific_lookup: dict[str, dict[str, Any]] = {
            sub["participant_id"]: sub for sub in prolific_submissions
        }

        merged: list[dict[str, Any]] = []

        for result in jatos_results:
            # Extract Prolific PID from JATOS data
            prolific_pid = self._extract_prolific_pid(result)

            if prolific_pid and prolific_pid in prolific_lookup:
                # Merge
                merged_record: dict[str, Any] = {
                    "jatos_data": result,
                    "prolific_metadata": prolific_lookup[prolific_pid],
                    "merged": True,
                }
            else:
                # No match
                merged_record = {
                    "jatos_data": result,
                    "prolific_metadata": None,
                    "merged": False,
                }

            merged.append(merged_record)

        return merged

    def _extract_prolific_pid(
        self,
        jatos_result: dict[str, Any],
    ) -> str | None:
        """Extract Prolific PID from JATOS result.

        Searches for the participant ID in both the data and metadata
        fields of the JATOS result.

        Parameters
        ----------
        jatos_result : dict[str, Any]
            JATOS result dictionary.

        Returns
        -------
        str | None
            Prolific PID if found, None otherwise.

        Examples
        --------
        ::

            result = {"data": {"PROLIFIC_PID": "abc123"}}
            pid = merger._extract_prolific_pid(result)
            assert pid == "abc123"
        """
        # Check in data field
        data = jatos_result.get("data")
        if isinstance(data, dict):
            # Extract value from untyped JSON and verify type at runtime
            value = data.get(self.merge_key)  # type: ignore[reportUnknownMemberType]
            if isinstance(value, str):
                return value

        # Check in metadata field
        metadata = jatos_result.get("metadata")
        if isinstance(metadata, dict):
            # Extract value from untyped JSON and verify type at runtime
            value = metadata.get(self.merge_key)  # type: ignore[reportUnknownMemberType]
            if isinstance(value, str):
                return value

        return None
