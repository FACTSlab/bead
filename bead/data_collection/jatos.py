"""JATOS data collection for model training.

This module provides the JATOSDataCollector class for downloading experimental
results from JATOS servers. It wraps the existing JATOSClient and adds
functionality for:
- Downloading all results for a study
- Filtering by component and worker type
- Adding metadata (timestamps, etc.)
- Saving to JSONLines format
"""

from __future__ import annotations

import json
import zipfile
from io import BytesIO
from pathlib import Path, PurePosixPath

from bead.data.base import JsonValue
from bead.data.timestamps import now_iso8601
from bead.deployment.jatos.api import JATOSClient


def load_jatos_result_archive(
    source: Path | BytesIO,
) -> list[dict[str, JsonValue]]:
    """Load each component's raw data and metadata from a ZIP export.

    JSON arrays are decoded for legacy consumers. Appended JSONL remains text
    for sequence-aware parsing, including the last line of interrupted runs.
    Components without submitted data are included with a null payload.
    """
    results: list[dict[str, JsonValue]] = []
    with zipfile.ZipFile(source) as archive:
        files = set(archive.namelist())
        metadata = json.loads(archive.read("metadata.json"))
        for study in metadata["data"]:
            for result in study["studyResults"]:
                for component in result["componentResults"]:
                    folder = component.get("path") or (
                        f"study_result_{result['id']}/comp_result_{component['id']}"
                    )
                    path = str(PurePosixPath(folder.lstrip("/")) / "data.txt")
                    data: JsonValue = None
                    if path in files:
                        text = archive.read(path).decode("utf-8")
                        try:
                            parsed: JsonValue = json.loads(text)
                            data = (
                                text
                                if isinstance(parsed, dict) and "record_type" in parsed
                                else parsed
                            )
                        except json.JSONDecodeError:
                            data = text
                    results.append(
                        {
                            "result_id": result["id"],
                            "component_result_id": component["id"],
                            "data": data,
                            "metadata": {
                                **result,
                                **component,
                                "studyId": study["studyId"],
                                "studyResultId": result["id"],
                            },
                        }
                    )
    return results


class JATOSDataCollector:
    """Collects experimental data from JATOS API.

    This class wraps the existing JATOSClient to provide data collection
    functionality specifically for model training. It downloads results,
    adds metadata, and saves in JSONLines format.

    Parameters
    ----------
    base_url : str
        JATOS instance URL (e.g., https://jatos.example.com).
    api_token : str
        API authentication token.
    study_id : int
        JATOS study ID to collect data from.

    Attributes
    ----------
    study_id : int
        JATOS study ID to collect data from.
    client : JATOSClient
        Underlying JATOS API client.

    Examples
    --------
    Create a collector and download results::

        collector = JATOSDataCollector(
            base_url="https://jatos.example.com",
            api_token="my-token",
            study_id=123
        )
        results = collector.download_results(Path("results.jsonl"))
    """

    def __init__(
        self,
        base_url: str,
        api_token: str,
        study_id: int,
    ) -> None:
        self.study_id = study_id
        self.client = JATOSClient(base_url, api_token)

    def download_results(
        self,
        output_path: Path,
        component_id: int | None = None,
        worker_type: str | None = None,
    ) -> list[dict[str, JsonValue]]:
        """Download all results for the study.

        Downloads results from JATOS, optionally filtering by component ID
        and worker type. Each result is enriched with download timestamp
        metadata and saved to a JSONLines file (one result per line).

        Parameters
        ----------
        output_path : Path
            Path to save results (JSONLines format).
        component_id : int | None
            Filter by component ID (optional).
        worker_type : str | None
            Filter by worker type (optional).

        Returns
        -------
        list[dict[str, JsonValue]]
            Downloaded results with metadata.

        Raises
        ------
        requests.HTTPError
            If the API request fails.

        Examples
        --------
        Download all results::

            results = collector.download_results(Path("results.jsonl"))

        Download with filters::

            results = collector.download_results(
                Path("results.jsonl"),
                component_id=1,
                worker_type="Prolific"
            )
        """
        archive = self.client.download_result_archive(self.study_id)
        results = []
        for result in load_jatos_result_archive(BytesIO(archive)):
            metadata = result["metadata"]
            if not isinstance(metadata, dict):
                continue
            if component_id is not None and metadata.get("componentId") != component_id:
                continue
            if worker_type is not None and metadata.get("workerType") != worker_type:
                continue
            results.append({**result, "download_timestamp": now_iso8601().isoformat()})

        # Save to JSONLines file (one result per line)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")

        return results

    def get_study_info(self) -> dict[str, JsonValue]:
        """Get study information.

        Delegates to the underlying JATOSClient.

        Returns
        -------
        dict[str, JsonValue]
            Study details dictionary.

        Raises
        ------
        requests.HTTPError
            If the API request fails.

        Examples
        --------
        ::

            info = collector.get_study_info()
            print(info["title"])
        """
        return self.client.get_study(self.study_id)

    def get_result_count(self) -> int:
        """Get count of results.

        Returns
        -------
        int
            Number of results available for the study.

        Raises
        ------
        requests.HTTPError
            If the API request fails.

        Examples
        --------
        ::

            count = collector.get_result_count()
            print(f"Found {count} results")
        """
        result_ids = self.client.get_results(self.study_id)
        return len(result_ids)
