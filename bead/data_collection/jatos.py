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
from pathlib import Path
from typing import Any

from bead.data.timestamps import now_iso8601
from bead.deployment.jatos.api import JATOSClient


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
    ) -> list[dict[str, Any]]:
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
        list[dict[str, Any]]
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
        # Get result IDs from JATOS API
        result_ids = self.client.get_results(self.study_id)

        results: list[dict[str, Any]] = []

        # Download each result with metadata
        for result_id in result_ids:
            result = self._download_single_result(result_id)

            # Apply filters
            if component_id is not None:
                result_component_id = result.get("metadata", {}).get("componentId")
                if result_component_id != component_id:
                    continue

            if worker_type is not None:
                result_worker_type = result.get("metadata", {}).get("workerType")
                if result_worker_type != worker_type:
                    continue

            results.append(result)

        # Save to JSONLines file (one result per line)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for result in results:
                f.write(json.dumps(result) + "\n")

        return results

    def _download_single_result(self, result_id: int) -> dict[str, Any]:
        """Download a single result with metadata.

        Parameters
        ----------
        result_id : int
            Result ID to download.

        Returns
        -------
        dict[str, Any]
            Result data with metadata and download timestamp.

        Raises
        ------
        requests.HTTPError
            If the API request fails.
        """
        # Get result data
        data_url = f"{self.client.base_url}/api/v1/results/{result_id}/data"
        data_response = self.client.session.get(data_url)
        data_response.raise_for_status()

        # Get result metadata
        meta_url = f"{self.client.base_url}/api/v1/results/{result_id}"
        meta_response = self.client.session.get(meta_url)
        meta_response.raise_for_status()

        metadata = meta_response.json()

        return {
            "result_id": result_id,
            "data": data_response.json(),
            "metadata": metadata,
            "download_timestamp": now_iso8601().isoformat(),
        }

    def get_study_info(self) -> dict[str, Any]:
        """Get study information.

        Delegates to the underlying JATOSClient.

        Returns
        -------
        dict[str, Any]
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
