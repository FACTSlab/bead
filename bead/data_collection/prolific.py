"""Prolific data collection for model training.

This module provides the ProlificDataCollector class for downloading participant
metadata and submissions from Prolific. It supports:
- Downloading participant submissions with pagination
- Filtering by submission status
- Approving submissions
- Getting study metadata
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import requests

from bead.data.timestamps import now_iso8601


class ProlificDataCollector:
    """Collects participant data from Prolific API.

    This class interfaces with the Prolific API v1 to download participant
    submissions, demographics, and metadata for model training.

    Parameters
    ----------
    api_key : str
        Prolific API key for authentication.
    study_id : str
        Prolific study ID to collect data from.

    Attributes
    ----------
    api_key : str
        Prolific API key for authentication.
    study_id : str
        Prolific study ID to collect data from.
    base_url : str
        Prolific API base URL.
    session : requests.Session
        HTTP session with authentication headers.

    Examples
    --------
    Create a collector and download submissions::

        collector = ProlificDataCollector(
            api_key="my-api-key",
            study_id="abc123"
        )
        submissions = collector.download_submissions(Path("submissions.json"))
    """

    def __init__(
        self,
        api_key: str,
        study_id: str,
    ) -> None:
        self.api_key = api_key
        self.study_id = study_id
        self.base_url = "https://api.prolific.co/api/v1"

        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Token {api_key}"})

    def download_submissions(
        self,
        output_path: Path,
        status: str | None = None,
    ) -> list[dict[str, Any]]:
        """Download participant submissions.

        Downloads all submissions for the study, handling pagination automatically.
        Each submission is enriched with a download timestamp.

        Parameters
        ----------
        output_path : Path
            Path to save submissions (JSON format).
        status : str | None
            Filter by status (e.g., "APPROVED", "AWAITING REVIEW").

        Returns
        -------
        list[dict[str, Any]]
            Downloaded submissions with metadata.

        Raises
        ------
        requests.HTTPError
            If the API request fails.

        Examples
        --------
        Download all submissions::

            submissions = collector.download_submissions(Path("submissions.json"))

        Download with status filter::

            submissions = collector.download_submissions(
                Path("approved.json"),
                status="APPROVED"
            )
        """
        submissions: list[dict[str, Any]] = []
        page = 1

        while True:
            url = f"{self.base_url}/studies/{self.study_id}/submissions/"
            params: dict[str, str | int] = {"page": page}

            if status:
                params["status"] = status

            response = self.session.get(url, params=params)
            response.raise_for_status()

            data: dict[str, Any] = response.json()
            page_submissions: list[Any] = data.get("results", [])

            if not page_submissions:
                break

            # Enrich with metadata
            for sub in page_submissions:
                if isinstance(sub, dict):
                    sub["download_timestamp"] = now_iso8601().isoformat()
                    submissions.append(sub)

            page += 1

        # Save to JSON file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(submissions, f, indent=2)

        return submissions

    def get_study_info(self) -> dict[str, Any]:
        """Get study information.

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
            print(info["name"])
        """
        url = f"{self.base_url}/studies/{self.study_id}/"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def approve_submissions(
        self,
        submission_ids: list[str],
    ) -> None:
        """Approve submissions.

        Approves multiple submissions by transitioning their status to APPROVED.

        Parameters
        ----------
        submission_ids : list[str]
            Submission IDs to approve.

        Raises
        ------
        requests.HTTPError
            If the API request fails.

        Examples
        --------
        ::

            collector.approve_submissions(["sub1", "sub2", "sub3"])
        """
        for submission_id in submission_ids:
            url = f"{self.base_url}/submissions/{submission_id}/transition/"
            data = {"action": "APPROVE"}
            response = self.session.post(url, json=data)
            response.raise_for_status()
