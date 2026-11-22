"""JATOS REST API client.

This module provides the JATOSClient class for interacting with JATOS servers
via the REST API.
"""

from __future__ import annotations

from pathlib import Path

import requests

from bead.data.base import JsonValue


class JATOSClient:
    """Client for JATOS REST API.

    Supports uploading study packages (.jzip), listing studies, deleting
    studies, and retrieving study results.

    Parameters
    ----------
    base_url
        Base URL for JATOS instance (e.g., "https://jatos.example.com").
    api_token
        API token for authentication.

    Attributes
    ----------
    base_url : str
        Base URL for JATOS instance (trailing slash removed).
    api_token : str
        API token for authentication.
    session : requests.Session
        HTTP session with authentication headers configured.

    Examples
    --------
    >>> client = JATOSClient("https://jatos.example.com", "my-api-token")
    >>> # studies = client.list_studies()
    """

    def __init__(self, base_url: str, api_token: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_token}"})

    @property
    def _api_url(self) -> str:
        if self.base_url.endswith("/jatos"):
            return f"{self.base_url}/api/v1"
        return f"{self.base_url}/jatos/api/v1"

    def upload_study(self, jzip_path: Path) -> dict[str, JsonValue]:
        """Upload study package to JATOS.

        POST /jatos/api/v1/study

        Parameters
        ----------
        jzip_path : Path
            Path to .jzip file to upload.

        Returns
        -------
        dict[str, JsonValue]
            Response with study ID, UUID, and URL.

        Raises
        ------
        requests.HTTPError
            If the upload fails.
        FileNotFoundError
            If the .jzip file does not exist.

        Examples
        --------
        >>> client = JATOSClient("https://jatos.example.com", "token")
        >>> # result = client.upload_study(Path("study.jzip"))
        >>> # print(result["id"])
        """
        if not jzip_path.exists():
            raise FileNotFoundError(f".jzip file not found: {jzip_path}")

        url = f"{self._api_url}/study"

        with open(jzip_path, "rb") as f:
            files = {"study": (jzip_path.name, f, "application/zip")}
            response = self.session.post(url, files=files)

        response.raise_for_status()
        payload: dict[str, JsonValue] = response.json()
        data = payload.get("data")
        return data if isinstance(data, dict) else payload

    def download_result_archive(self, study_id: int) -> bytes:
        """Download the JATOS result archive for a study as ZIP bytes."""
        response = self.session.post(
            f"{self._api_url}/results",
            params={"studyId": study_id},
            headers={"Accept": "application/zip"},
            timeout=120,
        )
        response.raise_for_status()
        return response.content

    def list_studies(self) -> list[dict[str, JsonValue]]:
        """List all studies.

        GET /api/v1/studies

        Returns
        -------
        list[dict[str, JsonValue]]
            List of study dictionaries.

        Raises
        ------
        requests.HTTPError
            If the request fails.

        Examples
        --------
        >>> client = JATOSClient("https://jatos.example.com", "token")
        >>> # studies = client.list_studies()
        >>> # print(len(studies))
        """
        url = f"{self.base_url}/api/v1/studies"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def get_study(self, study_id: int) -> dict[str, JsonValue]:
        """Get study details.

        GET /api/v1/studies/{study_id}

        Parameters
        ----------
        study_id : int
            Study ID.

        Returns
        -------
        dict[str, JsonValue]
            Study details dictionary.

        Raises
        ------
        requests.HTTPError
            If the request fails.

        Examples
        --------
        >>> client = JATOSClient("https://jatos.example.com", "token")
        >>> # study = client.get_study(123)
        >>> # print(study["title"])
        """
        url = f"{self.base_url}/api/v1/studies/{study_id}"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def delete_study(self, study_id: int) -> None:
        """Delete study.

        DELETE /api/v1/studies/{study_id}

        Parameters
        ----------
        study_id : int
            Study ID to delete.

        Raises
        ------
        requests.HTTPError
            If the request fails.

        Examples
        --------
        >>> client = JATOSClient("https://jatos.example.com", "token")
        >>> # client.delete_study(123)
        """
        url = f"{self.base_url}/api/v1/studies/{study_id}"
        response = self.session.delete(url)
        response.raise_for_status()

    def get_results(self, study_id: int) -> list[int]:
        """Get all result IDs for a study.

        GET /api/v1/studies/{study_id}/results

        Parameters
        ----------
        study_id : int
            Study ID.

        Returns
        -------
        list[int]
            List of result IDs.

        Raises
        ------
        requests.HTTPError
            If the request fails.

        Examples
        --------
        >>> client = JATOSClient("https://jatos.example.com", "token")
        >>> # result_ids = client.get_results(123)
        >>> # print(len(result_ids))
        """
        url = f"{self.base_url}/api/v1/studies/{study_id}/results"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()
