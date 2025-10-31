"""JATOS REST API client.

This module provides the JATOSClient class for interacting with JATOS servers
via the REST API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import requests


class JATOSClient:
    """Client for JATOS REST API.

    Supports:
    - Uploading study packages (.jzip)
    - Listing studies
    - Deleting studies
    - Getting study results

    Attributes
    ----------
    base_url : str
        Base URL for JATOS instance (e.g., https://jatos.example.com).
    api_token : str
        API token for authentication.

    Examples
    --------
    >>> client = JATOSClient("https://jatos.example.com", "my-api-token")
    >>> # studies = client.list_studies()
    """

    def __init__(self, base_url: str, api_token: str) -> None:
        """Initialize JATOS API client.

        Parameters
        ----------
        base_url : str
            Base URL for JATOS instance.
        api_token : str
            API token for authentication.
        """
        self.base_url = base_url.rstrip("/")
        self.api_token = api_token
        self.session = requests.Session()
        self.session.headers.update({"Authorization": f"Bearer {api_token}"})

    def upload_study(self, jzip_path: Path) -> dict[str, Any]:
        """Upload study package to JATOS.

        POST /api/v1/studies

        Parameters
        ----------
        jzip_path : Path
            Path to .jzip file to upload.

        Returns
        -------
        dict[str, Any]
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

        url = f"{self.base_url}/api/v1/studies"

        with open(jzip_path, "rb") as f:
            files = {"file": (jzip_path.name, f, "application/zip")}
            response = self.session.post(url, files=files)

        response.raise_for_status()
        return response.json()

    def list_studies(self) -> list[dict[str, Any]]:
        """List all studies.

        GET /api/v1/studies

        Returns
        -------
        list[dict[str, Any]]
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

    def get_study(self, study_id: int) -> dict[str, Any]:
        """Get study details.

        GET /api/v1/studies/{study_id}

        Parameters
        ----------
        study_id : int
            Study ID.

        Returns
        -------
        dict[str, Any]
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
