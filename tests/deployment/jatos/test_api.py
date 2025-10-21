"""Tests for JATOS API client."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import requests

from sash.deployment.jatos.api import JATOSClient


def test_jatos_client_creation() -> None:
    """Test JATOS client can be created."""
    client = JATOSClient("https://jatos.example.com", "test-token")
    assert client.base_url == "https://jatos.example.com"
    assert client.api_token == "test-token"


def test_base_url_trailing_slash_removed() -> None:
    """Test that trailing slash is removed from base_url."""
    client = JATOSClient("https://jatos.example.com/", "test-token")
    assert client.base_url == "https://jatos.example.com"


def test_upload_study(tmp_path: Path) -> None:
    """Test study upload."""
    # Create test .jzip file
    jzip_path = tmp_path / "test.jzip"
    jzip_path.write_bytes(b"test data")

    # Mock the response
    with patch("sash.deployment.jatos.api.requests.Session") as mock_session_cls:
        mock_session = mock_session_cls.return_value
        mock_response = Mock()
        mock_response.json.return_value = {"id": 1, "uuid": "test-uuid-123"}
        mock_session.post.return_value = mock_response

        client = JATOSClient("https://jatos.example.com", "test-token")
        result = client.upload_study(jzip_path)

        # Verify the result
        assert result["id"] == 1
        assert result["uuid"] == "test-uuid-123"

        # Verify the request was made correctly
        mock_session.post.assert_called_once()
        call_args = mock_session.post.call_args
        assert call_args[0][0] == "https://jatos.example.com/api/v1/studies"


def test_upload_study_missing_file(tmp_path: Path) -> None:
    """Test upload raises error for missing file."""
    client = JATOSClient("https://jatos.example.com", "test-token")

    nonexistent_file = tmp_path / "nonexistent.jzip"

    with pytest.raises(FileNotFoundError, match=".jzip file not found"):
        client.upload_study(nonexistent_file)


def test_upload_study_http_error(tmp_path: Path) -> None:
    """Test upload raises HTTP error on failure."""
    jzip_path = tmp_path / "test.jzip"
    jzip_path.write_bytes(b"test data")

    with patch("sash.deployment.jatos.api.requests.Session") as mock_session_cls:
        mock_session = mock_session_cls.return_value
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_session.post.return_value = mock_response

        client = JATOSClient("https://jatos.example.com", "test-token")

        with pytest.raises(requests.HTTPError):
            client.upload_study(jzip_path)


def test_list_studies() -> None:
    """Test listing studies."""
    with patch("sash.deployment.jatos.api.requests.Session") as mock_session_cls:
        mock_session = mock_session_cls.return_value
        mock_response = Mock()
        mock_response.json.return_value = [
            {"id": 1, "title": "Study 1"},
            {"id": 2, "title": "Study 2"},
        ]
        mock_session.get.return_value = mock_response

        client = JATOSClient("https://jatos.example.com", "test-token")
        studies = client.list_studies()

        assert len(studies) == 2
        assert studies[0]["id"] == 1
        assert studies[1]["title"] == "Study 2"

        # Verify the request
        mock_session.get.assert_called_once_with(
            "https://jatos.example.com/api/v1/studies"
        )


def test_get_study() -> None:
    """Test getting study details."""
    with patch("sash.deployment.jatos.api.requests.Session") as mock_session_cls:
        mock_session = mock_session_cls.return_value
        mock_response = Mock()
        mock_response.json.return_value = {
            "id": 123,
            "title": "Test Study",
            "uuid": "test-uuid",
        }
        mock_session.get.return_value = mock_response

        client = JATOSClient("https://jatos.example.com", "test-token")
        study = client.get_study(123)

        assert study["id"] == 123
        assert study["title"] == "Test Study"

        # Verify the request
        mock_session.get.assert_called_once_with(
            "https://jatos.example.com/api/v1/studies/123"
        )


def test_delete_study() -> None:
    """Test deleting a study."""
    with patch("sash.deployment.jatos.api.requests.Session") as mock_session_cls:
        mock_session = mock_session_cls.return_value
        mock_response = Mock()
        mock_session.delete.return_value = mock_response

        client = JATOSClient("https://jatos.example.com", "test-token")
        client.delete_study(123)

        # Verify the request
        mock_session.delete.assert_called_once_with(
            "https://jatos.example.com/api/v1/studies/123"
        )
        mock_response.raise_for_status.assert_called_once()


def test_get_results() -> None:
    """Test getting study results."""
    with patch("sash.deployment.jatos.api.requests.Session") as mock_session_cls:
        mock_session = mock_session_cls.return_value
        mock_response = Mock()
        mock_response.json.return_value = [
            {"id": 1, "data": {"response": "yes"}},
            {"id": 2, "data": {"response": "no"}},
        ]
        mock_session.get.return_value = mock_response

        client = JATOSClient("https://jatos.example.com", "test-token")
        results = client.get_results(123)

        assert len(results) == 2
        assert results[0]["data"]["response"] == "yes"

        # Verify the request
        mock_session.get.assert_called_once_with(
            "https://jatos.example.com/api/v1/studies/123/results"
        )


def test_api_token_in_headers() -> None:
    """Test that API token is included in headers."""
    with patch("sash.deployment.jatos.api.requests.Session") as mock_session_cls:
        mock_session = mock_session_cls.return_value

        JATOSClient("https://jatos.example.com", "my-secret-token")

        # Verify the session headers were updated
        mock_session.headers.update.assert_called_once_with(
            {"Authorization": "Bearer my-secret-token"}
        )


def test_session_reuse() -> None:
    """Test that the same session is reused for multiple requests."""
    with patch("sash.deployment.jatos.api.requests.Session") as mock_session_cls:
        mock_session = mock_session_cls.return_value
        mock_response = Mock()
        mock_response.json.return_value = []
        mock_session.get.return_value = mock_response

        client = JATOSClient("https://jatos.example.com", "test-token")

        # Make multiple requests
        client.list_studies()
        client.get_study(123)

        # Verify Session was only instantiated once
        mock_session_cls.assert_called_once()


def test_http_error_raised_on_list_studies() -> None:
    """Test that HTTP errors are raised for list_studies."""
    with patch("sash.deployment.jatos.api.requests.Session") as mock_session_cls:
        mock_session = mock_session_cls.return_value
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError(
            "500 Server Error"
        )
        mock_session.get.return_value = mock_response

        client = JATOSClient("https://jatos.example.com", "test-token")

        with pytest.raises(requests.HTTPError):
            client.list_studies()


def test_http_error_raised_on_get_results() -> None:
    """Test that HTTP errors are raised for get_results."""
    with patch("sash.deployment.jatos.api.requests.Session") as mock_session_cls:
        mock_session = mock_session_cls.return_value
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("403 Forbidden")
        mock_session.get.return_value = mock_response

        client = JATOSClient("https://jatos.example.com", "test-token")

        with pytest.raises(requests.HTTPError):
            client.get_results(123)


def test_delete_study_http_error() -> None:
    """Test that HTTP errors are raised for delete_study."""
    with patch("sash.deployment.jatos.api.requests.Session") as mock_session_cls:
        mock_session = mock_session_cls.return_value
        mock_response = Mock()
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Not Found")
        mock_session.delete.return_value = mock_response

        client = JATOSClient("https://jatos.example.com", "test-token")

        with pytest.raises(requests.HTTPError):
            client.delete_study(999)
