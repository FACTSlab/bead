"""Root pytest configuration for bead package tests."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def tests_dir() -> Path:
    """Get tests directory path.

    Returns
    -------
    Path
        Path to tests directory
    """
    return Path(__file__).parent


@pytest.fixture
def sample_data_dir(tmp_path: Path) -> Path:
    """Create temporary directory for test data.

    Parameters
    ----------
    tmp_path : Path
        Pytest's tmp_path fixture

    Returns
    -------
    Path
        Path to temporary data directory
    """
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir
