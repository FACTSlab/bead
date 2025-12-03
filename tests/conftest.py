"""Root pytest configuration for bead package tests."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest


def pytest_runtest_setup(item):
    """Hook to set up environment before each test.

    This is called by pytest before running each test item.
    For pytest-codeblocks tests, we need to ensure PATH is set.
    """
    # Get project paths
    tests_dir = Path(__file__).parent
    project_root = tests_dir.parent
    fixtures_dir = tests_dir / "fixtures" / "api_docs"
    work_dir = tests_dir / "fixtures" / "cli_work"
    bead_bin = project_root / ".venv" / "bin"

    # Only set up for codeblocks tests
    if "line" in item.nodeid and "docs/user-guide/cli" in item.nodeid:
        # Ensure work directory exists with fixtures
        if not work_dir.exists():
            work_dir.mkdir(parents=True, exist_ok=True)
            for src_item in fixtures_dir.iterdir():
                dest = work_dir / src_item.name
                if src_item.is_dir():
                    shutil.copytree(src_item, dest, dirs_exist_ok=True)
                else:
                    shutil.copy2(src_item, dest)

        # Change to work directory
        os.chdir(work_dir)

        # Add bead to PATH
        if str(bead_bin) not in os.environ.get("PATH", ""):
            os.environ["PATH"] = f"{bead_bin}:{os.environ.get('PATH', '')}"


@pytest.fixture(scope="session", autouse=True)
def setup_cli_docs_test_environment():
    """Set up environment for CLI documentation tests.

    This fixture:
    1. Creates a temporary working directory for CLI tests
    2. Copies all fixtures from api_docs to cli_work
    3. Adds bead CLI to PATH globally
    4. Cleans up after all tests complete
    """
    fixtures_dir = Path(__file__).parent / "fixtures" / "api_docs"
    work_dir = Path(__file__).parent / "fixtures" / "cli_work"

    # Create working directory
    work_dir.mkdir(parents=True, exist_ok=True)

    # Copy fixtures to working directory
    for item in fixtures_dir.iterdir():
        dest = work_dir / item.name
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()

        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)

    # Save original directory
    original_dir = os.getcwd()

    # Add bead CLI to PATH permanently
    bead_bin = Path(__file__).parent.parent / ".venv" / "bin"
    os.environ["PATH"] = f"{bead_bin}:{os.environ.get('PATH', '')}"

    yield

    # Cleanup
    os.chdir(original_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir, ignore_errors=True)


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
