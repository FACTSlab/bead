"""Pytest configuration for CLI documentation tests.

This conftest.py sets up the environment for testing bash code blocks
in CLI documentation using pytest-codeblocks.
"""

import os
import shutil
from pathlib import Path

import pytest

# Paths relative to this file
DOCS_CLI_DIR = Path(__file__).parent
PROJECT_ROOT = DOCS_CLI_DIR.parent.parent.parent
FIXTURES_SRC = PROJECT_ROOT / "tests" / "fixtures" / "api_docs"
FIXTURES_WORK = PROJECT_ROOT / "tests" / "fixtures" / "cli_work"


def pytest_configure(config: pytest.Config) -> None:
    """Set up environment before test collection.

    This hook runs before test collection starts. We set up the fixtures
    directory but do NOT change the cwd yet, as that would break collection.
    """
    # Create working directory and copy fixtures
    FIXTURES_WORK.mkdir(parents=True, exist_ok=True)

    for item in FIXTURES_SRC.iterdir():
        dest = FIXTURES_WORK / item.name
        if dest.exists():
            if dest.is_dir():
                shutil.rmtree(dest)
            else:
                dest.unlink()

        if item.is_dir():
            shutil.copytree(item, dest)
        else:
            shutil.copy2(item, dest)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item: pytest.Item) -> None:
    """Change to fixtures directory before each test runs.

    This hook runs just before each test executes. We change the cwd here
    so that bash commands in the markdown files can find the fixture files.
    """
    # Store original directory on the item for restoration
    item._original_cwd = os.getcwd()  # type: ignore[attr-defined]

    # Change to fixtures directory
    os.chdir(FIXTURES_WORK)

    # Add .venv/bin to PATH for bead CLI
    venv_bin = PROJECT_ROOT / ".venv" / "bin"
    original_path = os.environ.get("PATH", "")
    if str(venv_bin) not in original_path:
        os.environ["PATH"] = f"{venv_bin}:{original_path}"
        item._original_path = original_path  # type: ignore[attr-defined]


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item: pytest.Item) -> None:
    """Restore original directory after each test."""
    if hasattr(item, "_original_cwd"):
        os.chdir(item._original_cwd)

    if hasattr(item, "_original_path"):
        os.environ["PATH"] = item._original_path


def pytest_unconfigure(config: pytest.Config) -> None:
    """Clean up fixtures directory after all tests complete."""
    if FIXTURES_WORK.exists():
        shutil.rmtree(FIXTURES_WORK, ignore_errors=True)
