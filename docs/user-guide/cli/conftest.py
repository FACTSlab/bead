"""Pytest configuration for CLI documentation tests.

This conftest.py sets up the environment for testing bash code blocks
in CLI documentation using pytest-codeblocks.

The fixtures are copied to a temporary directory to avoid modifying
the committed fixture files.
"""

import os
import shutil
import tempfile
from pathlib import Path

import pytest

# Paths relative to this file
DOCS_CLI_DIR = Path(__file__).parent
PROJECT_ROOT = DOCS_CLI_DIR.parent.parent.parent
FIXTURES_SRC = PROJECT_ROOT / "tests" / "fixtures" / "api_docs"

# Global temp directory for the test session
_temp_dir: Path | None = None


def pytest_configure(config: pytest.Config) -> None:
    """Set up temporary fixtures directory before test collection."""
    global _temp_dir
    _temp_dir = Path(tempfile.mkdtemp(prefix="bead_cli_test_"))

    # Copy fixtures to temp directory
    for item in FIXTURES_SRC.iterdir():
        dest = _temp_dir / item.name
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
    if _temp_dir is None:
        return

    # Store original directory on the item for restoration
    item._original_cwd = os.getcwd()  # type: ignore[attr-defined]

    # Change to temp fixtures directory
    os.chdir(_temp_dir)

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
    """Clean up temporary fixtures directory after all tests complete."""
    global _temp_dir
    if _temp_dir is not None and _temp_dir.exists():
        shutil.rmtree(_temp_dir, ignore_errors=True)
        _temp_dir = None
