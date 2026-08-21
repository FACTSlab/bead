"""Tests that the declared version stays consistent across the release artifacts.

``pyproject.toml`` is the single source of truth for the version: ``bead.__version__``
reads it back out of the installed distribution metadata rather than restating it. The
remaining manual steps of a release are the CHANGELOG entry and its compare link, so
these tests guard those against drifting from the declared version.
"""

from __future__ import annotations

import re
import tomllib
from importlib.metadata import version
from pathlib import Path

import pytest

import bead

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"
CHANGELOG = REPO_ROOT / "CHANGELOG.md"

# "## [0.9.0] - 2026-08-21", but not "## [Unreleased]"
RELEASE_HEADING = re.compile(
    r"^## \[(\d+\.\d+\.\d+)\] - \d{4}-\d{2}-\d{2}$", re.MULTILINE
)


def declared_version() -> str:
    """Read the version declared in ``pyproject.toml``."""
    if not PYPROJECT.is_file():
        pytest.skip("pyproject.toml is not available outside a source checkout")

    with PYPROJECT.open("rb") as handle:
        return str(tomllib.load(handle)["project"]["version"])


def changelog_text() -> str:
    """Read the CHANGELOG."""
    if not CHANGELOG.is_file():
        pytest.skip("CHANGELOG.md is not available outside a source checkout")

    return CHANGELOG.read_text(encoding="utf-8")


def test_version_comes_from_distribution_metadata() -> None:
    """bead.__version__ is read from the installed metadata, not restated in source."""
    assert bead.__version__ == version("bead")


def test_version_matches_pyproject() -> None:
    """The importable version matches the version declared in pyproject.toml."""
    assert bead.__version__ == declared_version()


def test_changelog_documents_the_current_version() -> None:
    """The newest dated CHANGELOG entry is the version being shipped."""
    releases = RELEASE_HEADING.findall(changelog_text())

    assert releases, "CHANGELOG.md has no dated release headings"
    assert releases[0] == bead.__version__


def test_changelog_links_the_current_version() -> None:
    """The CHANGELOG defines a compare link for the version being shipped."""
    text = changelog_text()

    assert f"\n[{bead.__version__}]: https://" in text
    assert f"compare/v{bead.__version__}...HEAD" in text
