"""Test fixtures for JATOS tests."""

from pathlib import Path

import pytest


@pytest.fixture
def sample_experiment_dir(tmp_path: Path) -> Path:
    """Create a sample experiment directory for testing.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary path fixture.

    Returns
    -------
    Path
        Path to sample experiment directory.
    """
    exp_dir = tmp_path / "experiment"
    exp_dir.mkdir()

    # Create index.html
    (exp_dir / "index.html").write_text(
        """<!DOCTYPE html>
<html>
<head>
    <title>Test Experiment</title>
</head>
<body>
    <div id="jspsych-target"></div>
</body>
</html>"""
    )

    # Create css directory and file
    css_dir = exp_dir / "css"
    css_dir.mkdir()
    (css_dir / "experiment.css").write_text("body { margin: 0; }")

    # Create js directory and file
    js_dir = exp_dir / "js"
    js_dir.mkdir()
    (js_dir / "experiment.js").write_text("console.log('test');")

    # Create data directory and files
    data_dir = exp_dir / "data"
    data_dir.mkdir()
    (data_dir / "config.json").write_text('{"title": "Test"}')
    (data_dir / "timeline.json").write_text('{"trials": []}')

    return exp_dir


@pytest.fixture
def jzip_output_path(tmp_path: Path) -> Path:
    """Create path for .jzip output file.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary path fixture.

    Returns
    -------
    Path
        Path for .jzip output file.
    """
    return tmp_path / "test_study.jzip"
