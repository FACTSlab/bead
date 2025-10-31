"""Tests for deployment CLI commands."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from bead.cli.deployment import deployment


def test_validate_missing_directory(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test validating nonexistent experiment directory."""
    bad_dir = tmp_path / "nonexistent"

    result = cli_runner.invoke(
        deployment,
        ["validate", str(bad_dir)],
    )

    assert result.exit_code != 0


def test_validate_missing_files(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test validating experiment with missing files."""
    exp_dir = tmp_path / "experiment"
    exp_dir.mkdir()

    result = cli_runner.invoke(
        deployment,
        ["validate", str(exp_dir)],
    )

    assert result.exit_code == 1
    assert "Missing required files" in result.output


def test_validate_valid_experiment(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test validating valid experiment structure."""
    exp_dir = tmp_path / "experiment"
    exp_dir.mkdir()

    # Create required files
    (exp_dir / "index.html").write_text("<html></html>")

    css_dir = exp_dir / "css"
    css_dir.mkdir()
    (css_dir / "experiment.css").write_text("body {}")

    js_dir = exp_dir / "js"
    js_dir.mkdir()
    (js_dir / "experiment.js").write_text("const x = 1;")

    data_dir = exp_dir / "data"
    data_dir.mkdir()
    (data_dir / "timeline.json").write_text("[]")

    result = cli_runner.invoke(
        deployment,
        ["validate", str(exp_dir)],
    )

    assert result.exit_code == 0
    assert "is valid" in result.output


def test_validate_invalid_timeline(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test validating experiment with invalid timeline.json."""
    exp_dir = tmp_path / "experiment"
    exp_dir.mkdir()

    (exp_dir / "index.html").write_text("<html></html>")

    css_dir = exp_dir / "css"
    css_dir.mkdir()
    (css_dir / "experiment.css").write_text("body {}")

    js_dir = exp_dir / "js"
    js_dir.mkdir()
    (js_dir / "experiment.js").write_text("const x = 1;")

    data_dir = exp_dir / "data"
    data_dir.mkdir()
    (data_dir / "timeline.json").write_text("{}")  # Should be list, not dict

    result = cli_runner.invoke(
        deployment,
        ["validate", str(exp_dir)],
    )

    assert result.exit_code == 1
    assert "must be a list" in result.output


def test_deployment_help(cli_runner: CliRunner) -> None:
    """Test deployment command help."""
    result = cli_runner.invoke(deployment, ["--help"])

    assert result.exit_code == 0
    assert "Deployment commands" in result.output


def test_generate_help(cli_runner: CliRunner) -> None:
    """Test generate command help."""
    result = cli_runner.invoke(deployment, ["generate", "--help"])

    assert result.exit_code == 0
    assert "Generate jsPsych experiment" in result.output


def test_export_jatos_help(cli_runner: CliRunner) -> None:
    """Test export-jatos command help."""
    result = cli_runner.invoke(deployment, ["export-jatos", "--help"])

    assert result.exit_code == 0
    assert "Export experiment to JATOS" in result.output
