"""Tests for training CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from bead.cli.training import training


def test_show_data_stats_empty_file(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test showing stats for empty file."""
    data_file = tmp_path / "empty.jsonl"
    data_file.write_text("")

    result = cli_runner.invoke(
        training,
        ["show-data-stats", str(data_file)],
    )

    assert result.exit_code == 1
    assert "No data found" in result.output


def test_show_data_stats_valid(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test showing statistics for valid data."""
    data_file = tmp_path / "results.jsonl"

    # Create mock results
    results = [
        {"worker_id": "w1", "data": {"response": "1"}},
        {"worker_id": "w2", "data": {"response": "2"}},
        {"worker_id": "w1", "data": {"response": "3"}},
    ]

    with open(data_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    result = cli_runner.invoke(
        training,
        ["show-data-stats", str(data_file)],
    )

    assert result.exit_code == 0
    assert "Total Results" in result.output
    assert "3" in result.output
    assert "Unique Workers" in result.output
    assert "2" in result.output


def test_show_data_stats_invalid_json(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test showing stats for file with invalid JSON."""
    data_file = tmp_path / "invalid.jsonl"
    data_file.write_text("not valid json\n")

    result = cli_runner.invoke(
        training,
        ["show-data-stats", str(data_file)],
    )

    assert result.exit_code == 1
    assert "Invalid JSON" in result.output


def test_training_help(cli_runner: CliRunner) -> None:
    """Test training command help."""
    result = cli_runner.invoke(training, ["--help"])

    assert result.exit_code == 0
    assert "Training commands" in result.output


def test_collect_data_help(cli_runner: CliRunner) -> None:
    """Test collect-data command help."""
    result = cli_runner.invoke(training, ["collect-data", "--help"])

    assert result.exit_code == 0
    assert "Collect judgment data" in result.output


def test_show_data_stats_help(cli_runner: CliRunner) -> None:
    """Test show-data-stats command help."""
    result = cli_runner.invoke(training, ["show-data-stats", "--help"])

    assert result.exit_code == 0
    assert "Show statistics" in result.output
