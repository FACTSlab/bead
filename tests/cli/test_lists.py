"""Tests for list CLI commands."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from bead.cli.lists import lists


def test_partition_balanced(
    cli_runner: CliRunner, tmp_path: Path, mock_items_file: Path
) -> None:
    """Test partitioning items with balanced strategy."""
    output_file = tmp_path / "lists.jsonl"

    result = cli_runner.invoke(
        lists,
        [
            "partition",
            str(mock_items_file),
            str(output_file),
            "--n-lists",
            "2",
            "--strategy",
            "balanced",
        ],
    )

    assert result.exit_code == 0, f"Command failed: {result.output}"
    assert output_file.exists()
    # Verify the file contains 2 lists (one per line)
    lines = output_file.read_text().strip().split("\n")
    assert len(lines) == 2


def test_partition_random(
    cli_runner: CliRunner, tmp_path: Path, mock_items_file: Path
) -> None:
    """Test partitioning items with random strategy."""
    output_file = tmp_path / "lists.jsonl"

    result = cli_runner.invoke(
        lists,
        [
            "partition",
            str(mock_items_file),
            str(output_file),
            "--n-lists",
            "3",
            "--strategy",
            "random",
            "--random-seed",
            "42",
        ],
    )

    assert result.exit_code == 0, f"Command failed: {result.output}"
    assert output_file.exists()
    # Verify the file contains 3 lists (one per line)
    lines = output_file.read_text().strip().split("\n")
    assert len(lines) == 3


def test_partition_invalid_n_lists(
    cli_runner: CliRunner, tmp_path: Path, mock_items_file: Path
) -> None:
    """Test error with invalid n_lists."""
    output_file = tmp_path / "lists.jsonl"

    result = cli_runner.invoke(
        lists,
        [
            "partition",
            str(mock_items_file),
            str(output_file),
            "--n-lists",
            "0",
        ],
    )

    assert result.exit_code == 1
    assert "must be >= 1" in result.output


def test_list_empty_file(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test listing lists from empty JSONL file."""
    empty_file = tmp_path / "empty.jsonl"
    empty_file.write_text("")

    result = cli_runner.invoke(
        lists,
        ["list", str(empty_file)],
    )

    assert result.exit_code == 0


def test_list_experiment_lists(
    cli_runner: CliRunner, mock_experiment_lists_file: Path
) -> None:
    """Test listing experiment lists."""
    result = cli_runner.invoke(
        lists,
        ["list", str(mock_experiment_lists_file)],
    )

    assert result.exit_code == 0
    assert "list_1" in result.output


def test_validate_valid(
    cli_runner: CliRunner, mock_experiment_lists_file: Path
) -> None:
    """Test validating valid experiment list."""
    result = cli_runner.invoke(
        lists,
        ["validate", str(mock_experiment_lists_file)],
    )

    assert result.exit_code == 0
    assert "is valid" in result.output


def test_validate_empty_file(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test validating empty file."""
    list_file = tmp_path / "empty.jsonl"
    list_file.write_text("")

    result = cli_runner.invoke(
        lists,
        ["validate", str(list_file)],
    )

    assert result.exit_code == 1
    assert "empty" in result.output.lower()


def test_show_stats(cli_runner: CliRunner, mock_experiment_lists_file: Path) -> None:
    """Test showing statistics for experiment lists."""
    result = cli_runner.invoke(
        lists,
        ["show-stats", str(mock_experiment_lists_file)],
    )

    assert result.exit_code == 0
    assert "Total Lists" in result.output


def test_lists_help(cli_runner: CliRunner) -> None:
    """Test lists command help."""
    result = cli_runner.invoke(lists, ["--help"])

    assert result.exit_code == 0
    assert "List construction commands" in result.output
