"""Tests for list CLI commands."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from bead.cli.lists import lists


def test_partition_balanced(
    cli_runner: CliRunner, tmp_path: Path, mock_items_file: Path
) -> None:
    """Test partitioning items with balanced strategy."""
    output_dir = tmp_path / "lists"

    result = cli_runner.invoke(
        lists,
        [
            "partition",
            str(mock_items_file),
            str(output_dir),
            "--n-lists",
            "2",
            "--strategy",
            "balanced",
        ],
    )

    assert result.exit_code == 0
    assert output_dir.exists()
    assert (output_dir / "list_0.jsonl").exists()
    assert (output_dir / "list_1.jsonl").exists()


def test_partition_random(
    cli_runner: CliRunner, tmp_path: Path, mock_items_file: Path
) -> None:
    """Test partitioning items with random strategy."""
    output_dir = tmp_path / "lists"

    result = cli_runner.invoke(
        lists,
        [
            "partition",
            str(mock_items_file),
            str(output_dir),
            "--n-lists",
            "3",
            "--strategy",
            "random",
            "--random-seed",
            "42",
        ],
    )

    assert result.exit_code == 0
    assert output_dir.exists()


def test_partition_invalid_n_lists(
    cli_runner: CliRunner, tmp_path: Path, mock_items_file: Path
) -> None:
    """Test error with invalid n_lists."""
    output_dir = tmp_path / "lists"

    result = cli_runner.invoke(
        lists,
        [
            "partition",
            str(mock_items_file),
            str(output_dir),
            "--n-lists",
            "0",
        ],
    )

    assert result.exit_code == 1
    assert "must be >= 1" in result.output


def test_list_empty_directory(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test listing lists in empty directory."""
    result = cli_runner.invoke(
        lists,
        ["list", "--directory", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert "No files found" in result.output


def test_list_experiment_lists(
    cli_runner: CliRunner, tmp_path: Path, mock_experiment_lists_dir: Path
) -> None:
    """Test listing experiment lists."""
    result = cli_runner.invoke(
        lists,
        ["list", "--directory", str(mock_experiment_lists_dir)],
    )

    assert result.exit_code == 0
    assert "list_1.jsonl" in result.output


def test_validate_valid(cli_runner: CliRunner, mock_experiment_lists_dir: Path) -> None:
    """Test validating valid experiment list."""
    list_file = mock_experiment_lists_dir / "list_1.jsonl"

    result = cli_runner.invoke(
        lists,
        ["validate", str(list_file)],
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


def test_show_stats(cli_runner: CliRunner, mock_experiment_lists_dir: Path) -> None:
    """Test showing statistics for experiment lists."""
    result = cli_runner.invoke(
        lists,
        ["show-stats", str(mock_experiment_lists_dir)],
    )

    assert result.exit_code == 0
    assert "Total Lists" in result.output


def test_lists_help(cli_runner: CliRunner) -> None:
    """Test lists command help."""
    result = cli_runner.invoke(lists, ["--help"])

    assert result.exit_code == 0
    assert "List construction commands" in result.output
