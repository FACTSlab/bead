"""Tests for template filling CLI commands."""

from __future__ import annotations

import shutil
from pathlib import Path

from click.testing import CliRunner

from bead.cli.templates import templates
from bead.resources.lexicon import LexicalItem
from bead.templates.filler import FilledTemplate


def test_fill_exhaustive(
    cli_runner: CliRunner,
    tmp_path: Path,
    mock_lexicon_file: Path,
    mock_template_file: Path,
) -> None:
    """Test filling templates with exhaustive strategy."""
    output_file = tmp_path / "filled.jsonl"

    result = cli_runner.invoke(
        templates,
        [
            "fill",
            str(mock_template_file),
            str(mock_lexicon_file),
            str(output_file),
            "--strategy",
            "exhaustive",
        ],
    )

    assert result.exit_code == 0
    assert output_file.exists()
    assert "Created" in result.output
    assert "filled templates" in result.output


def test_fill_random(
    cli_runner: CliRunner,
    tmp_path: Path,
    mock_lexicon_file: Path,
    mock_template_file: Path,
) -> None:
    """Test filling templates with random strategy."""
    output_file = tmp_path / "filled.jsonl"

    result = cli_runner.invoke(
        templates,
        [
            "fill",
            str(mock_template_file),
            str(mock_lexicon_file),
            str(output_file),
            "--strategy",
            "random",
            "--max-combinations",
            "5",
            "--random-seed",
            "42",
        ],
    )

    assert result.exit_code == 0
    assert output_file.exists()

    # Verify we got 5 or fewer combinations
    with open(output_file) as f:
        lines = [line for line in f if line.strip()]
        assert len(lines) <= 5


def test_fill_stratified(
    cli_runner: CliRunner,
    tmp_path: Path,
    mock_lexicon_file: Path,
    mock_template_file: Path,
) -> None:
    """Test filling templates with stratified strategy."""
    output_file = tmp_path / "filled.jsonl"

    result = cli_runner.invoke(
        templates,
        [
            "fill",
            str(mock_template_file),
            str(mock_lexicon_file),
            str(output_file),
            "--strategy",
            "stratified",
            "--max-combinations",
            "5",
            "--grouping-property",
            "pos",
        ],
    )

    assert result.exit_code == 0
    assert output_file.exists()


def test_fill_with_language_code(
    cli_runner: CliRunner,
    tmp_path: Path,
    mock_lexicon_file: Path,
    mock_template_file: Path,
) -> None:
    """Test filling templates with language code filter."""
    output_file = tmp_path / "filled.jsonl"

    result = cli_runner.invoke(
        templates,
        [
            "fill",
            str(mock_template_file),
            str(mock_lexicon_file),
            str(output_file),
            "--strategy",
            "exhaustive",
            "--language-code",
            "eng",
        ],
    )

    assert result.exit_code == 0
    assert output_file.exists()


def test_fill_random_missing_max_combinations(
    cli_runner: CliRunner,
    tmp_path: Path,
    mock_lexicon_file: Path,
    mock_template_file: Path,
) -> None:
    """Test error when max-combinations not provided for random strategy."""
    output_file = tmp_path / "filled.jsonl"

    result = cli_runner.invoke(
        templates,
        [
            "fill",
            str(mock_template_file),
            str(mock_lexicon_file),
            str(output_file),
            "--strategy",
            "random",
        ],
    )

    assert result.exit_code == 1
    assert "--max-combinations required" in result.output


def test_fill_stratified_missing_grouping_property(
    cli_runner: CliRunner,
    tmp_path: Path,
    mock_lexicon_file: Path,
    mock_template_file: Path,
) -> None:
    """Test error when grouping-property not provided for stratified strategy."""
    output_file = tmp_path / "filled.jsonl"

    result = cli_runner.invoke(
        templates,
        [
            "fill",
            str(mock_template_file),
            str(mock_lexicon_file),
            str(output_file),
            "--strategy",
            "stratified",
            "--max-combinations",
            "5",
        ],
    )

    assert result.exit_code == 1
    assert "--grouping-property required" in result.output


def test_fill_nonexistent_lexicon(
    cli_runner: CliRunner, tmp_path: Path, mock_template_file: Path
) -> None:
    """Test error with nonexistent lexicon file."""
    output_file = tmp_path / "filled.jsonl"
    bad_lexicon = tmp_path / "nonexistent.jsonl"

    result = cli_runner.invoke(
        templates,
        [
            "fill",
            str(mock_template_file),
            str(bad_lexicon),
            str(output_file),
            "--strategy",
            "exhaustive",
        ],
    )

    assert result.exit_code != 0


def test_fill_nonexistent_template(
    cli_runner: CliRunner, tmp_path: Path, mock_lexicon_file: Path
) -> None:
    """Test error with nonexistent template file."""
    output_file = tmp_path / "filled.jsonl"
    bad_template = tmp_path / "nonexistent.jsonl"

    result = cli_runner.invoke(
        templates,
        [
            "fill",
            str(bad_template),
            str(mock_lexicon_file),
            str(output_file),
            "--strategy",
            "exhaustive",
        ],
    )

    assert result.exit_code != 0


def test_list_filled_empty_directory(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test listing filled templates in empty directory."""
    result = cli_runner.invoke(
        templates,
        ["list-filled", "--directory", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert "No files found" in result.output


def test_list_filled(
    cli_runner: CliRunner, tmp_path: Path, mock_filled_templates_file: Path
) -> None:
    """Test listing filled templates."""
    dest = tmp_path / "filled.jsonl"
    shutil.copy(mock_filled_templates_file, dest)

    result = cli_runner.invoke(
        templates,
        ["list-filled", "--directory", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert "filled.jsonl" in result.output


def test_list_filled_with_pattern(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test listing filled templates with pattern."""
    # Create multiple filled template files
    for i in range(3):
        file_path = tmp_path / f"filled_{i}.jsonl"
        subject = LexicalItem(
            lemma="cat", language_code="eng", features={"pos": "NOUN"}
        )
        verb = LexicalItem(lemma="ran", language_code="eng", features={"pos": "VERB"})
        obj = LexicalItem(lemma="fast", language_code="eng", features={"pos": "ADV"})

        filled = FilledTemplate(
            template_id="test_id",
            template_name="test_template",
            slot_fillers={"subject": subject, "verb": verb, "object": obj},
            rendered_text="cat ran fast",
            strategy_name="exhaustive",
        )

        with open(file_path, "w") as f:
            f.write(filled.model_dump_json() + "\n")

    result = cli_runner.invoke(
        templates,
        ["list-filled", "--directory", str(tmp_path), "--pattern", "filled_0*.jsonl"],
    )

    assert result.exit_code == 0
    assert "filled_0.jsonl" in result.output
    assert "filled_1.jsonl" not in result.output


def test_validate_filled_valid(
    cli_runner: CliRunner, mock_filled_templates_file: Path
) -> None:
    """Test validating valid filled templates file."""
    result = cli_runner.invoke(
        templates,
        ["validate-filled", str(mock_filled_templates_file)],
    )

    assert result.exit_code == 0
    assert "is valid" in result.output


def test_validate_filled_invalid_json(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test validating filled templates with invalid JSON."""
    filled_file = tmp_path / "invalid.jsonl"
    filled_file.write_text("not valid json\n")

    result = cli_runner.invoke(
        templates,
        ["validate-filled", str(filled_file)],
    )

    assert result.exit_code == 1
    assert "Validation failed" in result.output


def test_validate_filled_invalid_data(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test validating filled templates with invalid data."""
    filled_file = tmp_path / "invalid.jsonl"
    # Missing required fields
    filled_file.write_text('{"template_id": "test"}\n')

    result = cli_runner.invoke(
        templates,
        ["validate-filled", str(filled_file)],
    )

    assert result.exit_code == 1
    assert "Validation failed" in result.output


def test_show_stats(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test showing statistics for filled templates."""
    filled_file = tmp_path / "filled.jsonl"

    # Create multiple filled templates
    for i in range(10):
        subject = LexicalItem(
            lemma=f"noun{i}", language_code="eng", features={"pos": "NOUN"}
        )
        verb = LexicalItem(
            lemma=f"verb{i}", language_code="eng", features={"pos": "VERB"}
        )
        obj = LexicalItem(
            lemma=f"obj{i}", language_code="eng", features={"pos": "NOUN"}
        )

        filled = FilledTemplate(
            template_id=f"template_{i % 3}",
            template_name=f"template_{i % 3}",
            slot_fillers={"subject": subject, "verb": verb, "object": obj},
            rendered_text=f"noun{i} verb{i} obj{i}",
            strategy_name="exhaustive" if i < 5 else "random",
        )

        with open(filled_file, "a") as f:
            f.write(filled.model_dump_json() + "\n")

    result = cli_runner.invoke(
        templates,
        ["show-stats", str(filled_file)],
    )

    assert result.exit_code == 0
    assert "Total Filled Templates" in result.output
    assert "10" in result.output
    assert "Unique Template Names" in result.output
    assert "Strategy: exhaustive" in result.output
    assert "Strategy: random" in result.output


def test_show_stats_empty_file(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test showing stats for empty file."""
    filled_file = tmp_path / "empty.jsonl"
    filled_file.write_text("")

    result = cli_runner.invoke(
        templates,
        ["show-stats", str(filled_file)],
    )

    assert result.exit_code == 1
    assert "No valid filled templates found" in result.output


def test_show_stats_with_varied_lengths(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test statistics calculation with varied text lengths."""
    filled_file = tmp_path / "filled.jsonl"

    # Create filled templates with different text lengths
    texts = ["short", "medium length text", "very long text that goes on and on"]

    for _i, text in enumerate(texts):
        subject = LexicalItem(
            lemma="cat", language_code="eng", features={"pos": "NOUN"}
        )
        verb = LexicalItem(lemma="ran", language_code="eng", features={"pos": "VERB"})
        obj = LexicalItem(lemma="fast", language_code="eng", features={"pos": "ADV"})

        filled = FilledTemplate(
            template_id="test",
            template_name="test",
            slot_fillers={"subject": subject, "verb": verb, "object": obj},
            rendered_text=text,
            strategy_name="exhaustive",
        )

        with open(filled_file, "a") as f:
            f.write(filled.model_dump_json() + "\n")

    result = cli_runner.invoke(
        templates,
        ["show-stats", str(filled_file)],
    )

    assert result.exit_code == 0
    assert "Avg Text Length" in result.output
    assert "Min Text Length" in result.output
    assert "Max Text Length" in result.output


def test_templates_help(cli_runner: CliRunner) -> None:
    """Test templates command help."""
    result = cli_runner.invoke(templates, ["--help"])

    assert result.exit_code == 0
    assert "Template filling commands" in result.output


def test_fill_help(cli_runner: CliRunner) -> None:
    """Test fill command help."""
    result = cli_runner.invoke(templates, ["fill", "--help"])

    assert result.exit_code == 0
    assert "Fill templates with lexical items" in result.output


def test_list_filled_help(cli_runner: CliRunner) -> None:
    """Test list-filled command help."""
    result = cli_runner.invoke(templates, ["list-filled", "--help"])

    assert result.exit_code == 0
    assert "List filled template files" in result.output


def test_validate_filled_help(cli_runner: CliRunner) -> None:
    """Test validate-filled command help."""
    result = cli_runner.invoke(templates, ["validate-filled", "--help"])

    assert result.exit_code == 0
    assert "Validate a filled templates file" in result.output


def test_show_stats_help(cli_runner: CliRunner) -> None:
    """Test show-stats command help."""
    result = cli_runner.invoke(templates, ["show-stats", "--help"])

    assert result.exit_code == 0
    assert "Show statistics" in result.output
