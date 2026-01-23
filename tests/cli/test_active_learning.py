"""Integration tests for bead.cli.active_learning CLI commands.

Tests the ONLY fully implemented active learning command:
1. check-convergence - FULLY IMPLEMENTED with actual ConvergenceDetector

DEFERRED COMMANDS (not tested - awaiting full package implementation):
- select-items (requires model.predict_proba)
- run (requires data collection)
- monitor-convergence (requires checkpoint loading)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from bead.cli.active_learning import active_learning


@pytest.fixture
def runner() -> CliRunner:
    """Create Click test runner."""
    return CliRunner()


@pytest.fixture
def sample_predictions_file(tmp_path: Path) -> Path:
    """Create sample predictions file."""
    predictions_file = tmp_path / "predictions.jsonl"

    predictions = []
    for i in range(50):
        pred = {"item_id": f"item_{i}", "prediction": i % 2}
        predictions.append(json.dumps(pred) + "\n")

    predictions_file.write_text("".join(predictions))
    return predictions_file


@pytest.fixture
def sample_labels_file(tmp_path: Path) -> Path:
    """Create sample human labels file with multiple raters."""
    labels_file = tmp_path / "labels.jsonl"

    labels = []
    # Create 3 raters rating 50 items
    for rater_idx in range(3):
        for item_idx in range(50):
            # Add some variation in labels
            label_value = (item_idx + rater_idx) % 2
            label = {
                "item_id": f"item_{item_idx}",
                "label": label_value,
                "rater_id": f"rater_{rater_idx}",
            }
            labels.append(json.dumps(label) + "\n")

    labels_file.write_text("".join(labels))
    return labels_file


class TestCheckConvergence:
    """Test check-convergence command (FULLY IMPLEMENTED)."""

    def test_check_convergence_krippendorff_alpha(
        self,
        runner: CliRunner,
        sample_predictions_file: Path,
        sample_labels_file: Path,
    ) -> None:
        """Test convergence check with Krippendorff's alpha."""
        result = runner.invoke(
            active_learning,
            [
                "check-convergence",
                "--predictions",
                str(sample_predictions_file),
                "--human-labels",
                str(sample_labels_file),
                "--metric",
                "krippendorff_alpha",
                "--threshold",
                "0.80",
            ],
        )

        # Exit code 0 (converged) or 1 (not converged) both valid
        assert result.exit_code in [0, 1]
        assert "Convergence Results" in result.output
        assert "Human Baseline" in result.output
        assert "Model Agreement" in result.output
        assert "krippendorff_alpha" in result.output or "Krippendorff" in result.output

    def test_check_convergence_percentage_agreement(
        self,
        runner: CliRunner,
        sample_predictions_file: Path,
        sample_labels_file: Path,
    ) -> None:
        """Test convergence check with percentage agreement."""
        result = runner.invoke(
            active_learning,
            [
                "check-convergence",
                "--predictions",
                str(sample_predictions_file),
                "--human-labels",
                str(sample_labels_file),
                "--metric",
                "percentage_agreement",
                "--threshold",
                "0.75",
            ],
        )

        assert result.exit_code in [0, 1]
        assert "Convergence Results" in result.output

    def test_check_convergence_custom_threshold(
        self,
        runner: CliRunner,
        sample_predictions_file: Path,
        sample_labels_file: Path,
    ) -> None:
        """Test convergence check with custom threshold."""
        result = runner.invoke(
            active_learning,
            [
                "check-convergence",
                "--predictions",
                str(sample_predictions_file),
                "--human-labels",
                str(sample_labels_file),
                "--metric",
                "krippendorff_alpha",
                "--threshold",
                "0.95",  # Very high threshold
                "--min-iterations",
                "1",
            ],
        )

        assert result.exit_code in [0, 1]
        assert "0.95" in result.output or "0.9500" in result.output

    def test_check_convergence_missing_predictions_file(
        self,
        runner: CliRunner,
        tmp_path: Path,
        sample_labels_file: Path,
    ) -> None:
        """Test error when predictions file is missing."""
        nonexistent_file = tmp_path / "nonexistent.jsonl"

        result = runner.invoke(
            active_learning,
            [
                "check-convergence",
                "--predictions",
                str(nonexistent_file),
                "--human-labels",
                str(sample_labels_file),
                "--metric",
                "krippendorff_alpha",
                "--threshold",
                "0.80",
            ],
        )

        assert result.exit_code == 2  # Click exits with 2 for invalid paths

    def test_check_convergence_invalid_json(
        self,
        runner: CliRunner,
        tmp_path: Path,
        sample_labels_file: Path,
    ) -> None:
        """Test error handling for invalid JSON."""
        invalid_file = tmp_path / "invalid.jsonl"
        invalid_file.write_text("not valid json\n{incomplete")

        result = runner.invoke(
            active_learning,
            [
                "check-convergence",
                "--predictions",
                str(invalid_file),
                "--human-labels",
                str(sample_labels_file),
                "--metric",
                "krippendorff_alpha",
                "--threshold",
                "0.80",
            ],
        )

        assert result.exit_code == 1
        assert "Invalid JSON" in result.output or "failed" in result.output.lower()

    def test_check_convergence_missing_required_field(
        self,
        runner: CliRunner,
        tmp_path: Path,
        sample_labels_file: Path,
    ) -> None:
        """Test error when required field is missing."""
        bad_predictions = tmp_path / "bad_predictions.jsonl"
        bad_predictions.write_text(
            '{"item_id": "item_0"}\n'  # Missing 'prediction' field
        )

        result = runner.invoke(
            active_learning,
            [
                "check-convergence",
                "--predictions",
                str(bad_predictions),
                "--human-labels",
                str(sample_labels_file),
                "--metric",
                "krippendorff_alpha",
                "--threshold",
                "0.80",
            ],
        )

        assert result.exit_code == 1
        assert (
            "Missing required field" in result.output
            or "KeyError" in result.output
            or "failed" in result.output.lower()
        )


class TestActiveLearnHelpCommand:
    """Test help command and documentation."""

    def test_active_learning_help(self, runner: CliRunner) -> None:
        """Test active-learning group help text."""
        result = runner.invoke(active_learning, ["--help"])

        assert result.exit_code == 0
        assert "Active learning commands" in result.output
        assert "check-convergence" in result.output
        assert "select-items" in result.output
        assert "run" in result.output

    def test_check_convergence_help(self, runner: CliRunner) -> None:
        """Test check-convergence help text."""
        result = runner.invoke(active_learning, ["check-convergence", "--help"])

        assert result.exit_code == 0
        assert "convergence" in result.output.lower()
        assert "--predictions" in result.output
        assert "--human-labels" in result.output
        assert "--metric" in result.output
        assert "--threshold" in result.output
