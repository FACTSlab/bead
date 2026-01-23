"""Tests for workflow CLI commands."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from bead.cli.workflow import workflow


class TestWorkflowInit:
    """Tests for workflow init command."""

    def test_init_acceptability_template(
        self, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test initializing project with acceptability-study template."""
        project_dir = tmp_path / "test-project"
        project_dir.mkdir()

        result = cli_runner.invoke(
            workflow,
            [
                "init",
                "acceptability-study",
                "--output-dir",
                str(project_dir),
                "--force",
            ],
        )

        assert result.exit_code == 0, f"Output: {result.output}"
        assert (project_dir / "bead.yaml").exists()
        assert (project_dir / "lexicons").exists()
        assert (project_dir / "templates").exists()

    def test_init_forced_choice_template(
        self, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test initializing project with forced-choice template."""
        project_dir = tmp_path / "fc-project"
        project_dir.mkdir()

        result = cli_runner.invoke(
            workflow,
            ["init", "forced-choice", "--output-dir", str(project_dir), "--force"],
        )

        assert result.exit_code == 0
        assert (project_dir / "bead.yaml").exists()

    def test_init_ordinal_scale_template(
        self, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test initializing project with ordinal-scale template."""
        project_dir = tmp_path / "os-project"
        project_dir.mkdir()

        result = cli_runner.invoke(
            workflow,
            ["init", "ordinal-scale", "--output-dir", str(project_dir), "--force"],
        )

        assert result.exit_code == 0
        assert (project_dir / "bead.yaml").exists()

    def test_init_active_learning_template(
        self, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test initializing project with active-learning template."""
        project_dir = tmp_path / "al-project"
        project_dir.mkdir()

        result = cli_runner.invoke(
            workflow,
            ["init", "active-learning", "--output-dir", str(project_dir), "--force"],
        )

        assert result.exit_code == 0
        assert (project_dir / "bead.yaml").exists()

    def test_init_invalid_template(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test initializing with invalid template name."""
        project_dir = tmp_path / "invalid-project"
        project_dir.mkdir()

        result = cli_runner.invoke(
            workflow,
            ["init", "invalid-template", "--output-dir", str(project_dir)],
        )

        assert result.exit_code != 0

    def test_init_existing_directory(
        self, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test initializing in existing directory without force."""
        project_dir = tmp_path / "existing"
        project_dir.mkdir()
        (project_dir / "bead.yaml").write_text("test")

        result = cli_runner.invoke(
            workflow,
            ["init", "acceptability-study", "--output-dir", str(project_dir)],
        )

        assert result.exit_code != 0

    def test_init_with_force(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test initializing with --force flag."""
        project_dir = tmp_path / "force-test"
        project_dir.mkdir()
        (project_dir / "bead.yaml").write_text("old")

        result = cli_runner.invoke(
            workflow,
            [
                "init",
                "acceptability-study",
                "--output-dir",
                str(project_dir),
                "--force",
            ],
        )

        assert result.exit_code == 0
        assert (project_dir / "bead.yaml").exists()


class TestWorkflowRun:
    """Tests for workflow run command."""

    def test_run_with_stages(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test running specific stages."""
        project_dir = tmp_path / "run-stages"
        project_dir.mkdir()
        config_file = project_dir / "bead.yaml"
        config_file.write_text(
            """
profile: test
paths:
  data_dir: .
  lexicons_dir: lexicons
  templates_dir: templates
"""
        )

        result = cli_runner.invoke(
            workflow,
            [
                "run",
                "--config",
                str(config_file),
                "--stages",
                "resources,templates",
                "--dry-run",
            ],
        )

        # Dry run should succeed even without actual files
        assert result.exit_code == 0 or "DRY RUN" in result.output

    def test_run_from_stage(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test running from a specific stage onward."""
        project_dir = tmp_path / "run-from"
        project_dir.mkdir()
        config_file = project_dir / "bead.yaml"
        config_file.write_text("profile: test\npaths:\n  data_dir: .\n")

        result = cli_runner.invoke(
            workflow,
            ["run", "--config", str(config_file), "--from-stage", "items", "--dry-run"],
        )

        assert result.exit_code == 0 or "DRY RUN" in result.output

    def test_run_no_config_file(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test running without config file fails."""
        project_dir = tmp_path / "no-config"
        project_dir.mkdir()
        config_file = project_dir / "nonexistent.yaml"

        result = cli_runner.invoke(
            workflow,
            ["run", "--config", str(config_file)],
        )

        assert result.exit_code != 0


class TestWorkflowStatus:
    """Tests for workflow status command."""

    def test_status_no_state(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test status with no workflow state."""
        project_dir = tmp_path / "no-state"
        project_dir.mkdir()
        config_file = project_dir / "bead.yaml"
        config_file.write_text("profile: test\n")

        result = cli_runner.invoke(
            workflow,
            ["status", "--config", str(config_file)],
        )

        assert result.exit_code == 0

    def test_status_with_completed_stages(
        self, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test status with completed stages."""
        project_dir = tmp_path / "completed"
        project_dir.mkdir()
        config_file = project_dir / "bead.yaml"
        config_file.write_text("profile: test\n")

        # Create state file
        state_dir = project_dir / ".bead"
        state_dir.mkdir()
        state_file = state_dir / "workflow_state.json"
        state_data = {
            "stages": {
                "resources": {
                    "status": "completed",
                    "timestamp": "2025-01-01T00:00:00",
                },
                "templates": {
                    "status": "completed",
                    "timestamp": "2025-01-01T00:01:00",
                },
            },
            "last_run": "2025-01-01T00:01:00",
        }
        state_file.write_text(json.dumps(state_data))

        result = cli_runner.invoke(
            workflow,
            ["status", "--config", str(config_file)],
        )

        assert result.exit_code == 0
        assert "resources" in result.output or "Resources" in result.output


class TestWorkflowResume:
    """Tests for workflow resume command."""

    def test_resume_no_state(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test resume with no prior state."""
        project_dir = tmp_path / "no-resume"
        project_dir.mkdir()
        config_file = project_dir / "bead.yaml"
        config_file.write_text("profile: test\npaths:\n  data_dir: .\n")

        result = cli_runner.invoke(
            workflow,
            ["resume", "--config", str(config_file)],
        )

        # Should handle gracefully (may exit with error or message)
        assert "resume" in result.output.lower() or result.exit_code != 0

    def test_resume_from_last_stage(
        self, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test resuming from last incomplete stage."""
        project_dir = tmp_path / "resume"
        project_dir.mkdir()
        config_file = project_dir / "bead.yaml"
        config_file.write_text("profile: test\npaths:\n  data_dir: .\n")

        # Create state with some completed stages
        state_dir = project_dir / ".bead"
        state_dir.mkdir()
        state_file = state_dir / "workflow_state.json"
        state_data = {
            "stages": {
                "resources": {
                    "status": "completed",
                    "timestamp": "2025-01-01T00:00:00",
                },
            },
            "last_run": "2025-01-01T00:00:00",
        }
        state_file.write_text(json.dumps(state_data))

        result = cli_runner.invoke(
            workflow,
            ["resume", "--config", str(config_file), "--dry-run"],
        )

        # Dry run or actual resume attempt
        assert result.exit_code == 0 or "resume" in result.output.lower()


class TestWorkflowRollback:
    """Tests for workflow rollback command."""

    def test_rollback_to_stage(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test rolling back to a specific stage."""
        project_dir = tmp_path / "rollback"
        project_dir.mkdir()
        config_file = project_dir / "bead.yaml"
        config_file.write_text("profile: test\n")

        # Create output directories
        for d in ["templates", "items"]:
            (project_dir / d).mkdir()
            (project_dir / d / "test.jsonl").write_text("test\n")

        # Create state
        state_dir = project_dir / ".bead"
        state_dir.mkdir()
        state_file = state_dir / "workflow_state.json"
        state_data = {
            "stages": {
                "resources": {"status": "completed"},
                "templates": {"status": "completed"},
                "items": {"status": "completed"},
            },
            "last_run": "2025-01-01T00:00:00",
        }
        state_file.write_text(json.dumps(state_data))

        result = cli_runner.invoke(
            workflow,
            ["rollback", "templates", "--config", str(config_file), "--force"],
        )

        assert result.exit_code == 0 or "rollback" in result.output.lower()

    def test_rollback_with_dry_run(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test rollback with --dry-run flag."""
        project_dir = tmp_path / "dry-rollback"
        project_dir.mkdir()
        config_file = project_dir / "bead.yaml"
        config_file.write_text("profile: test\n")

        items_dir = project_dir / "items"
        items_dir.mkdir()
        (items_dir / "test.jsonl").write_text("test\n")

        state_dir = project_dir / ".bead"
        state_dir.mkdir()
        state_file = state_dir / "workflow_state.json"
        state_data = {
            "stages": {
                "items": {"status": "completed"},
            },
            "last_run": "2025-01-01T00:00:00",
        }
        state_file.write_text(json.dumps(state_data))

        result = cli_runner.invoke(
            workflow,
            ["rollback", "resources", "--config", str(config_file), "--dry-run"],
        )

        # Should show what would be deleted
        assert result.exit_code == 0 or "DRY RUN" in result.output
        # Files should still exist
        assert (items_dir / "test.jsonl").exists()


class TestWorkflowListTemplates:
    """Tests for workflow list-templates command."""

    def test_list_templates_basic(self, cli_runner: CliRunner) -> None:
        """Test listing available templates."""
        result = cli_runner.invoke(workflow, ["list-templates"])

        assert result.exit_code == 0
        assert "acceptability-study" in result.output
        assert "forced-choice" in result.output
        assert "ordinal-scale" in result.output
        assert "active-learning" in result.output


class TestWorkflowHelp:
    """Tests for workflow help and usage."""

    def test_workflow_help(self, cli_runner: CliRunner) -> None:
        """Test workflow command help."""
        result = cli_runner.invoke(workflow, ["--help"])

        assert result.exit_code == 0
        assert "workflow" in result.output.lower() or "Workflow" in result.output

    def test_init_help(self, cli_runner: CliRunner) -> None:
        """Test workflow init help."""
        result = cli_runner.invoke(workflow, ["init", "--help"])

        assert result.exit_code == 0

    def test_run_help(self, cli_runner: CliRunner) -> None:
        """Test workflow run help."""
        result = cli_runner.invoke(workflow, ["run", "--help"])

        assert result.exit_code == 0

    def test_status_help(self, cli_runner: CliRunner) -> None:
        """Test workflow status help."""
        result = cli_runner.invoke(workflow, ["status", "--help"])

        assert result.exit_code == 0

    def test_resume_help(self, cli_runner: CliRunner) -> None:
        """Test workflow resume help."""
        result = cli_runner.invoke(workflow, ["resume", "--help"])

        assert result.exit_code == 0

    def test_rollback_help(self, cli_runner: CliRunner) -> None:
        """Test workflow rollback help."""
        result = cli_runner.invoke(workflow, ["rollback", "--help"])

        assert result.exit_code == 0


class TestWorkflowStateManagement:
    """Tests for workflow state file management."""

    def test_state_file_creation(self, tmp_path: Path) -> None:
        """Test state file is created in correct location."""
        from bead.cli.workflow import get_state_file  # noqa: PLC0415

        project_dir = tmp_path / "state-test"
        project_dir.mkdir()

        state_file = get_state_file(project_dir)

        assert state_file == project_dir / ".bead" / "workflow_state.json"
        assert state_file.parent.exists()

    def test_load_empty_state(self, tmp_path: Path) -> None:
        """Test loading state when no file exists."""
        from bead.cli.workflow import load_state  # noqa: PLC0415

        project_dir = tmp_path / "empty"
        project_dir.mkdir()

        state = load_state(project_dir)

        assert "stages" in state
        assert "last_run" in state
        assert state["last_run"] is None

    def test_save_and_load_state(self, tmp_path: Path) -> None:
        """Test saving and loading state."""
        from bead.cli.workflow import load_state, save_state  # noqa: PLC0415

        project_dir = tmp_path / "save-load"
        project_dir.mkdir()

        test_state = {
            "stages": {
                "resources": {"status": "completed", "timestamp": "2025-01-01T00:00:00"}
            },
            "last_run": "2025-01-01T00:00:00",
        }

        save_state(project_dir, test_state)
        loaded_state = load_state(project_dir)

        assert loaded_state["stages"]["resources"]["status"] == "completed"  # type: ignore
        assert loaded_state["last_run"] == "2025-01-01T00:00:00"

    def test_update_stage_state(self, tmp_path: Path) -> None:
        """Test updating stage status in state."""
        from bead.cli.workflow import load_state, update_stage_state  # noqa: PLC0415

        project_dir = tmp_path / "update"
        project_dir.mkdir()

        update_stage_state(project_dir, "resources", "completed")

        state = load_state(project_dir)
        stages = state.get("stages")
        assert isinstance(stages, dict)
        assert stages["resources"]["status"] == "completed"  # type: ignore
        assert "timestamp" in stages["resources"]  # type: ignore
