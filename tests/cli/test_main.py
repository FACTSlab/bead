"""Tests for main CLI commands."""

from __future__ import annotations

import os
from pathlib import Path

from click.testing import CliRunner

from bead import __version__
from bead.cli.main import cli


def test_cli_version(cli_runner: CliRunner) -> None:
    """Test --version option."""
    result = cli_runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert __version__ in result.output


def test_cli_help(cli_runner: CliRunner) -> None:
    """Test --help option."""
    result = cli_runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "bead" in result.output.lower()
    assert "linguistic" in result.output.lower() or "judgment" in result.output.lower()


def test_cli_with_config_file(cli_runner: CliRunner, mock_config_file: Path) -> None:
    """Test CLI with --config-file option."""
    result = cli_runner.invoke(cli, ["--config-file", str(mock_config_file), "--help"])
    assert result.exit_code == 0


def test_cli_with_profile(cli_runner: CliRunner) -> None:
    """Test CLI with --profile option."""
    result = cli_runner.invoke(cli, ["--profile", "dev", "--help"])
    assert result.exit_code == 0


def test_cli_with_verbose(cli_runner: CliRunner) -> None:
    """Test CLI with --verbose option."""
    result = cli_runner.invoke(cli, ["--verbose", "--help"])
    assert result.exit_code == 0


def test_cli_with_quiet(cli_runner: CliRunner) -> None:
    """Test CLI with --quiet option."""
    result = cli_runner.invoke(cli, ["--quiet", "--help"])
    assert result.exit_code == 0


def test_init_command_help(cli_runner: CliRunner) -> None:
    """Test init command help."""
    result = cli_runner.invoke(cli, ["init", "--help"])
    assert result.exit_code == 0
    assert "Initialize" in result.output or "init" in result.output.lower()


def test_init_command_creates_project(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test init command creates project structure."""
    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        result = cli_runner.invoke(
            cli,
            ["init", "my_project"],
            catch_exceptions=False,
        )
    finally:
        os.chdir(old_cwd)

    project_dir = tmp_path / "my_project"

    # Check exit code
    assert result.exit_code == 0

    # Check project directory created
    assert project_dir.exists()

    # Check subdirectories
    expected_dirs = [
        "lexicons",
        "templates",
        "filled_templates",
        "items",
        "lists",
        "experiments",
        "data",
        "models",
    ]

    for subdir in expected_dirs:
        assert (project_dir / subdir).exists(), f"Missing directory: {subdir}"

    # Check files
    assert (project_dir / "bead.yaml").exists()
    assert (project_dir / ".gitignore").exists()

    # Check config content
    config_content = (project_dir / "bead.yaml").read_text()
    assert "profile:" in config_content
    assert "paths:" in config_content


def test_init_command_with_profile(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test init command with profile option."""
    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        result = cli_runner.invoke(
            cli,
            ["init", "test_project", "--profile", "dev"],
        )
    finally:
        os.chdir(old_cwd)

    assert result.exit_code == 0
    project_dir = tmp_path / "test_project"
    config_content = (project_dir / "bead.yaml").read_text()
    assert "profile: dev" in config_content


def test_init_command_current_directory(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test init command in current directory."""
    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        result = cli_runner.invoke(cli, ["init"])
    finally:
        os.chdir(old_cwd)

    assert result.exit_code == 0
    assert (tmp_path / "bead.yaml").exists()


def test_init_command_existing_directory_with_force(
    cli_runner: CliRunner, tmp_path: Path
) -> None:
    """Test init command with --force on existing directory."""
    project_dir = tmp_path / "existing_project"
    project_dir.mkdir()

    # Create existing file
    (project_dir / "existing.txt").write_text("existing")

    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        result = cli_runner.invoke(
            cli,
            ["init", "existing_project", "--force"],
        )
    finally:
        os.chdir(old_cwd)

    assert result.exit_code == 0
    assert (project_dir / "bead.yaml").exists()


def test_init_command_invalid_project_name(
    cli_runner: CliRunner, tmp_path: Path
) -> None:
    """Test init command with invalid project name."""
    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        result = cli_runner.invoke(
            cli,
            ["init", "invalid name with spaces"],
        )
    finally:
        os.chdir(old_cwd)

    # Command should fail with exit code 1
    assert result.exit_code != 0
    assert "Invalid project name" in result.output


def test_init_command_existing_nonempty_directory_no_force(
    cli_runner: CliRunner, tmp_path: Path
) -> None:
    """Test init command on existing non-empty directory without --force."""
    project_dir = tmp_path / "nonempty"
    project_dir.mkdir()
    (project_dir / "file.txt").write_text("content")

    # Simulate user declining confirmation
    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        result = cli_runner.invoke(
            cli,
            ["init", "nonempty"],
            input="n\n",  # Answer 'no' to confirmation
        )
    finally:
        os.chdir(old_cwd)

    # Should exit gracefully
    assert result.exit_code == 0


def test_gitignore_content(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test .gitignore file content."""
    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        result = cli_runner.invoke(
            cli,
            ["init", "test_gitignore"],
        )
    finally:
        os.chdir(old_cwd)

    assert result.exit_code == 0

    gitignore_path = tmp_path / "test_gitignore" / ".gitignore"
    gitignore_content = gitignore_path.read_text()

    # Check for important entries
    assert ".cache/" in gitignore_content
    assert "__pycache__/" in gitignore_content
    assert "*.pyc" in gitignore_content
    assert ".DS_Store" in gitignore_content


def test_config_yaml_structure(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test generated bead.yaml structure."""
    old_cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        result = cli_runner.invoke(
            cli,
            ["init", "test_config"],
        )
    finally:
        os.chdir(old_cwd)

    assert result.exit_code == 0

    config_path = tmp_path / "test_config" / "bead.yaml"
    config_content = config_path.read_text()

    # Check for all major sections
    sections = [
        "profile:",
        "logging:",
        "paths:",
        "resources:",
        "templates:",
        "models:",
        "items:",
        "lists:",
        "deployment:",
        "training:",
    ]

    for section in sections:
        assert section in config_content, f"Missing section: {section}"
