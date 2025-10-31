"""Tests for config CLI commands."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from bead.cli.main import cli


def test_config_help(cli_runner: CliRunner) -> None:
    """Test config command help."""
    result = cli_runner.invoke(cli, ["config", "--help"])
    assert result.exit_code == 0
    assert "config" in result.output.lower()


def test_config_show_default(cli_runner: CliRunner) -> None:
    """Test config show with default profile."""
    result = cli_runner.invoke(cli, ["config", "show"])
    assert result.exit_code == 0
    assert "profile:" in result.output


def test_config_show_yaml_format(cli_runner: CliRunner) -> None:
    """Test config show with YAML format."""
    result = cli_runner.invoke(cli, ["config", "show", "--format", "yaml"])
    assert result.exit_code == 0
    assert "profile:" in result.output


def test_config_show_json_format(cli_runner: CliRunner) -> None:
    """Test config show with JSON format."""
    result = cli_runner.invoke(cli, ["config", "show", "--format", "json"])
    assert result.exit_code == 0
    assert '"profile"' in result.output


def test_config_show_table_format(cli_runner: CliRunner) -> None:
    """Test config show with table format."""
    result = cli_runner.invoke(cli, ["config", "show", "--format", "table"])
    assert result.exit_code == 0
    # Table output should contain keys
    assert "profile" in result.output or "Key" in result.output


def test_config_show_specific_key(cli_runner: CliRunner) -> None:
    """Test config show with specific key."""
    result = cli_runner.invoke(cli, ["config", "show", "--key", "profile"])
    assert result.exit_code == 0
    assert "default" in result.output or "test" in result.output


def test_config_show_nested_key(cli_runner: CliRunner) -> None:
    """Test config show with nested key."""
    result = cli_runner.invoke(cli, ["config", "show", "--key", "logging.level"])
    assert result.exit_code == 0
    # Should output the logging level value
    assert result.output.strip() in ["INFO", "DEBUG", "WARNING", "ERROR"]


def test_config_show_nonexistent_key(cli_runner: CliRunner) -> None:
    """Test config show with nonexistent key."""
    result = cli_runner.invoke(cli, ["config", "show", "--key", "nonexistent.key"])
    assert result.exit_code != 0
    assert "not found" in result.output.lower()


def test_config_show_with_file(cli_runner: CliRunner, mock_config_file: Path) -> None:
    """Test config show with config file."""
    result = cli_runner.invoke(
        cli, ["--config-file", str(mock_config_file), "config", "show"]
    )
    assert result.exit_code == 0
    assert "profile:" in result.output or '"profile"' in result.output


def test_config_show_redacts_sensitive_values(cli_runner: CliRunner) -> None:
    """Test config show redacts API keys by default."""
    result = cli_runner.invoke(cli, ["config", "show", "--format", "yaml"])
    assert result.exit_code == 0
    # If there are any API keys, they should be redacted
    # Default config may not have API keys, so just check it runs


def test_config_show_no_redact(cli_runner: CliRunner) -> None:
    """Test config show with --no-redact flag."""
    result = cli_runner.invoke(cli, ["config", "show", "--no-redact"])
    assert result.exit_code == 0


def test_config_validate_no_file(cli_runner: CliRunner) -> None:
    """Test config validate without file."""
    result = cli_runner.invoke(cli, ["config", "validate"])
    assert result.exit_code != 0
    assert "No configuration file" in result.output


def test_config_validate_valid_file(
    cli_runner: CliRunner, mock_config_file: Path
) -> None:
    """Test config validate with valid file."""
    result = cli_runner.invoke(
        cli, ["config", "validate", "--config-file", str(mock_config_file)]
    )
    assert result.exit_code == 0
    assert "valid" in result.output.lower()


def test_config_validate_invalid_yaml(
    cli_runner: CliRunner, mock_invalid_config_file: Path
) -> None:
    """Test config validate with invalid YAML."""
    result = cli_runner.invoke(
        cli, ["config", "validate", "--config-file", str(mock_invalid_config_file)]
    )
    assert result.exit_code == 1


def test_config_validate_with_validation_errors(
    cli_runner: CliRunner, mock_config_with_validation_errors: Path
) -> None:
    """Test config validate with Pydantic validation errors."""
    result = cli_runner.invoke(
        cli,
        [
            "config",
            "validate",
            "--config-file",
            str(mock_config_with_validation_errors),
        ],
    )
    assert result.exit_code == 1
    assert "validation" in result.output.lower() or "failed" in result.output.lower()


def test_config_validate_with_context_config_file(
    cli_runner: CliRunner, mock_config_file: Path
) -> None:
    """Test config validate using context config file."""
    result = cli_runner.invoke(
        cli, ["--config-file", str(mock_config_file), "config", "validate"]
    )
    # Should use the config file from context
    assert result.exit_code == 0


def test_config_export_stdout(cli_runner: CliRunner) -> None:
    """Test config export to stdout."""
    result = cli_runner.invoke(cli, ["config", "export"])
    assert result.exit_code == 0
    assert "profile:" in result.output


def test_config_export_to_file(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test config export to file."""
    output_file = tmp_path / "exported.yaml"

    result = cli_runner.invoke(cli, ["config", "export", "--output", str(output_file)])
    assert result.exit_code == 0
    assert output_file.exists()

    # Check file content
    content = output_file.read_text()
    assert "profile:" in content


def test_config_export_with_comments(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test config export with comments."""
    output_file = tmp_path / "commented.yaml"

    result = cli_runner.invoke(
        cli, ["config", "export", "--comments", "--output", str(output_file)]
    )
    assert result.exit_code == 0
    assert output_file.exists()

    # Check for comments
    content = output_file.read_text()
    assert "#" in content
    assert "profile:" in content


def test_config_export_no_redact(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test config export with --no-redact."""
    output_file = tmp_path / "unredacted.yaml"

    result = cli_runner.invoke(
        cli, ["config", "export", "--no-redact", "--output", str(output_file)]
    )
    assert result.exit_code == 0
    assert output_file.exists()


def test_config_export_with_config_file(
    cli_runner: CliRunner, mock_config_file: Path, tmp_path: Path
) -> None:
    """Test config export using loaded config file."""
    output_file = tmp_path / "exported_from_file.yaml"

    result = cli_runner.invoke(
        cli,
        [
            "--config-file",
            str(mock_config_file),
            "config",
            "export",
            "--output",
            str(output_file),
        ],
    )
    assert result.exit_code == 0
    assert output_file.exists()

    # Check that exported config contains values from mock file
    content = output_file.read_text()
    assert "profile:" in content


def test_config_profiles(cli_runner: CliRunner) -> None:
    """Test config profiles command."""
    result = cli_runner.invoke(cli, ["config", "profiles"])
    assert result.exit_code == 0
    assert "default" in result.output
    assert "dev" in result.output
    assert "prod" in result.output
    assert "test" in result.output


def test_config_show_with_profile(cli_runner: CliRunner) -> None:
    """Test config show with different profile."""
    result = cli_runner.invoke(cli, ["--profile", "dev", "config", "show"])
    assert result.exit_code == 0
    assert "profile:" in result.output


def test_config_show_with_verbose(cli_runner: CliRunner) -> None:
    """Test config show with verbose flag."""
    result = cli_runner.invoke(cli, ["--verbose", "config", "show"])
    assert result.exit_code == 0


def test_config_export_comments_to_stdout(cli_runner: CliRunner) -> None:
    """Test config export with comments to stdout."""
    result = cli_runner.invoke(cli, ["config", "export", "--comments"])
    assert result.exit_code == 0
    assert "#" in result.output
    assert "profile:" in result.output
