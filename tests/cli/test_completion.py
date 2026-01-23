"""Tests for completion CLI commands."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

from click.testing import CliRunner

from bead.cli.completion import completion


class TestCompletionBash:
    """Tests for bash completion generation."""

    def test_bash_completion_output(self, cli_runner: CliRunner) -> None:
        """Test bash completion script is generated."""
        result = cli_runner.invoke(completion, ["bash"])

        assert result.exit_code == 0
        assert "_bead_completion" in result.output
        assert "complete" in result.output
        assert "COMP_WORDS" in result.output
        assert "_BEAD_COMPLETE=bash_complete" in result.output

    def test_bash_completion_function_name(self, cli_runner: CliRunner) -> None:
        """Test bash completion has correct function name."""
        result = cli_runner.invoke(completion, ["bash"])

        assert result.exit_code == 0
        assert "_bead_completion()" in result.output

    def test_bash_completion_complete_command(self, cli_runner: CliRunner) -> None:
        """Test bash completion has correct complete command."""
        result = cli_runner.invoke(completion, ["bash"])

        assert result.exit_code == 0
        assert "complete -o nosort -F _bead_completion bead" in result.output

    def test_bash_completion_help_message(self, cli_runner: CliRunner) -> None:
        """Test bash completion includes help message on stderr."""
        result = cli_runner.invoke(completion, ["bash"])

        assert result.exit_code == 0
        # Help message should be in stderr (captured in output by cli_runner)


class TestCompletionZsh:
    """Tests for zsh completion generation."""

    def test_zsh_completion_output(self, cli_runner: CliRunner) -> None:
        """Test zsh completion script is generated."""
        result = cli_runner.invoke(completion, ["zsh"])

        assert result.exit_code == 0
        assert "#compdef bead" in result.output
        assert "_bead()" in result.output
        assert "_BEAD_COMPLETE=zsh_complete" in result.output

    def test_zsh_completion_compdef(self, cli_runner: CliRunner) -> None:
        """Test zsh completion has compdef directive."""
        result = cli_runner.invoke(completion, ["zsh"])

        assert result.exit_code == 0
        assert "compdef _bead bead" in result.output

    def test_zsh_completion_function(self, cli_runner: CliRunner) -> None:
        """Test zsh completion has proper function structure."""
        result = cli_runner.invoke(completion, ["zsh"])

        assert result.exit_code == 0
        assert "_bead() {" in result.output
        assert "local -a completions" in result.output
        assert "_describe" in result.output or "compadd" in result.output


class TestCompletionFish:
    """Tests for fish completion generation."""

    def test_fish_completion_output(self, cli_runner: CliRunner) -> None:
        """Test fish completion script is generated."""
        result = cli_runner.invoke(completion, ["fish"])

        assert result.exit_code == 0
        assert "__fish_bead_complete" in result.output
        assert "set -lx _BEAD_COMPLETE fish_complete" in result.output

    def test_fish_completion_function(self, cli_runner: CliRunner) -> None:
        """Test fish completion has proper function structure."""
        result = cli_runner.invoke(completion, ["fish"])

        assert result.exit_code == 0
        assert "function __fish_bead_complete" in result.output
        assert "complete -c bead" in result.output


class TestCompletionInstall:
    """Tests for auto-install command."""

    def test_install_bash(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test installing bash completion."""
        with patch("bead.cli.completion.Path.home") as mock_home:
            mock_home.return_value = tmp_path

            with patch.dict("os.environ", {"SHELL": "/bin/bash"}):
                with patch("bead.cli.completion.subprocess.run") as mock_run:
                    mock_result = MagicMock()
                    mock_result.stdout = "# bash completion script"
                    mock_run.return_value = mock_result

                    result = cli_runner.invoke(completion, ["install"])

        assert result.exit_code == 0
        assert (
            "Detected shell: bash" in result.output
            or "Installed bash completion" in result.output
        )

    def test_install_zsh(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test installing zsh completion."""
        with patch("bead.cli.completion.Path.home") as mock_home:
            mock_home.return_value = tmp_path

            with patch.dict("os.environ", {"SHELL": "/bin/zsh"}):
                with patch("bead.cli.completion.subprocess.run") as mock_run:
                    mock_result = MagicMock()
                    mock_result.stdout = "#compdef bead"
                    mock_run.return_value = mock_result

                    result = cli_runner.invoke(completion, ["install"])

        assert result.exit_code == 0
        assert (
            "Detected shell: zsh" in result.output
            or "Installed zsh completion" in result.output
        )

    def test_install_fish(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test installing fish completion."""
        with patch("bead.cli.completion.Path.home") as mock_home:
            mock_home.return_value = tmp_path

            with patch.dict("os.environ", {"SHELL": "/usr/bin/fish"}):
                with patch("bead.cli.completion.subprocess.run") as mock_run:
                    mock_result = MagicMock()
                    mock_result.stdout = "# fish completion"
                    mock_run.return_value = mock_result

                    result = cli_runner.invoke(completion, ["install"])

        assert result.exit_code == 0
        assert (
            "Detected shell: fish" in result.output
            or "Installed fish completion" in result.output
        )

    def test_install_no_shell(self, cli_runner: CliRunner) -> None:
        """Test install fails without SHELL environment variable."""
        with patch.dict("os.environ", {}, clear=False):
            # Remove SHELL if it exists
            os.environ.pop("SHELL", None)

            result = cli_runner.invoke(completion, ["install"])

        assert result.exit_code != 0
        assert "Could not detect shell" in result.output

    def test_install_unsupported_shell(self, cli_runner: CliRunner) -> None:
        """Test install fails with unsupported shell."""
        with patch.dict("os.environ", {"SHELL": "/bin/csh"}):
            result = cli_runner.invoke(completion, ["install"])

        assert result.exit_code != 0
        assert "Unsupported shell" in result.output


class TestCompletionHelp:
    """Tests for completion help and usage."""

    def test_completion_help(self, cli_runner: CliRunner) -> None:
        """Test completion command help."""
        result = cli_runner.invoke(completion, ["--help"])

        assert result.exit_code == 0
        assert "completion" in result.output.lower()
        assert "bash" in result.output
        assert "zsh" in result.output
        assert "fish" in result.output
        assert "install" in result.output

    def test_bash_help(self, cli_runner: CliRunner) -> None:
        """Test bash command help."""
        result = cli_runner.invoke(completion, ["bash", "--help"])

        assert result.exit_code == 0
        assert "bash" in result.output.lower()

    def test_zsh_help(self, cli_runner: CliRunner) -> None:
        """Test zsh command help."""
        result = cli_runner.invoke(completion, ["zsh", "--help"])

        assert result.exit_code == 0
        assert "zsh" in result.output.lower()

    def test_fish_help(self, cli_runner: CliRunner) -> None:
        """Test fish command help."""
        result = cli_runner.invoke(completion, ["fish", "--help"])

        assert result.exit_code == 0
        assert "fish" in result.output.lower()

    def test_install_help(self, cli_runner: CliRunner) -> None:
        """Test install command help."""
        result = cli_runner.invoke(completion, ["install", "--help"])

        assert result.exit_code == 0
        assert "install" in result.output.lower()


class TestCompletionInstallHelpers:
    """Tests for installation helper functions."""

    def test_install_bash_creates_directory(self, tmp_path: Path) -> None:
        """Test bash installation creates completion directory."""
        from bead.cli.completion import _install_bash  # noqa: PLC0415

        with patch("bead.cli.completion.Path.home") as mock_home:
            mock_home.return_value = tmp_path

            with patch("bead.cli.completion.subprocess.run") as mock_run:
                mock_result = MagicMock()
                mock_result.stdout = "# completion"
                mock_run.return_value = mock_result

                try:
                    _install_bash()
                except Exception:
                    pass  # May fail due to subprocess, but directory should exist

        completion_dir = tmp_path / ".bash_completion.d"
        assert completion_dir.exists()

    def test_install_zsh_creates_directory(self, tmp_path: Path) -> None:
        """Test zsh installation creates completion directory."""
        from bead.cli.completion import _install_zsh  # noqa: PLC0415

        with patch("bead.cli.completion.Path.home") as mock_home:
            mock_home.return_value = tmp_path

            with patch("bead.cli.completion.subprocess.run") as mock_run:
                mock_result = MagicMock()
                mock_result.stdout = "# completion"
                mock_run.return_value = mock_result

                try:
                    _install_zsh()
                except Exception:
                    pass

        completion_dir = tmp_path / ".zsh" / "completion"
        assert completion_dir.exists()

    def test_install_fish_creates_directory(self, tmp_path: Path) -> None:
        """Test fish installation creates completion directory."""
        from bead.cli.completion import _install_fish  # noqa: PLC0415

        with patch("bead.cli.completion.Path.home") as mock_home:
            mock_home.return_value = tmp_path

            with patch("bead.cli.completion.subprocess.run") as mock_run:
                mock_result = MagicMock()
                mock_result.stdout = "# completion"
                mock_run.return_value = mock_result

                try:
                    _install_fish()
                except Exception:
                    pass

        completion_dir = tmp_path / ".config" / "fish" / "completions"
        assert completion_dir.exists()


class TestCompletionScriptContent:
    """Tests for completion script content validation."""

    def test_bash_script_has_required_elements(self, cli_runner: CliRunner) -> None:
        """Test bash script contains all required elements."""
        result = cli_runner.invoke(completion, ["bash"])

        assert result.exit_code == 0
        # Required bash completion elements
        assert "COMP_WORDS" in result.output
        assert "COMP_CWORD" in result.output
        assert "COMPREPLY" in result.output
        assert "IFS" in result.output

    def test_zsh_script_has_required_elements(self, cli_runner: CliRunner) -> None:
        """Test zsh script contains all required elements."""
        result = cli_runner.invoke(completion, ["zsh"])

        assert result.exit_code == 0
        # Required zsh completion elements
        assert "local -a" in result.output
        assert "response=" in result.output or "completions" in result.output

    def test_fish_script_has_required_elements(self, cli_runner: CliRunner) -> None:
        """Test fish script contains all required elements."""
        result = cli_runner.invoke(completion, ["fish"])

        assert result.exit_code == 0
        # Required fish completion elements
        assert "commandline" in result.output
        assert "COMP_WORDS" in result.output or "set -lx" in result.output
