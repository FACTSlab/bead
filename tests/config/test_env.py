"""Tests for environment variable configuration support."""

import os
from pathlib import Path

import pytest

from bead.config.env import env_to_nested_dict, load_from_env, parse_env_value


class TestParseEnvValue:
    """Tests for parse_env_value function."""

    def test_parse_boolean_true_lowercase(self) -> None:
        """Test parsing boolean true (lowercase)."""
        assert parse_env_value("true") is True

    def test_parse_boolean_true_uppercase(self) -> None:
        """Test parsing boolean true (uppercase)."""
        assert parse_env_value("TRUE") is True

    def test_parse_boolean_true_mixed_case(self) -> None:
        """Test parsing boolean true (mixed case)."""
        assert parse_env_value("TrUe") is True

    def test_parse_boolean_true_numeric(self) -> None:
        """Test parsing boolean true as numeric."""
        assert parse_env_value("1") is True

    def test_parse_boolean_true_yes(self) -> None:
        """Test parsing boolean true as 'yes'."""
        assert parse_env_value("yes") is True

    def test_parse_boolean_true_on(self) -> None:
        """Test parsing boolean true as 'on'."""
        assert parse_env_value("on") is True

    def test_parse_boolean_false_lowercase(self) -> None:
        """Test parsing boolean false (lowercase)."""
        assert parse_env_value("false") is False

    def test_parse_boolean_false_uppercase(self) -> None:
        """Test parsing boolean false (uppercase)."""
        assert parse_env_value("FALSE") is False

    def test_parse_boolean_false_numeric(self) -> None:
        """Test parsing boolean false as numeric."""
        assert parse_env_value("0") is False

    def test_parse_boolean_false_no(self) -> None:
        """Test parsing boolean false as 'no'."""
        assert parse_env_value("no") is False

    def test_parse_boolean_false_off(self) -> None:
        """Test parsing boolean false as 'off'."""
        assert parse_env_value("off") is False

    def test_parse_integer(self) -> None:
        """Test parsing integer values."""
        assert parse_env_value("42") == 42
        assert isinstance(parse_env_value("42"), int)

    def test_parse_negative_integer(self) -> None:
        """Test parsing negative integer."""
        assert parse_env_value("-100") == -100

    def test_parse_float(self) -> None:
        """Test parsing float values."""
        assert parse_env_value("3.14") == 3.14
        assert isinstance(parse_env_value("3.14"), float)

    def test_parse_negative_float(self) -> None:
        """Test parsing negative float."""
        assert parse_env_value("-2.5") == -2.5

    def test_parse_absolute_path(self) -> None:
        """Test parsing absolute path."""
        result = parse_env_value("/path/to/file")
        assert isinstance(result, Path)
        assert str(result) == "/path/to/file"

    def test_parse_relative_path_dot_slash(self) -> None:
        """Test parsing relative path with ./."""
        result = parse_env_value("./relative/path")
        assert isinstance(result, Path)

    def test_parse_home_path(self) -> None:
        """Test parsing home directory path."""
        result = parse_env_value("~/documents")
        assert isinstance(result, Path)
        # Should expand ~ to actual home directory
        assert "~" not in str(result)

    def test_parse_parent_path(self) -> None:
        """Test parsing parent directory path."""
        result = parse_env_value("../parent")
        assert isinstance(result, Path)

    def test_parse_comma_separated_list(self) -> None:
        """Test parsing comma-separated list."""
        result = parse_env_value("a,b,c")
        assert result == ["a", "b", "c"]

    def test_parse_comma_separated_list_with_spaces(self) -> None:
        """Test parsing comma-separated list with spaces."""
        result = parse_env_value("a, b, c")
        assert result == ["a", "b", "c"]

    def test_parse_string_fallback(self) -> None:
        """Test that non-special values are parsed as strings."""
        assert parse_env_value("hello") == "hello"
        assert isinstance(parse_env_value("hello"), str)

    def test_parse_string_with_special_chars(self) -> None:
        """Test parsing string with special characters."""
        assert parse_env_value("hello-world_123") == "hello-world_123"


class TestEnvToNestedDict:
    """Tests for env_to_nested_dict function."""

    def test_env_to_nested_dict_single_level(self) -> None:
        """Test converting single-level env vars."""
        env_vars = {"BEAD_PROFILE": "dev"}
        result = env_to_nested_dict(env_vars, "BEAD_")
        assert result == {"profile": "dev"}

    def test_env_to_nested_dict_two_levels(self) -> None:
        """Test converting two-level nested env vars."""
        env_vars = {"BEAD_LOGGING__LEVEL": "DEBUG"}
        result = env_to_nested_dict(env_vars, "BEAD_")
        assert result == {"logging": {"level": "DEBUG"}}

    def test_env_to_nested_dict_three_levels(self) -> None:
        """Test converting three-level nested env vars."""
        env_vars = {"BEAD_PATHS__DATA__DIR": "/data"}
        result = env_to_nested_dict(env_vars, "BEAD_")
        # Note: This creates paths -> data -> dir structure
        # The actual config uses paths__data_dir, not paths__data__dir
        assert "paths" in result
        assert "data" in result["paths"]

    def test_env_to_nested_dict_multiple_vars(self) -> None:
        """Test converting multiple env vars."""
        env_vars = {
            "BEAD_LOGGING__LEVEL": "DEBUG",
            "BEAD_LOGGING__CONSOLE": "true",
            "BEAD_PATHS__DATA_DIR": "/data",
        }
        result = env_to_nested_dict(env_vars, "BEAD_")
        assert result["logging"]["level"] == "DEBUG"
        assert result["logging"]["console"] is True
        assert isinstance(result["paths"]["data_dir"], Path)

    def test_env_to_nested_dict_ignores_other_prefixes(self) -> None:
        """Test that vars with different prefixes are ignored."""
        env_vars = {
            "BEAD_LOGGING__LEVEL": "DEBUG",
            "OTHER_VAR": "value",
            "PATH": "/usr/bin",
        }
        result = env_to_nested_dict(env_vars, "BEAD_")
        assert "logging" in result
        assert "other_var" not in result
        assert "path" not in result

    def test_env_to_nested_dict_empty(self) -> None:
        """Test with no matching env vars."""
        env_vars = {"OTHER_VAR": "value"}
        result = env_to_nested_dict(env_vars, "BEAD_")
        assert result == {}

    def test_env_to_nested_dict_custom_prefix(self) -> None:
        """Test with custom prefix."""
        env_vars = {"MYAPP_DATABASE__HOST": "localhost"}
        result = env_to_nested_dict(env_vars, "MYAPP_")
        assert result == {"database": {"host": "localhost"}}

    def test_env_to_nested_dict_case_conversion(self) -> None:
        """Test that keys are converted to lowercase."""
        env_vars = {"BEAD_LOGGING__LEVEL": "DEBUG"}
        result = env_to_nested_dict(env_vars, "BEAD_")
        assert "logging" in result
        assert "level" in result["logging"]
        # Should not have uppercase keys
        assert "LOGGING" not in result

    def test_env_to_nested_dict_with_type_parsing(self) -> None:
        """Test that values are parsed to correct types."""
        env_vars = {
            "BEAD_TEMPLATES__BATCH_SIZE": "500",
            "BEAD_LOGGING__CONSOLE": "false",
            "BEAD_PATHS__DATA_DIR": "/data",
        }
        result = env_to_nested_dict(env_vars, "BEAD_")
        assert result["templates"]["batch_size"] == 500
        assert isinstance(result["templates"]["batch_size"], int)
        assert result["logging"]["console"] is False
        assert isinstance(result["paths"]["data_dir"], Path)


class TestLoadFromEnv:
    """Tests for load_from_env function."""

    def test_load_from_env_default_prefix(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test loading from env with default prefix."""
        monkeypatch.setenv("BEAD_LOGGING__LEVEL", "DEBUG")
        result = load_from_env()
        assert result == {"logging": {"level": "DEBUG"}}

    def test_load_from_env_custom_prefix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test loading from env with custom prefix."""
        monkeypatch.setenv("CUSTOM_DATABASE__HOST", "localhost")
        result = load_from_env(prefix="CUSTOM_")
        assert result == {"database": {"host": "localhost"}}

    def test_load_from_env_multiple_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test loading multiple env vars."""
        monkeypatch.setenv("BEAD_LOGGING__LEVEL", "ERROR")
        monkeypatch.setenv("BEAD_LOGGING__CONSOLE", "false")
        monkeypatch.setenv("BEAD_PATHS__DATA_DIR", "/custom/data")
        result = load_from_env()
        assert result["logging"]["level"] == "ERROR"
        assert result["logging"]["console"] is False
        assert isinstance(result["paths"]["data_dir"], Path)

    def test_load_from_env_no_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test loading when no matching env vars exist."""
        # Clear any BEAD_ vars that might exist
        for key in list(os.environ.keys()):
            if key.startswith("BEAD_"):
                monkeypatch.delenv(key, raising=False)
        result = load_from_env()
        assert result == {}

    def test_load_from_env_mixed_vars(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test loading with mix of BEAD_ and other vars."""
        monkeypatch.setenv("BEAD_PROFILE", "dev")
        monkeypatch.setenv("OTHER_VAR", "ignored")
        monkeypatch.setenv("PATH", "/usr/bin")
        result = load_from_env()
        assert "profile" in result
        assert "other_var" not in result
        assert "path" not in result

    def test_load_from_env_nested_structure(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that nested structure is created correctly."""
        monkeypatch.setenv("BEAD_LOGGING__LEVEL", "INFO")
        monkeypatch.setenv("BEAD_LOGGING__CONSOLE", "true")
        monkeypatch.setenv("BEAD_LOGGING__FILE", "/var/log/app.log")
        result = load_from_env()
        assert "logging" in result
        assert len(result["logging"]) == 3
        assert result["logging"]["level"] == "INFO"
        assert result["logging"]["console"] is True
        assert isinstance(result["logging"]["file"], Path)
