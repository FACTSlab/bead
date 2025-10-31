"""Tests for CLI utility functions."""

from __future__ import annotations

from pathlib import Path

import pytest

from bead.cli.utils import (
    format_output,
    get_nested_value,
    load_config_for_cli,
    redact_sensitive_values,
)


def test_load_config_for_cli_default() -> None:
    """Test loading config with default profile."""
    config = load_config_for_cli(None, "default", False)
    assert config.profile == "default"


def test_load_config_for_cli_with_file(mock_config_file: Path) -> None:
    """Test loading config from file."""
    config = load_config_for_cli(str(mock_config_file), "default", False)
    assert config.logging.level == "DEBUG"
    assert config.resources.cache_external is True


def test_load_config_for_cli_nonexistent_file() -> None:
    """Test loading config from nonexistent file."""
    with pytest.raises(SystemExit):
        load_config_for_cli("/nonexistent/config.yaml", "default", False)


def test_load_config_for_cli_invalid_yaml(mock_invalid_config_file: Path) -> None:
    """Test loading config with invalid YAML."""
    with pytest.raises(SystemExit):
        load_config_for_cli(str(mock_invalid_config_file), "default", False)


def test_format_output_yaml() -> None:
    """Test YAML output formatting."""
    data = {"key": "value", "nested": {"foo": "bar"}}
    output = format_output(data, "yaml")
    assert "key: value" in output
    assert "nested:" in output
    assert "foo: bar" in output


def test_format_output_json() -> None:
    """Test JSON output formatting."""
    data = {"key": "value", "nested": {"foo": "bar"}}
    output = format_output(data, "json")
    assert '"key": "value"' in output
    assert '"nested"' in output
    assert '"foo": "bar"' in output


def test_format_output_table() -> None:
    """Test table output formatting."""
    data = {"key1": "value1", "key2": "value2"}
    output = format_output(data, "table")
    assert "key1" in output
    assert "value1" in output
    assert "key2" in output
    assert "value2" in output


def test_format_output_table_with_nested_dict() -> None:
    """Test table formatting with nested dictionary."""
    data = {"outer": {"inner": "value"}}
    output = format_output(data, "table")
    assert "outer" in output
    assert "inner" in output


def test_format_output_invalid_format() -> None:
    """Test invalid format type."""
    data = {"key": "value"}
    with pytest.raises(ValueError, match="Invalid format type"):
        format_output(data, "invalid")  # type: ignore[arg-type]


def test_format_output_table_requires_dict() -> None:
    """Test table format requires dict data."""
    data = ["item1", "item2"]
    with pytest.raises(ValueError, match="Table format requires dict data"):
        format_output(data, "table")


def test_get_nested_value_simple() -> None:
    """Test getting simple nested value."""
    data = {"a": {"b": {"c": 42}}}
    assert get_nested_value(data, "a.b.c") == 42


def test_get_nested_value_top_level() -> None:
    """Test getting top-level value."""
    data = {"key": "value"}
    assert get_nested_value(data, "key") == "value"


def test_get_nested_value_missing_key() -> None:
    """Test getting nonexistent key."""
    data = {"a": {"b": 1}}
    with pytest.raises(KeyError, match="not found"):
        get_nested_value(data, "a.c")


def test_get_nested_value_non_dict() -> None:
    """Test getting value from non-dict."""
    data = {"a": "string"}
    with pytest.raises(KeyError, match="Cannot access key"):
        get_nested_value(data, "a.b")


def test_redact_sensitive_values_api_key() -> None:
    """Test redacting API keys."""
    data = {"openai_api_key": "sk-12345", "normal_key": "value"}
    redacted = redact_sensitive_values(data)
    assert redacted["openai_api_key"] == "***REDACTED***"
    assert redacted["normal_key"] == "value"


def test_redact_sensitive_values_nested() -> None:
    """Test redacting sensitive values in nested dict."""
    data = {
        "outer": {
            "api_key": "secret123",
            "password": "pass456",
            "normal": "visible",
        }
    }
    redacted = redact_sensitive_values(data)
    assert redacted["outer"]["api_key"] == "***REDACTED***"
    assert redacted["outer"]["password"] == "***REDACTED***"
    assert redacted["outer"]["normal"] == "visible"


def test_redact_sensitive_values_none() -> None:
    """Test redacting None sensitive values."""
    data = {"api_key": None, "token": None}
    redacted = redact_sensitive_values(data)
    assert redacted["api_key"] is None
    assert redacted["token"] is None


def test_redact_sensitive_values_various_keys() -> None:
    """Test redacting various sensitive key patterns."""
    data = {
        "secret": "value1",
        "my_token": "value2",
        "anthropic_api_key": "value3",
        "google_api_key": "value4",
    }
    redacted = redact_sensitive_values(data)
    assert all(v == "***REDACTED***" for v in redacted.values())
