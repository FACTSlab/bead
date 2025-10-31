"""CLI utility functions for bead package.

This module provides utility functions for the CLI including configuration loading,
output formatting, error handling, and user prompts.
"""

from __future__ import annotations

import json
import sys
from io import StringIO
from pathlib import Path
from typing import Any, Literal

import click
import yaml
from rich.console import Console
from rich.table import Table

from bead.config import BeadConfig, load_config

console = Console()


def load_config_for_cli(
    config_file: str | None,
    profile: str,
    verbose: bool,
) -> BeadConfig:
    """Load configuration with CLI options.

    Parameters
    ----------
    config_file : str | None
        Path to configuration file (None to use profile defaults).
    profile : str
        Configuration profile name (default, dev, prod, test).
    verbose : bool
        Whether to enable verbose output.

    Returns
    -------
    BeadConfig
        Loaded configuration object.

    Raises
    ------
    FileNotFoundError
        If config_file is specified but doesn't exist.
    ValidationError
        If configuration is invalid.
    """
    config_path = Path(config_file) if config_file else None

    try:
        config = load_config(config_path=config_path, profile=profile)

        if verbose:
            console.print(
                f"[green]✓[/green] Loaded configuration from profile: {profile}"
            )
            if config_file:
                console.print(f"[green]✓[/green] Applied overrides from: {config_file}")

        return config
    except FileNotFoundError:
        print_error(f"Configuration file not found: {config_file}", exit_code=1)
        raise  # For type checking
    except Exception as e:
        print_error(f"Failed to load configuration: {e}", exit_code=1)
        raise  # For type checking


def format_output(
    data: dict[str, Any] | list[Any],
    format_type: Literal["yaml", "json", "table"],
) -> str:
    """Format data for CLI output.

    Parameters
    ----------
    data : dict[str, Any] | list[Any]
        Data to format.
    format_type : {"yaml", "json", "table"}
        Output format type.

    Returns
    -------
    str
        Formatted output string.

    Raises
    ------
    ValueError
        If format_type is invalid or data cannot be formatted.
    """
    if format_type == "yaml":
        return yaml.dump(data, default_flow_style=False, sort_keys=False)
    elif format_type == "json":
        # Convert Path objects to strings for JSON serialization
        def convert_paths(obj: Any) -> Any:
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, dict):
                return {str(k): convert_paths(v) for k, v in obj.items()}  # type: ignore[misc]
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]  # type: ignore[misc]
            return obj

        converted_data: Any = convert_paths(data)
        return json.dumps(converted_data, indent=2)
    elif format_type == "table":
        if not isinstance(data, dict):
            raise ValueError("Table format requires dict data")
        return _dict_to_table(data)
    else:
        raise ValueError(f"Invalid format type: {format_type}")


def _dict_to_table(data: dict[str, Any], title: str | None = None) -> str:
    """Convert dictionary to rich table string.

    Parameters
    ----------
    data : dict[str, Any]
        Dictionary to convert.
    title : str | None
        Optional table title.

    Returns
    -------
    str
        Rendered table as string.
    """
    table = Table(title=title, show_header=True, header_style="bold cyan")
    table.add_column("Key", style="yellow", no_wrap=True)
    table.add_column("Value", style="white")

    for key, value in data.items():
        # Handle nested dicts
        if isinstance(value, dict):
            value_str = _format_nested_dict(value)  # type: ignore[arg-type]
        elif isinstance(value, list):
            value_str = "\n".join(str(item) for item in value)  # type: ignore[var-annotated]
        else:
            value_str = str(value)

        table.add_row(key, value_str)

    # Capture table output
    string_io = StringIO()
    temp_console = Console(file=string_io, force_terminal=True, width=120)
    temp_console.print(table)
    return string_io.getvalue()


def _format_nested_dict(data: dict[str, Any], indent: int = 0) -> str:
    """Format nested dictionary for display.

    Parameters
    ----------
    data : dict[str, Any]
        Dictionary to format.
    indent : int
        Indentation level.

    Returns
    -------
    str
        Formatted string.
    """
    lines: list[str] = []
    for key, value in data.items():
        prefix = "  " * indent
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(_format_nested_dict(value, indent + 1))  # type: ignore[arg-type]
        else:
            lines.append(f"{prefix}{key}: {value}")
    return "\n".join(lines)


def print_error(message: str, exit_code: int = 1) -> None:
    """Print error message and exit.

    Parameters
    ----------
    message : str
        Error message to display.
    exit_code : int
        Exit code (default: 1). Pass 0 to not exit.
    """
    console.print(f"[red]✗ Error:[/red] {message}")
    if exit_code != 0:
        sys.exit(exit_code)


def print_success(message: str) -> None:
    """Print success message.

    Parameters
    ----------
    message : str
        Success message to display.
    """
    console.print(f"[green]✓ {message}[/green]")


def print_warning(message: str) -> None:
    """Print warning message.

    Parameters
    ----------
    message : str
        Warning message to display.
    """
    console.print(f"[yellow]⚠ Warning:[/yellow] {message}")


def print_info(message: str) -> None:
    """Print info message.

    Parameters
    ----------
    message : str
        Info message to display.
    """
    console.print(f"[blue]ℹ Info:[/blue] {message}")


def confirm(prompt: str, default: bool = False) -> bool:
    """Prompt user for confirmation.

    Parameters
    ----------
    prompt : str
        Confirmation prompt.
    default : bool
        Default value if user just presses Enter.

    Returns
    -------
    bool
        True if user confirmed, False otherwise.
    """
    return click.confirm(prompt, default=default)


def get_nested_value(data: dict[str, Any], key_path: str) -> Any:
    """Get nested dictionary value using dot notation.

    Parameters
    ----------
    data : dict[str, Any]
        Dictionary to search.
    key_path : str
        Dot-separated key path (e.g., "paths.data_dir").

    Returns
    -------
    Any
        Value at key path.

    Raises
    ------
    KeyError
        If key path doesn't exist.

    Examples
    --------
    >>> data = {"a": {"b": {"c": 42}}}
    >>> get_nested_value(data, "a.b.c")
    42
    """
    keys = key_path.split(".")
    current = data
    for key in keys:
        if not isinstance(current, dict):
            raise KeyError(
                f"Cannot access key '{key}' in non-dict value at path '{key_path}'"
            )
        if key not in current:
            raise KeyError(f"Key '{key}' not found in path '{key_path}'")
        current = current[key]
    return current


def redact_sensitive_values(data: dict[str, Any]) -> dict[str, Any]:
    """Redact sensitive values in configuration.

    Parameters
    ----------
    data : dict[str, Any]
        Configuration data.

    Returns
    -------
    dict[str, Any]
        Data with sensitive values redacted.
    """
    sensitive_keys = {
        "api_key",
        "secret",
        "password",
        "token",
        "openai_api_key",
        "anthropic_api_key",
        "google_api_key",
    }

    result: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            result[key] = redact_sensitive_values(value)  # type: ignore[arg-type]
        elif any(sensitive in key.lower() for sensitive in sensitive_keys):
            result[key] = "***REDACTED***" if value else None
        else:
            result[key] = value

    return result
