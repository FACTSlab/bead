"""CLI utility functions for bead package.

This module provides utility functions for the CLI including configuration loading,
output formatting, error handling, and user prompts.
"""

from __future__ import annotations

import json
import sys
from datetime import date, datetime
from io import StringIO
from pathlib import Path
from typing import TYPE_CHECKING, Literal, cast
from uuid import UUID

import click
import yaml
from rich.console import Console
from rich.table import Table

if TYPE_CHECKING:
    from bead.config import BeadConfig

from bead.config import load_config
from bead.data.base import JsonValue

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
    data: dict[str, JsonValue] | list[JsonValue],
    format_type: Literal["yaml", "json", "table"],
) -> str:
    """Format data for CLI output.

    Parameters
    ----------
    data : dict[str, JsonValue] | list[JsonValue]
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
        # Convert Path, UUID, and datetime objects to strings for JSON serialization
        def convert_non_serializable(
            obj: JsonValue | Path | UUID | datetime | date,
        ) -> JsonValue:
            if isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, UUID):
                return str(obj)
            elif isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, date):
                return obj.isoformat()
            elif isinstance(obj, dict):
                result: dict[str, JsonValue] = {}
                k: str
                v: JsonValue
                for k, v in obj.items():
                    result[str(k)] = convert_non_serializable(v)
                return result
            elif isinstance(obj, list):
                converted_list: list[JsonValue] = []
                item: JsonValue
                for item in obj:
                    converted_list.append(convert_non_serializable(item))
                return converted_list
            return obj

        converted_data: JsonValue = convert_non_serializable(data)
        return json.dumps(converted_data, indent=2)
    elif format_type == "table":
        if not isinstance(data, dict):
            raise ValueError("Table format requires dict data")
        return _dict_to_table(data)
    else:
        raise ValueError(f"Invalid format type: {format_type}")


def _dict_to_table(data: dict[str, JsonValue], title: str | None = None) -> str:
    """Convert dictionary to rich table string.

    Parameters
    ----------
    data : dict[str, JsonValue]
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


def _format_nested_dict(data: dict[str, JsonValue], indent: int = 0) -> str:
    """Format nested dictionary for display.

    Parameters
    ----------
    data : dict[str, JsonValue]
        Dictionary to format.
    indent : int
        Indentation level.

    Returns
    -------
    str
        Formatted string.
    """
    lines: list[str] = []
    key: str
    value: JsonValue
    for key, value in data.items():
        prefix: str = "  " * indent
        if isinstance(value, dict):
            lines.append(f"{prefix}{key}:")
            lines.append(_format_nested_dict(value, indent + 1))
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


def get_nested_value(data: dict[str, JsonValue], key_path: str) -> JsonValue:
    """Get nested dictionary value using dot notation.

    Parameters
    ----------
    data : dict[str, JsonValue]
        Dictionary to search.
    key_path : str
        Dot-separated key path (e.g., "paths.data_dir").

    Returns
    -------
    JsonValue
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


def redact_sensitive_values(data: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Redact sensitive values in configuration.

    Parameters
    ----------
    data : dict[str, JsonValue]
        Configuration data.

    Returns
    -------
    dict[str, JsonValue]
        Data with sensitive values redacted.
    """
    sensitive_keys: set[str] = {
        "api_key",
        "secret",
        "password",
        "token",
        "openai_api_key",
        "anthropic_api_key",
        "google_api_key",
    }

    result: dict[str, JsonValue] = {}
    key: str
    value: JsonValue
    for key, value in data.items():
        if isinstance(value, dict):
            result[key] = redact_sensitive_values(value)
        elif any(sensitive in key.lower() for sensitive in sensitive_keys):
            result[key] = "***REDACTED***" if value else None
        else:
            result[key] = value

    return result


def parse_json_option(
    json_str: str,
    option_name: str,
) -> dict[str, JsonValue]:
    """Parse JSON string from CLI option with helpful error messages.

    Parameters
    ----------
    json_str : str
        JSON string to parse.
    option_name : str
        Name of the CLI option (for error messages).

    Returns
    -------
    dict[str, JsonValue]
        Parsed JSON dictionary.

    Raises
    ------
    ValueError
        If JSON is invalid, with helpful error message.

    Examples
    --------
    >>> parse_json_option('{"key": "value"}', "--config")
    {'key': 'value'}
    """
    try:
        result: JsonValue = json.loads(json_str)
        if not isinstance(result, dict):
            raise ValueError(
                f"{option_name} must be a JSON object (dictionary), "
                f"not {type(result).__name__}. "
                f'Wrap your JSON in curly braces: \'{{"key": "value"}}\''
            )
        # At this point, we've validated result is a dict
        # Cast to the proper return type
        return cast(dict[str, JsonValue], result)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in {option_name}: {e}\n"
            f"Provided: {json_str}\n"
            f'Example: \'{{"key": "value"}}\''
        ) from e


def parse_key_value_pairs(
    pairs_str: str,
    separator: str = ",",
    kv_separator: str = "=",
) -> dict[str, str]:
    """Parse key=value pairs from string.

    Parameters
    ----------
    pairs_str : str
        String containing key=value pairs.
    separator : str, optional
        Separator between pairs (default: ",").
    kv_separator : str, optional
        Separator between key and value (default: "=").

    Returns
    -------
    dict[str, str]
        Dictionary of parsed key-value pairs.

    Raises
    ------
    ValueError
        If format is invalid.

    Examples
    --------
    >>> parse_key_value_pairs("key1=val1,key2=val2")
    {'key1': 'val1', 'key2': 'val2'}
    """
    result: dict[str, str] = {}
    if not pairs_str or not pairs_str.strip():
        return result

    for pair in pairs_str.split(separator):
        pair = pair.strip()
        if not pair:
            continue

        if kv_separator not in pair:
            raise ValueError(
                f"Invalid key=value pair: '{pair}'. "
                f"Expected format: key{kv_separator}value"
            )

        key, value = pair.split(kv_separator, 1)
        key = key.strip()
        value = value.strip()

        if not key:
            raise ValueError(f"Empty key in pair: '{pair}'")

        result[key] = value

    return result


def parse_list_option(
    list_str: str,
    separator: str = ",",
    allow_empty: bool = False,
) -> list[str]:
    """Parse comma-separated list from string.

    Parameters
    ----------
    list_str : str
        String containing comma-separated values.
    separator : str, optional
        Separator between values (default: ",").
    allow_empty : bool, optional
        Whether to allow empty lists (default: False).

    Returns
    -------
    list[str]
        List of parsed values.

    Raises
    ------
    ValueError
        If list is empty and allow_empty is False.

    Examples
    --------
    >>> parse_list_option("a,b,c")
    ['a', 'b', 'c']
    """
    if not list_str or not list_str.strip():
        if allow_empty:
            return []
        raise ValueError("List cannot be empty")

    values = [v.strip() for v in list_str.split(separator) if v.strip()]

    if not values and not allow_empty:
        raise ValueError("List cannot be empty after parsing")

    return values


def validate_file_exists(
    file_path: Path,
    file_description: str = "File",
) -> None:
    """Validate that a file exists.

    Parameters
    ----------
    file_path : Path
        Path to file.
    file_description : str, optional
        Description of file for error message (default: "File").

    Raises
    ------
    FileNotFoundError
        If file doesn't exist.

    Examples
    --------
    >>> validate_file_exists(Path("config.yaml"), "Config file")
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"{file_description} not found: {file_path}\n"
            f"Please check the path and try again."
        )

    if not file_path.is_file():
        raise ValueError(
            f"{file_description} is not a file: {file_path}\n"
            f"Expected a file, got a directory."
        )


def validate_directory_exists(
    dir_path: Path,
    dir_description: str = "Directory",
    create_if_missing: bool = False,
) -> None:
    """Validate that a directory exists.

    Parameters
    ----------
    dir_path : Path
        Path to directory.
    dir_description : str, optional
        Description of directory for error message (default: "Directory").
    create_if_missing : bool, optional
        Whether to create directory if it doesn't exist (default: False).

    Raises
    ------
    FileNotFoundError
        If directory doesn't exist and create_if_missing is False.
    ValueError
        If path exists but is not a directory.

    Examples
    --------
    >>> validate_directory_exists(Path("data/"), "Data directory")
    """
    if not dir_path.exists():
        if create_if_missing:
            dir_path.mkdir(parents=True, exist_ok=True)
            print_info(f"Created {dir_description}: {dir_path}")
        else:
            raise FileNotFoundError(
                f"{dir_description} not found: {dir_path}\n"
                f"Please create the directory or use --create flag."
            )
    elif not dir_path.is_dir():
        raise ValueError(
            f"{dir_description} is not a directory: {dir_path}\n"
            f"Expected a directory, got a file."
        )


def merge_config_dicts(
    base: dict[str, JsonValue],
    override: dict[str, JsonValue],
) -> dict[str, JsonValue]:
    """Merge two configuration dictionaries recursively.

    Parameters
    ----------
    base : dict[str, JsonValue]
        Base configuration dictionary.
    override : dict[str, JsonValue]
        Override configuration dictionary.

    Returns
    -------
    dict[str, JsonValue]
        Merged configuration dictionary.

    Examples
    --------
    >>> base = {"a": 1, "b": {"c": 2}}
    >>> override = {"b": {"c": 3, "d": 4}}
    >>> merge_config_dicts(base, override)
    {'a': 1, 'b': {'c': 3, 'd': 4}}
    """
    result: dict[str, JsonValue] = base.copy()

    key: str
    value: JsonValue
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            base_val: dict[str, JsonValue] = cast(dict[str, JsonValue], result[key])
            override_val: dict[str, JsonValue] = cast(dict[str, JsonValue], value)
            result[key] = merge_config_dicts(base_val, override_val)
        else:
            result[key] = value

    return result
