"""Environment variable support for configuration.

This module provides functionality for loading configuration values from
environment variables, with support for nested configuration paths and
automatic type parsing.
"""

import os
from pathlib import Path
from typing import Any


def parse_env_value(value: str) -> Any:
    """Parse environment variable value to appropriate Python type.

    Handles: bool, int, float, Path, list (comma-separated), string

    Parameters
    ----------
    value : str
        Raw environment variable value.

    Returns
    -------
    Any
        Parsed value with appropriate type.

    Examples
    --------
    >>> parse_env_value("true")
    True
    >>> parse_env_value("42")
    42
    >>> parse_env_value("/path/to/file")
    PosixPath('/path/to/file')
    >>> parse_env_value("a,b,c")
    ['a', 'b', 'c']
    """
    # Handle boolean values
    if value.lower() in ("true", "1", "yes", "on"):
        return True
    if value.lower() in ("false", "0", "no", "off"):
        return False

    # Handle numeric values
    # Try int first
    try:
        return int(value)
    except ValueError:
        pass

    # Try float
    try:
        return float(value)
    except ValueError:
        pass

    # Handle path-like strings
    if value.startswith(("/", "./", "~/", "../")):
        return Path(value).expanduser()

    # Handle comma-separated lists
    if "," in value:
        return [item.strip() for item in value.split(",")]

    # Default to string
    return value


def env_to_nested_dict(env_vars: dict[str, str], prefix: str) -> dict[str, Any]:
    """Convert flat environment variables to nested dictionary.

    Parameters
    ----------
    env_vars : dict[str, str]
        Environment variables to convert.
    prefix : str
        Prefix to strip from variable names.

    Returns
    -------
    dict[str, Any]
        Nested configuration dictionary.

    Examples
    --------
    >>> env_vars = {"BEAD_LOGGING__LEVEL": "DEBUG"}
    >>> env_to_nested_dict(env_vars, "BEAD_")
    {'logging': {'level': 'DEBUG'}}
    """
    result: dict[str, Any] = {}

    for key, value in env_vars.items():
        if not key.startswith(prefix):
            continue

        # Remove prefix
        key_without_prefix = key[len(prefix) :]

        # Split on double underscore for nesting
        parts = key_without_prefix.split("__")

        # Convert to lowercase for config keys
        parts = [part.lower() for part in parts]

        # Parse the value
        parsed_value = parse_env_value(value)

        # Navigate/create nested structure
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]

        # Set the final value
        current[parts[-1]] = parsed_value

    return result


def load_from_env(prefix: str = "BEAD_") -> dict[str, Any]:
    """Load configuration values from environment variables.

    Converts environment variables with the given prefix to a nested
    configuration dictionary.

    Parameters
    ----------
    prefix : str
        Environment variable prefix to filter on.

    Returns
    -------
    dict[str, Any]
        Nested configuration dictionary from environment.

    Examples
    --------
    >>> # With env var: BEAD_LOGGING__LEVEL=DEBUG
    >>> load_from_env()
    {'logging': {'level': 'DEBUG'}}
    """
    # Get all environment variables with the prefix
    env_vars = {k: v for k, v in os.environ.items() if k.startswith(prefix)}

    # Convert to nested dict
    return env_to_nested_dict(env_vars, prefix)
