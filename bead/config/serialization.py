"""Configuration serialization to YAML format.

This module provides functionality for serializing BeadConfig objects to YAML format,
including conversion to dictionaries and saving to files.
"""

from pathlib import Path
from typing import Any

import yaml

from bead.config.defaults import get_default_config
from bead.config.models import BeadConfig


def config_to_dict(
    config: BeadConfig, include_defaults: bool = False
) -> dict[str, Any]:
    """Convert BeadConfig to dictionary for YAML serialization.

    Parameters
    ----------
    config : BeadConfig
        Configuration to convert.
    include_defaults : bool
        Whether to include default values.

    Returns
    -------
    dict[str, Any]
        Dictionary representation suitable for YAML.

    Examples
    --------
    >>> from bead.config import get_default_config
    >>> config = get_default_config()
    >>> config_dict = config_to_dict(config)
    >>> 'profile' in config_dict
    True
    """
    # Get dictionary with all values, excluding unset fields
    config_dict: dict[str, Any] = config.model_dump(
        mode="json", exclude_unset=not include_defaults
    )

    if not include_defaults:
        # Get default config to compare against
        default_config = get_default_config()
        default_dict: dict[str, Any] = default_config.model_dump(mode="json")
        # Remove values that match defaults
        config_dict = _remove_defaults(config_dict, default_dict)  # type: ignore[arg-type]

    # Convert Path objects to strings
    config_dict = _convert_paths_to_strings(config_dict)  # type: ignore[arg-type]

    return config_dict


def _remove_defaults(
    config_dict: dict[str, Any], default_dict: dict[str, Any]
) -> dict[str, Any]:
    """Remove values that match defaults from config dictionary.

    Parameters
    ----------
    config_dict : dict[str, Any]
        Configuration dictionary.
    default_dict : dict[str, Any]
        Default configuration dictionary.

    Returns
    -------
    dict[str, Any]
        Configuration dictionary with defaults removed.
    """
    result: dict[str, Any] = {}
    for key, value in config_dict.items():
        if key not in default_dict:
            # Keep values not in defaults
            result[key] = value
        elif isinstance(value, dict) and isinstance(default_dict[key], dict):
            # Recursively remove defaults from nested dicts
            nested_result = _remove_defaults(value, default_dict[key])  # type: ignore[arg-type]
            if nested_result:  # Only include if not empty after removing defaults
                result[key] = nested_result
        elif value != default_dict[key]:
            # Keep values that differ from defaults
            result[key] = value
    return result


def _convert_paths_to_strings(data: dict[str, Any]) -> dict[str, Any]:
    """Convert Path objects to strings in dictionary.

    Parameters
    ----------
    data : dict[str, Any]
        Dictionary potentially containing Path objects.

    Returns
    -------
    dict[str, Any]
        Dictionary with Path objects converted to strings.
    """
    result: dict[str, Any] = {}
    for key, value in data.items():
        if isinstance(value, dict):
            result[key] = _convert_paths_to_strings(value)  # type: ignore[arg-type]
        elif isinstance(value, Path):
            result[key] = str(value)
        elif isinstance(value, list):
            converted_list: list[Any] = [
                str(item) if isinstance(item, Path) else item
                for item in value  # type: ignore[misc]
            ]
            result[key] = converted_list
        else:
            result[key] = value
    return result


def to_yaml(config: BeadConfig, include_defaults: bool = False) -> str:
    """Serialize configuration to YAML string.

    Parameters
    ----------
    config : BeadConfig
        Configuration to serialize.
    include_defaults : bool
        If True, include all fields even if they have default values.
        If False, only include non-default values.

    Returns
    -------
    str
        YAML representation of configuration.

    Examples
    --------
    >>> from bead.config import get_default_config
    >>> config = get_default_config()
    >>> yaml_str = to_yaml(config)
    >>> 'profile: default' in yaml_str
    True
    """
    config_dict = config_to_dict(config, include_defaults=include_defaults)

    # Configure YAML dumper for clean output
    return yaml.dump(
        config_dict,
        default_flow_style=False,
        sort_keys=True,
        allow_unicode=True,
        indent=2,
    )


def save_yaml(
    config: BeadConfig,
    path: Path | str,
    include_defaults: bool = False,
    create_dirs: bool = True,
) -> None:
    """Save configuration to YAML file.

    Parameters
    ----------
    config : BeadConfig
        Configuration to save.
    path : Path | str
        Path where YAML file should be saved.
    include_defaults : bool
        If True, include all fields even if they have default values.
    create_dirs : bool
        If True, create parent directories if they don't exist.

    Raises
    ------
    IOError
        If file cannot be written.
    FileNotFoundError
        If create_dirs is False and parent directory doesn't exist.

    Examples
    --------
    >>> from pathlib import Path
    >>> from bead.config import get_default_config
    >>> config = get_default_config()
    >>> save_yaml(config, Path("config.yaml"))
    """
    path = Path(path) if isinstance(path, str) else path

    # Ensure parent directory exists
    if create_dirs:
        path.parent.mkdir(parents=True, exist_ok=True)
    elif not path.parent.exists():
        raise FileNotFoundError(
            f"Parent directory does not exist: {path.parent}. "
            f"Set create_dirs=True to create it automatically."
        )

    # Get YAML string
    yaml_str = to_yaml(config, include_defaults=include_defaults)

    # Write to file
    try:
        with open(path, "w") as f:
            f.write(yaml_str)
    except OSError as e:
        raise OSError(f"Failed to write YAML file {path}: {e}") from e
