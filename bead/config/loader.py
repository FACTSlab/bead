"""Configuration loading from YAML files.

This module provides functionality for loading configurations from YAML files,
merging configurations from multiple sources, and applying configuration overrides.
"""

from pathlib import Path
from typing import Any

import yaml

from bead.config.models import BeadConfig
from bead.config.profiles import get_profile


def merge_configs(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep merge two configuration dictionaries.

    Recursively merges override into base, with override values taking precedence.

    Parameters
    ----------
    base : dict[str, Any]
        Base configuration dictionary.
    override : dict[str, Any]
        Override configuration dictionary.

    Returns
    -------
    dict[str, Any]
        Merged configuration dictionary.

    Examples
    --------
    >>> base = {"a": 1, "b": {"c": 2}}
    >>> override = {"b": {"d": 3}}
    >>> merge_configs(base, override)
    {'a': 1, 'b': {'c': 2, 'd': 3}}
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)  # type: ignore[arg-type]
        else:
            result[key] = value
    return result


def load_yaml_file(path: Path | str) -> dict[str, Any]:
    """Load YAML file and return as dictionary.

    Parameters
    ----------
    path : Path | str
        Path to YAML file.

    Returns
    -------
    dict[str, Any]
        Parsed YAML content.

    Raises
    ------
    FileNotFoundError
        If file doesn't exist.
    yaml.YAMLError
        If YAML is malformed.
    """
    path = Path(path) if isinstance(path, str) else path

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    try:
        with open(path) as f:
            content = yaml.safe_load(f)
            # Handle empty files
            return content if content is not None else {}
    except yaml.YAMLError as e:
        raise yaml.YAMLError(f"Failed to parse YAML file {path}: {e}") from e


def load_config(
    config_path: Path | str | None = None,
    profile: str = "default",
    **overrides: Any,
) -> BeadConfig:
    """Load configuration from YAML file with optional overrides.

    Precedence (lowest to highest):
    1. Profile defaults
    2. YAML file values
    3. Keyword overrides

    Parameters
    ----------
    config_path : Path | str | None
        Path to YAML config file. If None, uses profile defaults.
    profile : str
        Profile to use as base (default, dev, prod, test).
    **overrides : Any
        Direct overrides for config values.

    Returns
    -------
    BeadConfig
        Loaded and merged configuration.

    Raises
    ------
    FileNotFoundError
        If config_path is specified but doesn't exist.
    yaml.YAMLError
        If YAML file is malformed.
    ValidationError
        If configuration is invalid.

    Examples
    --------
    >>> config = load_config(profile="dev")
    >>> config.profile
    'dev'
    >>> config = load_config(config_path="config.yaml", logging__level="DEBUG")
    >>> config.logging.level
    'DEBUG'
    """
    # Start with profile defaults
    base_config: dict[str, Any] = get_profile(profile).model_dump()

    # Merge with YAML file if provided
    if config_path is not None:
        yaml_config = load_yaml_file(config_path)
        base_config = merge_configs(base_config, yaml_config)

    # Convert overrides with __ syntax to nested dicts
    if overrides:
        override_dict: dict[str, Any] = {}
        for key, value in overrides.items():
            parts = key.split("__")
            current = override_dict
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
        base_config = merge_configs(base_config, override_dict)

    # Construct and validate BeadConfig
    return BeadConfig(**base_config)
