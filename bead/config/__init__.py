"""Configuration system for bead package.

This module provides configuration models, default settings, and profiles
for all aspects of the bead pipeline.

Examples
--------
>>> from bead.config import BeadConfig, get_default_config, get_profile
>>> # Use default configuration
>>> config = get_default_config()
>>> config.profile
'default'
>>> # Use a profile
>>> dev_config = get_profile("dev")
>>> dev_config.logging.level
'DEBUG'
>>> # Create custom configuration
>>> from bead.config import PathsConfig
>>> from pathlib import Path
>>> custom = BeadConfig(paths=PathsConfig(data_dir=Path("my_data")))
"""

from __future__ import annotations

from bead.config.active_learning import ActiveLearningConfig
from bead.config.config import BeadConfig
from bead.config.defaults import DEFAULT_CONFIG, get_default_config
from bead.config.deployment import DeploymentConfig
from bead.config.env import load_from_env
from bead.config.item import ItemConfig
from bead.config.list import ListConfig
from bead.config.loader import load_config, load_yaml_file, merge_configs
from bead.config.logging import LoggingConfig
from bead.config.model import ModelConfig
from bead.config.paths import PathsConfig
from bead.config.profiles import (
    DEV_CONFIG,
    PROD_CONFIG,
    PROFILES,
    TEST_CONFIG,
    get_profile,
    list_profiles,
)
from bead.config.resources import ResourceConfig
from bead.config.serialization import save_yaml, to_yaml
from bead.config.template import TemplateConfig
from bead.config.validation import validate_config

__all__ = [
    # Main config
    "BeadConfig",
    # Config sections
    "PathsConfig",
    "ResourceConfig",
    "TemplateConfig",
    "ModelConfig",
    "ItemConfig",
    "ListConfig",
    "DeploymentConfig",
    "ActiveLearningConfig",
    "LoggingConfig",
    # Defaults
    "DEFAULT_CONFIG",
    "get_default_config",
    # Profiles
    "DEV_CONFIG",
    "PROD_CONFIG",
    "TEST_CONFIG",
    "PROFILES",
    "get_profile",
    "list_profiles",
    # Loading
    "load_config",
    "load_yaml_file",
    "merge_configs",
    # Environment
    "load_from_env",
    # Validation
    "validate_config",
    # Serialization
    "to_yaml",
    "save_yaml",
]
