"""Default configurations for the bead package.

This module provides default configuration instances and helper functions
for retrieving default configurations.
"""

from __future__ import annotations

from pydantic import BaseModel

from bead.config.models import (
    ActiveLearningConfig,
    DeploymentConfig,
    ItemConfig,
    ListConfig,
    LoggingConfig,
    PathsConfig,
    ResourceConfig,
    BeadConfig,
    TemplateConfig,
)

DEFAULT_CONFIG = BeadConfig(
    profile="default",
    paths=PathsConfig(),
    resources=ResourceConfig(),
    templates=TemplateConfig(),
    items=ItemConfig(),
    lists=ListConfig(),
    deployment=DeploymentConfig(),
    active_learning=ActiveLearningConfig(),
    logging=LoggingConfig(),
)
"""Default configuration instance.

This configuration uses all default values from each config model.
It's the base configuration used when no config file is provided.

Examples
--------
>>> from bead.config.defaults import DEFAULT_CONFIG
>>> DEFAULT_CONFIG.profile
'default'
>>> DEFAULT_CONFIG.paths.data_dir
PosixPath('data')
"""


def get_default_config() -> BeadConfig:
    """Get a copy of the default configuration.

    Returns
    -------
    BeadConfig
        A deep copy of the default configuration.

    Examples
    --------
    >>> from bead.config.defaults import get_default_config
    >>> config = get_default_config()
    >>> config.profile
    'default'
    >>> config.templates.batch_size
    1000

    Notes
    -----
    Returns a deep copy to ensure modifications don't affect the original
    DEFAULT_CONFIG instance.
    """
    return DEFAULT_CONFIG.model_copy(deep=True)


def get_default_for_model[T: BaseModel](model_type: type[T]) -> T:
    """Get default instance of any config model.

    Parameters
    ----------
    model_type : type[BaseModel]
        The configuration model type to instantiate.

    Returns
    -------
    BaseModel
        Default instance of the specified model type.

    Examples
    --------
    >>> from bead.config.defaults import get_default_for_model
    >>> from bead.config.models import PathsConfig
    >>> paths = get_default_for_model(PathsConfig)
    >>> paths.data_dir
    PosixPath('data')

    Raises
    ------
    TypeError
        If model_type is not a valid Pydantic model class.

    Notes
    -----
    This function provides runtime validation to ensure the input is a valid
    Pydantic model class, even though the type system constrains it.
    """
    # Runtime validation for cases where type checking is bypassed
    try:
        if not isinstance(model_type, type):  # type: ignore[reportUnnecessaryIsInstance]
            msg = f"model_type must be a Pydantic BaseModel class, got {model_type}"
            raise TypeError(msg)
        if not issubclass(model_type, BaseModel):  # type: ignore[reportUnnecessaryIsInstance]
            msg = f"model_type must be a Pydantic BaseModel class, got {model_type}"
            raise TypeError(msg)
    except TypeError as e:
        # Re-raise TypeError with our custom message
        if "must be a Pydantic BaseModel class" in str(e):
            raise
        msg = f"model_type must be a Pydantic BaseModel class, got {model_type}"
        raise TypeError(msg) from e

    return model_type()
