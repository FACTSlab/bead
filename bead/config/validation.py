"""Configuration validation utilities.

This module provides pre-flight validation for configuration objects,
checking for common issues before the configuration is used.
"""

from __future__ import annotations

from bead.config.models import BeadConfig


def check_paths_exist(config: BeadConfig) -> list[str]:
    """Check that all configured paths exist or can be created.

    Parameters
    ----------
    config : BeadConfig
        Configuration to check.

    Returns
    -------
    list[str]
        List of path validation errors.

    Examples
    --------
    >>> from bead.config import get_default_config
    >>> config = get_default_config()
    >>> errors = check_paths_exist(config)
    >>> isinstance(errors, list)
    True
    """
    errors: list[str] = []

    # Check main paths if they should exist and are absolute
    if config.paths.data_dir.is_absolute() and not config.paths.data_dir.exists():
        errors.append(f"data_dir does not exist: {config.paths.data_dir}")

    if config.paths.output_dir.is_absolute() and not config.paths.output_dir.exists():
        errors.append(f"output_dir does not exist: {config.paths.output_dir}")

    if config.paths.cache_dir.is_absolute() and not config.paths.cache_dir.exists():
        errors.append(f"cache_dir does not exist: {config.paths.cache_dir}")

    # Check resource paths
    if (
        config.resources.lexicon_path is not None
        and config.resources.lexicon_path.is_absolute()
        and not config.resources.lexicon_path.exists()
    ):
        errors.append(f"lexicon_path does not exist: {config.resources.lexicon_path}")

    if (
        config.resources.templates_path is not None
        and config.resources.templates_path.is_absolute()
        and not config.resources.templates_path.exists()
    ):
        errors.append(
            f"templates_path does not exist: {config.resources.templates_path}"
        )

    if (
        config.resources.constraints_path is not None
        and config.resources.constraints_path.is_absolute()
        and not config.resources.constraints_path.exists()
    ):
        errors.append(
            f"constraints_path does not exist: {config.resources.constraints_path}"
        )

    # Check training logging dir
    if (
        config.active_learning.trainer.logging_dir.is_absolute()
        and not config.active_learning.trainer.logging_dir.exists()
    ):
        errors.append(
            f"logging_dir does not exist: {config.active_learning.trainer.logging_dir}"
        )

    # Check logging file parent directory
    if (
        config.logging.file is not None
        and config.logging.file.is_absolute()
        and not config.logging.file.parent.exists()
    ):
        parent_dir = config.logging.file.parent
        errors.append(f"logging file parent directory does not exist: {parent_dir}")

    return errors


def check_resource_compatibility(config: BeadConfig) -> list[str]:
    """Verify resources are compatible with templates.

    Parameters
    ----------
    config : BeadConfig
        Configuration to check.

    Returns
    -------
    list[str]
        List of resource compatibility errors.
    """
    errors: list[str] = []

    # Check that if templates_path is specified, lexicon_path should also be specified
    if (
        config.resources.templates_path is not None
        and config.resources.lexicon_path is None
    ):
        errors.append(
            "templates_path is specified but lexicon_path is not. "
            "Templates require a lexicon."
        )

    return errors


def check_model_configuration(config: BeadConfig) -> list[str]:
    """Verify model settings are valid.

    Parameters
    ----------
    config : BeadConfig
        Configuration to check.

    Returns
    -------
    list[str]
        List of model configuration errors.
    """
    try:
        import torch  # noqa: PLC0415
    except ImportError:
        torch = None  # type: ignore[assignment]

    errors: list[str] = []

    # Check CUDA availability if device is set to cuda
    if config.items.model.device == "cuda":
        if torch is None:
            errors.append(
                "Model device is set to 'cuda' but PyTorch is not installed. "
                "Install PyTorch or set device to 'cpu'."
            )
        elif not torch.cuda.is_available():  # type: ignore[no-untyped-call]
            errors.append(
                "Model device is set to 'cuda' but CUDA is not available. "
                "Set device to 'cpu' or install CUDA."
            )

    # Check MPS availability if device is set to mps
    if config.items.model.device == "mps":
        if torch is None:
            errors.append(
                "Model device is set to 'mps' but PyTorch is not installed. "
                "Install PyTorch or set device to 'cpu'."
            )
        elif not torch.backends.mps.is_available():  # type: ignore[no-untyped-call]
            errors.append(
                "Model device is set to 'mps' but MPS is not available. "
                "Set device to 'cpu' or use a macOS system with MPS support."
            )

    return errors


def check_training_configuration(config: BeadConfig) -> list[str]:
    """Verify training settings are compatible.

    Parameters
    ----------
    config : BeadConfig
        Configuration to check.

    Returns
    -------
    list[str]
        List of training configuration errors.
    """
    errors: list[str] = []

    # Check that batch size is positive
    if config.active_learning.forced_choice_model.batch_size <= 0:
        batch_size = config.active_learning.forced_choice_model.batch_size
        errors.append(f"Training batch size must be positive, got {batch_size}")

    # Check that epochs is positive
    if config.active_learning.trainer.epochs <= 0:
        epochs = config.active_learning.trainer.epochs
        errors.append(f"Training epochs must be positive, got {epochs}")

    # Check that learning rate is positive
    if config.active_learning.forced_choice_model.learning_rate <= 0:
        lr = config.active_learning.forced_choice_model.learning_rate
        errors.append(f"Training learning rate must be positive, got {lr}")

    return errors


def check_deployment_configuration(config: BeadConfig) -> list[str]:
    """Verify deployment settings are valid.

    Parameters
    ----------
    config : BeadConfig
        Configuration to check.

    Returns
    -------
    list[str]
        List of deployment configuration errors.
    """
    errors: list[str] = []

    # Check jsPsych version format if platform is jspsych
    if config.deployment.platform == "jspsych":
        version = config.deployment.jspsych_version
        if version is None:  # type: ignore[reportUnnecessaryComparison]
            errors.append("jsPsych platform requires jspsych_version to be specified")
        elif not isinstance(version, str):  # type: ignore[reportUnnecessaryIsInstance]
            errors.append(
                f"jspsych_version must be a string, got {type(version).__name__}"
            )

    return errors


def validate_config(config: BeadConfig) -> list[str]:
    """Perform pre-flight validation on configuration.

    Checks:
    - All paths exist (if absolute paths are specified)
    - Resource paths exist (if specified)
    - Model configurations are compatible
    - Training configurations are valid
    - No conflicting settings

    Parameters
    ----------
    config : BeadConfig
        Configuration to validate.

    Returns
    -------
    list[str]
        List of validation errors. Empty if valid.

    Examples
    --------
    >>> from bead.config import get_default_config
    >>> config = get_default_config()
    >>> errors = validate_config(config)
    >>> len(errors)
    0
    """
    errors: list[str] = []

    # Run all validation checks
    errors.extend(check_paths_exist(config))
    errors.extend(check_resource_compatibility(config))
    errors.extend(check_model_configuration(config))
    errors.extend(check_training_configuration(config))
    errors.extend(check_deployment_configuration(config))

    return errors
