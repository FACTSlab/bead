"""Main configuration model for the bead package."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from bead.config.active_learning import ActiveLearningConfig
from bead.config.deployment import DeploymentConfig
from bead.config.item import ItemConfig
from bead.config.list import ListConfig
from bead.config.logging import LoggingConfig
from bead.config.paths import PathsConfig
from bead.config.resources import ResourceConfig
from bead.config.template import TemplateConfig


class BeadConfig(BaseModel):
    """Main configuration for the bead package.

    Reflects the actual bead/ module structure:
    - active_learning: Active learning (models, trainers, loop, selection)
    - data_collection: Human data collection (JATOS, Prolific)
    - deployment: Experiment deployment (jsPsych, JATOS)
    - evaluation: Model evaluation and metrics
    - items: Item generation and management
    - lists: List construction and balancing
    - resources: Linguistic resources (VerbNet, PropBank, UniMorph)
    - templates: Template management

    Parameters
    ----------
    profile : str
        Configuration profile name.
    paths : PathsConfig
        Paths configuration.
    resources : ResourceConfig
        Resources configuration.
    templates : TemplateConfig
        Templates configuration.
    items : ItemConfig
        Items configuration.
    lists : ListConfig
        Lists configuration.
    deployment : DeploymentConfig
        Deployment configuration.
    active_learning : ActiveLearningConfig
        Active learning configuration (models, trainers, loop, selection).
    logging : LoggingConfig
        Logging configuration.

    Examples
    --------
    >>> config = BeadConfig()
    >>> config.profile
    'default'
    >>> config.paths.data_dir
    PosixPath('data')
    >>> config.active_learning.forced_choice_model.model_name
    'bert-base-uncased'
    >>> config.active_learning.trainer.trainer_type
    'huggingface'
    >>> config.active_learning.loop.max_iterations
    10
    """

    profile: str = Field(default="default", description="Configuration profile name")
    paths: PathsConfig = Field(
        default_factory=PathsConfig, description="Paths configuration"
    )
    resources: ResourceConfig = Field(
        default_factory=ResourceConfig, description="Resources configuration"
    )
    templates: TemplateConfig = Field(
        default_factory=TemplateConfig, description="Templates configuration"
    )
    items: ItemConfig = Field(
        default_factory=ItemConfig, description="Items configuration"
    )
    lists: ListConfig = Field(
        default_factory=ListConfig, description="Lists configuration"
    )
    deployment: DeploymentConfig = Field(
        default_factory=DeploymentConfig, description="Deployment configuration"
    )
    active_learning: ActiveLearningConfig = Field(
        default_factory=ActiveLearningConfig,
        description="Active learning configuration",
    )
    logging: LoggingConfig = Field(
        default_factory=LoggingConfig, description="Logging configuration"
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary.

        Returns
        -------
        dict[str, Any]
            Configuration as a dictionary.

        Examples
        --------
        >>> config = BeadConfig()
        >>> d = config.to_dict()
        >>> d["profile"]
        'default'
        """
        return self.model_dump()

    def to_yaml(self) -> str:
        """Convert configuration to YAML string.

        Returns
        -------
        str
            Configuration as YAML string.

        Examples
        --------
        >>> config = BeadConfig()
        >>> yaml_str = config.to_yaml()
        >>> 'profile: default' in yaml_str
        True
        """
        from bead.config.serialization import to_yaml  # noqa: PLC0415

        return to_yaml(self, include_defaults=False)

    def validate_paths(self) -> list[str]:
        """Validate all path fields exist.

        Returns
        -------
        list[str]
            List of validation errors. Empty if all paths are valid.

        Examples
        --------
        >>> config = BeadConfig()
        >>> errors = config.validate_paths()
        >>> len(errors)
        0
        """
        errors: list[str] = []

        # Check paths config
        if not self.paths.data_dir.exists() and self.paths.data_dir.is_absolute():
            errors.append(f"data_dir does not exist: {self.paths.data_dir}")
        if not self.paths.output_dir.exists() and self.paths.output_dir.is_absolute():
            errors.append(f"output_dir does not exist: {self.paths.output_dir}")
        if not self.paths.cache_dir.exists() and self.paths.cache_dir.is_absolute():
            errors.append(f"cache_dir does not exist: {self.paths.cache_dir}")
        if self.paths.temp_dir is not None and not self.paths.temp_dir.exists():
            errors.append(f"temp_dir does not exist: {self.paths.temp_dir}")

        # Check resource paths
        if (
            self.resources.lexicon_path is not None
            and not self.resources.lexicon_path.exists()
        ):
            errors.append(f"lexicon_path does not exist: {self.resources.lexicon_path}")
        if (
            self.resources.templates_path is not None
            and not self.resources.templates_path.exists()
        ):
            errors.append(
                f"templates_path does not exist: {self.resources.templates_path}"
            )
        if (
            self.resources.constraints_path is not None
            and not self.resources.constraints_path.exists()
        ):
            errors.append(
                f"constraints_path does not exist: {self.resources.constraints_path}"
            )

        # Check training logging dir
        if (
            not self.active_learning.trainer.logging_dir.exists()
            and self.active_learning.trainer.logging_dir.is_absolute()
        ):
            log_dir = self.active_learning.trainer.logging_dir
            errors.append(f"logging_dir does not exist: {log_dir}")

        # Check logging file
        if self.logging.file is not None and not self.logging.file.parent.exists():
            parent_dir = self.logging.file.parent
            errors.append(f"logging file parent directory does not exist: {parent_dir}")

        return errors
