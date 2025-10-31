"""Configuration models for the bead package.

This module provides Pydantic models for all configuration sections
with comprehensive validation rules.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class PathsConfig(BaseModel):
    """Configuration for file system paths.

    Parameters
    ----------
    data_dir : Path
        Base directory for data files.
    output_dir : Path
        Base directory for outputs.
    cache_dir : Path
        Cache directory.
    temp_dir : Path | None
        Temporary directory. If None, uses system temp.
    create_dirs : bool
        Whether to create directories if they don't exist.

    Examples
    --------
    >>> config = PathsConfig()
    >>> config.data_dir
    PosixPath('data')
    >>> config = PathsConfig(data_dir=Path("/absolute/path"))
    >>> config.data_dir
    PosixPath('/absolute/path')
    """

    data_dir: Path = Field(
        default=Path("data"), description="Base directory for data files"
    )
    output_dir: Path = Field(
        default=Path("output"), description="Base directory for outputs"
    )
    cache_dir: Path = Field(default=Path(".cache"), description="Cache directory")
    temp_dir: Path | None = Field(default=None, description="Temporary directory")
    create_dirs: bool = Field(
        default=True, description="Create directories if they don't exist"
    )


class ResourceConfig(BaseModel):
    """Configuration for external resources.

    Parameters
    ----------
    lexicon_path : Path | None
        Path to lexicon file.
    templates_path : Path | None
        Path to templates file.
    constraints_path : Path | None
        Path to constraints file.
    external_adapters : list[str]
        List of external adapters to enable.
    cache_external : bool
        Whether to cache external resource lookups.

    Examples
    --------
    >>> config = ResourceConfig()
    >>> config.cache_external
    True
    >>> config.external_adapters
    []
    """

    lexicon_path: Path | None = Field(default=None, description="Path to lexicon file")
    templates_path: Path | None = Field(
        default=None, description="Path to templates file"
    )
    constraints_path: Path | None = Field(
        default=None, description="Path to constraints file"
    )
    external_adapters: list[str] = Field(
        default_factory=list, description="External adapters to enable"
    )
    cache_external: bool = Field(
        default=True, description="Cache external resource lookups"
    )


class TemplateConfig(BaseModel):
    """Configuration for template filling.

    Parameters
    ----------
    filling_strategy : str
        Strategy name for filling templates ("exhaustive", "random", "stratified", "mlm", "mixed").
    batch_size : int
        Batch size for filling operations.
    max_combinations : int | None
        Maximum combinations to generate.
    random_seed : int | None
        Random seed for reproducibility.
    stream_mode : bool
        Use streaming for large templates.
    use_csp_solver : bool
        Use CSP solver for templates with multi-slot constraints.
    mlm_model_name : str | None
        HuggingFace model name for MLM filling.
    mlm_beam_size : int
        Beam search width for MLM strategy.
    mlm_fill_direction : str
        Direction for filling slots in MLM strategy.
    mlm_custom_order : list[int] | None
        Custom slot fill order for MLM strategy.
    mlm_top_k : int
        Number of top candidates per slot in MLM.
    mlm_device : str
        Device for MLM inference.
    mlm_cache_enabled : bool
        Enable content-addressable caching for MLM predictions.
    mlm_cache_dir : Path | None
        Directory for MLM prediction cache.
    slot_strategies : dict[str, dict[str, Any]] | None
        Per-slot strategy configuration for mixed filling.
        Maps slot names to strategy configs with format:
        {'slot_name': {'strategy': 'exhaustive|random|stratified|mlm', ...strategy_config...}}

    Examples
    --------
    >>> config = TemplateConfig()
    >>> config.filling_strategy
    'exhaustive'
    >>> config.batch_size
    1000
    >>> # MLM configuration
    >>> config_mlm = TemplateConfig(
    ...     filling_strategy="mlm", mlm_model_name="bert-base-uncased"
    ... )
    >>> config_mlm.mlm_beam_size
    5
    >>> # Mixed strategy configuration
    >>> config_mixed = TemplateConfig(
    ...     filling_strategy="mixed",
    ...     mlm_model_name="bert-base-uncased",
    ...     slot_strategies={
    ...         "noun": {"strategy": "exhaustive"},
    ...         "verb": {"strategy": "exhaustive"},
    ...         "adjective": {"strategy": "mlm"}
    ...     }
    ... )
    >>> config_mixed.slot_strategies
    {'noun': {'strategy': 'exhaustive'}, 'verb': {'strategy': 'exhaustive'}, 'adjective': {'strategy': 'mlm'}}
    """

    filling_strategy: Literal["exhaustive", "random", "stratified", "mlm", "mixed"] = (
        Field(default="exhaustive", description="Strategy for filling templates")
    )
    batch_size: int = Field(default=1000, description="Batch size for filling", gt=0)
    max_combinations: int | None = Field(
        default=None, description="Max combinations to generate"
    )
    random_seed: int | None = Field(
        default=None, description="Random seed for reproducibility"
    )
    stream_mode: bool = Field(
        default=False, description="Use streaming for large templates"
    )
    use_csp_solver: bool = Field(
        default=False,
        description="Use CSP solver for templates with multi-slot constraints",
    )

    # MLM-specific settings
    mlm_model_name: str | None = Field(
        default=None, description="HuggingFace model name for MLM filling"
    )
    mlm_beam_size: int = Field(
        default=5, description="Beam search width for MLM strategy", gt=0
    )
    mlm_fill_direction: Literal[
        "left_to_right", "right_to_left", "inside_out", "outside_in", "custom"
    ] = Field(
        default="left_to_right",
        description="Direction for filling slots in MLM strategy",
    )
    mlm_custom_order: list[int] | None = Field(
        default=None, description="Custom slot fill order for MLM strategy"
    )
    mlm_top_k: int = Field(
        default=20, description="Number of top candidates per slot in MLM", gt=0
    )
    mlm_device: str = Field(default="cpu", description="Device for MLM inference")
    mlm_cache_enabled: bool = Field(
        default=True, description="Enable caching for MLM predictions"
    )
    mlm_cache_dir: Path | None = Field(
        default=None, description="Directory for MLM prediction cache"
    )

    # Mixed strategy settings
    slot_strategies: dict[str, dict[str, Any]] | None = Field(
        default=None,
        description="Per-slot strategy configuration for mixed filling. "
        "Format: {'slot_name': {'strategy': 'exhaustive|random|stratified|mlm', ...config...}}",
    )

    @field_validator("max_combinations")
    @classmethod
    def validate_max_combinations(cls, v: int | None) -> int | None:
        """Validate max_combinations is positive.

        Parameters
        ----------
        v : int | None
            Max combinations value.

        Returns
        -------
        int | None
            Validated value.

        Raises
        ------
        ValueError
            If value is not positive.
        """
        if v is not None and v <= 0:
            msg = f"max_combinations must be positive, got {v}"
            raise ValueError(msg)
        return v

    @model_validator(mode="after")
    def validate_mlm_config(self) -> TemplateConfig:
        """Validate MLM configuration is consistent.

        Returns
        -------
        TemplateConfig
            Validated config.

        Raises
        ------
        ValueError
            If MLM config is inconsistent.
        """
        if self.filling_strategy == "mlm" and self.mlm_model_name is None:
            msg = "mlm_model_name must be specified when filling_strategy is 'mlm'"
            raise ValueError(msg)

        if self.mlm_fill_direction == "custom" and self.mlm_custom_order is None:
            msg = (
                "mlm_custom_order must be specified when mlm_fill_direction is 'custom'"
            )
            raise ValueError(msg)

        # Validate mixed strategy configuration
        if self.filling_strategy == "mixed" and self.slot_strategies is None:
            msg = "slot_strategies must be specified when filling_strategy is 'mixed'"
            raise ValueError(msg)

        if self.slot_strategies is not None:
            for slot_name, config in self.slot_strategies.items():
                if "strategy" not in config:
                    msg = f"'strategy' key required for slot '{slot_name}' in slot_strategies"
                    raise ValueError(msg)

                strategy_name = config["strategy"]
                if strategy_name not in ["exhaustive", "random", "stratified", "mlm"]:
                    msg = f"Invalid strategy '{strategy_name}' for slot '{slot_name}'"
                    raise ValueError(msg)

                # If MLM, check model config is available
                if strategy_name == "mlm" and self.mlm_model_name is None:
                    msg = f"mlm_model_name must be specified when slot '{slot_name}' uses MLM"
                    raise ValueError(msg)

        return self


class ModelConfig(BaseModel):
    """Configuration for language models.

    Parameters
    ----------
    provider : str
        Model provider name.
    model_name : str
        Model identifier.
    batch_size : int
        Inference batch size.
    device : str
        Device to use for computation.
    max_length : int
        Maximum sequence length.
    temperature : float
        Sampling temperature.
    cache_outputs : bool
        Whether to cache model outputs.

    Examples
    --------
    >>> config = ModelConfig()
    >>> config.provider
    'huggingface'
    >>> config.device
    'cpu'
    """

    provider: Literal["huggingface", "openai", "anthropic"] = Field(
        default="huggingface", description="Model provider"
    )
    model_name: str = Field(default="gpt2", description="Model identifier")
    batch_size: int = Field(default=8, description="Inference batch size", gt=0)
    device: Literal["cpu", "cuda", "mps"] = Field(
        default="cpu", description="Device to use"
    )
    max_length: int = Field(default=512, description="Max sequence length", gt=0)
    temperature: float = Field(default=1.0, description="Sampling temperature", ge=0)
    cache_outputs: bool = Field(default=True, description="Cache model outputs")


class ItemConfig(BaseModel):
    """Configuration for item generation.

    Parameters
    ----------
    model : ModelConfig
        Model configuration.
    apply_constraints : bool
        Whether to apply model-based constraints.
    track_metadata : bool
        Whether to track item metadata.
    parallel_processing : bool
        Whether to use parallel processing.
    num_workers : int
        Number of workers for parallel processing.

    Examples
    --------
    >>> config = ItemConfig()
    >>> config.apply_constraints
    True
    >>> config.num_workers
    4
    """

    model: ModelConfig = Field(
        default_factory=ModelConfig, description="Model configuration"
    )
    apply_constraints: bool = Field(
        default=True, description="Apply model-based constraints"
    )
    track_metadata: bool = Field(default=True, description="Track item metadata")
    parallel_processing: bool = Field(
        default=False, description="Use parallel processing"
    )
    num_workers: int = Field(default=4, description="Number of workers", gt=0)


class ListConfig(BaseModel):
    """Configuration for list partitioning.

    Parameters
    ----------
    partitioning_strategy : str
        Strategy name for partitioning.
    num_lists : int
        Number of lists to create.
    items_per_list : int | None
        Items per list.
    balance_by : list[str]
        Fields to balance on.
    ensure_uniqueness : bool
        Whether to ensure items are unique across lists.
    random_seed : int | None
        Random seed for reproducibility.
    batch_constraints : list[BatchConstraintConfig] | None
        Batch-level constraints to apply across all lists.

    Examples
    --------
    >>> config = ListConfig()
    >>> config.partitioning_strategy
    'balanced'
    >>> config.num_lists
    1
    """

    partitioning_strategy: str = Field(
        default="balanced", description="Partitioning strategy"
    )
    num_lists: int = Field(default=1, description="Number of lists to create", gt=0)
    items_per_list: int | None = Field(default=None, description="Items per list")
    balance_by: list[str] = Field(
        default_factory=list, description="Fields to balance on"
    )
    ensure_uniqueness: bool = Field(
        default=True, description="Ensure items unique across lists"
    )
    random_seed: int | None = Field(default=None, description="Random seed")
    batch_constraints: list[BatchConstraintConfig] | None = Field(
        default=None, description="Batch-level constraints"
    )

    @field_validator("items_per_list")
    @classmethod
    def validate_items_per_list(cls, v: int | None) -> int | None:
        """Validate items_per_list is positive.

        Parameters
        ----------
        v : int | None
            Items per list value.

        Returns
        -------
        int | None
            Validated value.

        Raises
        ------
        ValueError
            If value is not positive.
        """
        if v is not None and v <= 0:
            msg = f"items_per_list must be positive, got {v}"
            raise ValueError(msg)
        return v


class DeploymentConfig(BaseModel):
    """Configuration for experiment deployment.

    Parameters
    ----------
    platform : str
        Deployment platform.
    jspsych_version : str
        jsPsych version to use.
    apply_material_design : bool
        Whether to use Material Design.
    include_demographics : bool
        Whether to include demographics survey.
    include_attention_checks : bool
        Whether to include attention checks.
    jatos_export : bool
        Whether to export to JATOS.

    Examples
    --------
    >>> config = DeploymentConfig()
    >>> config.platform
    'jspsych'
    >>> config.jspsych_version
    '7.3.0'
    """

    platform: str = Field(default="jspsych", description="Deployment platform")
    jspsych_version: str = Field(default="7.3.0", description="jsPsych version")
    apply_material_design: bool = Field(default=True, description="Use Material Design")
    include_demographics: bool = Field(
        default=True, description="Include demographics survey"
    )
    include_attention_checks: bool = Field(
        default=True, description="Include attention checks"
    )
    jatos_export: bool = Field(default=False, description="Export to JATOS")


class ForcedChoiceModelConfig(BaseModel):
    """Configuration for forced choice active learning models.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    max_length : int
        Maximum sequence length for tokenization.
    encoder_mode : Literal["single_encoder", "dual_encoder"]
        Encoding strategy for options.
    include_instructions : bool
        Whether to include task instructions.
    learning_rate : float
        Learning rate for AdamW optimizer.
    batch_size : int
        Batch size for training.
    num_epochs : int
        Number of training epochs.
    device : Literal["cpu", "cuda", "mps"]
        Device to train on.

    Examples
    --------
    >>> config = ForcedChoiceModelConfig()
    >>> config.model_name
    'bert-base-uncased'
    >>> config.batch_size
    16
    """

    model_name: str = Field(
        default="bert-base-uncased",
        description="HuggingFace model identifier",
    )
    max_length: int = Field(
        default=128,
        description="Maximum sequence length for tokenization",
        gt=0,
    )
    encoder_mode: Literal["single_encoder", "dual_encoder"] = Field(
        default="single_encoder",
        description="Encoding strategy for options",
    )
    include_instructions: bool = Field(
        default=False,
        description="Whether to include task instructions",
    )
    learning_rate: float = Field(
        default=2e-5,
        description="Learning rate for AdamW optimizer",
        gt=0,
    )
    batch_size: int = Field(
        default=16,
        description="Batch size for training",
        gt=0,
    )
    num_epochs: int = Field(
        default=3,
        description="Number of training epochs",
        gt=0,
    )
    device: Literal["cpu", "cuda", "mps"] = Field(
        default="cpu",
        description="Device to train on",
    )


class UncertaintySamplerConfig(BaseModel):
    """Configuration for uncertainty sampling strategies.

    Parameters
    ----------
    method : str
        Uncertainty method to use ("entropy", "margin", "least_confidence").
    batch_size : int | None
        Number of items to select per iteration. If None, uses the
        budget_per_iteration from ActiveLearningLoopConfig.

    Examples
    --------
    >>> config = UncertaintySamplerConfig()
    >>> config.method
    'entropy'
    >>> config = UncertaintySamplerConfig(method="margin", batch_size=50)
    >>> config.method
    'margin'
    """

    method: Literal["entropy", "margin", "least_confidence"] = Field(
        default="entropy",
        description="Uncertainty sampling method",
    )
    batch_size: int | None = Field(
        default=None,
        description="Number of items to select per iteration",
        gt=0,
    )


class ActiveLearningLoopConfig(BaseModel):
    """Configuration for active learning loop orchestration.

    Parameters
    ----------
    max_iterations : int
        Maximum number of AL iterations to run.
    budget_per_iteration : int
        Number of items to select per iteration.
    stopping_criterion : str
        Stopping criterion.
    performance_threshold : float | None
        Performance threshold for stopping.
    metric_name : str
        Metric name for convergence/threshold checks.
    convergence_patience : int
        Iterations to wait before declaring convergence.
    convergence_threshold : float
        Minimum improvement to avoid convergence.

    Examples
    --------
    >>> config = ActiveLearningLoopConfig()
    >>> config.max_iterations
    10
    >>> config.budget_per_iteration
    100
    """

    max_iterations: int = Field(
        default=10,
        description="Maximum number of iterations",
        gt=0,
    )
    budget_per_iteration: int = Field(
        default=100,
        description="Number of items to select per iteration",
        gt=0,
    )
    stopping_criterion: Literal[
        "max_iterations", "convergence", "performance_threshold"
    ] = Field(
        default="max_iterations",
        description="Stopping criterion for the loop",
    )
    performance_threshold: float | None = Field(
        default=None,
        description="Performance threshold for stopping",
        ge=0,
        le=1,
    )
    metric_name: str = Field(
        default="accuracy",
        description="Metric name for convergence/threshold checks",
    )
    convergence_patience: int = Field(
        default=3,
        description="Iterations to wait before declaring convergence",
        gt=0,
    )
    convergence_threshold: float = Field(
        default=0.01,
        description="Minimum improvement to avoid convergence",
        gt=0,
    )


class TrainerConfig(BaseModel):
    """Configuration for active learning trainers (HuggingFace, Lightning, etc.).

    Parameters
    ----------
    trainer_type : str
        Trainer type ("huggingface", "lightning").
    epochs : int
        Number of training epochs.
    eval_strategy : str
        Evaluation strategy.
    save_strategy : str
        Save strategy.
    logging_dir : Path
        Logging directory.
    use_wandb : bool
        Whether to use Weights & Biases.
    wandb_project : str | None
        W&B project name.

    Examples
    --------
    >>> config = TrainerConfig()
    >>> config.trainer_type
    'huggingface'
    >>> config.epochs
    3
    """

    trainer_type: Literal["huggingface", "lightning"] = Field(
        default="huggingface",
        description="Trainer type",
    )
    epochs: int = Field(default=3, description="Training epochs", gt=0)
    eval_strategy: str = Field(default="epoch", description="Evaluation strategy")
    save_strategy: str = Field(default="epoch", description="Save strategy")
    logging_dir: Path = Field(default=Path("logs"), description="Logging directory")
    use_wandb: bool = Field(default=False, description="Use Weights & Biases")
    wandb_project: str | None = Field(default=None, description="W&B project name")


class ActiveLearningConfig(BaseModel):
    """Configuration for active learning infrastructure.

    Reflects the bead/active_learning/ module structure:
    - models: Active learning models (ForcedChoiceModel, etc.)
    - trainers: Training infrastructure (HuggingFace, Lightning)
    - loop: Active learning loop orchestration
    - selection: Item selection strategies (uncertainty sampling, etc.)

    Parameters
    ----------
    forced_choice_model : ForcedChoiceModelConfig
        Configuration for forced choice models.
    trainer : TrainerConfig
        Configuration for trainers (HuggingFace, Lightning).
    loop : ActiveLearningLoopConfig
        Configuration for active learning loop.
    uncertainty_sampler : UncertaintySamplerConfig
        Configuration for uncertainty sampling strategies.

    Examples
    --------
    >>> config = ActiveLearningConfig()
    >>> config.forced_choice_model.model_name
    'bert-base-uncased'
    >>> config.trainer.trainer_type
    'huggingface'
    >>> config.loop.max_iterations
    10
    >>> config.uncertainty_sampler.method
    'entropy'
    """

    forced_choice_model: ForcedChoiceModelConfig = Field(
        default_factory=ForcedChoiceModelConfig,
        description="Forced choice model configuration",
    )
    trainer: TrainerConfig = Field(
        default_factory=TrainerConfig,
        description="Trainer configuration",
    )
    loop: ActiveLearningLoopConfig = Field(
        default_factory=ActiveLearningLoopConfig,
        description="Active learning loop configuration",
    )
    uncertainty_sampler: UncertaintySamplerConfig = Field(
        default_factory=UncertaintySamplerConfig,
        description="Uncertainty sampler configuration",
    )


class BatchConstraintConfig(BaseModel):
    """Configuration for batch-level constraints.

    Batch constraints operate across all lists in a batch to ensure global
    properties like coverage, balance, and diversity.

    Attributes
    ----------
    type : Literal["coverage", "balance", "diversity", "min_occurrence"]
        Type of batch constraint.
    property_expression : str
        Expression to extract property (e.g., "item['template_id']").
    target_values : list[str | int | float] | None
        Target values for coverage constraint. Default: None.
    min_coverage : float
        Minimum coverage fraction for coverage constraint (0.0-1.0). Default: 1.0.
    target_distribution : dict[str, float] | None
        Target distribution for balance constraint (values sum to 1.0). Default: None.
    tolerance : float
        Tolerance for balance constraint (0.0-1.0). Default: 0.1.
    max_lists_per_value : int | None
        Maximum lists per value for diversity constraint. Default: None.
    min_occurrences : int | None
        Minimum occurrences per value for min_occurrence constraint. Default: None.
    priority : int
        Constraint priority (higher = more important). Default: 1.

    Examples
    --------
    >>> # Coverage constraint
    >>> config = BatchConstraintConfig(
    ...     type="coverage",
    ...     property_expression="item['template_id']",
    ...     target_values=list(range(26)),
    ...     min_coverage=1.0
    ... )
    >>> # Balance constraint
    >>> config = BatchConstraintConfig(
    ...     type="balance",
    ...     property_expression="item['pair_type']",
    ...     target_distribution={"same_verb": 0.5, "different_verb": 0.5},
    ...     tolerance=0.05
    ... )
    >>> # Diversity constraint
    >>> config = BatchConstraintConfig(
    ...     type="diversity",
    ...     property_expression="item['verb_lemma']",
    ...     max_lists_per_value=3
    ... )
    >>> # Min occurrence constraint
    >>> config = BatchConstraintConfig(
    ...     type="min_occurrence",
    ...     property_expression="item['quantile']",
    ...     min_occurrences=50
    ... )
    """

    type: Literal["coverage", "balance", "diversity", "min_occurrence"] = Field(
        ..., description="Type of batch constraint"
    )
    property_expression: str = Field(
        ..., description="Expression to extract property"
    )
    target_values: list[str | int | float] | None = Field(
        default=None, description="Target values for coverage constraint"
    )
    min_coverage: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Minimum coverage fraction"
    )
    target_distribution: dict[str, float] | None = Field(
        default=None, description="Target distribution for balance constraint"
    )
    tolerance: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Tolerance for balance constraint"
    )
    max_lists_per_value: int | None = Field(
        default=None, ge=1, description="Maximum lists per value for diversity"
    )
    min_occurrences: int | None = Field(
        default=None, ge=1, description="Minimum occurrences for min_occurrence"
    )
    priority: int = Field(
        default=1, ge=1, description="Constraint priority"
    )

    @field_validator("property_expression")
    @classmethod
    def validate_property_expression(cls, v: str) -> str:
        """Validate property expression is non-empty."""
        if not v or not v.strip():
            raise ValueError("property_expression must be non-empty")
        return v.strip()

    @model_validator(mode="after")
    def validate_constraint_params(self) -> BatchConstraintConfig:
        """Validate constraint-specific parameters are provided."""
        if self.type == "coverage":
            # Coverage requires target_values (can be None for auto-detection)
            pass
        elif self.type == "balance":
            if self.target_distribution is None:
                raise ValueError("target_distribution required for balance constraint")
        elif self.type == "diversity":
            if self.max_lists_per_value is None:
                raise ValueError("max_lists_per_value required for diversity constraint")
        elif self.type == "min_occurrence":
            if self.min_occurrences is None:
                raise ValueError("min_occurrences required for min_occurrence constraint")

        return self


class LoggingConfig(BaseModel):
    """Configuration for logging.

    Parameters
    ----------
    level : str
        Log level.
    format : str
        Log format string.
    file : Path | None
        Log file path.
    console : bool
        Whether to log to console.

    Examples
    --------
    >>> config = LoggingConfig()
    >>> config.level
    'INFO'
    >>> config.console
    True
    """

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO", description="Log level"
    )
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format",
    )
    file: Path | None = Field(default=None, description="Log file path")
    console: bool = Field(default=True, description="Log to console")


class BeadConfig(BaseModel):
    """Main configuration for the bead package.

    Reflects the actual bead/ module structure:
    - active_learning: Active learning infrastructure (models, trainers, loop, selection)
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
            errors.append(
                f"logging_dir does not exist: {self.active_learning.trainer.logging_dir}"
            )

        # Check logging file
        if self.logging.file is not None and not self.logging.file.parent.exists():
            parent_dir = self.logging.file.parent
            errors.append(f"logging file parent directory does not exist: {parent_dir}")

        return errors


# ============================================================================
# Simulation Configuration
# ============================================================================


class NoiseModelConfig(BaseModel):
    """Configuration for noise model in simulated judgments.

    Attributes
    ----------
    noise_type : Literal["temperature", "systematic", "random", "none"]
        Type of noise to apply.
    temperature : float
        Temperature for scaling (higher = more random). Default: 1.0.
    bias_strength : float
        Strength of systematic biases (0.0-1.0). Default: 0.0.
    bias_type : str | None
        Type of bias ("length", "frequency", "position"). Default: None.
    random_noise_stddev : float
        Standard deviation for random noise. Default: 0.0.

    Examples
    --------
    >>> # Temperature-scaled decisions (more random)
    >>> config = NoiseModelConfig(noise_type="temperature", temperature=2.0)
    >>>
    >>> # Systematic length bias (prefer shorter)
    >>> config = NoiseModelConfig(
    ...     noise_type="systematic",
    ...     bias_strength=0.3,
    ...     bias_type="length"
    ... )
    >>>
    >>> # Random noise injection
    >>> config = NoiseModelConfig(
    ...     noise_type="random",
    ...     random_noise_stddev=0.1
    ... )
    """

    noise_type: Literal["temperature", "systematic", "random", "none"] = Field(
        default="temperature",
        description="Type of noise model",
    )
    temperature: float = Field(
        default=1.0,
        ge=0.01,
        le=10.0,
        description="Temperature for scaling decisions",
    )
    bias_strength: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Strength of systematic biases",
    )
    bias_type: str | None = Field(
        default=None,
        description="Type of systematic bias",
    )
    random_noise_stddev: float = Field(
        default=0.0,
        ge=0.0,
        description="Standard deviation for random noise",
    )


class SimulatedAnnotatorConfig(BaseModel):
    """Configuration for simulated annotator.

    Attributes
    ----------
    strategy : Literal["lm_score", "distance", "random", "oracle", "dsl"]
        Base strategy for generating judgments.
    noise_model : NoiseModelConfig
        Noise model configuration.
    dsl_expression : str | None
        Custom DSL expression for simulation logic.
    random_state : int | None
        Random seed for reproducibility.
    model_output_key : str
        Key to extract from Item.model_outputs. Default: "lm_score".
    fallback_to_random : bool
        Whether to fallback to random if model outputs missing. Default: True.

    Examples
    --------
    >>> # LM score-based with temperature
    >>> config = SimulatedAnnotatorConfig(
    ...     strategy="lm_score",
    ...     noise_model=NoiseModelConfig(noise_type="temperature", temperature=1.5),
    ...     random_state=42
    ... )
    >>>
    >>> # Distance-based with embeddings
    >>> config = SimulatedAnnotatorConfig(
    ...     strategy="distance",
    ...     model_output_key="embedding",
    ...     noise_model=NoiseModelConfig(noise_type="none")
    ... )
    >>>
    >>> # Custom DSL logic
    >>> config = SimulatedAnnotatorConfig(
    ...     strategy="dsl",
    ...     dsl_expression="sample_categorical(softmax(model_scores) / temperature)",
    ...     noise_model=NoiseModelConfig(noise_type="temperature", temperature=1.0)
    ... )
    """

    strategy: Literal["lm_score", "distance", "random", "oracle", "dsl"] = Field(
        default="lm_score",
        description="Base simulation strategy",
    )
    noise_model: NoiseModelConfig = Field(
        default_factory=NoiseModelConfig,
        description="Noise model configuration",
    )
    dsl_expression: str | None = Field(
        default=None,
        description="Custom DSL expression for simulation",
    )
    random_state: int | None = Field(
        default=None,
        description="Random seed for reproducibility",
    )
    model_output_key: str = Field(
        default="lm_score",
        description="Key to extract from model outputs",
    )
    fallback_to_random: bool = Field(
        default=True,
        description="Fallback to random if model outputs missing",
    )


class SimulationRunnerConfig(BaseModel):
    """Configuration for simulation runner.

    Attributes
    ----------
    annotator_configs : list[SimulatedAnnotatorConfig]
        List of annotator configurations (for multi-annotator simulation).
    n_annotators : int
        Number of simulated annotators. Default: 1.
    inter_annotator_correlation : float | None
        Desired correlation between annotators (0.0-1.0). Default: None (independent).
    output_format : Literal["dict", "dataframe", "jsonl"]
        Output format for simulation results. Default: "dict".
    save_path : Path | None
        Path to save simulation results. Default: None.

    Examples
    --------
    >>> # Single annotator
    >>> config = SimulationRunnerConfig(
    ...     annotator_configs=[SimulatedAnnotatorConfig(strategy="lm_score")],
    ...     n_annotators=1
    ... )
    >>>
    >>> # Multiple independent annotators
    >>> config = SimulationRunnerConfig(
    ...     annotator_configs=[
    ...         SimulatedAnnotatorConfig(strategy="lm_score", random_state=1),
    ...         SimulatedAnnotatorConfig(strategy="lm_score", random_state=2),
    ...         SimulatedAnnotatorConfig(strategy="lm_score", random_state=3)
    ...     ],
    ...     n_annotators=3
    ... )
    >>>
    >>> # Correlated annotators
    >>> config = SimulationRunnerConfig(
    ...     annotator_configs=[SimulatedAnnotatorConfig(strategy="lm_score")],
    ...     n_annotators=5,
    ...     inter_annotator_correlation=0.7  # 70% agreement
    ... )
    """

    annotator_configs: list[SimulatedAnnotatorConfig] = Field(
        default_factory=lambda: [SimulatedAnnotatorConfig()],
        description="Annotator configurations",
    )
    n_annotators: int = Field(
        default=1,
        ge=1,
        le=100,
        description="Number of simulated annotators",
    )
    inter_annotator_correlation: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Inter-annotator correlation",
    )
    output_format: Literal["dict", "dataframe", "jsonl"] = Field(
        default="dict",
        description="Output format",
    )
    save_path: Path | None = Field(
        default=None,
        description="Path to save results",
    )
