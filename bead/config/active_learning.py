"""Active learning configuration models for the bead package."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from bead.active_learning.config import MixedEffectsConfig
from bead.data.range import Range


class BaseEncoderModelConfig(BaseModel):
    """Base configuration for encoder-based active learning models.

    Provides shared configuration fields for models that use transformer
    encoders with optional dual-encoder architecture and mixed effects.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    max_length : int
        Maximum sequence length for tokenization.
    encoder_mode : Literal["single_encoder", "dual_encoder"]
        Encoding strategy for input processing.
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
    mixed_effects : MixedEffectsConfig
        Mixed effects configuration for participant-level modeling.

    Examples
    --------
    >>> config = BaseEncoderModelConfig()
    >>> config.model_name
    'bert-base-uncased'
    >>> config.batch_size
    16
    >>> config.mixed_effects.mode
    'fixed'
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
        description="Encoding strategy for input processing",
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
    mixed_effects: MixedEffectsConfig = Field(
        default_factory=MixedEffectsConfig,
        description="Mixed effects configuration for participant-level modeling",
    )


class ForcedChoiceModelConfig(BaseEncoderModelConfig):
    """Configuration for forced choice active learning models.

    Inherits all fields from BaseEncoderModelConfig. Used for tasks where
    participants select one option from a set of alternatives.

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
    mixed_effects : MixedEffectsConfig
        Mixed effects configuration for participant-level modeling.

    Examples
    --------
    >>> config = ForcedChoiceModelConfig()
    >>> config.model_name
    'bert-base-uncased'
    >>> config.batch_size
    16
    >>> config.mixed_effects.mode
    'fixed'
    """


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


class JatosDataCollectionConfig(BaseModel):
    """Configuration for JATOS data collection.

    Parameters
    ----------
    base_url : str
        JATOS base URL (e.g., "https://jatos.example.com").
    api_token : str
        JATOS API token for authentication.
    study_id : int
        JATOS study ID to collect data from.

    Examples
    --------
    >>> config = JatosDataCollectionConfig(
    ...     base_url="https://jatos.example.com",
    ...     api_token="secret-token",
    ...     study_id=123,
    ... )
    >>> config.base_url
    'https://jatos.example.com'
    """

    base_url: str = Field(..., description="JATOS base URL")
    api_token: str = Field(..., description="JATOS API token")
    study_id: int = Field(..., description="JATOS study ID")


class ProlificDataCollectionConfig(BaseModel):
    """Configuration for Prolific data collection.

    Parameters
    ----------
    api_key : str
        Prolific API key for authentication.
    study_id : str
        Prolific study ID to collect data from.

    Examples
    --------
    >>> config = ProlificDataCollectionConfig(
    ...     api_key="secret-key",
    ...     study_id="abc123",
    ... )
    >>> config.study_id
    'abc123'
    """

    api_key: str = Field(..., description="Prolific API key")
    study_id: str = Field(..., description="Prolific study ID")


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
    jatos : JatosDataCollectionConfig | None
        Configuration for JATOS data collection. If None, JATOS integration
        is disabled.
    prolific : ProlificDataCollectionConfig | None
        Configuration for Prolific data collection. If None, Prolific
        integration is disabled.
    data_collection_timeout : int
        Timeout in seconds for data collection.

    Examples
    --------
    >>> config = ActiveLearningLoopConfig()
    >>> config.max_iterations
    10
    >>> config.budget_per_iteration
    100

    >>> # With JATOS integration
    >>> jatos_config = JatosDataCollectionConfig(
    ...     base_url="https://jatos.example.com",
    ...     api_token="secret-token",
    ...     study_id=123,
    ... )
    >>> config = ActiveLearningLoopConfig(jatos=jatos_config)
    >>> config.jatos.study_id
    123
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
    # Data collection configuration (optional)
    jatos: JatosDataCollectionConfig | None = Field(
        default=None,
        description="Configuration for JATOS data collection",
    )
    prolific: ProlificDataCollectionConfig | None = Field(
        default=None,
        description="Configuration for Prolific data collection",
    )
    data_collection_timeout: int = Field(
        default=3600,
        description="Timeout in seconds for data collection",
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


class CategoricalModelConfig(BaseEncoderModelConfig):
    """Configuration for categorical active learning models.

    Inherits all fields from BaseEncoderModelConfig. Used for tasks where
    participants select one category from a predefined set.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    max_length : int
        Maximum sequence length for tokenization.
    encoder_mode : Literal["single_encoder", "dual_encoder"]
        Encoding strategy for categories.
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
    mixed_effects : MixedEffectsConfig
        Mixed effects configuration for participant-level modeling.

    Examples
    --------
    >>> config = CategoricalModelConfig()
    >>> config.model_name
    'bert-base-uncased'
    >>> config.mixed_effects.mode
    'fixed'
    """


class BinaryModelConfig(BaseEncoderModelConfig):
    """Configuration for binary active learning models.

    Inherits all fields from BaseEncoderModelConfig. Used for binary
    classification tasks (yes/no, true/false, acceptable/unacceptable).

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    max_length : int
        Maximum sequence length for tokenization.
    encoder_mode : Literal["single_encoder", "dual_encoder"]
        Encoding strategy for binary classification.
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
    mixed_effects : MixedEffectsConfig
        Mixed effects configuration for participant-level modeling.

    Examples
    --------
    >>> config = BinaryModelConfig()
    >>> config.model_name
    'bert-base-uncased'
    >>> config.mixed_effects.mode
    'fixed'
    """


class MultiSelectModelConfig(BaseEncoderModelConfig):
    """Configuration for multi-select active learning models.

    Inherits all fields from BaseEncoderModelConfig. Used for tasks where
    participants can select multiple options from a set of alternatives.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    max_length : int
        Maximum sequence length for tokenization.
    encoder_mode : Literal["single_encoder", "dual_encoder"]
        Encoding strategy for multi-select options.
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
    mixed_effects : MixedEffectsConfig
        Mixed effects configuration for participant-level modeling.

    Examples
    --------
    >>> config = MultiSelectModelConfig()
    >>> config.model_name
    'bert-base-uncased'
    >>> config.mixed_effects.mode
    'fixed'
    """


class OrdinalScaleModelConfig(BaseModel):
    """Configuration for ordinal scale active learning models.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    max_length : int
        Maximum sequence length for tokenization.
    encoder_mode : Literal["single_encoder"]
        Encoding strategy for ordinal scale tasks.
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
    scale : Range[float]
        Numeric range for the ordinal scale (default: 0.0 to 1.0).
    distribution : Literal["truncated_normal"]
        Distribution for modeling bounded continuous responses.
    sigma : float
        Standard deviation for truncated normal distribution.
    mixed_effects : MixedEffectsConfig
        Mixed effects configuration for participant-level modeling.

    Examples
    --------
    >>> config = OrdinalScaleModelConfig()
    >>> config.model_name
    'bert-base-uncased'
    >>> config.scale.min
    0.0
    >>> config.scale.max
    1.0
    >>> config.mixed_effects.mode
    'fixed'

    >>> # Custom scale from 1.0 to 5.0
    >>> config = OrdinalScaleModelConfig(
    ...     scale=Range[float](min=1.0, max=5.0)
    ... )
    >>> config.scale.contains(3.5)
    True
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
    encoder_mode: Literal["single_encoder"] = Field(
        default="single_encoder",
        description="Encoding strategy for ordinal scale tasks",
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
    scale: Range[float] = Field(
        default_factory=lambda: Range[float](min=0.0, max=1.0),
        description="Numeric range for the ordinal scale",
    )
    distribution: Literal["truncated_normal"] = Field(
        default="truncated_normal",
        description="Distribution for modeling bounded continuous responses",
    )
    sigma: float = Field(
        default=0.1,
        description="Standard deviation for truncated normal distribution",
        gt=0,
    )
    mixed_effects: MixedEffectsConfig = Field(
        default_factory=MixedEffectsConfig,
        description="Mixed effects configuration for participant-level modeling",
    )


class MagnitudeModelConfig(BaseModel):
    """Configuration for magnitude active learning models.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier.
    max_length : int
        Maximum sequence length for tokenization.
    encoder_mode : Literal["single_encoder"]
        Encoding strategy for magnitude tasks.
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
    bounded : bool
        Whether magnitude values are bounded to a range.
    min_value : float | None
        Minimum value (for bounded case). Required if bounded=True.
    max_value : float | None
        Maximum value (for bounded case). Required if bounded=True.
    distribution : Literal["normal", "truncated_normal"]
        Distribution for modeling responses.
        "normal" for unbounded, "truncated_normal" for bounded.
    sigma : float
        Standard deviation for the distribution.
    mixed_effects : MixedEffectsConfig
        Mixed effects configuration for participant-level modeling.

    Examples
    --------
    >>> # Unbounded magnitude (e.g., reading time)
    >>> config = MagnitudeModelConfig(bounded=False, distribution="normal")
    >>> config.bounded
    False
    >>> config.distribution
    'normal'

    >>> # Bounded magnitude (e.g., confidence on 0-100 scale)
    >>> config = MagnitudeModelConfig(
    ...     bounded=True,
    ...     min_value=0.0,
    ...     max_value=100.0,
    ...     distribution="truncated_normal"
    ... )
    >>> config.min_value
    0.0
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
    encoder_mode: Literal["single_encoder"] = Field(
        default="single_encoder",
        description="Encoding strategy for magnitude tasks",
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
    bounded: bool = Field(
        default=False,
        description="Whether magnitude values are bounded to a range",
    )
    min_value: float | None = Field(
        default=None,
        description="Minimum value (required if bounded=True)",
    )
    max_value: float | None = Field(
        default=None,
        description="Maximum value (required if bounded=True)",
    )
    distribution: Literal["normal", "truncated_normal"] = Field(
        default="normal",
        description="Distribution for modeling responses",
    )
    sigma: float = Field(
        default=0.1,
        description="Standard deviation for the distribution",
        gt=0,
    )
    mixed_effects: MixedEffectsConfig = Field(
        default_factory=MixedEffectsConfig,
        description="Mixed effects configuration for participant-level modeling",
    )

    @model_validator(mode="after")
    def validate_bounded_configuration(self) -> MagnitudeModelConfig:
        """Validate bounded configuration consistency.

        Raises
        ------
        ValueError
            If bounded=True but min_value or max_value not set.
        ValueError
            If bounded=False but min_value or max_value is set.
        ValueError
            If min_value >= max_value.
        ValueError
            If distribution inconsistent with bounded setting.
        """
        if self.bounded:
            if self.min_value is None or self.max_value is None:
                raise ValueError(
                    "bounded=True requires both min_value and max_value to be set. "
                    f"Got min_value={self.min_value}, max_value={self.max_value}."
                )
            if self.min_value >= self.max_value:
                raise ValueError(
                    f"min_value ({self.min_value}) must be less than "
                    f"max_value ({self.max_value})."
                )
            if self.distribution != "truncated_normal":
                raise ValueError(
                    "bounded=True requires distribution='truncated_normal'. "
                    f"Got distribution='{self.distribution}'."
                )
        else:
            if self.min_value is not None or self.max_value is not None:
                raise ValueError(
                    "bounded=False but min_value or max_value is set. "
                    f"Got min_value={self.min_value}, max_value={self.max_value}. "
                    "Either set bounded=True or remove min_value/max_value."
                )
            if self.distribution != "normal":
                raise ValueError(
                    "bounded=False requires distribution='normal'. "
                    f"Got distribution='{self.distribution}'."
                )
        return self


class FreeTextModelConfig(BaseModel):
    """Configuration for free text generation with GLMM support.

    Implements seq2seq generation with participant-level random effects using
    LoRA (Low-Rank Adaptation) for random slopes mode.

    Parameters
    ----------
    model_name : str
        HuggingFace seq2seq model identifier (e.g., "t5-base", "facebook/bart-base").
    max_input_length : int
        Maximum input sequence length for tokenization.
    max_output_length : int
        Maximum output sequence length for generation.
    num_beams : int
        Beam search width (1 = greedy decoding).
    temperature : float
        Sampling temperature for generation.
    top_p : float
        Nucleus sampling probability cutoff.
    learning_rate : float
        Learning rate for AdamW optimizer.
    batch_size : int
        Batch size for training (typically smaller for seq2seq due to memory).
    num_epochs : int
        Number of training epochs.
    device : Literal["cpu", "cuda", "mps"]
        Device to train on.
    lora_rank : int
        LoRA rank r for low-rank decomposition (typical: 4-16).
    lora_alpha : float
        LoRA scaling factor α (typically 2*rank).
    lora_dropout : float
        Dropout probability for LoRA layers.
    lora_target_modules : list[str]
        Attention modules to apply LoRA (e.g., ["q_proj", "v_proj"]).
    eval_metric : Literal["exact_match", "token_accuracy", "bleu"]
        Evaluation metric for generation quality.
    mixed_effects : MixedEffectsConfig
        Mixed effects configuration for participant-level modeling.

    Examples
    --------
    >>> config = FreeTextModelConfig()
    >>> config.model_name
    't5-base'
    >>> config.lora_rank
    8
    >>> config.mixed_effects.mode
    'fixed'

    >>> # With random slopes (LoRA)
    >>> config = FreeTextModelConfig(
    ...     mixed_effects=MixedEffectsConfig(mode="random_slopes"),
    ...     lora_rank=8,
    ...     lora_alpha=16.0
    ... )
    """

    model_name: str = Field(
        default="t5-base",
        description="HuggingFace seq2seq model identifier",
    )
    max_input_length: int = Field(
        default=128,
        description="Maximum input sequence length",
        gt=0,
    )
    max_output_length: int = Field(
        default=64,
        description="Maximum output sequence length",
        gt=0,
    )
    num_beams: int = Field(
        default=4,
        description="Beam search width (1 = greedy)",
        gt=0,
    )
    temperature: float = Field(
        default=1.0,
        description="Sampling temperature",
        gt=0.0,
    )
    top_p: float = Field(
        default=0.9,
        description="Nucleus sampling probability cutoff",
        ge=0.0,
        le=1.0,
    )
    learning_rate: float = Field(
        default=2e-5,
        description="Learning rate for AdamW optimizer",
        gt=0,
    )
    batch_size: int = Field(
        default=8,
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
    lora_rank: int = Field(
        default=8,
        description="LoRA rank r for low-rank decomposition",
        gt=0,
    )
    lora_alpha: float = Field(
        default=16.0,
        description="LoRA scaling factor α",
        gt=0,
    )
    lora_dropout: float = Field(
        default=0.1,
        description="Dropout probability for LoRA layers",
        ge=0.0,
        lt=1.0,
    )
    lora_target_modules: list[str] = Field(
        default=["q", "v"],
        description="Attention modules to apply LoRA to",
    )
    eval_metric: Literal["exact_match", "token_accuracy", "bleu"] = Field(
        default="exact_match",
        description="Evaluation metric for generation quality",
    )
    mixed_effects: MixedEffectsConfig = Field(
        default_factory=MixedEffectsConfig,
        description="Mixed effects configuration for participant-level modeling",
    )


class ClozeModelConfig(BaseModel):
    """Configuration for cloze (fill-in-the-blank) models with GLMM support.

    Implements masked language modeling with participant-level random effects for
    predicting tokens at unfilled slots in partially-filled templates.

    Parameters
    ----------
    model_name : str
        HuggingFace masked LM model identifier.
        Examples: "bert-base-uncased", "roberta-base".
    max_length : int
        Maximum sequence length for tokenization.
    learning_rate : float
        Learning rate for AdamW optimizer.
    batch_size : int
        Batch size for training.
    num_epochs : int
        Number of training epochs.
    device : Literal["cpu", "cuda", "mps"]
        Device to train on.
    mask_token : str
        Token used for masking (model-specific, e.g., "[MASK]" for BERT).
    eval_metric : Literal["exact_match", "token_accuracy"]
        Evaluation metric for masked token prediction.
    mixed_effects : MixedEffectsConfig
        Mixed effects configuration for participant-level modeling.

    Examples
    --------
    >>> config = ClozeModelConfig()
    >>> config.model_name
    'bert-base-uncased'
    >>> config.mask_token
    '[MASK]'
    >>> config.mixed_effects.mode
    'fixed'

    >>> # With random intercepts
    >>> config = ClozeModelConfig(
    ...     mixed_effects=MixedEffectsConfig(mode="random_intercepts"),
    ...     num_epochs=5
    ... )
    """

    model_name: str = Field(
        default="bert-base-uncased",
        description="HuggingFace masked LM model identifier",
    )
    max_length: int = Field(
        default=128,
        description="Maximum sequence length for tokenization",
        gt=0,
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
    mask_token: str = Field(
        default="[MASK]",
        description="Token used for masking (model-specific)",
    )
    eval_metric: Literal["exact_match", "token_accuracy"] = Field(
        default="exact_match",
        description="Evaluation metric for masked token prediction",
    )
    mixed_effects: MixedEffectsConfig = Field(
        default_factory=MixedEffectsConfig,
        description="Mixed effects configuration for participant-level modeling",
    )


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
