"""Template configuration models for the bead package."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class SlotStrategyConfig(BaseModel):
    """Configuration for a single slot's filling strategy.

    Parameters
    ----------
    strategy
        Filling strategy for this slot. Must be one of "exhaustive",
        "random", "stratified", or "mlm".
    sample_size
        Sample size for random or stratified strategies. Only used when
        strategy is "random" or "stratified".
    stratify_by
        Feature name to stratify by. Only used when strategy is "stratified".
    beam_size
        Beam size for MLM strategy. Only used when strategy is "mlm".

    Examples
    --------
    >>> config = SlotStrategyConfig(strategy="exhaustive")
    >>> config.strategy
    'exhaustive'
    >>> config_random = SlotStrategyConfig(strategy="random", sample_size=100)
    >>> config_random.sample_size
    100
    >>> config_stratified = SlotStrategyConfig(
    ...     strategy="stratified", sample_size=50, stratify_by="pos"
    ... )
    >>> config_stratified.stratify_by
    'pos'
    >>> config_mlm = SlotStrategyConfig(strategy="mlm", beam_size=10)
    >>> config_mlm.beam_size
    10
    """

    strategy: Literal["exhaustive", "random", "stratified", "mlm"] = Field(
        ..., description="Filling strategy for this slot"
    )
    sample_size: int | None = Field(
        default=None, description="Sample size for random/stratified"
    )
    stratify_by: str | None = Field(default=None, description="Feature to stratify by")
    beam_size: int | None = Field(default=None, description="Beam size for MLM")


class TemplateConfig(BaseModel):
    """Configuration for template filling.

    Parameters
    ----------
    filling_strategy : str
        Strategy name for filling templates
        ("exhaustive", "random", "stratified", "mlm", "mixed").
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
    slot_strategies : dict[str, SlotStrategyConfig] | None
        Per-slot strategy configuration for mixed filling.
        Maps slot names to SlotStrategyConfig instances.

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
    ...         "noun": SlotStrategyConfig(strategy="exhaustive"),
    ...         "verb": SlotStrategyConfig(strategy="exhaustive"),
    ...         "adjective": SlotStrategyConfig(strategy="mlm", beam_size=10)
    ...     }
    ... )
    >>> config_mixed.slot_strategies["noun"].strategy
    'exhaustive'
    >>> config_mixed.slot_strategies["adjective"].beam_size
    10
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
    slot_strategies: dict[str, SlotStrategyConfig] | None = Field(
        default=None,
        description="Per-slot strategy configuration for mixed filling. "
        "Maps slot names to SlotStrategyConfig instances.",
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
            for slot_name, slot_config in self.slot_strategies.items():
                # If MLM strategy is used for a slot, check model config is available
                if slot_config.strategy == "mlm" and self.mlm_model_name is None:
                    msg = (
                        f"mlm_model_name must be specified when slot "
                        f"'{slot_name}' uses MLM"
                    )
                    raise ValueError(msg)

        return self
