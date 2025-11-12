"""Template configuration models for the bead package."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


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
    slot_strategies : dict[str, dict[str, Any]] | None
        Per-slot strategy configuration for mixed filling.
        Maps slot names to strategy configs with format:
        {'slot_name': {'strategy': 'exhaustive|random|stratified|mlm',
         ...strategy_config...}}

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
    >>> config_mixed.slot_strategies  # doctest: +SKIP
    {'noun': {'strategy': 'exhaustive'}, 'verb': {'strategy': 'exhaustive'},
     'adjective': {'strategy': 'mlm'}}
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
        "Format: {'slot_name': {'strategy': "
        "'exhaustive|random|stratified|mlm', ...config...}}",
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
                    msg = (
                        f"'strategy' key required for slot '{slot_name}' "
                        f"in slot_strategies"
                    )
                    raise ValueError(msg)

                strategy_name = config["strategy"]
                if strategy_name not in ["exhaustive", "random", "stratified", "mlm"]:
                    msg = f"Invalid strategy '{strategy_name}' for slot '{slot_name}'"
                    raise ValueError(msg)

                # If MLM, check model config is available
                if strategy_name == "mlm" and self.mlm_model_name is None:
                    msg = (
                        f"mlm_model_name must be specified when slot "
                        f"'{slot_name}' uses MLM"
                    )
                    raise ValueError(msg)

        return self
