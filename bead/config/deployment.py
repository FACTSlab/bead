"""Deployment configuration models for the bead package."""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator

from bead.deployment.distribution import (
    DistributionStrategyType,
    ListDistributionStrategy,
)


class SlopitKeystrokeConfig(BaseModel):
    """Configuration for slopit keystroke capture.

    Attributes
    ----------
    enabled
        Whether to capture keystroke events.
    capture_key_up
        Whether to capture keyup events in addition to keydown.
    include_modifiers
        Whether to record modifier key states.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=True, description="Capture keystroke events")
    capture_key_up: bool = Field(default=True, description="Capture keyup events")
    include_modifiers: bool = Field(default=True, description="Record modifier states")


class SlopitFocusConfig(BaseModel):
    """Configuration for slopit focus/blur capture.

    Attributes
    ----------
    enabled
        Whether to capture focus events.
    use_visibility_api
        Whether to use the Page Visibility API.
    use_blur_focus
        Whether to track blur and focus events.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=True, description="Capture focus events")
    use_visibility_api: bool = Field(default=True, description="Use Page Visibility API")
    use_blur_focus: bool = Field(default=True, description="Track blur/focus events")


class SlopitPasteConfig(BaseModel):
    """Configuration for slopit paste event capture.

    Attributes
    ----------
    enabled
        Whether to capture paste events.
    prevent
        Whether to block paste actions.
    capture_preview
        Whether to capture preview of pasted text.
    preview_length
        Number of characters to include in preview.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(default=True, description="Capture paste events")
    prevent: bool = Field(default=False, description="Block paste actions")
    capture_preview: bool = Field(default=True, description="Capture text preview")
    preview_length: int = Field(default=100, ge=0, description="Preview character limit")


class SlopitIntegrationConfig(BaseModel):
    """Configuration for slopit behavioral capture integration.

    Slopit captures behavioral signals during experiment trials,
    including keystroke dynamics, focus patterns, and paste events.
    These signals can be used to detect AI-assisted responses.

    Attributes
    ----------
    enabled
        Whether to enable slopit behavioral capture. Disabled by default.
    keystroke
        Keystroke capture configuration.
    focus
        Focus/blur capture configuration.
    paste
        Paste event capture configuration.
    target_selectors
        CSS selectors for capture targets by task type.

    Examples
    --------
    >>> config = SlopitIntegrationConfig(enabled=True)
    >>> config.keystroke.enabled
    True

    >>> config = SlopitIntegrationConfig(
    ...     enabled=True,
    ...     paste=SlopitPasteConfig(prevent=True),
    ... )
    >>> config.paste.prevent
    True
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = Field(
        default=False,
        description="Enable slopit behavioral capture (opt-in)",
    )
    keystroke: SlopitKeystrokeConfig = Field(
        default_factory=SlopitKeystrokeConfig,
        description="Keystroke capture settings",
    )
    focus: SlopitFocusConfig = Field(
        default_factory=SlopitFocusConfig,
        description="Focus/blur capture settings",
    )
    paste: SlopitPasteConfig = Field(
        default_factory=SlopitPasteConfig,
        description="Paste event capture settings",
    )
    target_selectors: dict[str, str] = Field(
        default_factory=lambda: {
            "likert_rating": ".bead-rating-button",
            "slider_rating": ".bead-slider",
            "forced_choice": ".bead-choice-button",
            "cloze": ".bead-cloze-field",
        },
        description="CSS selectors for capture targets by task type",
    )

    @model_validator(mode="after")
    def validate_slopit_bundle_exists(self) -> SlopitIntegrationConfig:
        """Validate that slopit bundle exists when enabled.

        Raises
        ------
        ValueError
            If slopit is enabled but the compiled bundle is not found.
        """
        if self.enabled:
            bundle_path = (
                Path(__file__).parent.parent
                / "deployment"
                / "jspsych"
                / "dist"
                / "slopit-bundle.js"
            )
            if not bundle_path.exists():
                msg = (
                    f"Slopit bundle not found at {bundle_path}. "
                    "Run 'pnpm build' in bead/deployment/jspsych to compile TypeScript."
                )
                raise ValueError(msg)
        return self


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
    distribution_strategy : ListDistributionStrategy
        List distribution strategy for batch experiments.
        Defaults to balanced assignment.

    Examples
    --------
    >>> config = DeploymentConfig()
    >>> config.platform
    'jspsych'
    >>> config.jspsych_version
    '7.3.0'
    >>> config.distribution_strategy.strategy_type
    <DistributionStrategyType.BALANCED: 'balanced'>
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
    distribution_strategy: ListDistributionStrategy = Field(
        default_factory=lambda: ListDistributionStrategy(
            strategy_type=DistributionStrategyType.BALANCED
        ),
        description="List distribution strategy for batch experiments",
    )
