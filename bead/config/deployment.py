"""Deployment configuration models for the bead package."""

from __future__ import annotations

from pydantic import BaseModel, Field

from bead.deployment.distribution import (
    DistributionStrategyType,
    ListDistributionStrategy,
)


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
