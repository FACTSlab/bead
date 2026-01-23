"""List distribution configuration and strategies for batch experiments.

This module provides Pydantic models for configuring list distribution strategies
in JATOS batch experiments. It supports 8 different distribution strategies for
assigning participants to experiment lists.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import Field, field_validator, model_validator

from bead.data.base import BeadBaseModel


class DistributionStrategyType(str, Enum):
    """Available distribution strategies for list assignment.

    Attributes
    ----------
    RANDOM : str
        Random selection from available lists.
    SEQUENTIAL : str
        Round-robin assignment (list 0, 1, 2, ..., N, 0, 1, ...).
    BALANCED : str
        Assign to least-used list (minimizes imbalance).
    LATIN_SQUARE : str
        Latin square counterbalancing for order effects.
    STRATIFIED : str
        Balance across multiple factors (e.g., condition × list).
    WEIGHTED_RANDOM : str
        Random assignment with non-uniform probabilities.
    QUOTA_BASED : str
        Fixed quota per list, stop when reached.
    METADATA_BASED : str
        Intelligent assignment based on list metadata properties.
    """

    RANDOM = "random"
    SEQUENTIAL = "sequential"
    BALANCED = "balanced"
    LATIN_SQUARE = "latin_square"
    STRATIFIED = "stratified"
    WEIGHTED_RANDOM = "weighted_random"
    QUOTA_BASED = "quota_based"
    METADATA_BASED = "metadata_based"


class QuotaConfig(BeadBaseModel):
    """Configuration for quota-based assignment.

    Assigns participants to lists until each list reaches a target quota.
    When all quotas are filled, either raises an error or allows overflow.

    Attributes
    ----------
    participants_per_list : int
        Target number of participants per list (must be > 0).
    allow_overflow : bool
        Whether to allow assignment after quota reached (default: False).
        If True, uses balanced assignment after quotas filled.
        If False, raises error when all quotas reached.

    Examples
    --------
    >>> config = QuotaConfig(participants_per_list=25, allow_overflow=False)
    >>> config.participants_per_list
    25

    Raises
    ------
    ValueError
        If participants_per_list <= 0.
    """

    participants_per_list: int = Field(..., gt=0)
    allow_overflow: bool = False


class WeightedRandomConfig(BeadBaseModel):
    """Configuration for weighted random assignment.

    Assigns lists with non-uniform probabilities based on metadata expressions.
    Useful for oversampling certain lists or adaptive designs.

    Attributes
    ----------
    weight_expression : str
        JavaScript expression to compute weight from list metadata.
        Expression is evaluated with 'list_metadata' in scope.
        Example: "list_metadata.priority || 1.0"
    normalize_weights : bool
        Whether to normalize weights to sum to 1.0 (default: True).

    Examples
    --------
    >>> config = WeightedRandomConfig(
    ...     weight_expression="list_metadata.priority || 1.0",
    ...     normalize_weights=True
    ... )
    >>> config.weight_expression
    'list_metadata.priority || 1.0'

    Raises
    ------
    ValueError
        If weight_expression is empty.
    """

    weight_expression: str = Field(..., min_length=1)
    normalize_weights: bool = True

    @field_validator("weight_expression")
    @classmethod
    def validate_weight_expression(cls, v: str) -> str:
        """Validate weight expression is non-empty."""
        if not v or not v.strip():
            raise ValueError(
                "weight_expression must be non-empty. "
                "Provide a JavaScript expression like 'list_metadata.priority || 1.0'. "
                "This expression will be evaluated for each list to compute weights."
            )
        return v.strip()


class LatinSquareConfig(BeadBaseModel):
    """Configuration for Latin square counterbalancing.

    Generates a Latin square design for systematic counterbalancing of order effects.
    Ensures each condition appears at each position across participants.

    Attributes
    ----------
    balanced : bool
        Use balanced Latin square vs. standard (default: True).
        Balanced squares use Bradley's (1958) algorithm.

    Examples
    --------
    >>> config = LatinSquareConfig(balanced=True)
    >>> config.balanced
    True
    """

    balanced: bool = True


class MetadataBasedConfig(BeadBaseModel):
    """Configuration for metadata-based assignment.

    Filters and ranks lists based on metadata expressions before assignment.
    Useful for assignment based on list properties like difficulty or priority.

    Attributes
    ----------
    filter_expression : str | None
        JavaScript boolean expression to filter lists (default: None).
        Expression is evaluated with 'list_metadata' in scope.
        Only lists where expression evaluates to true are eligible.
        Example: "list_metadata.difficulty === 'easy'"
    rank_expression : str | None
        JavaScript expression to rank/sort lists (default: None).
        Expression is evaluated with 'list_metadata' in scope.
        Lists are sorted by this value before assignment.
        Example: "list_metadata.priority || 0"
    rank_ascending : bool
        Sort ascending vs descending when using rank_expression (default: True).

    Examples
    --------
    >>> config = MetadataBasedConfig(
    ...     filter_expression="list_metadata.difficulty === 'easy'",
    ...     rank_expression="list_metadata.priority || 0",
    ...     rank_ascending=False
    ... )
    >>> config.filter_expression
    "list_metadata.difficulty === 'easy'"

    Raises
    ------
    ValueError
        If both filter_expression and rank_expression are None.
    """

    filter_expression: str | None = None
    rank_expression: str | None = None
    rank_ascending: bool = True

    @model_validator(mode="after")
    def validate_at_least_one_expression(self) -> MetadataBasedConfig:
        """Validate at least one expression is provided."""
        if self.filter_expression is None and self.rank_expression is None:
            raise ValueError(
                "MetadataBasedConfig requires at least one of 'filter_expression' "
                "or 'rank_expression'. Got neither. "
                "Provide 'filter_expression' to filter lists (e.g., "
                "\"list_metadata.difficulty === 'easy'\") or 'rank_expression' to "
                'rank lists (e.g., "list_metadata.priority || 0").'
            )
        return self


class StratifiedConfig(BeadBaseModel):
    """Configuration for stratified assignment.

    Balances assignment across multiple factors (e.g., list × condition).
    Ensures even distribution across factor combinations.

    Attributes
    ----------
    factors : list[str]
        List metadata keys to use as stratification factors (must be non-empty).
        Lists are grouped by unique combinations of these factor values.
        Example: ["condition", "verb_type"] groups by condition × verb_type.

    Examples
    --------
    >>> config = StratifiedConfig(factors=["condition", "verb_type"])
    >>> config.factors
    ['condition', 'verb_type']

    Raises
    ------
    ValueError
        If factors list is empty.
    """

    factors: list[str] = Field(..., min_length=1)

    @field_validator("factors")
    @classmethod
    def validate_factors(cls, v: list[str]) -> list[str]:
        """Validate factors list is non-empty and contains no duplicates."""
        if not v:
            raise ValueError(
                "StratifiedConfig requires at least one factor in 'factors' list. "
                "Got empty list. "
                "Provide a list of metadata keys to stratify by, e.g., "
                "['condition', 'verb_type']."
            )

        if len(v) != len(set(v)):
            duplicates = [x for x in v if v.count(x) > 1]
            raise ValueError(
                f"StratifiedConfig 'factors' contains duplicates: {duplicates}. "
                f"Each factor must appear only once. "
                f"Remove duplicate entries from your factors list."
            )

        return v


class ListDistributionStrategy(BeadBaseModel):
    """Configuration for list distribution strategy in batch experiments.

    Defines how participants are assigned to experiment lists using JATOS batch
    sessions for server-side state management.

    Attributes
    ----------
    strategy_type : DistributionStrategyType
        Type of distribution strategy (required, no default).
    strategy_config : dict[str, Any]
        Strategy-specific configuration parameters (default: empty dict).
        Required keys depend on strategy_type:
        - quota_based: requires 'participants_per_list'
        - weighted_random: requires 'weight_expression'
        - metadata_based: requires 'filter_expression' or 'rank_expression'
        - stratified: requires 'factors'
    max_participants : int | None
        Maximum total participants across all lists (None = unlimited) (default: None).
    error_on_exhaustion : bool
        Raise error when max_participants reached (default: True).
        If False, continues assignment (may exceed max_participants).
    debug_mode : bool
        Enable debug mode (always assign same list) (default: False).
        Useful for development testing without batch session state.
    debug_list_index : int
        List index to use in debug mode (default: 0, must be >= 0).

    Examples
    --------
    >>> # Balanced assignment
    >>> strategy = ListDistributionStrategy(
    ...     strategy_type=DistributionStrategyType.BALANCED,
    ...     max_participants=100
    ... )

    >>> # Quota-based with 25 per list
    >>> strategy = ListDistributionStrategy(
    ...     strategy_type=DistributionStrategyType.QUOTA_BASED,
    ...     strategy_config={"participants_per_list": 25, "allow_overflow": False},
    ...     max_participants=400
    ... )

    >>> # Debug mode (always list 0)
    >>> strategy = ListDistributionStrategy(
    ...     strategy_type=DistributionStrategyType.RANDOM,
    ...     debug_mode=True,
    ...     debug_list_index=0
    ... )

    Raises
    ------
    ValueError
        If strategy_config doesn't match requirements for strategy_type.
    """

    strategy_type: DistributionStrategyType
    strategy_config: dict[str, Any] = Field(default_factory=dict)
    max_participants: int | None = Field(default=None, ge=1)
    error_on_exhaustion: bool = True
    debug_mode: bool = False
    debug_list_index: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_strategy_config_matches_type(self) -> ListDistributionStrategy:
        """Validate strategy_config has required keys for strategy_type.

        Raises
        ------
        ValueError
            If required configuration keys are missing for the strategy type.
        """
        strategy = self.strategy_type
        config = self.strategy_config

        # Quota-based requires participants_per_list
        if strategy == DistributionStrategyType.QUOTA_BASED:
            if "participants_per_list" not in config:
                raise ValueError(
                    f"QuotaConfig requires 'participants_per_list'. "
                    f"Got keys: {list(config.keys())}. "
                    f"Add 'participants_per_list: <int>' to strategy_config."
                )

            # Validate it's a positive integer
            ppl = config["participants_per_list"]
            if not isinstance(ppl, int) or ppl <= 0:
                raise ValueError(
                    f"'participants_per_list' must be positive int. "
                    f"Got: {ppl} ({type(ppl).__name__})."
                )

        # Weighted random requires weight_expression
        elif strategy == DistributionStrategyType.WEIGHTED_RANDOM:
            if "weight_expression" not in config:
                raise ValueError(
                    f"WeightedRandomConfig requires 'weight_expression'. "
                    f"Got keys: {list(config.keys())}. "
                    f"Add 'weight_expression: <string>' to strategy_config."
                )

            expr = config["weight_expression"]
            if not isinstance(expr, str) or not expr.strip():
                raise ValueError(
                    f"'weight_expression' must be a non-empty string. "
                    f"Got: {expr!r} ({type(expr).__name__})."
                )

        # Metadata-based requires filter_expression or rank_expression
        elif strategy == DistributionStrategyType.METADATA_BASED:
            has_filter = "filter_expression" in config and config["filter_expression"]
            has_rank = "rank_expression" in config and config["rank_expression"]

            if not has_filter and not has_rank:
                raise ValueError(
                    f"MetadataBasedConfig requires 'filter_expression' "
                    f"or 'rank_expression'. Got keys: {list(config.keys())}."
                )

        # Stratified requires factors list
        elif strategy == DistributionStrategyType.STRATIFIED:
            if "factors" not in config:
                raise ValueError(
                    f"StratifiedConfig requires 'factors'. "
                    f"Got keys: {list(config.keys())}. "
                    f"Add 'factors: [<list_of_keys>]' to strategy_config."
                )

            factors = config["factors"]
            if not isinstance(factors, list) or not factors:
                raise ValueError(
                    f"StratifiedConfig 'factors' must be a non-empty list of strings. "
                    f"Got: {factors!r} (type: {type(factors).__name__}). "
                    f"Provide a list like ['condition', 'verb_type']."
                )

        return self

    @field_validator("debug_list_index")
    @classmethod
    def validate_debug_list_index(cls, v: int) -> int:
        """Validate debug_list_index is non-negative."""
        if v < 0:
            raise ValueError(
                f"debug_list_index must be >= 0. Got: {v}. "
                f"Set to 0 for first list, 1 for second list, etc."
            )
        return v
