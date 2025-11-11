"""List configuration models for the bead package."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

if TYPE_CHECKING:
    pass


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
    property_expression: str = Field(..., description="Expression to extract property")
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
    priority: int = Field(default=1, ge=1, description="Constraint priority")

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
                raise ValueError(
                    "max_lists_per_value required for diversity constraint"
                )
        elif self.type == "min_occurrence":
            if self.min_occurrences is None:
                raise ValueError(
                    "min_occurrences required for min_occurrence constraint"
                )

        return self


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
