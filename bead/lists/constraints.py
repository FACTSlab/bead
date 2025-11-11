"""Constraint models for experimental list composition.

This module defines constraints that can be applied to experimental lists
to ensure balanced, well-distributed item selections. Constraints can specify:
- Uniqueness: No duplicate property values
- Balance: Balanced distribution across categories
- Quantile: Uniform distribution across quantiles
- Size: List size requirements
- Ordering: Item presentation order constraints (runtime enforcement)

All constraints inherit from BeadBaseModel and use Pydantic discriminated unions
for type-safe deserialization.
"""

from __future__ import annotations

from typing import Annotated, Literal
from uuid import UUID

from pydantic import Field, field_validator, model_validator

from bead.data.base import BeadBaseModel
from bead.resources.constraints import ContextValue

# Type alias for list constraint types
ListConstraintType = Literal[
    "uniqueness",  # No duplicate property values
    "conditional_uniqueness",  # Conditional uniqueness based on DSL expression
    "balance",  # Balanced distribution of property
    "quantile",  # Uniform across quantiles
    "grouped_quantile",  # Quantile distribution within groups
    "diversity",  # Minimum unique values for property
    "size",  # List size constraints
    "ordering",  # Presentation order constraints (runtime enforcement)
]

# Type alias for batch constraint types
BatchConstraintType = Literal[
    "coverage",  # All values appear somewhere in batch
    "balance",  # Balanced distribution across entire batch
    "diversity",  # Prevent values appearing in too many lists
    "min_occurrence",  # Minimum occurrences per value across batch
]


class UniquenessConstraint(BeadBaseModel):
    """Constraint requiring unique values for a property.

    Ensures that no two items in a list have the same value for the
    specified property. Useful for preventing duplicate target verbs,
    sentence structures, or other experimental materials.

    Attributes
    ----------
    constraint_type : Literal["uniqueness"]
        Discriminator field for constraint type (always "uniqueness").
    property_expression : str
        DSL expression that extracts the value that must be unique.
        The item is available as 'item' in the expression.
        Examples: "item.metadata.target_verb", "item.templates.sentence.text"
    context : dict[str, ContextValue]
        Additional context variables for DSL evaluation.
    allow_null : bool, default=False
        Whether to allow null/None values. If False, None values count
        as duplicates. If True, multiple None values are allowed.
    priority : int, default=1
        Constraint priority (higher = more important). When partitioning,
        violations of higher-priority constraints are penalized more heavily.

    Examples
    --------
    >>> # No two items with same target verb (high priority)
    >>> constraint = UniquenessConstraint(
    ...     property_expression="item.metadata.target_verb",
    ...     allow_null=False,
    ...     priority=5
    ... )
    >>> constraint.priority
    5
    """

    constraint_type: Literal["uniqueness"] = "uniqueness"
    property_expression: str = Field(
        ..., description="DSL expression for value to check"
    )
    context: dict[str, ContextValue] = Field(
        default_factory=dict, description="Additional context variables"
    )
    allow_null: bool = Field(
        default=False, description="Whether to allow multiple null values"
    )
    priority: int = Field(
        default=1, ge=1, description="Constraint priority (higher = more important)"
    )

    @field_validator("property_expression")
    @classmethod
    def validate_property_expression(cls, v: str) -> str:
        """Validate property expression is non-empty.

        Parameters
        ----------
        v : str
            Property expression to validate.

        Returns
        -------
        str
            Validated property expression.

        Raises
        ------
        ValueError
            If property expression is empty or contains only whitespace.
        """
        if not v or not v.strip():
            raise ValueError("property_expression must be non-empty")
        return v.strip()


class BalanceConstraint(BeadBaseModel):
    """Constraint requiring balanced distribution.

    Ensures balanced distribution of a categorical property across items
    in a list. Can specify target counts for each category or request
    equal distribution.

    Attributes
    ----------
    constraint_type : Literal["balance"]
        Discriminator field for constraint type (always "balance").
    property_expression : str
        DSL expression that extracts the category value to balance.
        The item is available as 'item' in the expression.
        Example: "item.metadata.transitivity"
    context : dict[str, ContextValue]
        Additional context variables for DSL evaluation.
    target_counts : dict[str, int] | None, default=None
        Target counts for each category value. If None, equal distribution
        is assumed. Keys are category values, values are target counts.
    tolerance : float, default=0.1
        Allowed deviation from target as a proportion (0.0-1.0).
        For example, 0.1 means up to 10% deviation is acceptable.
    priority : int, default=1
        Constraint priority (higher = more important). When partitioning,
        violations of higher-priority constraints are penalized more heavily.

    Examples
    --------
    >>> # Equal number of transitive and intransitive verbs
    >>> constraint = BalanceConstraint(
    ...     property_expression="item.metadata.transitivity",
    ...     tolerance=0.1
    ... )
    >>> # 2:1 ratio with high priority
    >>> constraint2 = BalanceConstraint(
    ...     property_expression="item.metadata.grammatical",
    ...     target_counts={"true": 20, "false": 10},
    ...     tolerance=0.05,
    ...     priority=3
    ... )
    """

    constraint_type: Literal["balance"] = "balance"
    property_expression: str = Field(
        ..., description="DSL expression for category value"
    )
    context: dict[str, ContextValue] = Field(
        default_factory=dict, description="Additional context variables"
    )
    target_counts: dict[str, int] | None = Field(
        default=None, description="Target counts per category (None = equal)"
    )
    tolerance: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Allowed deviation from target"
    )
    priority: int = Field(
        default=1, ge=1, description="Constraint priority (higher = more important)"
    )

    @field_validator("property_expression")
    @classmethod
    def validate_property_expression(cls, v: str) -> str:
        """Validate property expression is non-empty.

        Parameters
        ----------
        v : str
            Property expression to validate.

        Returns
        -------
        str
            Validated property expression.

        Raises
        ------
        ValueError
            If property expression is empty or contains only whitespace.
        """
        if not v or not v.strip():
            raise ValueError("property_expression must be non-empty")
        return v.strip()

    @field_validator("target_counts")
    @classmethod
    def validate_target_counts(cls, v: dict[str, int] | None) -> dict[str, int] | None:
        """Validate target counts are non-negative.

        Parameters
        ----------
        v : dict[str, int] | None
            Target counts to validate.

        Returns
        -------
        dict[str, int] | None
            Validated target counts.

        Raises
        ------
        ValueError
            If any count is negative.
        """
        if v is not None:
            for category, count in v.items():
                if count < 0:
                    raise ValueError(
                        f"target_counts values must be non-negative, "
                        f"got {count} for '{category}'"
                    )
        return v


class QuantileConstraint(BeadBaseModel):
    """Constraint requiring uniform distribution across quantiles.

    Ensures uniform distribution of items across quantiles of a numeric
    property. Useful for balancing language model probabilities, word
    frequencies, or other continuous variables. Supports complex DSL
    expressions for computing derived metrics.

    Attributes
    ----------
    constraint_type : Literal["quantile"]
        Discriminator field for constraint type (always "quantile").
    property_expression : str
        DSL expression that computes the numeric value to quantile.
        The item is available as 'item' in the expression.
        Can be simple (e.g., "item.metadata.lm_prob") or complex
        (e.g., "variance([item['val1'], item['val2'], item['val3']])")
    context : dict[str, ContextValue]
        Additional context variables for DSL evaluation.
        Example: {"hyp_keys": ["hyp1", "hyp2", "hyp3"]}
    n_quantiles : int, default=5
        Number of quantiles to create (must be >= 2).
    items_per_quantile : int, default=2
        Target number of items per quantile (must be >= 1).
    priority : int, default=1
        Constraint priority (higher = more important). When partitioning,
        violations of higher-priority constraints are penalized more heavily.

    Examples
    --------
    >>> # Uniform distribution of LM probabilities across 5 quantiles
    >>> constraint = QuantileConstraint(
    ...     property_expression="item.metadata.lm_prob",
    ...     n_quantiles=5,
    ...     items_per_quantile=2
    ... )
    >>> # Variance of precomputed NLI scores
    >>> constraint2 = QuantileConstraint(
    ...     property_expression="item['nli_variance']",
    ...     n_quantiles=5,
    ...     items_per_quantile=2
    ... )
    """

    constraint_type: Literal["quantile"] = "quantile"
    property_expression: str = Field(
        ..., description="DSL expression for numeric value"
    )
    context: dict[str, ContextValue] = Field(
        default_factory=dict, description="Additional context variables"
    )
    n_quantiles: int = Field(default=5, ge=2, description="Number of quantiles")
    items_per_quantile: int = Field(default=2, ge=1, description="Items per quantile")
    priority: int = Field(
        default=1, ge=1, description="Constraint priority (higher = more important)"
    )

    @field_validator("property_expression")
    @classmethod
    def validate_property_expression(cls, v: str) -> str:
        """Validate property expression is non-empty.

        Parameters
        ----------
        v : str
            Property expression to validate.

        Returns
        -------
        str
            Validated property expression.

        Raises
        ------
        ValueError
            If property expression is empty or contains only whitespace.
        """
        if not v or not v.strip():
            raise ValueError("property_expression must be non-empty")
        return v.strip()


class GroupedQuantileConstraint(BeadBaseModel):
    """Constraint requiring uniform quantile distribution within groups.

    Ensures uniform distribution across quantiles of a numeric property
    within each group defined by a grouping property. Useful for balancing
    a continuous variable independently within categorical groups.

    Attributes
    ----------
    constraint_type : Literal["grouped_quantile"]
        Discriminator field for constraint type (always "grouped_quantile").
    property_expression : str
        DSL expression that computes the numeric value to quantile.
        The item is available as 'item' in the expression.
        Example: "item.metadata.lm_prob"
    group_by_expression : str
        DSL expression that computes the grouping key.
        The item is available as 'item' in the expression.
        Example: "item.metadata.condition"
    context : dict[str, ContextValue]
        Additional context variables for DSL evaluation.
    n_quantiles : int, default=5
        Number of quantiles to create per group (must be >= 2).
    items_per_quantile : int, default=2
        Target number of items per quantile per group (must be >= 1).
    priority : int, default=1
        Constraint priority (higher = more important). When partitioning,
        violations of higher-priority constraints are penalized more heavily.

    Examples
    --------
    >>> # Balance LM probability quantiles within each condition
    >>> constraint = GroupedQuantileConstraint(
    ...     property_expression="item.metadata.lm_prob",
    ...     group_by_expression="item.metadata.condition",
    ...     n_quantiles=5,
    ...     items_per_quantile=2
    ... )
    >>> # Balance embedding similarity IQR within semantic categories
    >>> constraint2 = GroupedQuantileConstraint(
    ...     property_expression="item['embedding_iqr']",
    ...     group_by_expression="item['semantic_category']",
    ...     n_quantiles=4,
    ...     items_per_quantile=3
    ... )
    """

    constraint_type: Literal["grouped_quantile"] = "grouped_quantile"
    property_expression: str = Field(
        ..., description="DSL expression for numeric value"
    )
    group_by_expression: str = Field(..., description="DSL expression for grouping key")
    context: dict[str, ContextValue] = Field(
        default_factory=dict, description="Additional context variables"
    )
    n_quantiles: int = Field(
        default=5, ge=2, description="Number of quantiles per group"
    )
    items_per_quantile: int = Field(
        default=2, ge=1, description="Items per quantile per group"
    )
    priority: int = Field(
        default=1, ge=1, description="Constraint priority (higher = more important)"
    )

    @field_validator("property_expression", "group_by_expression")
    @classmethod
    def validate_expression(cls, v: str) -> str:
        """Validate expression is non-empty.

        Parameters
        ----------
        v : str
            Expression to validate.

        Returns
        -------
        str
            Validated expression.

        Raises
        ------
        ValueError
            If expression is empty or contains only whitespace.
        """
        if not v or not v.strip():
            raise ValueError("expression must be non-empty")
        return v.strip()


class ConditionalUniquenessConstraint(BeadBaseModel):
    """Constraint requiring uniqueness when a condition is met.

    Ensures that values are unique only when a boolean condition is satisfied.
    Useful for enforcing uniqueness on a subset of items while allowing
    duplicates in others.

    Attributes
    ----------
    constraint_type : Literal["conditional_uniqueness"]
        Discriminator field for constraint type (always "conditional_uniqueness").
    property_expression : str
        DSL expression that computes the value that must be unique.
        The item is available as 'item' in the expression.
        Example: "item.metadata.target_word"
    condition_expression : str
        DSL boolean expression that determines if constraint applies.
        The item is available as 'item' in the expression.
        Example: "item.metadata.is_critical == True"
    context : dict[str, ContextValue]
        Additional context variables for DSL evaluation.
    allow_null : bool, default=False
        Whether to allow multiple null values when condition is true.
    priority : int, default=1
        Constraint priority (higher = more important). When partitioning,
        violations of higher-priority constraints are penalized more heavily.

    Examples
    --------
    >>> # Unique target words only for critical items
    >>> constraint = ConditionalUniquenessConstraint(
    ...     property_expression="item.metadata.target_word",
    ...     condition_expression="item.metadata.is_critical == True",
    ...     allow_null=False,
    ...     priority=3
    ... )
    >>> # Unique sentences only when grammaticality is tested
    >>> constraint2 = ConditionalUniquenessConstraint(
    ...     property_expression="item.templates.sentence.text",
    ...     condition_expression="item.metadata.test_type in test_grammaticality",
    ...     context={"test_grammaticality": {"gram", "acceptability"}},
    ...     allow_null=True
    ... )
    """

    constraint_type: Literal["conditional_uniqueness"] = "conditional_uniqueness"
    property_expression: str = Field(
        ..., description="DSL expression for value to check"
    )
    condition_expression: str = Field(
        ..., description="DSL boolean expression for when to apply constraint"
    )
    context: dict[str, ContextValue] = Field(
        default_factory=dict, description="Additional context variables"
    )
    allow_null: bool = Field(
        default=False, description="Whether to allow multiple null values"
    )
    priority: int = Field(
        default=1, ge=1, description="Constraint priority (higher = more important)"
    )

    @field_validator("property_expression", "condition_expression")
    @classmethod
    def validate_expression(cls, v: str) -> str:
        """Validate expression is non-empty.

        Parameters
        ----------
        v : str
            Expression to validate.

        Returns
        -------
        str
            Validated expression.

        Raises
        ------
        ValueError
            If expression is empty or contains only whitespace.
        """
        if not v or not v.strip():
            raise ValueError("expression must be non-empty")
        return v.strip()


class DiversityConstraint(BeadBaseModel):
    """Constraint requiring minimum diversity (unique values) for a property.

    Ensures that a list contains at least a minimum number of unique values
    for a specified property. Useful for ensuring template diversity, verb
    diversity, or other experimental richness requirements.

    Attributes
    ----------
    constraint_type : Literal["diversity"]
        Discriminator field for constraint type (always "diversity").
    property_expression : str
        DSL expression that extracts the value to count for diversity.
        The item is available as 'item' in the expression.
        Examples: "item.metadata.template_id", "item.metadata.verb_lemma"
    min_unique_values : int
        Minimum number of unique values required in the list.
    context : dict[str, ContextValue]
        Additional context variables for DSL evaluation.
    priority : int, default=1
        Constraint priority (higher = more important). When partitioning,
        violations of higher-priority constraints are penalized more heavily.

    Examples
    --------
    >>> # Ensure at least 15 unique templates per list
    >>> constraint = DiversityConstraint(
    ...     property_expression="item.metadata.template_id",
    ...     min_unique_values=15,
    ...     priority=2
    ... )
    >>> constraint.min_unique_values
    15
    """

    constraint_type: Literal["diversity"] = "diversity"
    property_expression: str = Field(
        ..., description="DSL expression for value to check for diversity"
    )
    min_unique_values: int = Field(
        ..., ge=1, description="Minimum number of unique values required"
    )
    context: dict[str, ContextValue] = Field(
        default_factory=dict, description="Additional context variables"
    )
    priority: int = Field(
        default=1, ge=1, description="Constraint priority (higher = more important)"
    )

    @field_validator("property_expression")
    @classmethod
    def validate_property_expression(cls, v: str) -> str:
        """Validate property expression is non-empty.

        Parameters
        ----------
        v : str
            Property expression to validate.

        Returns
        -------
        str
            Validated property expression.

        Raises
        ------
        ValueError
            If property expression is empty or contains only whitespace.
        """
        if not v or not v.strip():
            raise ValueError("property_expression must be non-empty")
        return v.strip()


class SizeConstraint(BeadBaseModel):
    """Constraint on list size.

    Specifies size requirements for a list. Can specify exact size,
    minimum size, maximum size, or a range (min and max).

    Often used with high priority to ensure participants do equal work.

    Attributes
    ----------
    constraint_type : Literal["size"]
        Discriminator field for constraint type (always "size").
    min_size : int | None, default=None
        Minimum list size (must be >= 0 if set).
    max_size : int | None, default=None
        Maximum list size (must be >= 0 if set).
    exact_size : int | None, default=None
        Exact required size (must be >= 0 if set).
        Cannot be used with min_size or max_size.
    priority : int, default=1
        Constraint priority (higher = more important). When partitioning,
        violations of higher-priority constraints are penalized more heavily.
        Size constraints often use high priority (e.g., 10) to ensure
        participants do exactly equal amounts of work.

    Examples
    --------
    >>> # Exactly 40 items per list (highest priority)
    >>> constraint = SizeConstraint(exact_size=40, priority=10)
    >>> # Between 30-50 items per list
    >>> constraint2 = SizeConstraint(min_size=30, max_size=50)
    >>> # At least 20 items
    >>> constraint3 = SizeConstraint(min_size=20)
    >>> # At most 100 items
    >>> constraint4 = SizeConstraint(max_size=100)
    """

    constraint_type: Literal["size"] = "size"
    min_size: int | None = Field(default=None, ge=0, description="Minimum list size")
    max_size: int | None = Field(default=None, ge=0, description="Maximum list size")
    exact_size: int | None = Field(
        default=None, ge=0, description="Exact required size"
    )
    priority: int = Field(
        default=1, ge=1, description="Constraint priority (higher = more important)"
    )

    @model_validator(mode="after")
    def validate_size_params(self) -> SizeConstraint:
        """Validate size parameter combinations.

        Ensures that:
        - At least one size parameter is set
        - exact_size is not used with min_size or max_size
        - min_size <= max_size if both are set

        Returns
        -------
        SizeConstraint
            Validated constraint.

        Raises
        ------
        ValueError
            If validation fails.
        """
        # Check that at least one parameter is set
        if self.exact_size is None and self.min_size is None and self.max_size is None:
            raise ValueError(
                "Must specify at least one of: min_size, max_size, exact_size"
            )

        # Check that exact_size is not used with min/max
        if self.exact_size is not None:
            if self.min_size is not None or self.max_size is not None:
                raise ValueError("exact_size cannot be used with min_size or max_size")

        # Check that min <= max if both are set
        if self.min_size is not None and self.max_size is not None:
            if self.min_size > self.max_size:
                raise ValueError("min_size must be <= max_size")

        return self


class OrderingConstraint(BeadBaseModel):
    """Constraint on item presentation order.

    **CRITICAL**: This constraint is primarily enforced at **jsPsych runtime**,
    not during static list construction. The Python data model stores the
    constraint specification, which is then translated to JavaScript code
    for runtime enforcement during per-participant randomization.

    Attributes
    ----------
    constraint_type : Literal["ordering"]
        Discriminator for constraint type.
    precedence_pairs : list[tuple[UUID, UUID]]
        Pairs of (item_a_id, item_b_id) where item_a must appear before item_b.
    no_adjacent_property : str | None
        Property path; items with same value cannot be adjacent.
        Example: "item_metadata.condition" prevents AA, BB patterns.
    block_by_property : str | None
        Property path to group items into contiguous blocks.
        Example: "item_metadata.block_type" creates blocked design.
    min_distance : int | None
        Minimum number of items between items with same no_adjacent_property value.
    max_distance : int | None
        Maximum number of items between start and end of items with same
        block_by_property value (enforces tight blocking).
    practice_item_property : str | None
        Property path identifying practice items (should appear first).
        Example: "item_metadata.is_practice" with value True.
    randomize_within_blocks : bool
        Whether to randomize order within blocks (default True).
        Only applies when block_by_property is set.

    Examples
    --------
    >>> # No adjacent items with same condition
    >>> constraint = OrderingConstraint(
    ...     no_adjacent_property="item_metadata.condition"
    ... )

    >>> # Practice items first, then main items
    >>> constraint = OrderingConstraint(
    ...     practice_item_property="item_metadata.is_practice"
    ... )

    >>> # Blocked by condition, randomized within blocks
    >>> constraint = OrderingConstraint(
    ...     block_by_property="item_metadata.condition",
    ...     randomize_within_blocks=True
    ... )

    >>> # Item A before Item B
    >>> from uuid import uuid4
    >>> item_a, item_b = uuid4(), uuid4()
    >>> constraint = OrderingConstraint(
    ...     precedence_pairs=[(item_a, item_b)]
    ... )
    """

    constraint_type: Literal["ordering"] = "ordering"
    precedence_pairs: list[tuple[UUID, UUID]] = Field(
        default_factory=lambda: [], description="Pairs (a,b) where a must precede b"
    )
    no_adjacent_property: str | None = Field(
        default=None,
        description="Property that cannot have same value in adjacent items",
    )
    block_by_property: str | None = Field(
        default=None, description="Property to group into contiguous blocks"
    )
    min_distance: int | None = Field(
        default=None,
        ge=1,
        description="Minimum items between same no_adjacent_property values",
    )
    max_distance: int | None = Field(
        default=None, ge=1, description="Maximum distance for blocked items"
    )
    practice_item_property: str | None = Field(
        default=None, description="Property identifying practice items (shown first)"
    )
    randomize_within_blocks: bool = Field(
        default=True, description="Whether to randomize within blocks"
    )
    priority: int = Field(
        default=1,
        ge=1,
        description="Constraint priority (not used for static partitioning)",
    )

    @model_validator(mode="after")
    def validate_distance_constraints(self) -> OrderingConstraint:
        """Validate distance constraint combinations.

        Returns
        -------
        OrderingConstraint
            Validated constraint.

        Raises
        ------
        ValueError
            If validation fails.
        """
        if self.min_distance is not None and self.no_adjacent_property is None:
            raise ValueError("min_distance requires no_adjacent_property to be set")
        if self.max_distance is not None and self.block_by_property is None:
            raise ValueError("max_distance requires block_by_property to be set")
        if (
            self.min_distance
            and self.max_distance
            and self.min_distance > self.max_distance
        ):
            raise ValueError("min_distance cannot be greater than max_distance")
        return self


# Discriminated union for all list constraints
ListConstraint = Annotated[
    UniquenessConstraint
    | ConditionalUniquenessConstraint
    | BalanceConstraint
    | QuantileConstraint
    | GroupedQuantileConstraint
    | DiversityConstraint
    | SizeConstraint
    | OrderingConstraint,
    Field(discriminator="constraint_type"),
]


# ============================================================================
# Batch-Level Constraints
# ============================================================================


class BatchCoverageConstraint(BeadBaseModel):
    """Constraint ensuring all values appear somewhere in the batch.

    Ensures that all values of a property appear across the collection of lists.
    Useful for guaranteeing coverage of experimental conditions, templates, or
    stimulus categories across all participants.

    Attributes
    ----------
    constraint_type : Literal["coverage"]
        Discriminator field for constraint type (always "coverage").
    property_expression : str
        DSL expression that extracts the property value to check coverage.
        The item is available as 'item' in the expression (metadata dict).
        Example: "item['template_id']"
    context : dict[str, ContextValue]
        Additional context variables for DSL evaluation.
    target_values : list[str | int | float] | None
        Target values that must be covered. If None, uses all observed values.
    min_coverage : float, default=1.0
        Minimum coverage fraction (0.0-1.0). 1.0 means 100% of target values
        must appear.
    priority : int, default=1
        Constraint priority (higher = more important).

    Examples
    --------
    >>> # Ensure all 26 templates appear across all lists
    >>> constraint = BatchCoverageConstraint(
    ...     property_expression="item['template_id']",
    ...     target_values=list(range(26)),
    ...     min_coverage=1.0
    ... )
    >>> # Ensure at least 90% of verbs are covered
    >>> constraint = BatchCoverageConstraint(
    ...     property_expression="item['verb_lemma']",
    ...     target_values=["run", "jump", "eat", "sleep", "think"],
    ...     min_coverage=0.9
    ... )
    """

    constraint_type: Literal["coverage"] = "coverage"
    property_expression: str = Field(
        ..., description="DSL expression for property to check coverage"
    )
    context: dict[str, ContextValue] = Field(
        default_factory=dict, description="Additional context variables"
    )
    target_values: list[str | int | float] | None = Field(
        default=None, description="Target values that must be covered"
    )
    min_coverage: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Minimum coverage fraction"
    )
    priority: int = Field(
        default=1, ge=1, description="Constraint priority (higher = more important)"
    )

    @field_validator("property_expression")
    @classmethod
    def validate_property_expression(cls, v: str) -> str:
        """Validate property expression is non-empty."""
        if not v or not v.strip():
            raise ValueError("property_expression must be non-empty")
        return v.strip()


class BatchBalanceConstraint(BeadBaseModel):
    """Constraint ensuring balanced distribution across the entire batch.

    Ensures balanced distribution of a categorical property across all lists
    combined. Unlike per-list balance constraints, this operates on the
    aggregate distribution across the entire batch.

    Attributes
    ----------
    constraint_type : Literal["balance"]
        Discriminator field for constraint type (always "balance").
    property_expression : str
        DSL expression that extracts the category value to balance.
        Example: "item['pair_type']"
    context : dict[str, ContextValue]
        Additional context variables for DSL evaluation.
    target_distribution : dict[str, float]
        Target distribution (values sum to 1.0). Keys are category values,
        values are target proportions.
    tolerance : float, default=0.1
        Allowed deviation from target as a proportion (0.0-1.0).
    priority : int, default=1
        Constraint priority (higher = more important).

    Examples
    --------
    >>> # Ensure 50/50 balance of pair types across all lists
    >>> constraint = BatchBalanceConstraint(
    ...     property_expression="item['pair_type']",
    ...     target_distribution={"same_verb": 0.5, "different_verb": 0.5},
    ...     tolerance=0.05
    ... )
    >>> # Three-way split across conditions
    >>> constraint = BatchBalanceConstraint(
    ...     property_expression="item['condition']",
    ...     target_distribution={"A": 0.333, "B": 0.333, "C": 0.334},
    ...     tolerance=0.1
    ... )
    """

    constraint_type: Literal["balance"] = "balance"
    property_expression: str = Field(
        ..., description="DSL expression for category value"
    )
    context: dict[str, ContextValue] = Field(
        default_factory=dict, description="Additional context variables"
    )
    target_distribution: dict[str, float] = Field(
        ..., description="Target distribution (values sum to 1.0)"
    )
    tolerance: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Allowed deviation from target"
    )
    priority: int = Field(
        default=1, ge=1, description="Constraint priority (higher = more important)"
    )

    @field_validator("property_expression")
    @classmethod
    def validate_property_expression(cls, v: str) -> str:
        """Validate property expression is non-empty."""
        if not v or not v.strip():
            raise ValueError("property_expression must be non-empty")
        return v.strip()

    @field_validator("target_distribution")
    @classmethod
    def validate_target_distribution(cls, v: dict[str, float]) -> dict[str, float]:
        """Validate target distribution sums to ~1.0 and values are in [0, 1]."""
        if not v:
            raise ValueError("target_distribution must not be empty")

        for category, prob in v.items():
            if not 0.0 <= prob <= 1.0:
                raise ValueError(
                    f"target_distribution values must be in [0, 1], "
                    f"got {prob} for '{category}'"
                )

        total = sum(v.values())
        if not 0.99 <= total <= 1.01:
            raise ValueError(
                f"target_distribution values must sum to ~1.0, got {total}"
            )

        return v


class BatchDiversityConstraint(BeadBaseModel):
    """Constraint preventing values from appearing in too many lists.

    Ensures that no single value of a property appears in too many lists,
    promoting diversity across lists. Useful for ensuring that stimuli
    (e.g., verbs, nouns) are distributed across participants rather than
    concentrated in a few lists.

    Attributes
    ----------
    constraint_type : Literal["diversity"]
        Discriminator field for constraint type (always "diversity").
    property_expression : str
        DSL expression that extracts the property value to check diversity.
        Example: "item['verb_lemma']"
    context : dict[str, ContextValue]
        Additional context variables for DSL evaluation.
    max_lists_per_value : int
        Maximum number of lists any value can appear in.
    priority : int, default=1
        Constraint priority (higher = more important).

    Examples
    --------
    >>> # No verb should appear in more than 3 out of 8 lists
    >>> constraint = BatchDiversityConstraint(
    ...     property_expression="item['verb_lemma']",
    ...     max_lists_per_value=3
    ... )
    >>> # No template in more than half the lists
    >>> constraint = BatchDiversityConstraint(
    ...     property_expression="item['template_id']",
    ...     max_lists_per_value=4
    ... )
    """

    constraint_type: Literal["diversity"] = "diversity"
    property_expression: str = Field(
        ..., description="DSL expression for property value"
    )
    context: dict[str, ContextValue] = Field(
        default_factory=dict, description="Additional context variables"
    )
    max_lists_per_value: int = Field(
        ..., ge=1, description="Maximum lists any value can appear in"
    )
    priority: int = Field(
        default=1, ge=1, description="Constraint priority (higher = more important)"
    )

    @field_validator("property_expression")
    @classmethod
    def validate_property_expression(cls, v: str) -> str:
        """Validate property expression is non-empty."""
        if not v or not v.strip():
            raise ValueError("property_expression must be non-empty")
        return v.strip()


class BatchMinOccurrenceConstraint(BeadBaseModel):
    """Constraint ensuring minimum representation across the batch.

    Ensures that each value of a property appears at least a minimum number
    of times across all lists. Useful for guaranteeing sufficient data for
    each experimental condition or stimulus category.

    Attributes
    ----------
    constraint_type : Literal["min_occurrence"]
        Discriminator field for constraint type (always "min_occurrence").
    property_expression : str
        DSL expression that extracts the property value to check occurrences.
        Example: "item['quantile']"
    context : dict[str, ContextValue]
        Additional context variables for DSL evaluation.
    min_occurrences : int
        Minimum number of times each value must appear across all lists.
    priority : int, default=1
        Constraint priority (higher = more important).

    Examples
    --------
    >>> # Each quantile appears at least 50 times across all lists
    >>> constraint = BatchMinOccurrenceConstraint(
    ...     property_expression="item['quantile']",
    ...     min_occurrences=50
    ... )
    >>> # Each template at least 5 times
    >>> constraint = BatchMinOccurrenceConstraint(
    ...     property_expression="item['template_id']",
    ...     min_occurrences=5
    ... )
    """

    constraint_type: Literal["min_occurrence"] = "min_occurrence"
    property_expression: str = Field(
        ..., description="DSL expression for property value"
    )
    context: dict[str, ContextValue] = Field(
        default_factory=dict, description="Additional context variables"
    )
    min_occurrences: int = Field(
        ..., ge=1, description="Minimum occurrences per value across batch"
    )
    priority: int = Field(
        default=1, ge=1, description="Constraint priority (higher = more important)"
    )

    @field_validator("property_expression")
    @classmethod
    def validate_property_expression(cls, v: str) -> str:
        """Validate property expression is non-empty."""
        if not v or not v.strip():
            raise ValueError("property_expression must be non-empty")
        return v.strip()


# Discriminated union for all batch constraints
BatchConstraint = Annotated[
    BatchCoverageConstraint
    | BatchBalanceConstraint
    | BatchDiversityConstraint
    | BatchMinOccurrenceConstraint,
    Field(discriminator="constraint_type"),
]
