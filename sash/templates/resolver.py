"""Constraint resolution for template slot filling.

This module provides the ConstraintResolver class for evaluating constraints
against lexical items to determine which items satisfy template slot requirements.
"""

from __future__ import annotations

import hashlib
from typing import Any

from sash.adapters.registry import AdapterRegistry
from sash.data.language_codes import LanguageCode
from sash.dsl.context import EvaluationContext
from sash.dsl.evaluator import Evaluator
from sash.dsl.parser import parse
from sash.dsl.stdlib import register_stdlib
from sash.resources.constraints import (
    Constraint,
    DSLConstraint,
    ExtensionalConstraint,
    IntensionalConstraint,
    RelationalConstraint,
)
from sash.resources.lexicon import Lexicon
from sash.resources.models import LexicalItem


class ConstraintResolver:
    """Resolve constraints against lexical items.

    Evaluate template slot constraints to determine which lexical items
    from a lexicon satisfy the constraints. Supports all constraint types:
    extensional, intensional, relational, and DSL-based.

    Parameters
    ----------
    lexicon : Lexicon
        Lexicon containing lexical items to evaluate against.
    adapter_registry : AdapterRegistry | None
        Registry of external resource adapters (required for relational
        constraints that use external resources). If None, some relational
        constraints may not be evaluable.
    cache_results : bool
        Whether to cache constraint evaluation results. Default: True.

    Examples
    --------
    >>> from sash.resources.models import LexicalItem
    >>> from sash.resources.lexicon import Lexicon
    >>> from sash.resources.constraints import IntensionalConstraint
    >>> items = [
    ...     LexicalItem(lemma="break", pos="VERB"),
    ...     LexicalItem(lemma="cat", pos="NOUN")
    ... ]
    >>> lexicon = Lexicon(items=items)
    >>> resolver = ConstraintResolver(lexicon)
    >>> constraint = IntensionalConstraint(
    ...     property="pos", operator="==", value="VERB"
    ... )
    >>> matching = resolver.resolve(constraint)
    >>> len(matching)
    1
    >>> matching[0].lemma
    'break'
    """

    def __init__(
        self,
        lexicon: Lexicon,
        adapter_registry: AdapterRegistry | None = None,
        cache_results: bool = True,
    ) -> None:
        """Initialize constraint resolver.

        Parameters
        ----------
        lexicon : Lexicon
            Lexicon containing lexical items to evaluate.
        adapter_registry : AdapterRegistry | None
            Registry of external resource adapters. May be required for
            some relational constraints.
        cache_results : bool
            Whether to cache evaluation results. Default: True.
        """
        self.lexicon = lexicon
        self.adapter_registry = adapter_registry
        self.cache_results = cache_results

        # Initialize DSL evaluator for DSL constraints
        self._dsl_evaluator = Evaluator(use_cache=True)

        # Cache: (constraint_hash, item_id) -> bool
        self._cache: dict[tuple[str, str], bool] = {}

    def resolve(
        self,
        constraint: Constraint,
        language_code: LanguageCode | None = None,
    ) -> list[LexicalItem]:
        """Resolve constraint against lexicon.

        Evaluate the constraint against all items in the lexicon and
        return only those items that satisfy the constraint.

        Parameters
        ----------
        constraint : Constraint
            Constraint to evaluate (extensional, intensional, relational, or DSL).
        language_code : LanguageCode | None
            Optional language code to filter items before evaluation.
            If None, evaluates against all items in lexicon.

        Returns
        -------
        list[LexicalItem]
            Items from lexicon that satisfy the constraint.

        Raises
        ------
        ValueError
            If constraint type requires adapter_registry but none provided.
        RuntimeError
            If constraint evaluation fails.

        Examples
        --------
        >>> from sash.resources.constraints import IntensionalConstraint
        >>> constraint = IntensionalConstraint(
        ...     property="pos", operator="==", value="VERB"
        ... )
        >>> items = resolver.resolve(constraint, language_code="en")
        >>> all(item.language_code == "en" for item in items)
        True
        """
        # Get items from lexicon (items is a dict[UUID, LexicalItem])
        items_to_check = list(self.lexicon.items.values())

        # Filter by language code if specified
        if language_code is not None:
            items_to_check = [
                item for item in items_to_check if item.language_code == language_code
            ]

        # Evaluate constraint against each item
        matching_items: list[LexicalItem] = []
        for item in items_to_check:
            if self.evaluate(constraint, item):
                matching_items.append(item)

        return matching_items

    def evaluate(
        self,
        constraint: Constraint,
        item: LexicalItem,
    ) -> bool:
        """Evaluate whether an item satisfies a constraint.

        Check if a single lexical item satisfies the given constraint.
        Results are cached if cache_results=True.

        Parameters
        ----------
        constraint : Constraint
            Constraint to evaluate.
        item : LexicalItem
            Lexical item to check.

        Returns
        -------
        bool
            True if item satisfies constraint, False otherwise.

        Raises
        ------
        ValueError
            If constraint type is not supported or requires unavailable resources.
        RuntimeError
            If constraint evaluation encounters an error.

        Examples
        --------
        >>> from sash.resources.models import LexicalItem
        >>> from sash.resources.constraints import IntensionalConstraint
        >>> constraint = IntensionalConstraint(
        ...     property="pos", operator="==", value="VERB"
        ... )
        >>> item = LexicalItem(lemma="break", pos="VERB")
        >>> resolver.evaluate(constraint, item)
        True
        """
        # Check cache first
        if self.cache_results:
            cache_key = self._get_cache_key(constraint, item)
            if cache_key in self._cache:
                return self._cache[cache_key]

        # Evaluate based on constraint type
        try:
            if isinstance(constraint, ExtensionalConstraint):
                result = self._evaluate_extensional(constraint, item)
            elif isinstance(constraint, IntensionalConstraint):
                result = self._evaluate_intensional(constraint, item)
            elif isinstance(constraint, RelationalConstraint):
                result = self._evaluate_relational(constraint, item)
            else:
                # Must be DSLConstraint (only remaining type in union)
                result = self._evaluate_dsl(constraint, item)
        except Exception as e:
            raise RuntimeError(f"Error evaluating constraint: {e}") from e

        # Cache result
        if self.cache_results:
            cache_key = self._get_cache_key(constraint, item)
            self._cache[cache_key] = result

        return result

    def clear_cache(self) -> None:
        """Clear the constraint evaluation cache.

        Remove all cached evaluation results. Useful when lexicon
        contents change or constraints are updated.

        Examples
        --------
        >>> resolver = ConstraintResolver(lexicon)
        >>> resolver.clear_cache()
        >>> len(resolver._cache)
        0
        """
        self._cache.clear()

    def _evaluate_extensional(
        self,
        constraint: ExtensionalConstraint,
        item: LexicalItem,
    ) -> bool:
        """Evaluate extensional constraint against item.

        Check if item's ID is in the allowed/denied items list based on mode.

        Parameters
        ----------
        constraint : ExtensionalConstraint
            Extensional constraint to evaluate.
        item : LexicalItem
            Item to check.

        Returns
        -------
        bool
            True if item satisfies constraint.
        """
        if constraint.mode == "allow":
            # Item must be in the allowed list
            return item.id in constraint.items
        else:  # mode == "deny"
            # Item must not be in the denied list
            return item.id not in constraint.items

    def _evaluate_intensional(
        self,
        constraint: IntensionalConstraint,
        item: LexicalItem,
    ) -> bool:
        """Evaluate intensional constraint against item.

        Check if item's properties match the constraint property/operator/value.
        Supports nested property paths (e.g., "features.tense").

        Parameters
        ----------
        constraint : IntensionalConstraint
            Intensional constraint to evaluate.
        item : LexicalItem
            Item to check.

        Returns
        -------
        bool
            True if item satisfies the property constraint.
        """
        # Get the value from the item using property path
        try:
            item_value = self._get_nested_property(item, constraint.property)
        except (AttributeError, KeyError):
            # Property doesn't exist on item
            return False

        # Apply operator
        return self._apply_operator(item_value, constraint.operator, constraint.value)

    def _evaluate_relational(
        self,
        constraint: RelationalConstraint,
        item: LexicalItem,
    ) -> bool:
        """Evaluate relational constraint against item.

        Note: Relational constraints in the current implementation define
        relationships between slots (e.g., slot_a and slot_b must have the
        same/different property values). This type of constraint cannot be
        evaluated on a single item without context of other slots.

        This implementation returns False since we cannot evaluate slot
        relationships with only a single item. In a future implementation,
        this could be extended to support external resource queries.

        Parameters
        ----------
        constraint : RelationalConstraint
            Relational constraint to evaluate.
        item : LexicalItem
            Item to check.

        Returns
        -------
        bool
            Always returns False for single-item evaluation.

        Raises
        ------
        ValueError
            If the constraint requires adapter_registry but none provided.
        """
        # Current relational constraints are about slot relationships,
        # which cannot be evaluated on a single item.
        # This would require context about other slots being filled.
        # For now, return False as we can't evaluate without slot context.
        return False

    def _evaluate_dsl(
        self,
        constraint: DSLConstraint,
        item: LexicalItem,
    ) -> bool:
        """Evaluate DSL constraint against item.

        Parse and evaluate DSL expression using the DSL evaluator
        with item context.

        Parameters
        ----------
        constraint : DSLConstraint
            DSL constraint to evaluate.
        item : LexicalItem
            Item to check.

        Returns
        -------
        bool
            True if DSL expression evaluates to True.

        Raises
        ------
        RuntimeError
            If DSL evaluation fails.
        """
        # Create evaluation context and set variables
        context = EvaluationContext()

        # Register standard library functions
        register_stdlib(context)

        # Set item variables
        context.set_variable("lemma", item.lemma)
        context.set_variable("pos", item.pos)
        context.set_variable("language_code", item.language_code)
        context.set_variable("id", str(item.id))

        # Add features as nested dict and as top-level variables
        if item.features:
            context.set_variable("features", item.features)
            # Also add features as top-level for easier access
            for key, value in item.features.items():
                # Use feature_ prefix to avoid conflicts
                context.set_variable(key, value)

        # Add attributes
        if item.attributes:
            context.set_variable("attributes", item.attributes)

        # Parse and evaluate expression
        try:
            ast_node = parse(constraint.expression)
            result = self._dsl_evaluator.evaluate(ast_node, context)
            # Ensure result is boolean
            return bool(result)
        except Exception as e:
            # Check if this is an undefined variable error
            # If so, treat as False (item doesn't satisfy constraint)
            error_msg = str(e)
            if "Undefined variable" in error_msg or "Undefined function" in error_msg:
                return False
            # Re-raise other errors
            raise RuntimeError(
                f"Failed to evaluate DSL expression '{constraint.expression}': {e}"
            ) from e

    def _get_cache_key(
        self,
        constraint: Constraint,
        item: LexicalItem,
    ) -> tuple[str, str]:
        """Generate cache key for constraint-item pair.

        Create a deterministic cache key from constraint and item.

        Parameters
        ----------
        constraint : Constraint
            Constraint being evaluated.
        item : LexicalItem
            Item being checked.

        Returns
        -------
        tuple[str, str]
            (constraint_hash, item_id) tuple for cache lookup.
        """
        # Hash the constraint by serializing to JSON
        # Use model_dump_json() which handles datetime serialization
        constraint_json = constraint.model_dump_json()
        constraint_hash = hashlib.sha256(constraint_json.encode()).hexdigest()

        return (constraint_hash, str(item.id))

    def _get_nested_property(self, item: LexicalItem, property_path: str) -> Any:
        """Get nested property value from item using dot notation.

        Parameters
        ----------
        item : LexicalItem
            Item to get property from.
        property_path : str
            Property path (e.g., "pos" or "features.tense").

        Returns
        -------
        Any
            The property value.

        Raises
        ------
        AttributeError
            If property doesn't exist on item.
        KeyError
            If nested key doesn't exist in dict.
        """
        parts = property_path.split(".")
        current: Any = item

        for part in parts:
            if isinstance(current, dict):
                current = current[part]  # type: ignore[index]
            else:
                current = getattr(current, part)  # type: ignore[arg-type]

        return current  # type: ignore[return-value]

    def _apply_operator(
        self,
        left_value: Any,
        operator: str,
        right_value: Any,
    ) -> bool:
        """Apply comparison operator.

        Parameters
        ----------
        left_value : Any
            Left-hand side value (from item).
        operator : str
            Comparison operator.
        right_value : Any
            Right-hand side value (from constraint).

        Returns
        -------
        bool
            Result of comparison.
        """
        if operator == "==":
            return left_value == right_value
        elif operator == "!=":
            return left_value != right_value
        elif operator == "in":
            return left_value in right_value
        elif operator == "not in":
            return left_value not in right_value
        elif operator == "<":
            return left_value < right_value
        elif operator == ">":
            return left_value > right_value
        elif operator == "<=":
            return left_value <= right_value
        elif operator == ">=":
            return left_value >= right_value
        else:
            raise ValueError(f"Unsupported operator: {operator}")
