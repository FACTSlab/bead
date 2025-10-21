"""JavaScript randomizer code generator from OrderingConstraints.

This module converts Python OrderingConstraint models into JavaScript code
that performs constraint-aware trial randomization at jsPsych runtime. This
enables per-participant randomization while satisfying all ordering constraints.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from uuid import UUID

from jinja2 import Environment, FileSystemLoader

from sash.lists.constraints import OrderingConstraint


def generate_randomizer_function(
    item_ids: list[UUID],
    constraints: list[OrderingConstraint],
    metadata: dict[UUID, dict[str, Any]],
) -> str:
    """Generate JavaScript code for constraint-aware trial randomization.

    This function converts OrderingConstraints into JavaScript code that can
    randomize trial order at runtime while satisfying all constraints. The
    generated code uses seeded randomization for reproducibility and rejection
    sampling to satisfy constraints.

    Parameters
    ----------
    item_ids : list[UUID]
        List of item IDs included in the experiment.
    constraints : list[OrderingConstraint]
        Ordering constraints to enforce.
    metadata : dict[UUID, dict[str, Any]]
        Item metadata needed for constraint checking (keyed by item UUID).

    Returns
    -------
    str
        JavaScript code implementing randomizeTrials() function.

    Examples
    --------
    >>> from uuid import UUID
    >>> item1 = UUID("12345678-1234-5678-1234-567812345678")
    >>> item2 = UUID("87654321-4321-8765-4321-876543218765")
    >>> constraint = OrderingConstraint(
    ...     no_adjacent_property="item_metadata.condition"
    ... )
    >>> metadata = {
    ...     item1: {"condition": "A"},
    ...     item2: {"condition": "B"}
    ... }
    >>> js_code = generate_randomizer_function(
    ...     [item1, item2],
    ...     [constraint],
    ...     metadata
    ... )
    >>> "function randomizeTrials" in js_code
    True
    >>> "checkNoAdjacentConstraints" in js_code
    True
    """
    # Prepare template context
    context = _prepare_template_context(item_ids, constraints, metadata)

    # Load and render template
    template_dir = Path(__file__).parent / "templates"
    env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = env.get_template("randomizer.js.template")

    return template.render(**context)


def _prepare_template_context(
    item_ids: list[UUID],
    constraints: list[OrderingConstraint],
    metadata: dict[UUID, dict[str, Any]],
) -> dict[str, Any]:
    """Prepare Jinja2 template context from constraints.

    Parameters
    ----------
    item_ids : list[UUID]
        Item IDs in the experiment.
    constraints : list[OrderingConstraint]
        Ordering constraints.
    metadata : dict[UUID, dict[str, Any]]
        Item metadata.

    Returns
    -------
    dict[str, Any]
        Template context for Jinja2 rendering.
    """
    context: dict[str, Any] = {
        "metadata_json": _serialize_metadata(metadata),
        "has_practice_items": False,
        "practice_property": "",
        "has_blocking": False,
        "block_property": "",
        "randomize_within_blocks": True,
        "has_precedence": False,
        "precedence_pairs_json": "[]",
        "has_no_adjacent": False,
        "no_adjacent_property": "",
        "has_distance": False,
        "distance_constraints_json": "[]",
    }

    # Combine all constraints (multiple OrderingConstraints can be active)
    for constraint in constraints:
        # Practice items
        if constraint.practice_item_property:
            context["has_practice_items"] = True
            # Extract property name from path
            # (e.g., "item_metadata.is_practice" -> "is_practice")
            context["practice_property"] = _extract_property_name(
                constraint.practice_item_property
            )

        # Blocking
        if constraint.block_by_property:
            context["has_blocking"] = True
            context["block_property"] = _extract_property_name(
                constraint.block_by_property
            )
            context["randomize_within_blocks"] = constraint.randomize_within_blocks

        # Precedence
        if constraint.precedence_pairs:
            context["has_precedence"] = True
            # Convert UUID pairs to string pairs for JSON
            pairs = [[str(a), str(b)] for a, b in constraint.precedence_pairs]
            context["precedence_pairs_json"] = json.dumps(pairs)

        # No-adjacency
        if constraint.no_adjacent_property:
            context["has_no_adjacent"] = True
            # Extract property name since metadata is already extracted
            context["no_adjacent_property"] = _extract_property_name(
                constraint.no_adjacent_property
            )

        # Distance constraints
        if constraint.min_distance or constraint.max_distance:
            context["has_distance"] = True
            # Generate distance constraints for all item pairs
            distance_constraints = _generate_distance_constraints(
                item_ids, constraint, metadata
            )
            context["distance_constraints_json"] = json.dumps(distance_constraints)

    return context


def _serialize_metadata(metadata: dict[UUID, dict[str, Any]]) -> str:
    """Serialize metadata dictionary to JSON.

    Converts UUID keys to strings for JSON serialization.

    Parameters
    ----------
    metadata : dict[UUID, dict[str, Any]]
        Item metadata with UUID keys.

    Returns
    -------
    str
        JSON string of metadata.
    """
    # Convert UUID keys to strings
    serializable = {str(k): v for k, v in metadata.items()}
    return json.dumps(serializable, indent=2)


def _extract_property_name(property_path: str) -> str:
    """Extract final property name from dot-notation path.

    Parameters
    ----------
    property_path : str
        Dot-notation property path (e.g., "item_metadata.is_practice").

    Returns
    -------
    str
        Final property name (e.g., "is_practice").

    Examples
    --------
    >>> _extract_property_name("item_metadata.is_practice")
    'is_practice'
    >>> _extract_property_name("condition")
    'condition'
    """
    parts = property_path.split(".")
    return parts[-1]


def _generate_distance_constraints(
    item_ids: list[UUID],
    constraint: OrderingConstraint,
    metadata: dict[UUID, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Generate distance constraints for all relevant item pairs.

    Distance constraints are applied to items that share the same value
    for the no_adjacent_property (if specified).

    Parameters
    ----------
    item_ids : list[UUID]
        Item IDs in the experiment.
    constraint : OrderingConstraint
        Ordering constraint with distance specifications.
    metadata : dict[UUID, dict[str, Any]]
        Item metadata.

    Returns
    -------
    list[dict[str, Any]]
        List of distance constraint specifications.
    """
    distance_constraints: list[dict[str, Any]] = []

    # Group items by property value if no_adjacent_property is set
    if constraint.no_adjacent_property:
        property_path = constraint.no_adjacent_property
        # Extract just the property name from the path
        # (e.g., "condition" from "item_metadata.condition")
        # because metadata is already extracted from items
        property_name = _extract_property_name(property_path)

        # Group items by property value
        groups: dict[Any, list[UUID]] = {}
        for item_id in item_ids:
            item_meta = metadata.get(item_id, {})
            value = item_meta.get(property_name)

            if value is not None:
                if value not in groups:
                    groups[value] = []
                groups[value].append(item_id)

        # Create pairwise distance constraints within each group
        for _value, item_group in groups.items():
            if len(item_group) > 1:
                # Create constraints for all pairs in this group
                for i, item1 in enumerate(item_group):
                    for item2 in item_group[i + 1 :]:
                        distance_constraints.append(
                            {
                                "item1_id": str(item1),
                                "item2_id": str(item2),
                                "min_distance": constraint.min_distance,
                                "max_distance": constraint.max_distance,
                            }
                        )

    return distance_constraints


def _get_nested_property(obj: dict[str, Any], path: str) -> Any:  # pyright: ignore[reportUnusedFunction]
    """Get nested property from dictionary using dot notation.

    Parameters
    ----------
    obj : dict[str, Any]
        Object to query.
    path : str
        Property path (e.g., "item_metadata.condition").

    Returns
    -------
    Any
        Property value or None if not found.

    Examples
    --------
    >>> obj = {"item_metadata": {"condition": "A"}}
    >>> _get_nested_property(obj, "item_metadata.condition")
    'A'
    >>> _get_nested_property(obj, "missing.path") is None
    True
    """
    parts = path.split(".")
    current = obj

    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]

    return current
