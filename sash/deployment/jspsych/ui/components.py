"""Python helpers for generating UI components for jsPsych experiments.

This module provides functions to generate UI component configurations
from SASH models, inferring widget types from slot constraints.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from sash.items.models import UnfilledSlot
from sash.resources.constraints import (
    Constraint,
    DSLConstraint,
    ExtensionalConstraint,
    IntensionalConstraint,
)


def create_rating_scale(
    scale_min: int,
    scale_max: int,
    labels: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Generate jsPsych rating scale configuration.

    Parameters
    ----------
    scale_min : int
        Minimum value of the scale.
    scale_max : int
        Maximum value of the scale.
    labels : dict[int, str] | None
        Optional labels for specific scale points.

    Returns
    -------
    dict[str, Any]
        Rating scale configuration dictionary.

    Examples
    --------
    >>> config = create_rating_scale(
    ...     1, 7, {1: "Strongly Disagree", 7: "Strongly Agree"}
    ... )
    >>> config["scale_min"]
    1
    >>> config["scale_max"]
    7
    """
    return {
        "scale_min": scale_min,
        "scale_max": scale_max,
        "scale_labels": labels or {},
    }


def create_cloze_fields(
    unfilled_slots: list[UnfilledSlot],
    constraints: dict[UUID, Constraint],
) -> list[dict[str, Any]]:
    """Generate cloze field configurations from slots and constraints.

    Infers widget type from slot constraints:
    - ExtensionalConstraint with allow mode → dropdown with specific items
    - IntensionalConstraint with categorical property → text input with validation
    - DSLConstraint → text input
    - No constraints → free text input
    - RelationalConstraint → text input (requires dynamic validation)

    Parameters
    ----------
    unfilled_slots : list[UnfilledSlot]
        List of unfilled slots in the cloze task.
    constraints : dict[UUID, Constraint]
        Dictionary of constraints keyed by UUID (from slot.constraint_ids).

    Returns
    -------
    list[dict[str, Any]]
        List of field configuration dictionaries with keys:
        - slot_name: Name of the slot
        - position: Token position
        - type: Widget type ("dropdown" or "text")
        - options: List of allowed values (for dropdown)
        - placeholder: Placeholder text
        - validation_pattern: Regex pattern for text validation (optional)
        - validation_message: Message to show on validation failure (optional)

    Examples
    --------
    >>> from sash.items.models import UnfilledSlot
    >>> from sash.resources.constraints import ExtensionalConstraint
    >>> from uuid import uuid4
    >>> constraint_id = uuid4()
    >>> slot = UnfilledSlot(
    ...     slot_name="determiner",
    ...     position=0,
    ...     constraint_ids=[constraint_id]
    ... )
    >>> constraint = ExtensionalConstraint(
    ...     mode="allow",
    ...     items=[uuid4(), uuid4()]
    ... )
    >>> # Note: In real usage, constraint.items would be lexical item UUIDs
    >>> # and we'd need to look up their surface forms
    >>> fields = create_cloze_fields([slot], {constraint_id: constraint})
    >>> len(fields)
    1
    """
    fields: list[dict[str, Any]] = []

    for slot in unfilled_slots:
        field_config: dict[str, Any] = {
            "slot_name": slot.slot_name,
            "position": slot.position,
            "type": "text",  # Default to text input
            "options": [],
            "placeholder": slot.slot_name,
        }

        # Analyze constraints to determine widget type and options
        for constraint_id in slot.constraint_ids:
            if constraint_id not in constraints:
                continue

            constraint = constraints[constraint_id]

            if isinstance(constraint, ExtensionalConstraint):
                # Extensional constraint: explicit allow/deny list
                # For "allow" mode, render as dropdown with lexical items
                # For "deny" mode, render as text with validation
                if constraint.mode == "allow" and len(constraint.items) > 0:
                    field_config["type"] = "dropdown"
                    # Note: In a real implementation, we'd need to:
                    # 1. Look up the lexical items by UUID from constraint.items
                    # 2. Extract their surface forms (lemmas or inflected forms)
                    # 3. Add them to field_config["options"]
                    # For now, we mark it as dropdown type but leave options empty
                    # The JavaScript plugin will need to populate this from metadata
                    field_config["extensional_item_ids"] = [
                        str(item_id) for item_id in constraint.items
                    ]

            elif isinstance(constraint, IntensionalConstraint):
                # Intensional constraint: property-based filtering
                # We render as text input with optional pattern validation
                if constraint.operator == "==" and isinstance(constraint.value, str):
                    # Exact match constraint - could use autocomplete or validation
                    field_config["validation_pattern"] = f"^{constraint.value}$"
                    field_config["validation_message"] = (
                        f"Must be exactly '{constraint.value}'"
                    )
                # For other operators, we just use text input

            elif isinstance(constraint, DSLConstraint):
                # DSL constraint: complex expression
                # We can't easily infer UI from this, so use text input
                field_config["dsl_expression"] = constraint.expression

            else:
                # Must be RelationalConstraint (the only remaining type)
                # Relational constraint: requires relationship with another slot
                # Can't validate this in isolation, needs dynamic checking
                field_config["relational_constraint"] = {
                    "slot_a": constraint.slot_a,
                    "slot_b": constraint.slot_b,
                    "relation": constraint.relation,
                    "property": constraint.property,
                }

        fields.append(field_config)

    return fields


def create_forced_choice_config(
    alternatives: list[str],
    randomize_position: bool = True,
    enable_keyboard: bool = True,
) -> dict[str, Any]:
    """Generate forced choice configuration.

    Parameters
    ----------
    alternatives : list[str]
        List of alternative options to choose from.
    randomize_position : bool
        Whether to randomize left/right position.
    enable_keyboard : bool
        Whether to enable keyboard responses.

    Returns
    -------
    dict[str, Any]
        Forced choice configuration dictionary.

    Examples
    --------
    >>> config = create_forced_choice_config(["Option A", "Option B"])
    >>> config["randomize_position"]
    True
    >>> len(config["alternatives"])
    2
    """
    return {
        "alternatives": alternatives,
        "randomize_position": randomize_position,
        "enable_keyboard": enable_keyboard,
    }


def infer_widget_type(
    constraint_ids: list[UUID],
    constraints: dict[UUID, Constraint],
) -> str:
    """Infer UI widget type from slot constraints.

    Analyzes the constraint types to determine the most appropriate
    UI widget for collecting user input.

    Widget type inference logic:
    - ExtensionalConstraint with allow mode → "dropdown"
    - ExtensionalConstraint with deny mode → "text"
    - IntensionalConstraint → "text"
    - DSLConstraint → "text"
    - RelationalConstraint → "text"
    - No constraints → "text"

    Parameters
    ----------
    constraint_ids : list[UUID]
        List of constraint IDs for the slot.
    constraints : dict[UUID, Constraint]
        Dictionary of constraint objects keyed by UUID.

    Returns
    -------
    str
        Widget type: "dropdown" or "text".

    Examples
    --------
    >>> from sash.resources.constraints import ExtensionalConstraint
    >>> from uuid import uuid4
    >>> constraint_id = uuid4()
    >>> constraint = ExtensionalConstraint(mode="allow", items=[uuid4()])
    >>> widget = infer_widget_type([constraint_id], {constraint_id: constraint})
    >>> widget
    'dropdown'
    >>> widget2 = infer_widget_type([], {})
    >>> widget2
    'text'
    """
    if not constraint_ids:
        return "text"

    # Check each constraint
    for constraint_id in constraint_ids:
        if constraint_id not in constraints:
            continue

        constraint = constraints[constraint_id]

        # Extensional constraint with allow mode → dropdown
        if isinstance(constraint, ExtensionalConstraint):
            if constraint.mode == "allow" and len(constraint.items) > 0:
                return "dropdown"

    # Default to text input
    return "text"
