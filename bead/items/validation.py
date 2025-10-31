"""Validation utilities for constructed items.

This module provides validation functions to ensure constructed items
meet all requirements and contain complete, valid data.
"""

from __future__ import annotations

from bead.items.models import Item, ItemTemplate, ModelOutput


def validate_item(item: Item, item_template: ItemTemplate) -> list[str]:
    """Validate a constructed item against its template.

    Check that the item has all required fields, references valid templates,
    has consistent constraint satisfaction, and contains valid model outputs.

    Parameters
    ----------
    item : Item
        Item to validate.
    item_template : ItemTemplate
        Template the item was constructed from.

    Returns
    -------
    list[str]
        List of validation error messages. Empty list if valid.

    Examples
    --------
    >>> errors = validate_item(item, template)
    >>> if errors:
    ...     print(f"Item is invalid: {errors}")
    >>> else:
    ...     print("Item is valid")
    """
    errors: list[str] = []

    # Check item_template_id matches
    if item.item_template_id != item_template.id:
        errors.append(
            f"Item template ID mismatch: {item.item_template_id} != {item_template.id}"
        )

    # Check all elements are rendered
    expected_elements = {elem.element_name for elem in item_template.elements}
    actual_elements = set(item.rendered_elements.keys())

    missing = expected_elements - actual_elements
    if missing:
        errors.append(f"Missing rendered elements: {missing}")

    extra = actual_elements - expected_elements
    if extra:
        errors.append(f"Extra rendered elements: {extra}")

    # Check all constraints are evaluated
    expected_constraints = set(item_template.constraints)
    actual_constraints = set(item.constraint_satisfaction.keys())

    missing_constraints = expected_constraints - actual_constraints
    if missing_constraints:
        errors.append(f"Missing constraint evaluations: {missing_constraints}")

    # Check model outputs are valid
    for output in item.model_outputs:
        output_errors = validate_model_output(output)
        errors.extend(output_errors)

    return errors


def validate_model_output(output: ModelOutput) -> list[str]:
    """Validate a model output.

    Check that the model output has all required fields and valid values.

    Parameters
    ----------
    output : ModelOutput
        Model output to validate.

    Returns
    -------
    list[str]
        List of validation error messages. Empty list if valid.

    Examples
    --------
    >>> errors = validate_model_output(output)
    >>> if not errors:
    ...     print("Model output is valid")
    """
    errors: list[str] = []

    # Check required fields are not empty
    if not output.model_name or not output.model_name.strip():
        errors.append("Model output has empty model_name")

    if not output.operation or not output.operation.strip():
        errors.append("Model output has empty operation")

    if not output.cache_key or not output.cache_key.strip():
        errors.append("Model output has empty cache_key")

    # Check operation-specific output structure
    if output.operation == "nli":
        # NLI should return dict with entailment/neutral/contradiction
        if not isinstance(output.output, dict):
            errors.append(f"NLI output should be dict, got {type(output.output)}")
        else:
            expected_keys = {"entailment", "neutral", "contradiction"}
            actual_keys = set(output.output.keys())  # type: ignore[union-attr]
            if actual_keys != expected_keys:
                errors.append(
                    f"NLI output keys mismatch: expected {expected_keys}, "
                    f"got {actual_keys}"
                )

    elif output.operation in ("log_probability", "perplexity", "similarity"):
        # These should return numeric values
        if not isinstance(output.output, (int, float)):
            errors.append(
                f"{output.operation} output should be numeric, "
                f"got {type(output.output)}"
            )

    elif output.operation == "embedding":
        # Should return list or array
        if not isinstance(output.output, (list, dict)):
            # dict could be serialized ndarray
            errors.append(
                f"Embedding output should be list/array, got {type(output.output)}"
            )

    return errors


def validate_constraint_satisfaction(
    item: Item, item_template: ItemTemplate
) -> list[str]:
    """Validate constraint satisfaction consistency.

    Check that all constraints in the template have been evaluated and
    that the results are boolean values.

    Parameters
    ----------
    item : Item
        Item to validate.
    item_template : ItemTemplate
        Template with constraints.

    Returns
    -------
    list[str]
        List of validation error messages. Empty list if valid.

    Examples
    --------
    >>> errors = validate_constraint_satisfaction(item, template)
    >>> if not errors:
    ...     print("Constraint satisfaction is valid")
    """
    errors: list[str] = []

    # Check all template constraints are evaluated
    for constraint_id in item_template.constraints:
        if constraint_id not in item.constraint_satisfaction:
            errors.append(f"Constraint {constraint_id} not evaluated")
        else:
            # Check value is boolean
            value = item.constraint_satisfaction[constraint_id]
            if type(value) is not bool:
                errors.append(
                    f"Constraint {constraint_id} satisfaction should be bool, "
                    f"got {type(value)}"
                )

    return errors


def validate_metadata_completeness(item: Item) -> list[str]:
    """Validate that item metadata is complete.

    Check that the item has all expected metadata fields populated.
    Since Item inherits from BeadBaseModel, id, created_at, and modified_at
    are always present. This function is kept for consistency and future
    extensibility.

    Parameters
    ----------
    item : Item
        Item to validate.

    Returns
    -------
    list[str]
        List of validation error messages. Empty list if valid.

    Examples
    --------
    >>> errors = validate_metadata_completeness(item)
    >>> if not errors:
    ...     print("Metadata is complete")
    """
    errors: list[str] = []

    # Check base model fields (from BeadBaseModel)
    # These are always present due to Pydantic model initialization,
    # but we check for completeness
    if not hasattr(item, "id"):
        errors.append("Item missing id field")  # pragma: no cover

    if not hasattr(item, "created_at"):
        errors.append("Item missing created_at timestamp")  # pragma: no cover

    if not hasattr(item, "modified_at"):
        errors.append("Item missing modified_at timestamp")  # pragma: no cover

    return errors


def item_passes_all_constraints(item: Item) -> bool:
    """Check if item satisfies all constraints.

    Convenience function to check if all constraints are satisfied.

    Parameters
    ----------
    item : Item
        Item to check.

    Returns
    -------
    bool
        True if all constraints satisfied, False otherwise.

    Examples
    --------
    >>> if item_passes_all_constraints(item):
    ...     print("Item is valid")
    """
    return all(item.constraint_satisfaction.values())
