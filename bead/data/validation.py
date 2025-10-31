"""Validation utilities for data integrity checks.

This module provides validation functions beyond Pydantic's built-in validation,
including file validation, reference validation, and provenance chain validation.
"""

from __future__ import annotations

from pathlib import Path
from typing import get_type_hints
from uuid import UUID

from pydantic import BaseModel, Field, ValidationError

from bead.data.metadata import MetadataTracker


class ValidationReport(BaseModel):
    """Report of validation results.

    A lightweight model for collecting and reporting validation results,
    including errors, warnings, and statistics about validated objects.

    Attributes
    ----------
    valid : bool
        Overall validation status (False if any errors)
    errors : list[str]
        List of error messages (default: empty list)
    warnings : list[str]
        List of warning messages (default: empty list)
    object_count : int
        Number of objects validated (default: 0)

    Examples
    --------
    >>> report = ValidationReport(valid=True)
    >>> report.add_error("Invalid field")
    >>> report.valid
    False
    >>> report.has_errors()
    True
    >>> len(report.errors)
    1
    """

    valid: bool
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    object_count: int = 0

    def add_error(self, message: str) -> None:
        """Add an error message and set valid to False.

        Parameters
        ----------
        message : str
            Error message to add

        Examples
        --------
        >>> report = ValidationReport(valid=True)
        >>> report.add_error("Something went wrong")
        >>> report.valid
        False
        >>> "Something went wrong" in report.errors
        True
        """
        self.errors.append(message)
        self.valid = False

    def add_warning(self, message: str) -> None:
        """Add a warning message.

        Warnings do not affect the valid status.

        Parameters
        ----------
        message : str
            Warning message to add

        Examples
        --------
        >>> report = ValidationReport(valid=True)
        >>> report.add_warning("This might be an issue")
        >>> report.valid
        True
        >>> report.has_warnings()
        True
        """
        self.warnings.append(message)

    def has_errors(self) -> bool:
        """Check if report has any errors.

        Returns
        -------
        bool
            True if errors list is non-empty

        Examples
        --------
        >>> report = ValidationReport(valid=True)
        >>> report.has_errors()
        False
        >>> report.add_error("error")
        >>> report.has_errors()
        True
        """
        return len(self.errors) > 0

    def has_warnings(self) -> bool:
        """Check if report has any warnings.

        Returns
        -------
        bool
            True if warnings list is non-empty

        Examples
        --------
        >>> report = ValidationReport(valid=True)
        >>> report.has_warnings()
        False
        >>> report.add_warning("warning")
        >>> report.has_warnings()
        True
        """
        return len(self.warnings) > 0


def validate_jsonlines_file(
    path: Path, model_class: type[BaseModel], strict: bool = True
) -> ValidationReport:
    """Validate JSONLines file against Pydantic model schema.

    Reads and validates each line in a JSONLines file against the provided
    model class. Empty lines are skipped.

    Parameters
    ----------
    path : Path
        Path to JSONLines file to validate
    model_class : type[BaseModel]
        Pydantic model class to validate against
    strict : bool, optional
        If True, stop at first error. If False, collect all errors (default: True)

    Returns
    -------
    ValidationReport
        Validation report with results

    Examples
    --------
    >>> from pathlib import Path
    >>> from bead.data.base import BeadBaseModel
    >>> class TestModel(BeadBaseModel):
    ...     name: str
    >>> # Validate file
    >>> report = validate_jsonlines_file(
    ...     Path("data.jsonl"), TestModel
    ... )  # doctest: +SKIP
    >>> report.valid
    True
    """
    report = ValidationReport(valid=True)

    # Check if file exists
    if not path.exists():
        report.add_error(f"File not found: {path}")
        return report

    try:
        # Try to read the file
        with path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:  # Skip empty lines
                    continue

                try:
                    # Try to parse and validate
                    model_class.model_validate_json(line)
                    report.object_count += 1
                except ValidationError as e:
                    error_msg = f"Line {line_num}: Validation error - {e}"
                    report.add_error(error_msg)
                    if strict:
                        return report
                except Exception as e:
                    error_msg = f"Line {line_num}: Parse error - {e}"
                    report.add_error(error_msg)
                    if strict:
                        return report

    except OSError as e:
        report.add_error(f"Failed to read file: {e}")

    return report


def validate_uuid_references(
    objects: list[BaseModel], reference_pool: dict[UUID, BaseModel]
) -> ValidationReport:
    """Validate that UUID references point to existing objects.

    Checks all UUID fields in objects to ensure they reference valid objects
    in the reference pool. Supports both single UUID fields and list[UUID] fields.

    Parameters
    ----------
    objects : list[BaseModel]
        List of objects to validate
    reference_pool : dict[UUID, BaseModel]
        Dictionary of valid UUIDs to objects

    Returns
    -------
    ValidationReport
        Validation report with results

    Examples
    --------
    >>> from uuid import uuid4
    >>> from bead.data.base import BeadBaseModel
    >>> class Item(BeadBaseModel):
    ...     name: str
    >>> items = [Item(name="test")]
    >>> pool = {items[0].id: items[0]}
    >>> report = validate_uuid_references(items, pool)
    >>> report.valid
    True
    """
    report = ValidationReport(valid=True)
    report.object_count = len(objects)

    for obj in objects:
        # Get type hints for the object
        try:
            type_hints = get_type_hints(type(obj))
        except Exception:
            # If we can't get type hints, skip this object
            continue

        # Check each field
        for field_name, field_type in type_hints.items():
            # Skip 'id' field - it's the object's own ID, not a reference
            if field_name == "id":
                continue

            # Convert type to string for checking
            type_str = str(field_type)

            # Check if field contains UUID
            if "UUID" not in type_str:
                continue

            # Get field value
            try:
                field_value = getattr(obj, field_name)
            except AttributeError:
                continue

            # Check if it's a list of UUIDs
            if "list" in type_str.lower() or "List" in type_str:
                if isinstance(field_value, list):
                    for item in field_value:  # pyright: ignore[reportUnknownVariableType]
                        if not isinstance(item, UUID):
                            continue
                        if item not in reference_pool:
                            # Get object ID for error message
                            obj_id = getattr(obj, "id", "unknown")
                            report.add_error(
                                f"Object {obj_id}: "
                                f"Field '{field_name}' references "
                                f"missing UUID {item}"
                            )
            # Single UUID field
            elif isinstance(field_value, UUID):
                if field_value not in reference_pool:
                    # Get object ID for error message
                    obj_id = getattr(obj, "id", "unknown")
                    report.add_error(
                        f"Object {obj_id}: "
                        f"Field '{field_name}' references "
                        f"missing UUID {field_value}"
                    )

    return report


def validate_provenance_chain(
    metadata: MetadataTracker, repository: dict[UUID, BaseModel]
) -> ValidationReport:
    """Validate provenance chain references are valid.

    Checks that all parent_id references in the provenance chain exist in the
    repository and that parent_type matches the actual type.

    Parameters
    ----------
    metadata : MetadataTracker
        Metadata tracker with provenance chain to validate
    repository : dict[UUID, BaseModel]
        Dictionary of valid UUIDs to objects

    Returns
    -------
    ValidationReport
        Validation report with results

    Examples
    --------
    >>> from uuid import uuid4
    >>> from bead.data.base import BeadBaseModel
    >>> from bead.data.metadata import MetadataTracker
    >>> class Template(BeadBaseModel):
    ...     name: str
    >>> template = Template(name="test")
    >>> metadata = MetadataTracker()
    >>> metadata.add_provenance(template.id, "Template", "filled_from")
    >>> repo = {template.id: template}
    >>> report = validate_provenance_chain(metadata, repo)
    >>> report.valid
    True
    """
    report = ValidationReport(valid=True)
    report.object_count = len(metadata.provenance)

    for record in metadata.provenance:
        # Check if parent exists
        if record.parent_id not in repository:
            report.add_error(
                f"Provenance record references missing parent: {record.parent_id}"
            )
            continue

        # Check if parent_type matches actual type
        parent_obj = repository[record.parent_id]
        actual_type = type(parent_obj).__name__

        if record.parent_type != actual_type:
            report.add_error(
                f"Provenance record for {record.parent_id}: "
                f"Expected type '{record.parent_type}', got '{actual_type}'"
            )

    return report
