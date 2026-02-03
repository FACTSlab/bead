"""Metadata specification for participant attributes.

This module provides FieldSpec and ParticipantMetadataSpec for defining
configurable metadata fields with validation constraints (allowed values, ranges).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from bead.data.range import Range

if TYPE_CHECKING:
    from bead.deployment.jspsych.config import DemographicsConfig


def _empty_field_spec_list() -> list[FieldSpec]:
    """Return empty field spec list."""
    return []


class FieldSpec(BaseModel):
    """Specification for a single metadata field.

    Defines the constraints and display properties for a participant metadata
    field. Used for validation and for generating demographics forms.

    Attributes
    ----------
    name : str
        Field name (e.g., "age", "education"). Must be valid Python identifier.
    field_type : Literal["int", "float", "str", "bool"]
        Data type for the field.
    required : bool
        Whether this field is required (default: False).
    allowed_values : list[str | int | float | bool] | None
        Exhaustive list of allowed values (for categorical fields).
        If None, any value of the correct type is accepted.
    range : Range[int] | Range[float] | None
        Numeric range constraint (for int/float fields).
    label : str | None
        Display label for forms. If None, uses name with underscores replaced.
    description : str | None
        Help text / description for the field.

    Examples
    --------
    >>> age_spec = FieldSpec(
    ...     name="age",
    ...     field_type="int",
    ...     required=True,
    ...     range=Range[int](min=18, max=100),
    ...     label="Age",
    ...     description="Your age in years"
    ... )
    >>> education_spec = FieldSpec(
    ...     name="education",
    ...     field_type="str",
    ...     required=True,
    ...     allowed_values=["high_school", "bachelors", "masters", "phd"],
    ...     label="Highest Education Level"
    ... )
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str
    field_type: Literal["int", "float", "str", "bool"]
    required: bool = False
    allowed_values: list[str | int | float | bool] | None = None
    range: Range[int] | Range[float] | None = None
    label: str | None = None
    description: str | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate field name is non-empty and valid identifier.

        Parameters
        ----------
        v : str
            Field name to validate.

        Returns
        -------
        str
            Validated field name.

        Raises
        ------
        ValueError
            If field name is empty or not a valid Python identifier.
        """
        if not v or not v.strip():
            raise ValueError("Field name cannot be empty")
        v = v.strip()
        if not v.isidentifier():
            raise ValueError(f"Field name must be valid Python identifier: {v}")
        return v

    @model_validator(mode="after")
    def validate_constraints(self) -> FieldSpec:
        """Validate that constraints are consistent with field_type.

        Returns
        -------
        FieldSpec
            The validated FieldSpec instance.

        Raises
        ------
        ValueError
            If constraints are inconsistent with field_type.
        """
        # Range constraints only valid for numeric types
        if self.range is not None:
            if self.field_type not in ("int", "float"):
                raise ValueError(
                    f"range constraint only valid for numeric types, "
                    f"not {self.field_type}"
                )

        # Validate allowed_values types match field_type
        if self.allowed_values is not None:
            expected_type: type | tuple[type, ...]
            if self.field_type == "int":
                expected_type = int
            elif self.field_type == "float":
                expected_type = (int, float)
            elif self.field_type == "str":
                expected_type = str
            else:  # bool
                expected_type = bool

            for val in self.allowed_values:
                if not isinstance(val, expected_type):
                    raise ValueError(
                        f"allowed_values item {val!r} does not match "
                        f"field_type {self.field_type}"
                    )

        return self

    def validate_value(self, value: str | int | float | bool | None) -> bool:
        """Check if a value satisfies this field's constraints.

        Parameters
        ----------
        value : str | int | float | bool | None
            Value to validate.

        Returns
        -------
        bool
            True if value is valid, False otherwise.

        Examples
        --------
        >>> spec = FieldSpec(
        ...     name="age",
        ...     field_type="int",
        ...     range=Range[int](min=18, max=100)
        ... )
        >>> spec.validate_value(25)
        True
        >>> spec.validate_value(10)
        False
        """
        if value is None:
            return not self.required

        # Type check
        expected_type: type | tuple[type, ...]
        if self.field_type == "int":
            expected_type = int
        elif self.field_type == "float":
            expected_type = (int, float)
        elif self.field_type == "str":
            expected_type = str
        else:  # bool
            expected_type = bool

        if not isinstance(value, expected_type):
            return False

        # Allowed values check
        if self.allowed_values is not None and value not in self.allowed_values:
            return False

        # Range check
        if self.range is not None and isinstance(value, (int, float)):
            if not self.range.contains(value):  # type: ignore[arg-type]
                return False

        return True

    def get_display_label(self) -> str:
        """Get display label for forms.

        Returns
        -------
        str
            The label if set, otherwise name with underscores replaced by spaces
            and title-cased.

        Examples
        --------
        >>> spec = FieldSpec(name="native_speaker", field_type="bool")
        >>> spec.get_display_label()
        'Native Speaker'
        >>> spec = FieldSpec(name="age", field_type="int", label="Your Age")
        >>> spec.get_display_label()
        'Your Age'
        """
        if self.label:
            return self.label
        return self.name.replace("_", " ").title()


class ParticipantMetadataSpec(BaseModel):
    """Specification for participant metadata schema.

    Defines the allowed fields and their constraints for participant
    metadata. Used to validate participant data on ingestion and to
    generate demographics forms for experiments.

    Attributes
    ----------
    name : str
        Name of this specification (e.g., "prolific_demographics").
    version : str
        Version string for this spec.
    fields : list[FieldSpec]
        List of field specifications.

    Examples
    --------
    >>> spec = ParticipantMetadataSpec(
    ...     name="standard_demographics",
    ...     version="1.0.0",
    ...     fields=[
    ...         FieldSpec(
    ...             name="age",
    ...             field_type="int",
    ...             range=Range[int](min=18, max=100)
    ...         ),
    ...         FieldSpec(
    ...             name="education",
    ...             field_type="str",
    ...             allowed_values=["high_school", "bachelors", "masters", "phd"]
    ...         ),
    ...         FieldSpec(name="native_speaker", field_type="bool", required=True),
    ...     ]
    ... )
    >>> spec.get_field("age").range.min
    18
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str = "1.0.0"
    fields: list[FieldSpec] = Field(default_factory=_empty_field_spec_list)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate spec name is non-empty.

        Parameters
        ----------
        v : str
            Spec name to validate.

        Returns
        -------
        str
            Validated spec name.

        Raises
        ------
        ValueError
            If name is empty.
        """
        if not v or not v.strip():
            raise ValueError("Spec name cannot be empty")
        return v.strip()

    @field_validator("fields")
    @classmethod
    def validate_unique_field_names(cls, v: list[FieldSpec]) -> list[FieldSpec]:
        """Validate all field names are unique.

        Parameters
        ----------
        v : list[FieldSpec]
            List of field specs to validate.

        Returns
        -------
        list[FieldSpec]
            Validated list of field specs.

        Raises
        ------
        ValueError
            If duplicate field names found.
        """
        names = [f.name for f in v]
        if len(names) != len(set(names)):
            duplicates = [n for n in names if names.count(n) > 1]
            raise ValueError(f"Duplicate field names: {set(duplicates)}")
        return v

    def get_field(self, name: str) -> FieldSpec | None:
        """Get a field specification by name.

        Parameters
        ----------
        name : str
            Field name to look up.

        Returns
        -------
        FieldSpec | None
            The field spec if found, None otherwise.

        Examples
        --------
        >>> spec = ParticipantMetadataSpec(
        ...     name="test",
        ...     fields=[FieldSpec(name="age", field_type="int")]
        ... )
        >>> spec.get_field("age").field_type
        'int'
        >>> spec.get_field("unknown") is None
        True
        """
        for field in self.fields:
            if field.name == name:
                return field
        return None

    def get_required_fields(self) -> list[FieldSpec]:
        """Get all required field specifications.

        Returns
        -------
        list[FieldSpec]
            List of required fields.

        Examples
        --------
        >>> spec = ParticipantMetadataSpec(
        ...     name="test",
        ...     fields=[
        ...         FieldSpec(name="age", field_type="int", required=True),
        ...         FieldSpec(name="nickname", field_type="str", required=False),
        ...     ]
        ... )
        >>> [f.name for f in spec.get_required_fields()]
        ['age']
        """
        return [f for f in self.fields if f.required]

    def validate_metadata(
        self, metadata: dict[str, str | int | float | bool | None]
    ) -> tuple[bool, list[str]]:
        """Validate metadata against this specification.

        Parameters
        ----------
        metadata : dict[str, str | int | float | bool | None]
            Metadata dictionary to validate.

        Returns
        -------
        tuple[bool, list[str]]
            (is_valid, list of error messages). Empty list if valid.

        Examples
        --------
        >>> spec = ParticipantMetadataSpec(
        ...     name="test",
        ...     fields=[
        ...         FieldSpec(name="age", field_type="int", required=True),
        ...     ]
        ... )
        >>> spec.validate_metadata({"age": 25})
        (True, [])
        >>> spec.validate_metadata({})
        (False, ['Missing required field: age'])
        """
        errors: list[str] = []

        # Check required fields
        for field in self.get_required_fields():
            if field.name not in metadata or metadata[field.name] is None:
                errors.append(f"Missing required field: {field.name}")

        # Validate each provided field
        for key, value in metadata.items():
            field_spec = self.get_field(key)
            if field_spec is None:
                # Allow arbitrary fields not in spec (for flexibility)
                continue
            if not field_spec.validate_value(value):
                range_str = ""
                if field_spec.range is not None:
                    range_str = (
                        f", range=[{field_spec.range.min}, {field_spec.range.max}]"
                    )
                allowed_str = ""
                if field_spec.allowed_values is not None:
                    allowed_str = f", allowed={field_spec.allowed_values}"
                errors.append(
                    f"Invalid value for {key}: {value!r} "
                    f"(expected {field_spec.field_type}{range_str}{allowed_str})"
                )

        return len(errors) == 0, errors

    def to_demographics_config(self) -> DemographicsConfig:
        """Convert this spec to a DemographicsConfig for deployment.

        Creates a demographics form configuration that can be used in
        experiment deployment to collect participant data.

        Returns
        -------
        DemographicsConfig
            Demographics configuration for jsPsych deployment.

        Examples
        --------
        >>> spec = ParticipantMetadataSpec(
        ...     name="test",
        ...     fields=[
        ...         FieldSpec(name="age", field_type="int", required=True),
        ...     ]
        ... )
        >>> config = spec.to_demographics_config()  # doctest: +SKIP
        >>> config.enabled
        True
        """
        from bead.deployment.jspsych.config import (  # noqa: PLC0415
            DemographicsConfig,
            DemographicsFieldConfig,
        )

        fields: list[DemographicsFieldConfig] = []
        for field in self.fields:
            # Map field_type to form field_type
            form_field_type: Literal["text", "number", "dropdown", "radio", "checkbox"]
            if field.field_type == "int":
                form_field_type = "number"
            elif field.field_type == "float":
                form_field_type = "number"
            elif field.field_type == "bool":
                form_field_type = "checkbox"
            elif field.allowed_values is not None:
                # Categorical string with options
                form_field_type = "dropdown"
            else:
                form_field_type = "text"

            # Convert allowed_values to string options for dropdown
            options: list[str] | None = None
            if field.allowed_values is not None:
                options = [str(v) for v in field.allowed_values]

            fields.append(
                DemographicsFieldConfig(
                    name=field.name,
                    field_type=form_field_type,
                    label=field.get_display_label(),
                    required=field.required,
                    options=options,
                    range=field.range,
                    help_text=field.description,
                )
            )

        return DemographicsConfig(
            enabled=True,
            title="Participant Information",
            fields=fields,
        )
