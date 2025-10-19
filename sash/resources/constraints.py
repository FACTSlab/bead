"""Constraint models for lexical item selection.

This module provides constraint models that determine which lexical items
are valid fillers for template slots. Constraints can be:
- Extensional: Explicit lists of allowed/disallowed items
- Intensional: Property-based selection (e.g., pos == "VERB")
- Relational: Relationships between slots
- DSL: Complex constraints using the DSL evaluator
"""

from __future__ import annotations

from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import Field, field_validator

from sash.data.base import SashBaseModel


def _empty_uuid_list() -> list[UUID]:
    """Create an empty UUID list."""
    return []


class ExtensionalConstraint(SashBaseModel):
    """Constraint defined by explicit item lists.

    An extensional constraint specifies exactly which items are allowed
    or disallowed, using lists of item IDs.

    Attributes
    ----------
    constraint_type : Literal["extensional"]
        Discriminator field for constraint type.
    mode : Literal["allow", "deny"]
        Whether to allow only listed items or deny them.
    items : list[UUID]
        List of lexical item IDs.

    Examples
    --------
    >>> constraint = ExtensionalConstraint(
    ...     mode="allow",
    ...     items=[uuid1, uuid2, uuid3]
    ... )
    >>> constraint.mode
    'allow'
    """

    constraint_type: Literal["extensional"] = "extensional"
    mode: Literal["allow", "deny"]
    items: list[UUID] = Field(default_factory=_empty_uuid_list)


class IntensionalConstraint(SashBaseModel):
    """Constraint defined by item properties.

    An intensional constraint specifies items by their properties,
    such as part of speech, features, or custom attributes.

    Attributes
    ----------
    constraint_type : Literal["intensional"]
        Discriminator field for constraint type.
    property : str
        The property to check (e.g., "pos", "lemma", "features.tense").
    operator : Literal["==", "!=", "in", "not in", "<", ">", "<=", ">="]
        Comparison operator.
    value : str | int | float | bool | list[Any]
        Value to compare against.

    Examples
    --------
    >>> constraint = IntensionalConstraint(
    ...     property="pos",
    ...     operator="==",
    ...     value="VERB"
    ... )
    >>> constraint.property
    'pos'
    """

    constraint_type: Literal["intensional"] = "intensional"
    property: str
    operator: Literal["==", "!=", "in", "not in", "<", ">", "<=", ">="]
    value: str | int | float | bool | list[Any]

    @field_validator("property")
    @classmethod
    def validate_property(cls, v: str) -> str:
        """Validate that property is non-empty.

        Parameters
        ----------
        v : str
            The property name to validate.

        Returns
        -------
        str
            The validated property name.

        Raises
        ------
        ValueError
            If property is empty.
        """
        if not v or not v.strip():
            raise ValueError("property must be non-empty")
        return v


class RelationalConstraint(SashBaseModel):
    """Constraint defined by relationships between slots.

    A relational constraint specifies that two slots must have items
    with a specific relationship (e.g., same lemma, different pos).

    Attributes
    ----------
    constraint_type : Literal["relational"]
        Discriminator field for constraint type.
    slot_a : str
        Name of the first slot.
    slot_b : str
        Name of the second slot.
    relation : Literal["same", "different"]
        Type of relationship required.
    property : str
        Property to compare (e.g., "lemma", "pos").

    Examples
    --------
    >>> constraint = RelationalConstraint(
    ...     slot_a="subject",
    ...     slot_b="object",
    ...     relation="different",
    ...     property="lemma"
    ... )
    """

    constraint_type: Literal["relational"] = "relational"
    slot_a: str
    slot_b: str
    relation: Literal["same", "different"]
    property: str

    @field_validator("slot_a", "slot_b", "property")
    @classmethod
    def validate_non_empty(cls, v: str) -> str:
        """Validate that field is non-empty.

        Parameters
        ----------
        v : str
            The value to validate.

        Returns
        -------
        str
            The validated value.

        Raises
        ------
        ValueError
            If value is empty.
        """
        if not v or not v.strip():
            raise ValueError("field must be non-empty")
        return v


class DSLConstraint(SashBaseModel):
    """Constraint defined by DSL expression.

    A DSL constraint uses the constraint DSL to express complex
    conditions that must be satisfied by lexical items.

    Attributes
    ----------
    constraint_type : Literal["dsl"]
        Discriminator field for constraint type.
    expression : str
        DSL expression to evaluate.

    Examples
    --------
    >>> constraint = DSLConstraint(
    ...     expression="pos == 'VERB' and len(lemma) > 4"
    ... )
    >>> constraint.expression
    "pos == 'VERB' and len(lemma) > 4"
    """

    constraint_type: Literal["dsl"] = "dsl"
    expression: str

    @field_validator("expression")
    @classmethod
    def validate_expression(cls, v: str) -> str:
        """Validate that expression is non-empty.

        Parameters
        ----------
        v : str
            The expression to validate.

        Returns
        -------
        str
            The validated expression.

        Raises
        ------
        ValueError
            If expression is empty.
        """
        if not v or not v.strip():
            raise ValueError("expression must be non-empty")
        return v


# Union type for all constraints with discriminator
Constraint = Annotated[
    ExtensionalConstraint
    | IntensionalConstraint
    | RelationalConstraint
    | DSLConstraint,
    Field(discriminator="constraint_type"),
]
