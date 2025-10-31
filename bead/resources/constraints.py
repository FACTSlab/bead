"""Constraint models for lexical item selection.

This module provides a universal constraint model based on DSL expressions.
Constraints are pure DSL expressions with optional context variables.

Scope is determined by storage location:
- Slot.constraints → single-slot constraints (self = slot filler)
- Template.constraints → multi-slot constraints (slot names as variables)
- TemplateSequence.constraints → cross-template constraints
"""

from __future__ import annotations

from uuid import UUID

from pydantic import Field

from bead.data.base import BeadBaseModel
from bead.dsl.ast import ASTNode

# Type aliases for constraint context values
type ContextValue = str | int | float | bool | list[str] | set[str] | set[UUID]
type MetadataValue = (
    str | int | float | bool | list[str | int | float] | dict[str, str | int | float]
)


class Constraint(BeadBaseModel):
    """Universal constraint expressed via DSL.

    All constraints are DSL expressions evaluated with a context dictionary.
    The scope of the constraint is determined by where it is stored:
    - Slot.constraints: single-slot constraints where 'self' refers to the slot filler
    - Template.constraints: multi-slot constraints where slot names are variables
    - TemplateSequence.constraints: cross-template constraints

    Attributes
    ----------
    expression : str
        DSL expression to evaluate (must return boolean).
    context : dict[str, ContextValue]
        Context variables available during evaluation (e.g., whitelists, constants).
    description : str | None
        Optional human-readable description of the constraint.
    compiled : ASTNode | None
        Cached compiled AST after first compilation (optimization).

    Examples
    --------
    Extensional (whitelist):
    >>> constraint = Constraint(
    ...     expression="self.lemma in motion_verbs",
    ...     context={"motion_verbs": {"walk", "run", "jump"}}
    ... )

    Intensional (feature-based):
    >>> constraint = Constraint(
    ...     expression="self.pos == 'VERB' and self.features.number == 'singular'"
    ... )

    Binary agreement:
    >>> constraint = Constraint(
    ...     expression="subject.features.number == verb.features.number"
    ... )

    IF-THEN conditional:
    >>> constraint = Constraint(
    ...     expression="det.lemma != 'a' or noun.features.number == 'singular'"
    ... )
    """

    expression: str
    context: dict[str, ContextValue] = Field(default_factory=dict)
    description: str | None = None
    compiled: ASTNode | None = None
