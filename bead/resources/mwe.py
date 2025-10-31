"""Multi-word expression models.

This module provides models for multi-word expressions (MWEs) such as
phrasal verbs ("take off"), idioms ("spill the beans"), and other
multi-word lexical units. MWEs are treated as LexicalItem subclasses
to enable seamless integration with template filling and constraints.
"""

from __future__ import annotations

from pydantic import Field

from bead.resources.constraints import Constraint
from bead.resources.models import LexicalItem


def _empty_constraint_list() -> list[Constraint]:
    """Create an empty constraint list."""
    return []


class MWEComponent(LexicalItem):
    """A component of a multi-word expression.

    Components represent individual parts of an MWE (e.g., verb and particle
    in a phrasal verb). Each component has a role within the MWE and can
    have its own constraints.

    Attributes
    ----------
    role : str
        Role of this component in the MWE (e.g., "verb", "particle", "noun").
    required : bool
        Whether this component must be present (default: True).
    constraints : list[Constraint]
        Component-specific constraints (in addition to base LexicalItem constraints).

    Examples
    --------
    >>> # Verb component of "take off"
    >>> verb = MWEComponent(
    ...     lemma="take",
    ...     pos="VERB",
    ...     role="verb",
    ...     required=True
    ... )
    >>> # Particle component
    >>> particle = MWEComponent(
    ...     lemma="off",
    ...     pos="PART",
    ...     role="particle",
    ...     required=True
    ... )
    """

    role: str = Field(..., description="Component role in MWE")
    required: bool = Field(default=True, description="Whether component is required")
    constraints: list[Constraint] = Field(
        default_factory=_empty_constraint_list,
        description="Component-specific constraints",
    )


class MultiWordExpression(LexicalItem):
    """Multi-word expression as a lexical item.

    MWEs are lexical items composed of multiple components. They can be
    separable (components can be non-adjacent) or inseparable. MWEs
    support component-level constraints and adjacency patterns.

    Attributes
    ----------
    components : list[MWEComponent]
        Components that make up this MWE.
    separable : bool
        Whether components can be separated by other words (default: False).
        Example: "take the ball off" (separable) vs "kick the bucket" (inseparable).
    adjacency_pattern : str | None
        DSL expression defining valid adjacency patterns.
        Variables: component roles, 'distance' between components.
        Example: "distance(verb, particle) <= 3"

    Examples
    --------
    >>> # Inseparable phrasal verb "look after"
    >>> mwe1 = MultiWordExpression(
    ...     lemma="look after",
    ...     pos="VERB",
    ...     components=[
    ...         MWEComponent(lemma="look", pos="VERB", role="verb"),
    ...         MWEComponent(lemma="after", pos="ADP", role="particle")
    ...     ],
    ...     separable=False
    ... )
    >>>
    >>> # Separable phrasal verb "take off"
    >>> mwe2 = MultiWordExpression(
    ...     lemma="take off",
    ...     pos="VERB",
    ...     components=[
    ...         MWEComponent(lemma="take", pos="VERB", role="verb"),
    ...         MWEComponent(lemma="off", pos="PART", role="particle")
    ...     ],
    ...     separable=True,
    ...     adjacency_pattern="distance(verb, particle) <= 3"
    ... )
    >>>
    >>> # MWE with constraints on components
    >>> mwe3 = MultiWordExpression(
    ...     lemma="break down",
    ...     pos="VERB",
    ...     components=[
    ...         MWEComponent(
    ...             lemma="break",
    ...             pos="VERB",
    ...             role="verb",
    ...             constraints=[
    ...                 Constraint(
    ...                     expression="self.lemma in motion_verbs",
    ...                     context={"motion_verbs": {"break", "take", "give"}}
    ...                 )
    ...             ]
    ...         ),
    ...         MWEComponent(lemma="down", pos="PART", role="particle")
    ...     ],
    ...     separable=True
    ... )
    """

    components: list[MWEComponent] = Field(
        default_factory=list, description="MWE components"
    )
    separable: bool = Field(
        default=False, description="Whether components can be non-adjacent"
    )
    adjacency_pattern: str | None = Field(
        default=None, description="DSL expression for valid adjacency patterns"
    )
