"""Lexical item models for words and multi-word expressions.

This module provides data models for representing lexical items in the bead
system. Lexical items are the atomic units that fill template slots during
sentence generation. Includes support for single words and multi-word
expressions (MWEs).
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator

from bead.data.base import BeadBaseModel
from bead.data.language_codes import LanguageCode
from bead.resources.constraints import Constraint


def _empty_constraint_list() -> list[Constraint]:
    """Create an empty constraint list."""
    return []


class LexicalItem(BeadBaseModel):
    """A lexical item with linguistic features.

    Follows UniMorph structure: lemma, form, features bundle.
    - lemma: base/citation form
    - form: inflected surface form (None if same as lemma)
    - features: feature bundle (pos, tense, person, number, etc.)

    Attributes
    ----------
    lemma : str
        Base/citation form (e.g., "walk", "the").
    form : str | None
        Inflected surface form if different from lemma (e.g., "walked", "walking").
        None means form equals lemma.
    language_code : LanguageCode
        ISO 639-3 language code (e.g., "eng").
    features : dict[str, Any]
        Feature bundle with grammatical/linguistic features:
        - pos: str (e.g., "VERB", "DET", "NOUN", "ADJ", "ADP")
        - Morphological: tense, person, number, case, gender, etc.
        - unimorph_features: str (e.g., "V;PRS;3;SG")
        - Lexical resource info: verbnet_class, themroles, frame_info, etc.
    source : str | None
        Provenance (e.g., "VerbNet", "UniMorph", "manual").

    Examples
    --------
    >>> # Inflected verb
    >>> verb = LexicalItem(
    ...     lemma="walk",
    ...     form="walked",
    ...     language_code="eng",
    ...     features={"pos": "VERB", "tense": "PST"},
    ...     source="UniMorph"
    ... )
    >>> verb.form
    'walked'
    >>>
    >>> # Uninflected determiner
    >>> det = LexicalItem(
    ...     lemma="the",
    ...     form=None,
    ...     language_code="eng",
    ...     features={"pos": "DET"},
    ...     source="manual"
    ... )
    >>> det.form is None
    True
    """

    lemma: str
    form: str | None = None
    language_code: LanguageCode
    features: dict[str, Any] = Field(default_factory=dict)
    source: str | None = None

    @field_validator("lemma")
    @classmethod
    def validate_lemma(cls, v: str) -> str:
        """Validate that lemma is non-empty.

        Parameters
        ----------
        v : str
            The lemma value to validate.

        Returns
        -------
        str
            The validated lemma.

        Raises
        ------
        ValueError
            If lemma is empty or contains only whitespace.
        """
        if not v or not v.strip():
            raise ValueError("lemma must be non-empty")
        return v


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
