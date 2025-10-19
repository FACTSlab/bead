"""Core resource models for lexical items.

This module provides the fundamental data models for representing lexical
items in the sash system. Lexical items are the atomic units that fill
template slots during sentence generation.
"""

from __future__ import annotations

from typing import Any

from pydantic import Field, field_validator

from sash.data.base import SashBaseModel
from sash.data.language_codes import LanguageCode


class LexicalItem(SashBaseModel):
    """A lexical item with attributes and metadata.

    LexicalItems represent words or phrases that can be inserted into
    template slots. Each item has:
    - A unique identifier (inherited from SashBaseModel)
    - Core lexical attributes (lemma, pos, etc.)
    - Optional language code (ISO 639-1 or ISO 639-3)
    - Optional linguistic features
    - Optional custom attributes
    - Metadata tracking (provenance, processing history)

    Attributes
    ----------
    lemma : str
        The base form of the lexical item (e.g., "walk").
    pos : str | None
        Part of speech tag (e.g., "VERB", "NOUN").
    form : str | None
        Inflected surface form if different from lemma.
    language_code : LanguageCode | None
        ISO 639-1 (2-letter) or ISO 639-3 (3-letter) language code.
        Examples: "en", "eng", "ko", "kor", "zu", "zul".
        Required for cross-linguistic classification via LexicalItemClass (Phase 20).
    features : dict[str, Any]
        Linguistic features (e.g., {"tense": "past", "number": "plural"}).
    attributes : dict[str, Any]
        Custom attributes for constraint evaluation.
    source : str | None
        Source of the lexical item (e.g., "verbnet", "manual").

    Examples
    --------
    >>> item = LexicalItem(
    ...     lemma="walk",
    ...     pos="VERB",
    ...     features={"tense": "present", "transitive": True},
    ...     attributes={"frequency": 1000}
    ... )
    >>> item.lemma
    'walk'
    >>> item.features["transitive"]
    True
    """

    lemma: str
    pos: str | None = None
    form: str | None = None
    language_code: LanguageCode | None = None
    features: dict[str, Any] = Field(default_factory=dict)
    attributes: dict[str, Any] = Field(default_factory=dict)
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

    @field_validator("pos")
    @classmethod
    def validate_pos(cls, v: str | None) -> str | None:
        """Validate that pos is uppercase if provided.

        Parameters
        ----------
        v : str | None
            The POS tag to validate.

        Returns
        -------
        str | None
            The validated POS tag.

        Raises
        ------
        ValueError
            If POS tag is not uppercase.
        """
        if v is not None and v != v.upper():
            raise ValueError("pos must be uppercase")
        return v
