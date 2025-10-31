"""Template and structure models for sentence generation.

This module provides models for sentence templates and their structures.
Templates contain slots that are filled with lexical items during
sentence generation.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import Field, field_validator, model_validator

from bead.data.base import BeadBaseModel
from bead.data.language_codes import LanguageCode
from bead.resources.constraints import Constraint


def _empty_constraint_list() -> list[Constraint]:
    """Create an empty constraint list."""
    return []


def _empty_str_list() -> list[str]:
    """Create an empty string list."""
    return []


class Slot(BeadBaseModel):
    """A slot in a template that can be filled with a lexical item.

    Attributes
    ----------
    name : str
        Unique name for the slot within the template.
    description : str | None
        Human-readable description of the slot's purpose.
    constraints : list[Constraint]
        Constraints that determine valid fillers.
    required : bool
        Whether the slot must be filled.
    default_value : str | None
        Default value if slot is not filled.

    Examples
    --------
    >>> from bead.resources.constraints import IntensionalConstraint
    >>> slot = Slot(
    ...     name="subject",
    ...     description="The subject of the sentence",
    ...     constraints=[
    ...         IntensionalConstraint(property="pos", operator="==", value="NOUN")
    ...     ],
    ...     required=True
    ... )
    """

    name: str
    description: str | None = None
    constraints: list[Constraint] = Field(default_factory=_empty_constraint_list)
    required: bool = True
    default_value: str | None = None

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate that name is a valid Python identifier.

        Parameters
        ----------
        v : str
            The slot name to validate.

        Returns
        -------
        str
            The validated slot name.

        Raises
        ------
        ValueError
            If name is not a valid Python identifier.
        """
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        if not v.isidentifier():
            raise ValueError(f"name '{v}' must be a valid Python identifier")
        return v


class Template(BeadBaseModel):
    """A sentence template with slots for lexical items.

    Templates define the structure of generated sentences. They contain:
    - A template string with slot placeholders (e.g., "{subject} {verb} {object}")
    - Slot definitions with constraints
    - Optional language code
    - Optional metadata

    Attributes
    ----------
    name : str
        Unique name for the template.
    template_string : str
        Template with {slot_name} placeholders.
    slots : dict[str, Slot]
        Slot definitions keyed by slot name.
    constraints : list[Constraint]
        Multi-slot constraints (slot names as variables in DSL expressions).
    description : str | None
        Human-readable description.
    language_code : LanguageCode | None
        ISO 639-1 (2-letter) or ISO 639-3 (3-letter) language code.
        Examples: "en", "eng", "ko", "kor", "zu", "zul".
        Required for cross-linguistic classification via TemplateClass (Phase 20).
    tags : list[str]
        Tags for categorization.
    metadata : dict[str, Any]
        Additional metadata.

    Examples
    --------
    >>> template = Template(
    ...     name="simple_transitive",
    ...     template_string="{subject} {verb} {object}.",
    ...     slots={
    ...         "subject": Slot(name="subject", required=True),
    ...         "verb": Slot(name="verb", required=True),
    ...         "object": Slot(name="object", required=True)
    ...     },
    ...     tags=["transitive", "simple"]
    ... )
    """

    name: str
    template_string: str
    slots: dict[str, Slot] = Field(default_factory=dict)
    constraints: list[Constraint] = Field(default_factory=_empty_constraint_list)
    description: str | None = None
    language_code: LanguageCode | None = None
    tags: list[str] = Field(default_factory=_empty_str_list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate that name is non-empty.

        Parameters
        ----------
        v : str
            The template name to validate.

        Returns
        -------
        str
            The validated template name.

        Raises
        ------
        ValueError
            If name is empty.
        """
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v

    @field_validator("template_string")
    @classmethod
    def validate_template_string(cls, v: str) -> str:
        """Validate that template_string is non-empty.

        Parameters
        ----------
        v : str
            The template string to validate.

        Returns
        -------
        str
            The validated template string.

        Raises
        ------
        ValueError
            If template_string is empty.
        """
        if not v or not v.strip():
            raise ValueError("template_string must be non-empty")
        return v

    @model_validator(mode="after")
    def validate_slots_match_template(self) -> Template:
        """Validate that template_string and slots are consistent.

        Ensures that:
        - All slot names in template_string exist in slots dict
        - All slots in dict are referenced in template_string
        - Slot names match their keys in the dict

        Returns
        -------
        Template
            The validated template.

        Raises
        ------
        ValueError
            If template_string and slots are inconsistent.
        """
        # Extract slot names from template string
        template_slots = set(re.findall(r"\{(\w+)\}", self.template_string))

        # Get slot names from slots dict
        dict_slots = set(self.slots.keys())

        # Check that all template slots exist in dict
        missing_in_dict = template_slots - dict_slots
        if missing_in_dict:
            raise ValueError(
                f"Template references slots not in slots dict: {missing_in_dict}"
            )

        # Check that all dict slots are referenced in template
        missing_in_template = dict_slots - template_slots
        if missing_in_template:
            raise ValueError(
                f"Slots dict contains slots not referenced in template: "
                f"{missing_in_template}"
            )

        # Check that slot names match their keys
        for key, slot in self.slots.items():
            if slot.name != key:
                raise ValueError(
                    f"Slot key '{key}' does not match slot name '{slot.name}'"
                )

        return self


def _empty_template_list() -> list[Template]:
    """Create an empty template list."""
    return []


class TemplateSequence(BeadBaseModel):
    """A sequence of templates to be filled together.

    Template sequences allow multiple templates to be filled with
    related constraints (e.g., relational constraints across templates).

    Attributes
    ----------
    name : str
        Unique name for the sequence.
    templates : list[Template]
        Ordered list of templates.
    constraints : list[Constraint]
        Cross-template constraints (span multiple templates).

    Examples
    --------
    >>> sequence = TemplateSequence(
    ...     name="question_answer",
    ...     templates=[question_template, answer_template],
    ...     constraints=[...]
    ... )
    """

    name: str
    templates: list[Template] = Field(default_factory=_empty_template_list)
    constraints: list[Constraint] = Field(default_factory=_empty_constraint_list)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate that name is non-empty.

        Parameters
        ----------
        v : str
            The sequence name to validate.

        Returns
        -------
        str
            The validated sequence name.

        Raises
        ------
        ValueError
            If name is empty.
        """
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v


def _empty_tree_list() -> list[TemplateTree]:
    """Create an empty template tree list."""
    return []


class TemplateTree(BeadBaseModel):
    """A tree structure of templates.

    Template trees represent hierarchical relationships between
    templates (e.g., a discourse structure).

    Attributes
    ----------
    name : str
        Unique name for the tree.
    root : Template
        Root template.
    children : list[TemplateTree]
        Child subtrees.

    Examples
    --------
    >>> tree = TemplateTree(
    ...     name="discourse",
    ...     root=intro_template,
    ...     children=[
    ...         TemplateTree(name="body", root=body_template, children=[]),
    ...         TemplateTree(name="conclusion", root=conclusion_template, children=[])
    ...     ]
    ... )
    """

    name: str
    root: Template
    children: list[TemplateTree] = Field(default_factory=_empty_tree_list)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate that name is non-empty.

        Parameters
        ----------
        v : str
            The tree name to validate.

        Returns
        -------
        str
            The validated tree name.

        Raises
        ------
        ValueError
            If name is empty.
        """
        if not v or not v.strip():
            raise ValueError("name must be non-empty")
        return v
