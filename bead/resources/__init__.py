"""Resource models for bead.

This module provides data models for lexical items, templates, constraints,
and template structures.
"""

from bead.resources.constraints import Constraint
from bead.resources.lexicon import Lexicon
from bead.resources.models import LexicalItem
from bead.resources.structures import (
    Slot,
    Template,
    TemplateSequence,
    TemplateTree,
)
from bead.resources.template_collection import TemplateCollection

__all__ = [
    # Lexical items
    "LexicalItem",
    # Lexicon
    "Lexicon",
    # Constraints
    "Constraint",
    # Templates and structures
    "Slot",
    "Template",
    "TemplateSequence",
    "TemplateTree",
    # Template collection
    "TemplateCollection",
]
