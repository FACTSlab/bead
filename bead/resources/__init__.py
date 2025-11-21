"""Resource models.

Provides data models for lexical items, templates, constraints, and
template structures.
"""

from bead.resources.constraints import Constraint
from bead.resources.lexical_item import LexicalItem, MultiWordExpression, MWEComponent
from bead.resources.lexicon import Lexicon
from bead.resources.template import Slot, Template, TemplateSequence, TemplateTree
from bead.resources.template_collection import TemplateCollection

__all__ = [
    # Lexical items
    "LexicalItem",
    "MWEComponent",
    "MultiWordExpression",
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
