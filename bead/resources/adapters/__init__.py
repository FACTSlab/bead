"""External resource adapters for linguistic databases.

This module provides adapters for fetching lexical items from external
linguistic databases including VerbNet, PropBank, FrameNet (via glazing),
and UniMorph morphological paradigms.
"""

from bead.resources.adapters.base import ResourceAdapter
from bead.resources.adapters.cache import AdapterCache
from bead.resources.adapters.glazing import GlazingAdapter
from bead.resources.adapters.registry import AdapterRegistry
from bead.resources.adapters.unimorph import UniMorphAdapter

__all__ = [
    "ResourceAdapter",
    "AdapterCache",
    "GlazingAdapter",
    "UniMorphAdapter",
    "AdapterRegistry",
]
