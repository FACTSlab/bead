"""External resource adapters for linguistic databases.

This module provides adapters for fetching lexical items from external
linguistic databases including VerbNet, PropBank, FrameNet (via glazing),
and UniMorph morphological paradigms.
"""

from sash.adapters.base import ResourceAdapter
from sash.adapters.cache import AdapterCache
from sash.adapters.glazing import GlazingAdapter
from sash.adapters.registry import AdapterRegistry
from sash.adapters.unimorph import UniMorphAdapter

__all__ = [
    "ResourceAdapter",
    "AdapterCache",
    "GlazingAdapter",
    "UniMorphAdapter",
    "AdapterRegistry",
]
