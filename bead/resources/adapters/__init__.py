"""External resource adapters for linguistic databases.

Fetches lexical items from VerbNet, PropBank, FrameNet (via glazing), and
UniMorph morphological paradigms.
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
