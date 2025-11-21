"""Template filling model adapters.

Provides masked language model adapters for template filling (Stage 2).
Separate from judgment prediction models (Stage 3).
"""

from __future__ import annotations

from bead.templates.adapters.base import TemplateFillingModelAdapter
from bead.templates.adapters.cache import ModelOutputCache
from bead.templates.adapters.huggingface import HuggingFaceMLMAdapter

__all__ = [
    "TemplateFillingModelAdapter",
    "ModelOutputCache",
    "HuggingFaceMLMAdapter",
]
