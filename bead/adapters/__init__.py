"""Shared adapter utilities for bead.

This module contains shared utilities and base classes for adapters that
integrate with external ML frameworks like HuggingFace Transformers.
"""

from __future__ import annotations

from bead.adapters.huggingface import HuggingFaceAdapterMixin

__all__ = ["HuggingFaceAdapterMixin"]
