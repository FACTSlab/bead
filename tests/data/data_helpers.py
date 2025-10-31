"""Test helpers for data module tests."""

from __future__ import annotations

from bead.data.base import BeadBaseModel


class SimpleTestModel(BeadBaseModel):
    """Simple test model for serialization tests."""

    name: str
    value: int
