"""Fixtures for ordinal scale model tests."""

from __future__ import annotations

from uuid import uuid4

import pytest

from bead.items.item import Item


@pytest.fixture
def sample_items() -> list[Item]:
    """Create sample ordinal scale items."""
    items = []
    for i in range(20):
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={"text": f"Sentence variant {i}"},
        )
        items.append(item)
    return items


@pytest.fixture
def sample_labels() -> list[str]:
    """Create sample labels (continuous values on [0, 1])."""
    # Varied values across the scale
    return [str(i * 0.05) for i in range(20)]


@pytest.fixture
def sample_endpoint_labels() -> list[str]:
    """Create sample labels including endpoints 0.0 and 1.0."""
    labels = ["0.0", "1.0"] * 5 + ["0.5"] * 10
    return labels
