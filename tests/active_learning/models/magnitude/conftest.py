"""Fixtures for magnitude model tests."""

from __future__ import annotations

from uuid import uuid4

import pytest

from bead.items.item import Item


@pytest.fixture
def sample_items() -> list[Item]:
    """Create sample magnitude items."""
    items = []
    for i in range(20):
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={"text": f"Sentence variant {i}"},
        )
        items.append(item)
    return items


@pytest.fixture
def sample_unbounded_labels() -> list[str]:
    """Create sample labels for unbounded magnitude (e.g., reading times in ms)."""
    # Varied reading times from 150ms to 450ms
    base_times = [150, 200, 250, 300, 350, 400, 450]
    labels = []
    for i in range(20):
        time = base_times[i % len(base_times)] + (i * 10)
        labels.append(str(float(time)))
    return labels


@pytest.fixture
def sample_bounded_labels() -> list[str]:
    """Create sample labels for bounded magnitude (e.g., confidence on 0-100 scale)."""
    # Varied confidence values from 0 to 100
    labels = []
    for i in range(20):
        confidence = (i * 5) % 101  # 0, 5, 10, ..., 95, 0, 5, ...
        labels.append(str(float(confidence)))
    return labels


@pytest.fixture
def sample_bounded_endpoint_labels() -> list[str]:
    """Create sample labels including exact endpoints (0.0 and 100.0)."""
    labels = ["0.0", "100.0"] * 5 + ["50.0"] * 10
    return labels
