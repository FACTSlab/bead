"""Fixtures for MultiSelectModel tests."""

from __future__ import annotations

import json
from uuid import uuid4

import pytest

from bead.items.item import Item


@pytest.fixture
def sample_items() -> list[Item]:
    """Create sample multi-select items with 3 options each."""
    items = []
    for i in range(20):
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={
                "option_a": f"First option variant {i}",
                "option_b": f"Second option variant {i}",
                "option_c": f"Third option variant {i}",
            },
        )
        items.append(item)
    return items


@pytest.fixture
def sample_labels() -> list[str]:
    """Create sample multi-select labels (JSON-serialized lists).

    Returns
    -------
    list[str]
        JSON string of selected options, e.g., '["option_a", "option_c"]'.
    """
    labels = []
    # Vary selection patterns
    patterns = [
        ["option_a"],  # Single selection
        ["option_a", "option_b"],  # Two selections
        ["option_a", "option_b", "option_c"],  # All three
        ["option_b", "option_c"],  # Different pair
        [],  # No selection
    ]
    for i in range(20):
        selected = patterns[i % len(patterns)]
        labels.append(json.dumps(sorted(selected)))
    return labels


@pytest.fixture
def sample_participant_ids() -> list[str]:
    """Create sample participant IDs with 3 participants."""
    participant_ids = []
    participants = ["alice", "bob", "charlie"]
    for i in range(20):
        participant_ids.append(participants[i % 3])
    return participant_ids
