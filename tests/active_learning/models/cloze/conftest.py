"""Fixtures for cloze model tests."""

from __future__ import annotations

from uuid import uuid4

import pytest

from bead.items.item import Item, UnfilledSlot


@pytest.fixture
def sample_cloze_items() -> list[Item]:
    """Create sample cloze items with unfilled slots.

    Creates 6 items:
    - 3 items with 1 unfilled slot
    - 3 items with 2 unfilled slots
    """
    items = []

    # Items with single unfilled slot
    for i in range(3):
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={"text": f"The cat ___."},
            unfilled_slots=[
                UnfilledSlot(slot_name="verb", position=2, constraint_ids=[])
            ],
            item_metadata={"index": i, "n_slots": 1},
        )
        items.append(item)

    # Items with two unfilled slots
    for i in range(3):
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={"text": f"___ dog ___."},
            unfilled_slots=[
                UnfilledSlot(slot_name="det", position=0, constraint_ids=[]),
                UnfilledSlot(slot_name="verb", position=2, constraint_ids=[]),
            ],
            item_metadata={"index": i + 3, "n_slots": 2},
        )
        items.append(item)

    return items


@pytest.fixture
def sample_cloze_labels() -> list[list[str]]:
    """Create sample labels matching cloze items.

    Labels match the unfilled slots:
    - First 3 items: 1 token each
    - Last 3 items: 2 tokens each
    """
    labels = [
        ["ran"],
        ["jumped"],
        ["slept"],
        ["The", "barked"],
        ["A", "ran"],
        ["The", "jumped"],
    ]
    return labels


@pytest.fixture
def sample_short_cloze_items() -> list[Item]:
    """Create minimal cloze items for faster tests."""
    items = []
    for i in range(4):
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={"text": "The ___."},
            unfilled_slots=[
                UnfilledSlot(slot_name="noun", position=1, constraint_ids=[])
            ],
        )
        items.append(item)
    return items


@pytest.fixture
def sample_short_labels() -> list[list[str]]:
    """Create minimal labels for faster tests."""
    return [["cat"], ["dog"], ["bird"], ["fish"]]
