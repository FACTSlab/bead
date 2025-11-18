"""Fixtures for free text model tests."""

from __future__ import annotations

from uuid import uuid4

import pytest

from bead.items.item import Item


@pytest.fixture
def sample_items() -> list[Item]:
    """Create sample free text items (prompts for text generation)."""
    prompts = [
        "Summarize: The cat sat on the mat.",
        "Translate to French: Hello, how are you?",
        "Answer: What is 2 + 2?",
        "Complete: The quick brown fox",
        "Paraphrase: I am going to the store.",
    ] * 4  # 20 items

    items = []
    for i, prompt in enumerate(prompts):
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={"prompt": prompt},
            item_metadata={"index": i},
        )
        items.append(item)
    return items


@pytest.fixture
def sample_labels() -> list[str]:
    """Create sample labels (target text responses)."""
    targets = [
        "A cat sits on a mat.",
        "Bonjour, comment allez-vous?",
        "4",
        "jumps over the lazy dog.",
        "I'm heading to the shop.",
    ] * 4  # 20 labels
    return targets


@pytest.fixture
def sample_short_items() -> list[Item]:
    """Create smaller sample for faster tests."""
    prompts = ["Summarize: Text here."] * 6

    items = []
    for prompt in prompts:
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={"prompt": prompt},
        )
        items.append(item)
    return items


@pytest.fixture
def sample_short_labels() -> list[str]:
    """Create smaller sample labels."""
    return ["Summary."] * 6
