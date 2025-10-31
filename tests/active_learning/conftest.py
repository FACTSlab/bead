"""Shared test fixtures for training model tests."""

from uuid import uuid4

import pytest

from bead.items.models import (
    Item,
    ItemTemplate,
    PresentationSpec,
    TaskSpec,
)


@pytest.fixture
def test_items():
    """Create test items with simple two-option structure."""

    def _create_items(n: int = 10):
        items = []
        for i in range(n):
            item = Item(
                item_template_id=uuid4(),
                rendered_elements={
                    "option_a": f"The child played with toy {i}. This is a sentence.",
                    "option_b": f"The person walked to place {i}. This is different.",
                },
                item_metadata={"test_index": i},
            )
            items.append(item)
        return items

    return _create_items


@pytest.fixture
def varied_test_items():
    """Create varied test items with different text patterns."""

    def _create_items(n: int = 10):
        items = []
        patterns = [
            ("The", "A"),
            ("child", "person"),
            ("played", "walked"),
            ("with", "to"),
            ("toy", "place"),
        ]

        for i in range(n):
            pattern_idx = i % len(patterns)
            word1, word2 = patterns[pattern_idx]

            item = Item(
                item_template_id=uuid4(),
                rendered_elements={
                    "option_a": f"{word1} {word2} option {i}",
                    "option_b": f"Different text for option {i}",
                },
                item_metadata={"test_index": i, "pattern": pattern_idx},
            )
            items.append(item)

        return items

    return _create_items


@pytest.fixture
def forced_choice_template():
    """Create a forced choice ItemTemplate for testing."""
    template = ItemTemplate(
        name="test_2afc",
        judgment_type="preference",
        task_type="forced_choice",
        task_spec=TaskSpec(
            prompt="Which option sounds more natural?",
            options=["option_a", "option_b"],
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )
    return template
