"""Tests for free text simulation strategy."""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from bead.items.models import (
    Item,
    ItemTemplate,
    PresentationSpec,
    TaskSpec,
)
from bead.simulation.strategies.free_text import FreeTextStrategy


def test_strategy_instantiation() -> None:
    """Test that strategy can be instantiated."""
    strategy = FreeTextStrategy()
    assert strategy.supported_task_type == "free_text"


def test_validate_item_correct_task_type() -> None:
    """Test validation passes with correct task type."""
    strategy = FreeTextStrategy()

    template = ItemTemplate(
        name="test_free_text",
        judgment_type="comprehension",
        task_type="free_text",
        task_spec=TaskSpec(prompt="Describe what you see:"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        rendered_elements={"image": "image.jpg"},
    )

    # Should not raise
    strategy.validate_item(item, template)


def test_validate_item_wrong_task_type() -> None:
    """Test validation fails with wrong task type."""
    strategy = FreeTextStrategy()

    template = ItemTemplate(
        name="test_binary",
        judgment_type="acceptability",
        task_type="binary",
        task_spec=TaskSpec(prompt="Is this good?"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(item_template_id=uuid4())

    with pytest.raises(ValueError, match="Expected task_type 'free_text'"):
        strategy.validate_item(item, template)


def test_simulate_response_with_metadata_response() -> None:
    """Test response from item_metadata['response']."""
    strategy = FreeTextStrategy()

    template = ItemTemplate(
        name="test_free_text",
        judgment_type="comprehension",
        task_type="free_text",
        task_spec=TaskSpec(prompt="Describe:"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        item_metadata={"response": "This is a test response"},
    )

    rng = np.random.RandomState(42)
    response = strategy.simulate_response(item, template, "lm_score", rng)

    assert response == "This is a test response"


def test_simulate_response_with_response_template() -> None:
    """Test response from item_metadata['response_template']."""
    strategy = FreeTextStrategy()

    template = ItemTemplate(
        name="test_free_text",
        judgment_type="comprehension",
        task_type="free_text",
        task_spec=TaskSpec(prompt="Describe:"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        item_metadata={"response_template": "Template response"},
    )

    rng = np.random.RandomState(42)
    response = strategy.simulate_response(item, template, "lm_score", rng)

    assert response == "Template response"


def test_simulate_response_prefers_response_over_template() -> None:
    """Test that 'response' is preferred over 'response_template'."""
    strategy = FreeTextStrategy()

    template = ItemTemplate(
        name="test_free_text",
        judgment_type="comprehension",
        task_type="free_text",
        task_spec=TaskSpec(prompt="Describe:"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        item_metadata={
            "response": "Actual response",
            "response_template": "Template response",
        },
    )

    rng = np.random.RandomState(42)
    response = strategy.simulate_response(item, template, "lm_score", rng)

    assert response == "Actual response"


def test_simulate_response_from_rendered_elements() -> None:
    """Test fallback to rendered_elements."""
    strategy = FreeTextStrategy()

    template = ItemTemplate(
        name="test_free_text",
        judgment_type="comprehension",
        task_type="free_text",
        task_spec=TaskSpec(prompt="Describe:"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        rendered_elements={"sentence": "The cat sat on the mat."},
    )

    rng = np.random.RandomState(42)
    response = strategy.simulate_response(item, template, "lm_score", rng)

    assert response == "The cat sat on the mat."


def test_simulate_response_from_first_non_empty_element() -> None:
    """Test that first non-empty string is used from rendered_elements."""
    strategy = FreeTextStrategy()

    template = ItemTemplate(
        name="test_free_text",
        judgment_type="comprehension",
        task_type="free_text",
        task_spec=TaskSpec(prompt="Describe:"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        rendered_elements={
            "empty": "",
            "text1": "First text",
            "text2": "Second text",
        },
    )

    rng = np.random.RandomState(42)
    response = strategy.simulate_response(item, template, "lm_score", rng)

    # Should get one of the non-empty strings (dict iteration may vary)
    assert response in ["First text", "Second text"]


def test_simulate_response_fallback_to_default() -> None:
    """Test fallback to default when no content available."""
    strategy = FreeTextStrategy()

    template = ItemTemplate(
        name="test_free_text",
        judgment_type="comprehension",
        task_type="free_text",
        task_spec=TaskSpec(prompt="Describe:"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # Item with no useful content
    item = Item(item_template_id=uuid4())

    rng = np.random.RandomState(42)
    response = strategy.simulate_response(item, template, "lm_score", rng)

    assert response == "No response"


def test_simulate_response_with_empty_rendered_elements() -> None:
    """Test fallback when rendered_elements is empty."""
    strategy = FreeTextStrategy()

    template = ItemTemplate(
        name="test_free_text",
        judgment_type="comprehension",
        task_type="free_text",
        task_spec=TaskSpec(prompt="Describe:"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        rendered_elements={},
    )

    rng = np.random.RandomState(42)
    response = strategy.simulate_response(item, template, "lm_score", rng)

    assert response == "No response"


def test_simulate_response_with_only_empty_strings() -> None:
    """Test fallback when rendered_elements has only empty strings."""
    strategy = FreeTextStrategy()

    template = ItemTemplate(
        name="test_free_text",
        judgment_type="comprehension",
        task_type="free_text",
        task_spec=TaskSpec(prompt="Describe:"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        rendered_elements={"text1": "", "text2": ""},
    )

    rng = np.random.RandomState(42)
    response = strategy.simulate_response(item, template, "lm_score", rng)

    assert response == "No response"


def test_simulate_response_deterministic() -> None:
    """Test that responses are deterministic for same input."""
    strategy = FreeTextStrategy()

    template = ItemTemplate(
        name="test_free_text",
        judgment_type="comprehension",
        task_type="free_text",
        task_spec=TaskSpec(prompt="Describe:"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        item_metadata={"response": "Test response"},
    )

    # Generate responses with different seeds (should be same)
    rng1 = np.random.RandomState(123)
    response1 = strategy.simulate_response(item, template, "lm_score", rng1)

    rng2 = np.random.RandomState(456)
    response2 = strategy.simulate_response(item, template, "lm_score", rng2)

    # Should be identical (deterministic from content)
    assert response1 == response2


def test_free_text_response_type() -> None:
    """Test that response is always a string."""
    strategy = FreeTextStrategy()

    template = ItemTemplate(
        name="test_free_text",
        judgment_type="comprehension",
        task_type="free_text",
        task_spec=TaskSpec(prompt="Describe:"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        item_metadata={"response": "Test"},
    )

    rng = np.random.RandomState(42)
    response = strategy.simulate_response(item, template, "lm_score", rng)

    assert isinstance(response, str)


def test_simulate_response_with_non_string_rendered_elements() -> None:
    """Test handling of multiple rendered_elements."""
    strategy = FreeTextStrategy()

    template = ItemTemplate(
        name="test_free_text",
        judgment_type="comprehension",
        task_type="free_text",
        task_spec=TaskSpec(prompt="Describe:"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        rendered_elements={
            "number": "123",
            "text": "Valid text",
        },
    )

    rng = np.random.RandomState(42)
    response = strategy.simulate_response(item, template, "lm_score", rng)

    # Should use one of the string values from rendered_elements
    assert isinstance(response, str)
    assert response in ["123", "Valid text"]
