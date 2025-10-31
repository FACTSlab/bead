"""Tests for multi-select simulation strategy."""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from bead.items.models import (
    Item,
    ItemTemplate,
    ModelOutput,
    PresentationSpec,
    TaskSpec,
)
from bead.simulation.strategies.multi_select import MultiSelectStrategy


def _create_model_output(score: float, idx: int = 0) -> ModelOutput:
    """Create ModelOutput with required fields."""
    return ModelOutput(
        model_name="test_model",
        model_version="1.0",
        operation="lm_score",
        inputs={},
        output=score,
        cache_key=f"key_{score}_{idx}",
    )


def test_strategy_instantiation() -> None:
    """Test that strategy can be instantiated."""
    strategy = MultiSelectStrategy()
    assert strategy.supported_task_type == "multi_select"
    assert strategy.threshold == 0.5
    assert strategy.temperature == 1.0


def test_strategy_with_custom_parameters() -> None:
    """Test instantiation with custom parameters."""
    strategy = MultiSelectStrategy(threshold=0.7, temperature=2.0)
    assert strategy.threshold == 0.7
    assert strategy.temperature == 2.0


def test_validate_item_correct_task_type() -> None:
    """Test validation passes with correct task type."""
    strategy = MultiSelectStrategy()

    template = ItemTemplate(
        name="test_multi_select",
        judgment_type="preference",
        task_type="multi_select",
        task_spec=TaskSpec(
            prompt="Select all that apply:",
            options=["option_a", "option_b", "option_c"],
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        rendered_elements={"text": "Some content"},
    )

    # Should not raise
    strategy.validate_item(item, template)


def test_validate_item_wrong_task_type() -> None:
    """Test validation fails with wrong task type."""
    strategy = MultiSelectStrategy()

    template = ItemTemplate(
        name="test_forced_choice",
        judgment_type="preference",
        task_type="forced_choice",
        task_spec=TaskSpec(prompt="Which is better?", options=["a", "b"]),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(item_template_id=uuid4())

    with pytest.raises(ValueError, match="Expected task_type 'multi_select'"):
        strategy.validate_item(item, template)


def test_validate_item_no_options() -> None:
    """Test validation fails without options."""
    strategy = MultiSelectStrategy()

    template = ItemTemplate(
        name="test_multi_select",
        judgment_type="preference",
        task_type="multi_select",
        task_spec=TaskSpec(prompt="Select all that apply:"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(item_template_id=uuid4())

    with pytest.raises(ValueError, match="task_spec.options must be defined"):
        strategy.validate_item(item, template)


def test_validate_item_insufficient_options() -> None:
    """Test validation fails with only one option."""
    strategy = MultiSelectStrategy()

    template = ItemTemplate(
        name="test_multi_select",
        judgment_type="preference",
        task_type="multi_select",
        task_spec=TaskSpec(prompt="Select all that apply:", options=["only_one"]),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(item_template_id=uuid4())

    with pytest.raises(ValueError, match="multi_select requires at least 2 options"):
        strategy.validate_item(item, template)


def test_simulate_response_with_high_scores() -> None:
    """Test response with high scores tends to select options."""
    strategy = MultiSelectStrategy()

    template = ItemTemplate(
        name="test_multi_select",
        judgment_type="preference",
        task_type="multi_select",
        task_spec=TaskSpec(
            prompt="Select all that apply:",
            options=["option_a", "option_b", "option_c"],
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # All high positive scores
    item = Item(
        item_template_id=uuid4(),
        model_outputs=[
            _create_model_output(5.0, 0),
            _create_model_output(5.0, 1),
            _create_model_output(5.0, 2),
        ],
    )

    rng = np.random.RandomState(42)

    # Generate multiple responses
    all_selected = []
    for _ in range(100):
        response = strategy.simulate_response(item, template, "lm_score", rng)
        all_selected.extend(response)

    # Most responses should have selected all options
    assert all_selected.count("option_a") > 90
    assert all_selected.count("option_b") > 90
    assert all_selected.count("option_c") > 90


def test_simulate_response_with_low_scores() -> None:
    """Test response with low scores tends not to select options."""
    strategy = MultiSelectStrategy()

    template = ItemTemplate(
        name="test_multi_select",
        judgment_type="preference",
        task_type="multi_select",
        task_spec=TaskSpec(
            prompt="Select all that apply:",
            options=["option_a", "option_b", "option_c"],
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # All low negative scores
    item = Item(
        item_template_id=uuid4(),
        model_outputs=[
            _create_model_output(-5.0, 0),
            _create_model_output(-5.0, 1),
            _create_model_output(-5.0, 2),
        ],
    )

    rng = np.random.RandomState(42)

    # Generate multiple responses
    all_selected = []
    for _ in range(100):
        response = strategy.simulate_response(item, template, "lm_score", rng)
        all_selected.extend(response)

    # Most responses should have selected few/no options
    assert all_selected.count("option_a") < 10
    assert all_selected.count("option_b") < 10
    assert all_selected.count("option_c") < 10


def test_simulate_response_with_mixed_scores() -> None:
    """Test response with mixed scores gives mixed selections."""
    strategy = MultiSelectStrategy()

    template = ItemTemplate(
        name="test_multi_select",
        judgment_type="preference",
        task_type="multi_select",
        task_spec=TaskSpec(
            prompt="Select all that apply:",
            options=["option_a", "option_b", "option_c"],
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # Mixed scores: high, low, medium
    item = Item(
        item_template_id=uuid4(),
        model_outputs=[
            _create_model_output(5.0, 0),  # High - likely selected
            _create_model_output(-5.0, 1),  # Low - likely not selected
            _create_model_output(0.0, 2),  # Medium - 50/50
        ],
    )

    rng = np.random.RandomState(42)

    # Generate multiple responses
    all_selected = []
    for _ in range(100):
        response = strategy.simulate_response(item, template, "lm_score", rng)
        all_selected.extend(response)

    # option_a should be selected most often
    assert all_selected.count("option_a") > 90
    # option_b should be selected rarely
    assert all_selected.count("option_b") < 10
    # option_c should be selected around 50%
    assert 30 < all_selected.count("option_c") < 70


def test_simulate_response_without_model_outputs() -> None:
    """Test fallback to random when model outputs missing."""
    strategy = MultiSelectStrategy(threshold=0.5)

    template = ItemTemplate(
        name="test_multi_select",
        judgment_type="preference",
        task_type="multi_select",
        task_spec=TaskSpec(
            prompt="Select all that apply:",
            options=["option_a", "option_b", "option_c"],
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # Item without model outputs
    item = Item(item_template_id=uuid4())

    rng = np.random.RandomState(42)

    # Generate multiple responses
    all_selected = []
    for _ in range(1000):
        response = strategy.simulate_response(item, template, "lm_score", rng)
        all_selected.extend(response)

    # Each option should be selected approximately threshold% of time
    for option in ["option_a", "option_b", "option_c"]:
        count = all_selected.count(option)
        # Should be around 500 (50% of 1000)
        assert 400 < count < 600


def test_simulate_response_with_item_metadata() -> None:
    """Test extraction from item_metadata."""
    strategy = MultiSelectStrategy()

    template = ItemTemplate(
        name="test_multi_select",
        judgment_type="preference",
        task_type="multi_select",
        task_spec=TaskSpec(
            prompt="Select all that apply:",
            options=["option_a", "option_b"],
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # Scores in item_metadata
    item = Item(
        item_template_id=uuid4(),
        item_metadata={
            "lm_score1": 5.0,
            "lm_score2": -5.0,
        },
    )

    rng = np.random.RandomState(42)

    # Generate multiple responses
    all_selected = []
    for _ in range(100):
        response = strategy.simulate_response(item, template, "lm_score", rng)
        all_selected.extend(response)

    # option_a (high score) should be selected often
    assert all_selected.count("option_a") > 90
    # option_b (low score) should be selected rarely
    assert all_selected.count("option_b") < 10


def test_simulate_response_deterministic_with_seed() -> None:
    """Test that responses are deterministic with fixed seed."""
    strategy = MultiSelectStrategy()

    template = ItemTemplate(
        name="test_multi_select",
        judgment_type="preference",
        task_type="multi_select",
        task_spec=TaskSpec(
            prompt="Select all that apply:",
            options=["option_a", "option_b", "option_c"],
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        model_outputs=[
            _create_model_output(0.5, 0),
            _create_model_output(-0.5, 1),
            _create_model_output(1.0, 2),
        ],
    )

    # Generate responses with same seed
    rng1 = np.random.RandomState(123)
    response1 = strategy.simulate_response(item, template, "lm_score", rng1)

    rng2 = np.random.RandomState(123)
    response2 = strategy.simulate_response(item, template, "lm_score", rng2)

    # Should be identical
    assert response1 == response2


def test_simulate_response_temperature_effect() -> None:
    """Test that temperature affects selection probabilities."""
    template = ItemTemplate(
        name="test_multi_select",
        judgment_type="preference",
        task_type="multi_select",
        task_spec=TaskSpec(
            prompt="Select all that apply:",
            options=["option_a", "option_b"],
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        model_outputs=[
            _create_model_output(1.0, 0),
            _create_model_output(-1.0, 1),
        ],
    )

    # Low temperature (more deterministic)
    strategy_low = MultiSelectStrategy(temperature=0.5)
    rng = np.random.RandomState(42)
    selections_low = []
    for _ in range(100):
        response = strategy_low.simulate_response(item, template, "lm_score", rng)
        selections_low.extend(response)

    # High temperature (less deterministic, closer to 50/50)
    strategy_high = MultiSelectStrategy(temperature=5.0)
    rng = np.random.RandomState(42)
    selections_high = []
    for _ in range(100):
        response = strategy_high.simulate_response(item, template, "lm_score", rng)
        selections_high.extend(response)

    # Low temp should have stronger preference (more option_a, less option_b)
    option_a_low = selections_low.count("option_a")
    option_a_high = selections_high.count("option_a")

    # Low temp should select option_a more often than high temp
    assert option_a_low > option_a_high


def test_multi_select_response_type() -> None:
    """Test that response is always a list."""
    strategy = MultiSelectStrategy()

    template = ItemTemplate(
        name="test_multi_select",
        judgment_type="preference",
        task_type="multi_select",
        task_spec=TaskSpec(
            prompt="Select all that apply:",
            options=["option_a", "option_b"],
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        model_outputs=[
            _create_model_output(1.0, 0),
            _create_model_output(1.0, 1),
        ],
    )

    rng = np.random.RandomState(42)
    response = strategy.simulate_response(item, template, "lm_score", rng)

    assert isinstance(response, list)
    assert all(isinstance(item, str) for item in response)


def test_multi_select_can_select_none() -> None:
    """Test that it's possible to select no options."""
    strategy = MultiSelectStrategy()

    template = ItemTemplate(
        name="test_multi_select",
        judgment_type="preference",
        task_type="multi_select",
        task_spec=TaskSpec(
            prompt="Select all that apply:",
            options=["option_a", "option_b", "option_c"],
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # Very low scores
    item = Item(
        item_template_id=uuid4(),
        model_outputs=[
            _create_model_output(-10.0, 0),
            _create_model_output(-10.0, 1),
            _create_model_output(-10.0, 2),
        ],
    )

    rng = np.random.RandomState(42)

    # Generate multiple responses
    responses = [
        strategy.simulate_response(item, template, "lm_score", rng) for _ in range(100)
    ]

    # At least some should be empty
    empty_count = sum(1 for r in responses if len(r) == 0)
    assert empty_count > 50


def test_multi_select_can_select_all() -> None:
    """Test that it's possible to select all options."""
    strategy = MultiSelectStrategy()

    template = ItemTemplate(
        name="test_multi_select",
        judgment_type="preference",
        task_type="multi_select",
        task_spec=TaskSpec(
            prompt="Select all that apply:",
            options=["option_a", "option_b", "option_c"],
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # Very high scores
    item = Item(
        item_template_id=uuid4(),
        model_outputs=[
            _create_model_output(10.0, 0),
            _create_model_output(10.0, 1),
            _create_model_output(10.0, 2),
        ],
    )

    rng = np.random.RandomState(42)

    # Generate multiple responses
    responses = [
        strategy.simulate_response(item, template, "lm_score", rng) for _ in range(100)
    ]

    # Most should have all 3 options
    all_count = sum(1 for r in responses if len(r) == 3)
    assert all_count > 90
