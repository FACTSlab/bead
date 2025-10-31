"""Tests for cloze strategy."""

from __future__ import annotations

from uuid import uuid4

import numpy as np

from bead.items.models import (
    Item,
    ItemTemplate,
    ModelOutput,
    PresentationSpec,
    TaskSpec,
    UnfilledSlot,
)
from bead.simulation.strategies.cloze import ClozeStrategy


def _create_mlm_output(
    slot_name: str, candidate: str, score: float, idx: int = 0
) -> ModelOutput:
    """Create ModelOutput for MLM score."""
    return ModelOutput(
        model_name="bert_mlm",
        model_version="1.0",
        operation="mlm_score",
        inputs={"slot_name": slot_name, "candidate": candidate},
        output=score,
        cache_key=f"mlm_{slot_name}_{candidate}_{idx}",
    )


def test_strategy_instantiation() -> None:
    """Test that strategy can be instantiated."""
    strategy = ClozeStrategy()
    assert strategy.supported_task_type == "cloze"


def test_validate_item_correct_task_type() -> None:
    """Test validation passes for correct task type."""
    strategy = ClozeStrategy()

    template = ItemTemplate(
        name="test_cloze",
        judgment_type="comprehension",
        task_type="cloze",
        task_spec=TaskSpec(prompt="Fill in the blanks:"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        unfilled_slots=[
            UnfilledSlot(slot_name="determiner", position=0, constraint_ids=[])
        ],
    )

    # Should not raise
    strategy.validate_item(item, template)


def test_validate_item_wrong_task_type() -> None:
    """Test validation fails for wrong task type."""
    strategy = ClozeStrategy()

    template = ItemTemplate(
        name="test_wrong",
        judgment_type="plausibility",
        task_type="binary",
        task_spec=TaskSpec(prompt="Is this good?"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(item_template_id=uuid4())

    try:
        strategy.validate_item(item, template)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "Expected task_type 'cloze'" in str(e)


def test_validate_item_no_slots() -> None:
    """Test validation fails when no unfilled slots."""
    strategy = ClozeStrategy()

    template = ItemTemplate(
        name="test_cloze",
        judgment_type="comprehension",
        task_type="cloze",
        task_spec=TaskSpec(prompt="Fill in the blanks:"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(item_template_id=uuid4(), unfilled_slots=[])

    try:
        strategy.validate_item(item, template)
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        assert "at least one unfilled slot" in str(e)


def test_simulate_response_with_mlm_scores() -> None:
    """Test cloze response with MLM scores."""
    strategy = ClozeStrategy()

    template = ItemTemplate(
        name="test_cloze",
        judgment_type="comprehension",
        task_type="cloze",
        task_spec=TaskSpec(prompt="Fill in the blanks:"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # Item with one slot and MLM scores for different candidates
    item = Item(
        item_template_id=uuid4(),
        unfilled_slots=[
            UnfilledSlot(slot_name="determiner", position=0, constraint_ids=[])
        ],
        model_outputs=[
            _create_mlm_output("determiner", "the", 10.0, 0),
            _create_mlm_output("determiner", "a", 5.0, 1),
            _create_mlm_output("determiner", "an", 2.0, 2),
        ],
    )

    rng = np.random.RandomState(42)
    response = strategy.simulate_response(item, template, "mlm_score", rng)

    assert isinstance(response, dict)
    assert "determiner" in response
    assert response["determiner"] in ["the", "a", "an"]
    # "the" has highest score, should be selected most often
    responses = [
        strategy.simulate_response(
            item, template, "mlm_score", np.random.RandomState(i)
        )
        for i in range(100)
    ]
    the_count = sum(1 for r in responses if r["determiner"] == "the")
    assert the_count > 60  # Should strongly prefer "the"


def test_simulate_response_multiple_slots() -> None:
    """Test cloze response with multiple slots."""
    strategy = ClozeStrategy()

    template = ItemTemplate(
        name="test_cloze",
        judgment_type="comprehension",
        task_type="cloze",
        task_spec=TaskSpec(prompt="Fill in the blanks:"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        unfilled_slots=[
            UnfilledSlot(slot_name="determiner", position=0, constraint_ids=[]),
            UnfilledSlot(slot_name="noun", position=2, constraint_ids=[]),
        ],
        model_outputs=[
            # Determiner scores
            _create_mlm_output("determiner", "the", 10.0, 0),
            _create_mlm_output("determiner", "a", 5.0, 1),
            # Noun scores
            _create_mlm_output("noun", "cat", 8.0, 2),
            _create_mlm_output("noun", "dog", 4.0, 3),
        ],
    )

    rng = np.random.RandomState(42)
    response = strategy.simulate_response(item, template, "mlm_score", rng)

    assert isinstance(response, dict)
    assert "determiner" in response
    assert "noun" in response
    assert response["determiner"] in ["the", "a"]
    assert response["noun"] in ["cat", "dog"]


def test_simulate_response_without_mlm_scores() -> None:
    """Test fallback when MLM scores unavailable."""
    strategy = ClozeStrategy()

    template = ItemTemplate(
        name="test_cloze",
        judgment_type="comprehension",
        task_type="cloze",
        task_spec=TaskSpec(prompt="Fill in the blanks:"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # Item without MLM scores
    item = Item(
        item_template_id=uuid4(),
        unfilled_slots=[
            UnfilledSlot(slot_name="determiner", position=0, constraint_ids=[])
        ],
    )

    rng = np.random.RandomState(42)
    response = strategy.simulate_response(item, template, "mlm_score", rng)

    assert isinstance(response, dict)
    assert "determiner" in response
    # Should use fallback (common determiner words)
    assert isinstance(response["determiner"], str)
    assert len(response["determiner"]) > 0


def test_simulate_response_with_ground_truth_fallback() -> None:
    """Test using ground truth as fallback."""
    strategy = ClozeStrategy()

    template = ItemTemplate(
        name="test_cloze",
        judgment_type="comprehension",
        task_type="cloze",
        task_spec=TaskSpec(prompt="Fill in the blanks:"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # Item with ground truth but no MLM scores
    item = Item(
        item_template_id=uuid4(),
        unfilled_slots=[
            UnfilledSlot(slot_name="determiner", position=0, constraint_ids=[])
        ],
        item_metadata={"ground_truth": {"determiner": "the"}},
    )

    rng = np.random.RandomState(42)
    response = strategy.simulate_response(item, template, "mlm_score", rng)

    assert response["determiner"] == "the"


def test_simulate_response_deterministic_with_seed() -> None:
    """Test that responses are deterministic with same seed."""
    strategy = ClozeStrategy()

    template = ItemTemplate(
        name="test_cloze",
        judgment_type="comprehension",
        task_type="cloze",
        task_spec=TaskSpec(prompt="Fill in the blanks:"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        unfilled_slots=[
            UnfilledSlot(slot_name="determiner", position=0, constraint_ids=[])
        ],
        model_outputs=[
            _create_mlm_output("determiner", "the", 10.0, 0),
            _create_mlm_output("determiner", "a", 9.0, 1),
        ],
    )

    rng1 = np.random.RandomState(123)
    response1 = strategy.simulate_response(item, template, "mlm_score", rng1)

    rng2 = np.random.RandomState(123)
    response2 = strategy.simulate_response(item, template, "mlm_score", rng2)

    assert response1 == response2


def test_get_slot_scores() -> None:
    """Test extracting scores for a specific slot."""
    strategy = ClozeStrategy()

    item = Item(
        item_template_id=uuid4(),
        model_outputs=[
            _create_mlm_output("determiner", "the", 10.0, 0),
            _create_mlm_output("determiner", "a", 5.0, 1),
            _create_mlm_output("noun", "cat", 8.0, 2),
        ],
    )

    scores = strategy._get_slot_scores(item, "determiner", "mlm_score")

    assert len(scores) == 2
    assert scores["the"] == 10.0
    assert scores["a"] == 5.0
    assert "cat" not in scores  # Different slot


def test_get_slot_scores_empty() -> None:
    """Test extracting scores when none available."""
    strategy = ClozeStrategy()

    item = Item(item_template_id=uuid4(), model_outputs=[])

    scores = strategy._get_slot_scores(item, "determiner", "mlm_score")

    assert len(scores) == 0


def test_get_fallback_filler_by_slot_name() -> None:
    """Test fallback filler selection based on slot name."""
    strategy = ClozeStrategy()

    item = Item(item_template_id=uuid4())
    rng = np.random.RandomState(42)

    # Test different slot name patterns
    filler = strategy._get_fallback_filler(item, "determiner", rng)
    assert filler in ["the", "a", "an", "this", "that"]

    filler = strategy._get_fallback_filler(item, "verb", rng)
    assert filler in ["is", "was", "has", "can", "will"]

    filler = strategy._get_fallback_filler(item, "noun", rng)
    assert filler in ["thing", "person", "place", "time", "way"]

    filler = strategy._get_fallback_filler(item, "adjective", rng)
    assert filler in ["good", "new", "old", "big", "small"]


def test_get_fallback_filler_generic() -> None:
    """Test generic fallback for unknown slot names."""
    strategy = ClozeStrategy()

    item = Item(item_template_id=uuid4())
    rng = np.random.RandomState(42)

    filler = strategy._get_fallback_filler(item, "unknown_slot_type", rng)

    assert filler == "[unknown_slot_type]"


def test_simulate_response_mixed_slots() -> None:
    """Test with some slots having scores and others not."""
    strategy = ClozeStrategy()

    template = ItemTemplate(
        name="test_cloze",
        judgment_type="comprehension",
        task_type="cloze",
        task_spec=TaskSpec(prompt="Fill in the blanks:"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        unfilled_slots=[
            UnfilledSlot(slot_name="determiner", position=0, constraint_ids=[]),
            UnfilledSlot(slot_name="noun", position=2, constraint_ids=[]),
        ],
        model_outputs=[
            # Only determiner has MLM scores
            _create_mlm_output("determiner", "the", 10.0, 0),
            _create_mlm_output("determiner", "a", 5.0, 1),
        ],
    )

    rng = np.random.RandomState(42)
    response = strategy.simulate_response(item, template, "mlm_score", rng)

    assert isinstance(response, dict)
    assert "determiner" in response
    assert "noun" in response
    # Determiner should use MLM scores
    assert response["determiner"] in ["the", "a"]
    # Noun should use fallback
    assert isinstance(response["noun"], str)
