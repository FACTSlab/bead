"""Test filling strategies."""

from __future__ import annotations

from bead.resources.models import LexicalItem
from bead.templates.strategies import (
    ExhaustiveStrategy,
    RandomStrategy,
    StratifiedStrategy,
)


def test_exhaustive_strategy_two_slots() -> None:
    """Test exhaustive strategy with two slots."""
    strategy = ExhaustiveStrategy()
    slot_items = {
        "a": [
            LexicalItem(lemma="x", pos="A", language_code="en"),
            LexicalItem(lemma="y", pos="A", language_code="en"),
        ],
        "b": [
            LexicalItem(lemma="1", pos="B", language_code="en"),
            LexicalItem(lemma="2", pos="B", language_code="en"),
        ],
    }

    combinations = strategy.generate_combinations(slot_items)

    assert len(combinations) == 4
    assert all(isinstance(c, dict) for c in combinations)
    assert all("a" in c and "b" in c for c in combinations)


def test_exhaustive_strategy_three_slots() -> None:
    """Test exhaustive strategy with three slots."""
    strategy = ExhaustiveStrategy()
    slot_items = {
        "a": [LexicalItem(lemma="x", pos="A", language_code="en")],
        "b": [
            LexicalItem(lemma="1", pos="B", language_code="en"),
            LexicalItem(lemma="2", pos="B", language_code="en"),
        ],
        "c": [
            LexicalItem(lemma="α", pos="C", language_code="en"),
            LexicalItem(lemma="β", pos="C", language_code="en"),
        ],
    }

    combinations = strategy.generate_combinations(slot_items)

    assert len(combinations) == 4  # 1 * 2 * 2


def test_exhaustive_strategy_name() -> None:
    """Test exhaustive strategy name."""
    strategy = ExhaustiveStrategy()

    assert strategy.name == "exhaustive"


def test_exhaustive_strategy_empty_slot() -> None:
    """Test exhaustive strategy with empty slot."""
    strategy = ExhaustiveStrategy()
    slot_items = {
        "a": [LexicalItem(lemma="x", pos="A", language_code="en")],
        "b": [],
    }

    combinations = strategy.generate_combinations(slot_items)

    assert len(combinations) == 0


def test_exhaustive_strategy_empty_dict() -> None:
    """Test exhaustive strategy with empty slot items dict."""
    strategy = ExhaustiveStrategy()
    slot_items: dict[str, list[LexicalItem]] = {}

    combinations = strategy.generate_combinations(slot_items)

    assert len(combinations) == 0


def test_exhaustive_strategy_single_slot() -> None:
    """Test exhaustive strategy with single slot."""
    strategy = ExhaustiveStrategy()
    slot_items = {
        "a": [
            LexicalItem(lemma="x", pos="A", language_code="en"),
            LexicalItem(lemma="y", pos="A", language_code="en"),
            LexicalItem(lemma="z", pos="A", language_code="en"),
        ],
    }

    combinations = strategy.generate_combinations(slot_items)

    assert len(combinations) == 3
    assert all("a" in c for c in combinations)


def test_random_strategy_sample_size() -> None:
    """Test random strategy respects sample size."""
    strategy = RandomStrategy(n_samples=5, seed=42)
    slot_items = {
        "a": [
            LexicalItem(lemma=str(i), pos="A", language_code="en") for i in range(10)
        ],
        "b": [
            LexicalItem(lemma=str(i), pos="B", language_code="en") for i in range(10)
        ],
    }

    combinations = strategy.generate_combinations(slot_items)

    assert len(combinations) == 5


def test_random_strategy_deterministic() -> None:
    """Test random strategy is deterministic with seed."""
    strategy1 = RandomStrategy(n_samples=10, seed=42)
    strategy2 = RandomStrategy(n_samples=10, seed=42)
    slot_items = {
        "a": [LexicalItem(lemma=str(i), pos="A", language_code="en") for i in range(5)],
        "b": [LexicalItem(lemma=str(i), pos="B", language_code="en") for i in range(5)],
    }

    combinations1 = strategy1.generate_combinations(slot_items)
    combinations2 = strategy2.generate_combinations(slot_items)

    # Check that same items are selected in same order
    assert len(combinations1) == len(combinations2)
    for c1, c2 in zip(combinations1, combinations2, strict=True):
        assert c1["a"].lemma == c2["a"].lemma
        assert c1["b"].lemma == c2["b"].lemma


def test_random_strategy_name() -> None:
    """Test random strategy name."""
    strategy = RandomStrategy(n_samples=10, seed=42)

    assert strategy.name == "random"


def test_random_strategy_empty_slot() -> None:
    """Test random strategy with empty slot items dict."""
    strategy = RandomStrategy(n_samples=5, seed=42)
    slot_items: dict[str, list[LexicalItem]] = {}

    combinations = strategy.generate_combinations(slot_items)

    assert len(combinations) == 0


def test_random_strategy_single_slot() -> None:
    """Test random strategy with single slot."""
    strategy = RandomStrategy(n_samples=3, seed=42)
    slot_items = {
        "a": [
            LexicalItem(lemma=str(i), pos="A", language_code="en") for i in range(10)
        ],
    }

    combinations = strategy.generate_combinations(slot_items)

    assert len(combinations) == 3
    assert all("a" in c for c in combinations)


def test_stratified_strategy_balanced() -> None:
    """Test stratified strategy balances groups."""
    strategy = StratifiedStrategy(
        n_samples=10,
        grouping_property="pos",
        seed=42,
    )

    # Create items with different POS values
    verbs = [
        LexicalItem(lemma=f"v{i}", pos="VERB", language_code="en") for i in range(10)
    ]
    nouns = [
        LexicalItem(lemma=f"n{i}", pos="NOUN", language_code="en") for i in range(10)
    ]

    slot_items = {
        "slot": verbs + nouns,
    }

    combinations = strategy.generate_combinations(slot_items)

    assert len(combinations) == 10


def test_stratified_strategy_name() -> None:
    """Test stratified strategy name."""
    strategy = StratifiedStrategy(
        n_samples=10,
        grouping_property="pos",
        seed=42,
    )

    assert strategy.name == "stratified"


def test_stratified_strategy_deterministic() -> None:
    """Test stratified strategy is deterministic with seed."""
    strategy1 = StratifiedStrategy(n_samples=10, grouping_property="pos", seed=42)
    strategy2 = StratifiedStrategy(n_samples=10, grouping_property="pos", seed=42)

    verbs = [
        LexicalItem(lemma=f"v{i}", pos="VERB", language_code="en") for i in range(5)
    ]
    nouns = [
        LexicalItem(lemma=f"n{i}", pos="NOUN", language_code="en") for i in range(5)
    ]

    slot_items = {"slot": verbs + nouns}

    combinations1 = strategy1.generate_combinations(slot_items)
    combinations2 = strategy2.generate_combinations(slot_items)

    assert len(combinations1) == len(combinations2)
    for c1, c2 in zip(combinations1, combinations2, strict=True):
        assert c1["slot"].lemma == c2["slot"].lemma


def test_stratified_strategy_empty_slot() -> None:
    """Test stratified strategy with empty slot items dict."""
    strategy = StratifiedStrategy(n_samples=10, grouping_property="pos", seed=42)
    slot_items: dict[str, list[LexicalItem]] = {}

    combinations = strategy.generate_combinations(slot_items)

    assert len(combinations) == 0


def test_stratified_strategy_multiple_slots() -> None:
    """Test stratified strategy with multiple slots."""
    strategy = StratifiedStrategy(n_samples=10, grouping_property="pos", seed=42)

    slot_items = {
        "a": [
            LexicalItem(lemma="v1", pos="VERB", language_code="en"),
            LexicalItem(lemma="n1", pos="NOUN", language_code="en"),
        ],
        "b": [
            LexicalItem(lemma="v2", pos="VERB", language_code="en"),
            LexicalItem(lemma="n2", pos="NOUN", language_code="en"),
        ],
    }

    combinations = strategy.generate_combinations(slot_items)

    assert len(combinations) == 10
    assert all("a" in c and "b" in c for c in combinations)
