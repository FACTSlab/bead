"""Tests for MixedFillingStrategy."""

from __future__ import annotations

import pytest

from bead.resources.constraints import Constraint
from bead.resources.lexical_item import LexicalItem
from bead.resources.lexicon import Lexicon
from bead.resources.template import Slot, Template
from bead.templates.resolver import ConstraintResolver
from bead.templates.strategies import (
    ExhaustiveStrategy,
    MixedFillingStrategy,
    RandomStrategy,
)


@pytest.fixture
def simple_lexicon() -> Lexicon:
    """Create a simple lexicon for testing.

    Returns
    -------
    Lexicon
        Lexicon with nouns, verbs, and adjectives
    """
    items = [
        # Nouns
        LexicalItem(lemma="cat", language_code="eng", features={"pos": "NOUN"}),
        LexicalItem(lemma="dog", language_code="eng", features={"pos": "NOUN"}),
        # Verbs
        LexicalItem(lemma="runs", language_code="eng", features={"pos": "VERB"}),
        LexicalItem(lemma="jumps", language_code="eng", features={"pos": "VERB"}),
        # Adjectives
        LexicalItem(lemma="big", language_code="eng", features={"pos": "ADJ"}),
        LexicalItem(lemma="small", language_code="eng", features={"pos": "ADJ"}),
    ]
    return Lexicon(name="test", items={str(item.id): item for item in items})


@pytest.fixture
def simple_template() -> Template:
    """Create a simple template for testing.

    Returns
    -------
    Template
        Template with noun, verb, adjective slots
    """
    return Template(
        name="test_template",
        template_string="{det} {adjective} {noun} {verb}",
        slots={
            "det": Slot(
                name="det",
                constraints=[
                    Constraint(
                        expression="self.lemma in ['a', 'the']",
                        context={},
                    )
                ],
            ),
            "noun": Slot(
                name="noun",
                constraints=[
                    Constraint(
                        expression="self.features.get('pos') == 'NOUN'",
                        context={},
                    )
                ],
            ),
            "verb": Slot(
                name="verb",
                constraints=[
                    Constraint(
                        expression="self.features.get('pos') == 'VERB'",
                        context={},
                    )
                ],
            ),
            "adjective": Slot(
                name="adjective",
                constraints=[
                    Constraint(
                        expression="self.features.get('pos') == 'ADJ'",
                        context={},
                    )
                ],
            ),
        },
    )


def test_mixed_strategy_initialization():
    """Test MixedFillingStrategy initialization."""
    strategy = MixedFillingStrategy(
        slot_strategies={
            "noun": ("exhaustive", {}),
            "verb": ("exhaustive", {}),
            "adjective": ("exhaustive", {}),
        }
    )
    assert strategy.name == "mixed"
    assert len(strategy.phase1_slots) == 3
    assert len(strategy.phase2_slots) == 0


def test_mixed_strategy_with_mlm_slots():
    """Test MixedFillingStrategy with MLM slots."""
    # Create mock MLM config (without actual model)
    mlm_config = {
        "resolver": ConstraintResolver(),
        # These would need actual model in integration tests
        "model_adapter": None,  # Mock
        "beam_size": 5,
        "top_k": 10,
    }

    strategy = MixedFillingStrategy(
        slot_strategies={
            "noun": ("exhaustive", {}),
            "verb": ("exhaustive", {}),
            "adjective": ("mlm", mlm_config),
        }
    )
    assert len(strategy.phase1_slots) == 2
    assert len(strategy.phase2_slots) == 1
    assert "adjective" in strategy.phase2_slots


def test_mixed_strategy_only_exhaustive(simple_lexicon: Lexicon):
    """Test MixedFillingStrategy with only exhaustive strategies."""
    strategy = MixedFillingStrategy(
        slot_strategies={
            "noun": ("exhaustive", {}),
            "verb": ("exhaustive", {}),
        }
    )

    # Create simple slot items
    slot_items = {
        "noun": [
            LexicalItem(lemma="cat", language_code="eng", features={"pos": "NOUN"}),
            LexicalItem(lemma="dog", language_code="eng", features={"pos": "NOUN"}),
        ],
        "verb": [
            LexicalItem(lemma="runs", language_code="eng", features={"pos": "VERB"}),
            LexicalItem(lemma="jumps", language_code="eng", features={"pos": "VERB"}),
        ],
    }

    combinations = strategy.generate_combinations(slot_items)

    # Should generate cartesian product: 2 * 2 = 4 combinations
    assert len(combinations) == 4
    assert all("noun" in combo and "verb" in combo for combo in combinations)


def test_mixed_strategy_random_sampling():
    """Test MixedFillingStrategy with random sampling."""
    strategy = MixedFillingStrategy(
        slot_strategies={
            "noun": ("random", {"n_samples": 2, "seed": 42}),
            "verb": ("exhaustive", {}),
        }
    )

    slot_items = {
        "noun": [
            LexicalItem(lemma="cat", language_code="eng", features={"pos": "NOUN"}),
            LexicalItem(lemma="dog", language_code="eng", features={"pos": "NOUN"}),
            LexicalItem(lemma="bird", language_code="eng", features={"pos": "NOUN"}),
        ],
        "verb": [
            LexicalItem(lemma="runs", language_code="eng", features={"pos": "VERB"}),
            LexicalItem(lemma="jumps", language_code="eng", features={"pos": "VERB"}),
        ],
    }

    combinations = strategy.generate_combinations(slot_items)

    # Random strategy samples 2 nouns, exhaustive gives 2 verbs
    # So we expect 2 * 2 = 4 combinations
    assert len(combinations) == 4


def test_mixed_strategy_raises_with_mlm():
    """Test that MixedFillingStrategy raises when MLM used without template context."""
    mlm_config = {
        "resolver": ConstraintResolver(),
        "model_adapter": None,  # Mock
        "beam_size": 5,
        "top_k": 10,
    }

    strategy = MixedFillingStrategy(
        slot_strategies={
            "noun": ("exhaustive", {}),
            "adjective": ("mlm", mlm_config),
        }
    )

    slot_items = {
        "noun": [
            LexicalItem(lemma="cat", language_code="eng", features={"pos": "NOUN"})
        ],
        "adjective": [
            LexicalItem(lemma="big", language_code="eng", features={"pos": "ADJ"})
        ],
    }

    # Should raise because MLM requires template context
    with pytest.raises(NotImplementedError, match="requires template context"):
        strategy.generate_combinations(slot_items)


def test_mixed_strategy_instantiate_strategies():
    """Test strategy instantiation from names."""
    strategy = MixedFillingStrategy(
        slot_strategies={
            "noun": ("exhaustive", {}),
            "verb": ("random", {"n_samples": 10, "seed": 42}),
        }
    )

    # Check that strategies were instantiated correctly
    assert "noun" in strategy.phase1_strategies
    assert "verb" in strategy.phase1_strategies
    assert isinstance(strategy.phase1_strategies["noun"], ExhaustiveStrategy)
    assert isinstance(strategy.phase1_strategies["verb"], RandomStrategy)


def test_mixed_strategy_unknown_strategy_raises():
    """Test that unknown strategy names raise ValueError."""
    with pytest.raises(ValueError, match="Unknown strategy"):
        MixedFillingStrategy(
            slot_strategies={
                "noun": ("unknown_strategy", {}),
            }
        )


def test_mixed_strategy_default_strategy():
    """Test that default strategy is used for unspecified slots."""
    strategy = MixedFillingStrategy(
        slot_strategies={
            "noun": ("exhaustive", {}),
        },
        default_strategy=RandomStrategy(n_samples=5, seed=42),
    )

    # noun has explicit strategy, verb uses default
    slot_items = {
        "noun": [
            LexicalItem(lemma="cat", language_code="eng", features={"pos": "NOUN"}),
            LexicalItem(lemma="dog", language_code="eng", features={"pos": "NOUN"}),
        ],
        "verb": [
            LexicalItem(lemma="runs", language_code="eng", features={"pos": "VERB"}),
            LexicalItem(lemma="jumps", language_code="eng", features={"pos": "VERB"}),
            LexicalItem(lemma="walks", language_code="eng", features={"pos": "VERB"}),
        ],
    }

    combinations = strategy.generate_combinations(slot_items)

    # noun: 2 items exhaustive
    # verb: 5 samples from 3 items
    # Total: 2 * 5 = 10
    assert len(combinations) == 10


def test_mixed_strategy_empty_slots():
    """Test MixedFillingStrategy with empty slot items."""
    strategy = MixedFillingStrategy(
        slot_strategies={
            "noun": ("exhaustive", {}),
            "verb": ("exhaustive", {}),
        }
    )

    slot_items: dict[str, list[LexicalItem]] = {
        "noun": [],
        "verb": [
            LexicalItem(lemma="runs", language_code="eng", features={"pos": "VERB"})
        ],
    }

    combinations = strategy.generate_combinations(slot_items)

    # Empty noun slot means no combinations possible
    assert len(combinations) == 0


def test_mixed_strategy_single_slot():
    """Test MixedFillingStrategy with a single slot."""
    strategy = MixedFillingStrategy(
        slot_strategies={
            "noun": ("exhaustive", {}),
        }
    )

    slot_items = {
        "noun": [
            LexicalItem(lemma="cat", language_code="eng", features={"pos": "NOUN"}),
            LexicalItem(lemma="dog", language_code="eng", features={"pos": "NOUN"}),
        ],
    }

    combinations = strategy.generate_combinations(slot_items)

    assert len(combinations) == 2
    assert all("noun" in combo for combo in combinations)


def test_config_validation():
    """Test that configuration validation works."""
    from bead.config.template import TemplateConfig

    # Valid mixed configuration
    config = TemplateConfig(
        filling_strategy="mixed",
        slot_strategies={
            "noun": {"strategy": "exhaustive"},
            "verb": {"strategy": "exhaustive"},
            "adjective": {"strategy": "mlm"},
        },
        mlm_model_name="bert-base-uncased",
    )
    assert config.filling_strategy == "mixed"
    assert config.slot_strategies is not None

    # Invalid: mixed without slot_strategies
    with pytest.raises(ValueError, match="slot_strategies must be specified"):
        TemplateConfig(filling_strategy="mixed", slot_strategies=None)

    # Invalid: MLM slot without mlm_model_name
    with pytest.raises(ValueError, match="mlm_model_name must be specified"):
        TemplateConfig(
            filling_strategy="mixed",
            slot_strategies={"adjective": {"strategy": "mlm"}},
            mlm_model_name=None,
        )

    # Invalid: missing strategy key
    with pytest.raises(ValueError, match="'strategy' key required"):
        TemplateConfig(
            filling_strategy="mixed",
            slot_strategies={"noun": {}},
        )

    # Invalid: unknown strategy
    with pytest.raises(ValueError, match="Invalid strategy"):
        TemplateConfig(
            filling_strategy="mixed",
            slot_strategies={"noun": {"strategy": "unknown"}},
        )
