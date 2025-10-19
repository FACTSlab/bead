"""Test ConstraintResolver."""

from __future__ import annotations

import pytest

from sash.resources.constraints import (
    DSLConstraint,
    ExtensionalConstraint,
    IntensionalConstraint,
    RelationalConstraint,
)
from sash.resources.lexicon import Lexicon
from sash.resources.models import LexicalItem
from sash.templates.resolver import ConstraintResolver


def test_resolver_initialization(
    sample_lexicon: Lexicon,
    adapter_registry: object,
) -> None:
    """Test ConstraintResolver initialization."""
    resolver = ConstraintResolver(
        lexicon=sample_lexicon,
        adapter_registry=adapter_registry,
    )
    assert resolver.lexicon == sample_lexicon
    assert resolver.adapter_registry == adapter_registry
    assert resolver.cache_results is True


def test_resolver_initialization_no_registry(sample_lexicon: Lexicon) -> None:
    """Test ConstraintResolver can be created without adapter registry."""
    resolver = ConstraintResolver(lexicon=sample_lexicon)
    assert resolver.adapter_registry is None


# Extensional Constraints Tests


def test_resolve_extensional_allow_mode(resolver: ConstraintResolver) -> None:
    """Test resolving extensional constraint with allow mode."""
    # Get IDs of break and shatter items
    break_item = next(
        item for item in resolver.lexicon.items.values() if item.lemma == "break"
    )
    shatter_item = next(
        item for item in resolver.lexicon.items.values() if item.lemma == "shatter"
    )

    constraint = ExtensionalConstraint(
        mode="allow", items=[break_item.id, shatter_item.id]
    )
    items = resolver.resolve(constraint)

    assert len(items) == 2
    assert all(item.lemma in {"break", "shatter"} for item in items)


def test_resolve_extensional_deny_mode(resolver: ConstraintResolver) -> None:
    """Test resolving extensional constraint with deny mode."""
    # Get IDs of break and arrive items
    break_item = next(
        item for item in resolver.lexicon.items.values() if item.lemma == "break"
    )
    arrive_item = next(
        item for item in resolver.lexicon.items.values() if item.lemma == "arrive"
    )

    constraint = ExtensionalConstraint(
        mode="deny", items=[break_item.id, arrive_item.id]
    )
    items = resolver.resolve(constraint)

    # Should get all items except break and arrive
    assert len(items) > 0
    assert all(item.lemma not in {"break", "arrive"} for item in items)


def test_evaluate_extensional_single_item(resolver: ConstraintResolver) -> None:
    """Test evaluating extensional constraint on single item."""
    item = LexicalItem(lemma="break", pos="VERB")
    constraint = ExtensionalConstraint(mode="allow", items=[item.id])

    assert resolver.evaluate(constraint, item) is True

    item2 = LexicalItem(lemma="shatter", pos="VERB")
    assert resolver.evaluate(constraint, item2) is False


def test_extensional_empty_allow_list(resolver: ConstraintResolver) -> None:
    """Test extensional constraint with empty allow list matches nothing."""
    constraint = ExtensionalConstraint(mode="allow", items=[])
    items = resolver.resolve(constraint)

    assert len(items) == 0


def test_extensional_empty_deny_list(resolver: ConstraintResolver) -> None:
    """Test extensional constraint with empty deny list matches everything."""
    constraint = ExtensionalConstraint(mode="deny", items=[])
    items = resolver.resolve(constraint)

    # Should match all items in lexicon
    assert len(items) == len(resolver.lexicon.items)


# Intensional Constraints Tests


def test_resolve_intensional_single_property(
    resolver: ConstraintResolver,
) -> None:
    """Test resolving intensional constraint with single property."""
    constraint = IntensionalConstraint(property="pos", operator="==", value="VERB")
    items = resolver.resolve(constraint)

    assert len(items) > 0
    assert all(item.pos == "VERB" for item in items)


def test_resolve_intensional_not_equal_operator(
    resolver: ConstraintResolver,
) -> None:
    """Test resolving intensional constraint with != operator."""
    constraint = IntensionalConstraint(property="pos", operator="!=", value="VERB")
    items = resolver.resolve(constraint)

    assert len(items) > 0
    assert all(item.pos != "VERB" for item in items)


def test_resolve_intensional_nested_property(
    resolver: ConstraintResolver,
) -> None:
    """Test resolving intensional constraint with nested property."""
    constraint = IntensionalConstraint(
        property="features.transitivity", operator="==", value="transitive"
    )
    items = resolver.resolve(constraint)

    assert len(items) > 0
    assert all(item.features.get("transitivity") == "transitive" for item in items)


def test_resolve_intensional_feature_boolean(
    resolver: ConstraintResolver,
) -> None:
    """Test intensional constraint matching boolean feature."""
    constraint = IntensionalConstraint(
        property="features.causative", operator="==", value=True
    )
    items = resolver.resolve(constraint)

    assert len(items) > 0
    assert all(item.features.get("causative") is True for item in items)


def test_resolve_intensional_attribute_match(
    resolver: ConstraintResolver,
) -> None:
    """Test intensional constraint matching attributes."""
    constraint = IntensionalConstraint(
        property="attributes.frequency", operator="==", value="high"
    )
    items = resolver.resolve(constraint)

    assert len(items) > 0
    assert all(item.attributes.get("frequency") == "high" for item in items)


def test_resolve_intensional_in_operator(
    resolver: ConstraintResolver,
) -> None:
    """Test intensional constraint with 'in' operator."""
    constraint = IntensionalConstraint(
        property="pos", operator="in", value=["VERB", "NOUN"]
    )
    items = resolver.resolve(constraint)

    assert len(items) > 0
    assert all(item.pos in ["VERB", "NOUN"] for item in items)


def test_resolve_intensional_not_in_operator(
    resolver: ConstraintResolver,
) -> None:
    """Test intensional constraint with 'not in' operator."""
    constraint = IntensionalConstraint(
        property="pos", operator="not in", value=["VERB", "NOUN"]
    )
    items = resolver.resolve(constraint)

    assert len(items) > 0
    assert all(item.pos not in ["VERB", "NOUN"] for item in items)


def test_evaluate_intensional_no_match(resolver: ConstraintResolver) -> None:
    """Test intensional constraint evaluation returns False when no match."""
    constraint = IntensionalConstraint(
        property="features.nonexistent", operator="==", value="value"
    )
    item = LexicalItem(lemma="break", pos="VERB", features={})

    assert resolver.evaluate(constraint, item) is False


def test_evaluate_intensional_missing_property(resolver: ConstraintResolver) -> None:
    """Test intensional constraint with missing property returns False."""
    constraint = IntensionalConstraint(
        property="nonexistent_field", operator="==", value="test"
    )
    item = LexicalItem(lemma="test", pos="VERB")

    assert resolver.evaluate(constraint, item) is False


# Relational Constraints Tests


def test_evaluate_relational_returns_false(resolver: ConstraintResolver) -> None:
    """Test relational constraint evaluation returns False.

    Current implementation cannot evaluate slot relationships with single item.
    """
    constraint = RelationalConstraint(
        slot_a="subject",
        slot_b="object",
        relation="different",
        property="lemma",
    )
    item = LexicalItem(lemma="break", pos="VERB")

    # Relational constraints require multiple slots, so should return False
    assert resolver.evaluate(constraint, item) is False


def test_resolve_relational_no_matches(resolver: ConstraintResolver) -> None:
    """Test resolving relational constraint matches no items.

    Since relational constraints cannot be evaluated on single items,
    they should match nothing.
    """
    constraint = RelationalConstraint(
        slot_a="subject",
        slot_b="object",
        relation="same",
        property="pos",
    )
    items = resolver.resolve(constraint)

    assert len(items) == 0


# DSL Constraints Tests


def test_resolve_dsl_simple_expression(resolver: ConstraintResolver) -> None:
    """Test resolving DSL constraint with simple expression."""
    constraint = DSLConstraint(expression='pos == "VERB"')
    items = resolver.resolve(constraint)

    assert len(items) > 0
    assert all(item.pos == "VERB" for item in items)


def test_resolve_dsl_complex_expression(resolver: ConstraintResolver) -> None:
    """Test resolving DSL constraint with complex expression."""
    constraint = DSLConstraint(
        expression='pos == "VERB" and transitivity == "transitive"'
    )
    items = resolver.resolve(constraint)

    assert len(items) > 0
    assert all(item.pos == "VERB" for item in items)
    assert all(item.features.get("transitivity") == "transitive" for item in items)


def test_resolve_dsl_with_boolean_features(resolver: ConstraintResolver) -> None:
    """Test DSL constraint with boolean feature access."""
    constraint = DSLConstraint(expression="causative == true")
    items = resolver.resolve(constraint)

    assert len(items) > 0
    assert all(item.features.get("causative") is True for item in items)


def test_resolve_dsl_with_logical_operators(resolver: ConstraintResolver) -> None:
    """Test DSL constraint using logical AND/OR operators."""
    constraint = DSLConstraint(expression='pos == "VERB" or pos == "NOUN"')
    items = resolver.resolve(constraint)

    assert len(items) > 0
    assert all(item.pos in ["VERB", "NOUN"] for item in items)


def test_evaluate_dsl_with_functions(resolver: ConstraintResolver) -> None:
    """Test DSL constraint using standard library functions."""
    constraint = DSLConstraint(expression="len(lemma) > 5")
    item = LexicalItem(lemma="happiness", pos="NOUN")

    assert resolver.evaluate(constraint, item) is True

    item2 = LexicalItem(lemma="cat", pos="NOUN")
    assert resolver.evaluate(constraint, item2) is False


def test_resolve_dsl_with_not_operator(resolver: ConstraintResolver) -> None:
    """Test DSL constraint with NOT operator."""
    constraint = DSLConstraint(expression='not (pos == "VERB")')
    items = resolver.resolve(constraint)

    assert len(items) > 0
    assert all(item.pos != "VERB" for item in items)


def test_resolve_dsl_comparison_operators(resolver: ConstraintResolver) -> None:
    """Test DSL constraint with comparison operators."""
    # First add some numeric attributes to test
    for item in resolver.lexicon.items.values():
        if item.attributes.get("frequency") == "high":
            item.attributes["freq_count"] = 1000
        elif item.attributes.get("frequency") == "medium":
            item.attributes["freq_count"] = 500

    constraint = DSLConstraint(expression="attributes.freq_count > 700")
    items = resolver.resolve(constraint)

    assert len(items) > 0
    assert all(item.attributes.get("freq_count", 0) > 700 for item in items)


# Language Code Filtering Tests


def test_resolve_with_language_filter(resolver: ConstraintResolver) -> None:
    """Test resolving constraint filtered by language code."""
    constraint = IntensionalConstraint(property="pos", operator="==", value="VERB")
    items = resolver.resolve(constraint, language_code="en")

    assert len(items) > 0
    assert all(item.language_code == "en" for item in items)
    assert all(item.pos == "VERB" for item in items)


def test_resolve_language_filter_korean(resolver: ConstraintResolver) -> None:
    """Test resolving constraint filtered for Korean language."""
    constraint = IntensionalConstraint(property="pos", operator="==", value="VERB")
    items = resolver.resolve(constraint, language_code="ko")

    assert len(items) > 0
    assert all(item.language_code == "ko" for item in items)
    # Should only match kkakta
    assert all(item.lemma == "kkakta" for item in items)


def test_resolve_multilingual_no_filter(resolver: ConstraintResolver) -> None:
    """Test resolving constraint across all languages."""
    constraint = IntensionalConstraint(
        property="features.causative", operator="==", value=True
    )
    items = resolver.resolve(constraint)

    # Should get items from both English and Korean
    languages = {item.language_code for item in items}
    assert "en" in languages
    assert "ko" in languages


# Caching Tests


def test_cache_enabled(resolver: ConstraintResolver) -> None:
    """Test that caching works when enabled."""
    constraint = IntensionalConstraint(property="pos", operator="==", value="VERB")

    # First evaluation
    items1 = resolver.resolve(constraint)
    cache_size_1 = len(resolver._cache)

    # Second evaluation (should use cache)
    items2 = resolver.resolve(constraint)
    cache_size_2 = len(resolver._cache)

    assert items1 == items2
    assert cache_size_1 > 0
    assert cache_size_2 == cache_size_1  # Cache size unchanged


def test_cache_disabled(resolver_no_cache: ConstraintResolver) -> None:
    """Test that caching is disabled when cache_results=False."""
    constraint = IntensionalConstraint(property="pos", operator="==", value="VERB")

    resolver_no_cache.resolve(constraint)

    assert len(resolver_no_cache._cache) == 0


def test_clear_cache(resolver: ConstraintResolver) -> None:
    """Test clearing the cache."""
    constraint = IntensionalConstraint(property="pos", operator="==", value="VERB")

    # Populate cache
    resolver.resolve(constraint)
    assert len(resolver._cache) > 0

    # Clear cache
    resolver.clear_cache()
    assert len(resolver._cache) == 0


def test_cache_key_generation(resolver: ConstraintResolver) -> None:
    """Test cache key generation is deterministic."""
    constraint = IntensionalConstraint(property="pos", operator="==", value="VERB")
    item = LexicalItem(lemma="break", pos="VERB")

    key1 = resolver._get_cache_key(constraint, item)
    key2 = resolver._get_cache_key(constraint, item)

    assert key1 == key2
    assert isinstance(key1, tuple)
    assert len(key1) == 2


def test_cache_different_constraints_different_keys(
    resolver: ConstraintResolver,
) -> None:
    """Test different constraints generate different cache keys."""
    constraint1 = IntensionalConstraint(property="pos", operator="==", value="VERB")
    constraint2 = IntensionalConstraint(property="pos", operator="==", value="NOUN")
    item = LexicalItem(lemma="break", pos="VERB")

    key1 = resolver._get_cache_key(constraint1, item)
    key2 = resolver._get_cache_key(constraint2, item)

    assert key1 != key2


# Edge Cases and Error Handling


def test_resolve_empty_lexicon() -> None:
    """Test resolving constraint against empty lexicon."""
    empty_lexicon = Lexicon(name="empty")
    resolver = ConstraintResolver(lexicon=empty_lexicon)
    constraint = IntensionalConstraint(property="pos", operator="==", value="VERB")

    items = resolver.resolve(constraint)
    assert len(items) == 0


def test_resolve_no_matches(resolver: ConstraintResolver) -> None:
    """Test resolving constraint that matches no items."""
    constraint = IntensionalConstraint(
        property="pos", operator="==", value="NONEXISTENT"
    )
    items = resolver.resolve(constraint)

    assert len(items) == 0


def test_evaluate_item_without_features(resolver: ConstraintResolver) -> None:
    """Test evaluating intensional constraint on item without features."""
    constraint = IntensionalConstraint(
        property="features.transitivity", operator="==", value="transitive"
    )
    item = LexicalItem(lemma="test", pos="VERB")  # No features dict

    # Should return False (missing property)
    assert resolver.evaluate(constraint, item) is False


def test_resolve_dsl_invalid_expression(resolver: ConstraintResolver) -> None:
    """Test DSL constraint with invalid expression raises error."""
    constraint = DSLConstraint(expression="invalid syntax !!!")

    with pytest.raises(RuntimeError, match="Failed to evaluate DSL expression"):
        resolver.resolve(constraint)


def test_resolve_dsl_undefined_variable(resolver: ConstraintResolver) -> None:
    """Test DSL constraint with undefined variable returns no matches.

    Undefined variables are treated as missing features/attributes, so the
    constraint evaluates to False for all items.
    """
    constraint = DSLConstraint(expression="undefined_var == 'value'")
    items = resolver.resolve(constraint)

    # Should return no matches since undefined_var doesn't exist
    assert len(items) == 0


def test_resolve_multiple_constraint_types(
    resolver: ConstraintResolver,
) -> None:
    """Test resolving different constraint types in sequence."""
    # Get an item ID for extensional test
    break_item = next(
        item for item in resolver.lexicon.items.values() if item.lemma == "break"
    )

    # Extensional
    ext_items = resolver.resolve(
        ExtensionalConstraint(mode="allow", items=[break_item.id])
    )
    assert len(ext_items) == 1

    # Intensional
    int_items = resolver.resolve(
        IntensionalConstraint(property="pos", operator="==", value="VERB")
    )
    assert len(int_items) > 0

    # DSL
    dsl_items = resolver.resolve(DSLConstraint(expression='pos == "VERB"'))
    assert len(dsl_items) > 0

    # Intensional and DSL should have same results
    assert len(int_items) == len(dsl_items)


def test_intensional_less_than_operator() -> None:
    """Test intensional constraint with < operator."""
    items_list = [
        LexicalItem(lemma="a", pos="VERB", attributes={"frequency": 100}),
        LexicalItem(lemma="b", pos="VERB", attributes={"frequency": 200}),
        LexicalItem(lemma="c", pos="VERB", attributes={"frequency": 50}),
    ]
    lexicon = Lexicon(name="test")
    for item in items_list:
        lexicon.add(item)
    resolver = ConstraintResolver(lexicon=lexicon)

    constraint = IntensionalConstraint(
        property="attributes.frequency", operator="<", value=150
    )
    matching = resolver.resolve(constraint)

    assert len(matching) == 2
    assert all(item.attributes["frequency"] < 150 for item in matching)


def test_intensional_greater_than_operator() -> None:
    """Test intensional constraint with > operator."""
    items_list = [
        LexicalItem(lemma="a", pos="VERB", attributes={"frequency": 100}),
        LexicalItem(lemma="b", pos="VERB", attributes={"frequency": 200}),
        LexicalItem(lemma="c", pos="VERB", attributes={"frequency": 50}),
    ]
    lexicon = Lexicon(name="test")
    for item in items_list:
        lexicon.add(item)
    resolver = ConstraintResolver(lexicon=lexicon)

    constraint = IntensionalConstraint(
        property="attributes.frequency", operator=">", value=75
    )
    matching = resolver.resolve(constraint)

    assert len(matching) == 2
    assert all(item.attributes["frequency"] > 75 for item in matching)


def test_intensional_less_than_or_equal_operator() -> None:
    """Test intensional constraint with <= operator."""
    items_list = [
        LexicalItem(lemma="a", pos="VERB", attributes={"frequency": 100}),
        LexicalItem(lemma="b", pos="VERB", attributes={"frequency": 200}),
        LexicalItem(lemma="c", pos="VERB", attributes={"frequency": 100}),
    ]
    lexicon = Lexicon(name="test")
    for item in items_list:
        lexicon.add(item)
    resolver = ConstraintResolver(lexicon=lexicon)

    constraint = IntensionalConstraint(
        property="attributes.frequency", operator="<=", value=100
    )
    matching = resolver.resolve(constraint)

    assert len(matching) == 2
    assert all(item.attributes["frequency"] <= 100 for item in matching)


def test_intensional_greater_than_or_equal_operator() -> None:
    """Test intensional constraint with >= operator."""
    items_list = [
        LexicalItem(lemma="a", pos="VERB", attributes={"frequency": 100}),
        LexicalItem(lemma="b", pos="VERB", attributes={"frequency": 200}),
        LexicalItem(lemma="c", pos="VERB", attributes={"frequency": 200}),
    ]
    lexicon = Lexicon(name="test")
    for item in items_list:
        lexicon.add(item)
    resolver = ConstraintResolver(lexicon=lexicon)

    constraint = IntensionalConstraint(
        property="attributes.frequency", operator=">=", value=200
    )
    matching = resolver.resolve(constraint)

    assert len(matching) == 2
    assert all(item.attributes["frequency"] >= 200 for item in matching)


def test_dsl_with_attribute_access(resolver: ConstraintResolver) -> None:
    """Test DSL constraint with nested attribute access."""
    constraint = DSLConstraint(expression='attributes.frequency == "high"')
    items = resolver.resolve(constraint)

    assert len(items) > 0
    assert all(item.attributes.get("frequency") == "high" for item in items)


def test_resolve_with_none_language_code(resolver: ConstraintResolver) -> None:
    """Test resolve with language_code=None evaluates all items."""
    constraint = IntensionalConstraint(property="pos", operator="==", value="VERB")

    items_all = resolver.resolve(constraint, language_code=None)
    items_explicit_none = resolver.resolve(constraint)

    assert items_all == items_explicit_none
    # Should have items from multiple languages
    languages = {item.language_code for item in items_all}
    assert len(languages) > 1
