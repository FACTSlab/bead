"""Pytest fixtures for template resolver tests."""

from __future__ import annotations

import pytest

from sash.adapters.glazing import GlazingAdapter
from sash.adapters.registry import AdapterRegistry
from sash.adapters.unimorph import UniMorphAdapter
from sash.resources.lexicon import Lexicon
from sash.resources.models import LexicalItem
from sash.templates.resolver import ConstraintResolver


@pytest.fixture
def sample_lexicon() -> Lexicon:
    """Create sample lexicon with diverse items for testing.

    Returns
    -------
    Lexicon
        Lexicon with multiple items covering different languages,
        parts of speech, and features.
    """
    items_list = [
        LexicalItem(
            lemma="break",
            pos="VERB",
            language_code="en",
            features={"transitivity": "transitive", "causative": True},
            attributes={"frequency": "high"},
        ),
        LexicalItem(
            lemma="shatter",
            pos="VERB",
            language_code="en",
            features={"transitivity": "transitive", "causative": True},
            attributes={"frequency": "medium"},
        ),
        LexicalItem(
            lemma="arrive",
            pos="VERB",
            language_code="en",
            features={"transitivity": "intransitive", "causative": False},
            attributes={"frequency": "high"},
        ),
        LexicalItem(
            lemma="happiness",
            pos="NOUN",
            language_code="en",
            features={"number": "singular"},
            attributes={"frequency": "high"},
        ),
        LexicalItem(
            lemma="quickly",
            pos="ADV",
            language_code="en",
            attributes={"frequency": "high"},
        ),
        # Add multilingual items
        LexicalItem(
            lemma="kkakta",
            pos="VERB",
            language_code="ko",
            features={"transitivity": "transitive", "causative": True},
        ),
        LexicalItem(
            lemma="partir",
            pos="VERB",
            language_code="fr",
            features={"transitivity": "intransitive"},
        ),
    ]

    # Create lexicon and add items
    lexicon = Lexicon(name="test_lexicon")
    for item in items_list:
        lexicon.add(item)

    return lexicon


@pytest.fixture
def adapter_registry() -> AdapterRegistry:
    """Create adapter registry with glazing and unimorph adapters.

    Returns
    -------
    AdapterRegistry
        Registry with glazing and unimorph adapters registered.
    """
    registry = AdapterRegistry()
    registry.register("glazing", GlazingAdapter)
    registry.register("unimorph", UniMorphAdapter)
    return registry


@pytest.fixture
def resolver(
    sample_lexicon: Lexicon,
    adapter_registry: AdapterRegistry,
) -> ConstraintResolver:
    """Create constraint resolver with sample lexicon and adapters.

    Parameters
    ----------
    sample_lexicon : Lexicon
        Sample lexicon fixture.
    adapter_registry : AdapterRegistry
        Adapter registry fixture.

    Returns
    -------
    ConstraintResolver
        Resolver configured with caching enabled.
    """
    return ConstraintResolver(
        lexicon=sample_lexicon,
        adapter_registry=adapter_registry,
        cache_results=True,
    )


@pytest.fixture
def resolver_no_cache(
    sample_lexicon: Lexicon,
    adapter_registry: AdapterRegistry,
) -> ConstraintResolver:
    """Create constraint resolver without caching.

    Parameters
    ----------
    sample_lexicon : Lexicon
        Sample lexicon fixture.
    adapter_registry : AdapterRegistry
        Adapter registry fixture.

    Returns
    -------
    ConstraintResolver
        Resolver configured with caching disabled.
    """
    return ConstraintResolver(
        lexicon=sample_lexicon,
        adapter_registry=adapter_registry,
        cache_results=False,
    )
