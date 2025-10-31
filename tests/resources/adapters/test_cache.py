"""Test AdapterCache."""

from __future__ import annotations

from bead.resources.adapters.cache import AdapterCache
from bead.resources.models import LexicalItem


def test_cache_initialization(adapter_cache: AdapterCache) -> None:
    """Test cache initializes empty."""
    assert adapter_cache.get("nonexistent") is None


def test_cache_set_and_get(adapter_cache: AdapterCache) -> None:
    """Test setting and getting cached items."""
    items = [LexicalItem(lemma="walk", pos="VERB")]
    adapter_cache.set("key1", items)
    cached = adapter_cache.get("key1")
    assert cached == items


def test_cache_get_nonexistent(adapter_cache: AdapterCache) -> None:
    """Test getting nonexistent key returns None."""
    assert adapter_cache.get("missing") is None


def test_cache_clear(adapter_cache: AdapterCache) -> None:
    """Test clearing cache removes all entries."""
    adapter_cache.set("key1", [])
    adapter_cache.set("key2", [])
    adapter_cache.clear()
    assert adapter_cache.get("key1") is None
    assert adapter_cache.get("key2") is None


def test_make_key_consistency(adapter_cache: AdapterCache) -> None:
    """Test that make_key produces consistent keys."""
    key1 = adapter_cache.make_key("glazing", query="walk", language_code="en")
    key2 = adapter_cache.make_key("glazing", query="walk", language_code="en")
    assert key1 == key2


def test_make_key_different_params(adapter_cache: AdapterCache) -> None:
    """Test that different parameters produce different keys."""
    key1 = adapter_cache.make_key("glazing", query="walk", language_code="en")
    key2 = adapter_cache.make_key("glazing", query="run", language_code="en")
    key3 = adapter_cache.make_key("glazing", query="walk", language_code="ko")
    key4 = adapter_cache.make_key("unimorph", query="walk", language_code="en")

    assert key1 != key2  # Different query
    assert key1 != key3  # Different language
    assert key1 != key4  # Different adapter


def test_make_key_none_query(adapter_cache: AdapterCache) -> None:
    """Test make_key with None query."""
    key1 = adapter_cache.make_key("glazing", query=None, language_code="en")
    key2 = adapter_cache.make_key("glazing", query=None, language_code="en")
    assert key1 == key2
