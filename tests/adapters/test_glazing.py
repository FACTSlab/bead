"""Test GlazingAdapter."""

from __future__ import annotations

import pytest

from sash.adapters.cache import AdapterCache
from sash.adapters.glazing import GlazingAdapter


def test_glazing_adapter_initialization() -> None:
    """Test GlazingAdapter initialization."""
    adapter = GlazingAdapter(resource="verbnet")
    assert adapter.resource == "verbnet"
    assert adapter.cache is None


def test_glazing_adapter_with_cache(adapter_cache: AdapterCache) -> None:
    """Test GlazingAdapter with cache."""
    adapter = GlazingAdapter(resource="verbnet", cache=adapter_cache)
    assert adapter.cache is adapter_cache


def test_glazing_adapter_is_available(glazing_adapter: GlazingAdapter) -> None:
    """Test that glazing adapter is available."""
    assert glazing_adapter.is_available()


def test_glazing_adapter_fetch_items(glazing_adapter: GlazingAdapter) -> None:
    """Test fetching items from glazing."""
    items = glazing_adapter.fetch_items(query="break", language_code="en")
    assert len(items) > 0
    assert all(item.lemma == "break" for item in items)
    assert all(item.language_code == "en" for item in items)
    # Check VerbNet-specific attributes
    assert all("verbnet_class" in item.attributes for item in items)


def test_glazing_adapter_fetch_requires_query() -> None:
    """Test that glazing adapter requires query parameter."""
    adapter = GlazingAdapter(resource="verbnet")
    with pytest.raises(ValueError, match="requires a query string"):
        adapter.fetch_items(query=None, language_code="en")


def test_glazing_adapter_caching(
    glazing_adapter: GlazingAdapter, adapter_cache: AdapterCache
) -> None:
    """Test that glazing adapter uses cache."""
    # First fetch (miss)
    items1 = glazing_adapter.fetch_items(query="break", language_code="en")

    # Second fetch (hit) - should be same object from cache
    items2 = glazing_adapter.fetch_items(query="break", language_code="en")

    # Should be same (from cache)
    assert items1 == items2


def test_glazing_adapter_different_resources() -> None:
    """Test glazing adapter with different resources."""
    verbnet_adapter = GlazingAdapter(resource="verbnet")
    propbank_adapter = GlazingAdapter(resource="propbank")
    framenet_adapter = GlazingAdapter(resource="framenet")

    assert verbnet_adapter.resource == "verbnet"
    assert propbank_adapter.resource == "propbank"
    assert framenet_adapter.resource == "framenet"


def test_glazing_adapter_verbnet_attributes(glazing_adapter: GlazingAdapter) -> None:
    """Test that VerbNet items have expected attributes."""
    items = glazing_adapter.fetch_items(query="break", language_code="en")
    assert len(items) > 0

    # Check first item has VerbNet attributes
    item = items[0]
    assert "verbnet_class" in item.attributes
    assert "themroles" in item.attributes
    assert "frame_count" in item.attributes
    assert item.pos == "VERB"


def test_glazing_adapter_propbank() -> None:
    """Test PropBank adapter."""
    adapter = GlazingAdapter(resource="propbank")
    assert adapter.is_available()

    # Search for a common verb
    items = adapter.fetch_items(query="break", language_code="en")
    # PropBank should have framesets for "break"
    if len(items) > 0:
        item = items[0]
        assert item.lemma == "break"
        assert item.language_code == "en"
        assert "propbank_roleset_id" in item.attributes
        assert "roles" in item.attributes


def test_glazing_adapter_framenet() -> None:
    """Test FrameNet adapter."""
    adapter = GlazingAdapter(resource="framenet")
    assert adapter.is_available()

    # Search for a lexical unit
    items = adapter.fetch_items(query="break", language_code="en")
    # FrameNet should have lexical units for "break"
    if len(items) > 0:
        item = items[0]
        assert item.lemma == "break"
        assert item.language_code == "en"
        assert "framenet_frame" in item.attributes
        assert "lexical_unit_name" in item.attributes
