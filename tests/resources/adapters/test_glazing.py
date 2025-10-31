"""Test GlazingAdapter."""

from __future__ import annotations

import pytest

from bead.resources.adapters.cache import AdapterCache
from bead.resources.adapters.glazing import GlazingAdapter


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
    assert all(item.language_code == "eng" for item in items)
    # Check VerbNet-specific attributes
    assert all("verbnet_class" in item.attributes for item in items)


def test_glazing_adapter_fetch_all_verbnet() -> None:
    """Test that glazing adapter can fetch all VerbNet verbs."""
    adapter = GlazingAdapter(resource="verbnet")
    items = adapter.fetch_items(query=None, language_code="en")
    # VerbNet has 3000+ verb-class pairs
    assert len(items) > 3000
    assert all(item.pos == "VERB" for item in items)
    assert all("verbnet_class" in item.attributes for item in items)


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
        assert item.language_code == "eng"
        assert "propbank_roleset_id" in item.attributes
        assert "roleset_name" in item.attributes


def test_glazing_adapter_framenet() -> None:
    """Test FrameNet adapter."""
    adapter = GlazingAdapter(resource="framenet")
    assert adapter.is_available()

    # Search for a common verb
    items = adapter.fetch_items(query="break", language_code="en")
    # FrameNet should have multiple frames for "break"
    assert len(items) > 0

    # Check attributes
    item = items[0]
    assert item.lemma == "break"
    assert item.language_code == "eng"
    assert "framenet_frame" in item.attributes
    assert "framenet_frame_id" in item.attributes
    assert "lexical_unit_name" in item.attributes
    assert "lexical_unit_id" in item.attributes


def test_glazing_adapter_include_frames_verbnet(
    glazing_adapter: GlazingAdapter,
) -> None:
    """Test VerbNet adapter with include_frames parameter."""
    items = glazing_adapter.fetch_items(
        query="break", language_code="en", include_frames=True
    )
    assert len(items) > 0

    # Check that frames are included
    item = items[0]
    assert "frames" in item.attributes
    assert isinstance(item.attributes["frames"], list)
    assert len(item.attributes["frames"]) > 0

    # Check frame structure
    frame = item.attributes["frames"][0]
    assert "primary" in frame
    assert "secondary" in frame
    assert "syntax" in frame
    assert "examples" in frame


def test_glazing_adapter_include_frames_propbank() -> None:
    """Test PropBank adapter with include_frames parameter."""
    adapter = GlazingAdapter(resource="propbank")
    items = adapter.fetch_items(query="break", language_code="en", include_frames=True)

    if len(items) > 0:
        item = items[0]
        # Should have detailed role information
        if "roles" in item.attributes:
            assert isinstance(item.attributes["roles"], list)
            if len(item.attributes["roles"]) > 0:
                role = item.attributes["roles"][0]
                assert "arg" in role
                assert "description" in role


def test_glazing_adapter_include_frames_framenet() -> None:
    """Test FrameNet adapter with include_frames parameter."""
    adapter = GlazingAdapter(resource="framenet")

    items = adapter.fetch_items(query="break", language_code="en", include_frames=True)
    assert len(items) > 0

    # Check for detailed frame information
    item = items[0]
    assert "frame_definition" in item.attributes
    assert "frame_elements" in item.attributes
    # Frame elements should be a list
    assert isinstance(item.attributes["frame_elements"], list)


def test_glazing_adapter_fetch_all_propbank() -> None:
    """Test that glazing adapter can fetch all PropBank predicates."""
    adapter = GlazingAdapter(resource="propbank")
    items = adapter.fetch_items(query=None, language_code="en")
    # PropBank has many predicates
    assert len(items) > 100
    assert all("propbank_roleset_id" in item.attributes for item in items)


def test_glazing_adapter_fetch_all_framenet() -> None:
    """Test that glazing adapter can fetch all FrameNet lexical units."""
    adapter = GlazingAdapter(resource="framenet")

    items = adapter.fetch_items(query=None, language_code="en")
    # FrameNet has many lexical units across all frames
    assert len(items) > 1000
    assert all("framenet_frame" in item.attributes for item in items)
    assert all("lexical_unit_name" in item.attributes for item in items)
