"""Fixtures for adapter tests."""

from __future__ import annotations

import pytest

from bead.resources.adapters.cache import AdapterCache
from bead.resources.adapters.glazing import GlazingAdapter
from bead.resources.adapters.registry import AdapterRegistry
from bead.resources.adapters.unimorph import UniMorphAdapter


@pytest.fixture
def adapter_cache() -> AdapterCache:
    """Provide a fresh AdapterCache instance."""
    return AdapterCache()


@pytest.fixture
def glazing_adapter(adapter_cache: AdapterCache) -> GlazingAdapter:
    """Provide a GlazingAdapter with cache."""
    return GlazingAdapter(resource="verbnet", cache=adapter_cache)


@pytest.fixture
def unimorph_adapter(adapter_cache: AdapterCache) -> UniMorphAdapter:
    """Provide a UniMorphAdapter with cache."""
    return UniMorphAdapter(cache=adapter_cache)


@pytest.fixture
def adapter_registry() -> AdapterRegistry:
    """Provide a fresh AdapterRegistry."""
    return AdapterRegistry()
