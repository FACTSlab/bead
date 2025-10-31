"""Test ResourceAdapter abstract base class."""

from __future__ import annotations

import pytest

from bead.resources.adapters.base import ResourceAdapter
from bead.resources.models import LexicalItem


def test_cannot_instantiate_abstract_class() -> None:
    """Test that ResourceAdapter cannot be instantiated directly."""
    with pytest.raises(TypeError):
        ResourceAdapter()  # type: ignore[abstract]


def test_subclass_must_implement_fetch_items() -> None:
    """Test that subclasses must implement fetch_items()."""

    class IncompleteAdapter(ResourceAdapter):
        """Incomplete adapter missing fetch_items."""

        def is_available(self) -> bool:
            """Check availability."""
            return True

    with pytest.raises(TypeError):
        IncompleteAdapter()  # type: ignore[abstract]


def test_subclass_must_implement_is_available() -> None:
    """Test that subclasses must implement is_available()."""

    class IncompleteAdapter(ResourceAdapter):
        """Incomplete adapter missing is_available."""

        def fetch_items(
            self,
            query: str | None = None,
            language_code: str | None = None,
            **kwargs: object,
        ) -> list[LexicalItem]:
            """Fetch items."""
            return []

    with pytest.raises(TypeError):
        IncompleteAdapter()  # type: ignore[abstract]


def test_concrete_subclass_works() -> None:
    """Test that proper subclass implementation works."""

    class ConcreteAdapter(ResourceAdapter):
        """Concrete adapter implementation."""

        def fetch_items(
            self,
            query: str | None = None,
            language_code: str | None = None,
            **kwargs: object,
        ) -> list[LexicalItem]:
            """Fetch items."""
            return [LexicalItem(lemma="test", language_code=language_code)]

        def is_available(self) -> bool:
            """Check availability."""
            return True

    adapter = ConcreteAdapter()
    assert adapter.is_available()
    items = adapter.fetch_items(query="test", language_code="en")
    assert len(items) == 1
    assert items[0].lemma == "test"
