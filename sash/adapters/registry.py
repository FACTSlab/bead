"""Registry for managing resource adapters.

This module provides a registry for discovering and instantiating adapters
by name.
"""

from __future__ import annotations

from typing import Any

from sash.adapters.base import ResourceAdapter


class AdapterRegistry:
    """Registry for managing resource adapters.

    The registry allows adapters to be registered by name and retrieved
    with custom initialization parameters.

    Examples
    --------
    >>> from sash.adapters.glazing import GlazingAdapter
    >>> registry = AdapterRegistry()
    >>> registry.register("glazing", GlazingAdapter)
    >>> adapter = registry.get("glazing", resource="verbnet")
    >>> isinstance(adapter, GlazingAdapter)
    True
    """

    def __init__(self) -> None:
        """Initialize empty registry.

        Examples
        --------
        >>> registry = AdapterRegistry()
        >>> len(registry._adapters)
        0
        """
        self._adapters: dict[str, type[ResourceAdapter]] = {}

    def register(self, name: str, adapter_class: type[ResourceAdapter]) -> None:
        """Register an adapter class.

        Parameters
        ----------
        name : str
            Adapter name (e.g., "glazing", "unimorph").
        adapter_class : type[ResourceAdapter]
            Adapter class (not instance) that subclasses ResourceAdapter.

        Raises
        ------
        ValueError
            If name is empty or adapter_class is not a ResourceAdapter subclass.

        Examples
        --------
        >>> from sash.adapters.glazing import GlazingAdapter
        >>> registry = AdapterRegistry()
        >>> registry.register("glazing", GlazingAdapter)
        >>> "glazing" in registry.list_available()
        True
        """
        if not name or not name.strip():
            raise ValueError("Adapter name must be non-empty")
        if not issubclass(adapter_class, ResourceAdapter):
            raise ValueError(f"{adapter_class} must be a subclass of ResourceAdapter")
        self._adapters[name] = adapter_class

    def get(self, name: str, **kwargs: Any) -> ResourceAdapter:
        """Get adapter instance by name.

        Parameters
        ----------
        name : str
            Adapter name (must be registered).
        **kwargs : Any
            Arguments passed to adapter constructor.

        Returns
        -------
        ResourceAdapter
            Adapter instance.

        Raises
        ------
        KeyError
            If adapter name is not registered.

        Examples
        --------
        >>> from sash.adapters.glazing import GlazingAdapter
        >>> registry = AdapterRegistry()
        >>> registry.register("glazing", GlazingAdapter)
        >>> adapter = registry.get("glazing", resource="verbnet")
        >>> adapter.resource
        'verbnet'
        """
        if name not in self._adapters:
            raise KeyError(
                f"Adapter '{name}' not registered. Available: {self.list_available()}"
            )
        adapter_class = self._adapters[name]
        return adapter_class(**kwargs)

    def list_available(self) -> list[str]:
        """List names of available adapters.

        Returns
        -------
        list[str]
            Sorted list of registered adapter names.

        Examples
        --------
        >>> registry = AdapterRegistry()
        >>> registry.list_available()
        []
        >>> from sash.adapters.glazing import GlazingAdapter
        >>> registry.register("glazing", GlazingAdapter)
        >>> registry.list_available()
        ['glazing']
        """
        return sorted(self._adapters.keys())
