"""Model adapter registry for centralized adapter management.

This module provides a registry for managing all model adapters,
both local (HuggingFace) and API-based (OpenAI, Anthropic, etc.).
"""

from __future__ import annotations

from typing import Any

from bead.items.adapters.base import ModelAdapter


class ModelAdapterRegistry:
    """Registry for all model adapters (local and API-based).

    Provides centralized management of adapter types and instances,
    with automatic instance caching to avoid redundant initialization.

    Attributes
    ----------
    adapters : dict[str, type[ModelAdapter]]
        Registered adapter classes keyed by adapter type name.
    instances : dict[str, ModelAdapter]
        Cached adapter instances keyed by unique identifier.
    """

    def __init__(self) -> None:
        self.adapters: dict[str, type[ModelAdapter]] = {}
        self.instances: dict[str, ModelAdapter] = {}

    def register(self, name: str, adapter_class: type[ModelAdapter]) -> None:
        """Register an adapter class.

        Parameters
        ----------
        name : str
            Unique name for the adapter type (e.g., "openai", "huggingface_lm").
        adapter_class : type[ModelAdapter]
            Adapter class to register (must inherit from ModelAdapter).

        Raises
        ------
        ValueError
            If adapter class does not inherit from ModelAdapter.
        """
        if not issubclass(adapter_class, ModelAdapter):  # type: ignore[misc]
            raise ValueError(
                f"Adapter class {adapter_class.__name__} must inherit from ModelAdapter"
            )
        self.adapters[name] = adapter_class

    def get_adapter(
        self, adapter_type: str, model_name: str, **kwargs: Any
    ) -> ModelAdapter:
        """Get or create adapter instance (with caching).

        Creates a new adapter instance if not cached, otherwise returns
        the cached instance. Instances are cached by adapter type and model name.

        Parameters
        ----------
        adapter_type : str
            Type of adapter (must be registered).
        model_name : str
            Model identifier for the adapter.
        **kwargs : Any
            Additional keyword arguments to pass to adapter constructor.

        Returns
        -------
        ModelAdapter
            Adapter instance (cached or newly created).

        Raises
        ------
        ValueError
            If adapter type is not registered.

        Examples
        --------
        >>> registry = ModelAdapterRegistry()
        >>> registry.register("openai", OpenAIAdapter)
        >>> adapter = registry.get_adapter("openai", "gpt-4", api_key="...")
        """
        if adapter_type not in self.adapters:
            raise ValueError(
                f"Unknown adapter type: {adapter_type}. "
                f"Available types: {list(self.adapters.keys())}"
            )

        # Create cache key from adapter type and model name
        cache_key = f"{adapter_type}:{model_name}"

        # Return cached instance if available
        if cache_key in self.instances:
            return self.instances[cache_key]

        # Create new instance
        adapter_class = self.adapters[adapter_type]
        adapter = adapter_class(model_name=model_name, **kwargs)

        # Cache and return
        self.instances[cache_key] = adapter
        return adapter

    def clear_cache(self) -> None:
        """Clear all cached adapter instances.

        Useful for testing or when you want to force recreation of adapters
        with different parameters.
        """
        self.instances.clear()

    def list_adapters(self) -> list[str]:
        """List all registered adapter types.

        Returns
        -------
        list[str]
            List of registered adapter type names.
        """
        return list(self.adapters.keys())


# Create default registry with all built-in adapters
default_registry = ModelAdapterRegistry()

# Register HuggingFace adapters
try:
    from bead.items.adapters.huggingface import (
        HuggingFaceLanguageModel,
        HuggingFaceMaskedLanguageModel,
        HuggingFaceNLI,
    )

    default_registry.register("huggingface_lm", HuggingFaceLanguageModel)
    default_registry.register("huggingface_mlm", HuggingFaceMaskedLanguageModel)
    default_registry.register("huggingface_nli", HuggingFaceNLI)
except ImportError:
    # HuggingFace adapters not available (missing dependencies)
    pass

# Register sentence transformers
try:
    from bead.items.adapters.sentence_transformers import HuggingFaceSentenceTransformer

    default_registry.register("sentence_transformer", HuggingFaceSentenceTransformer)
except ImportError:
    # Sentence transformers not available
    pass

# Register API adapters (these are optional)
try:
    from bead.items.adapters.openai import OpenAIAdapter

    default_registry.register("openai", OpenAIAdapter)
except ImportError:
    # OpenAI not available
    pass

try:
    from bead.items.adapters.anthropic import AnthropicAdapter

    default_registry.register("anthropic", AnthropicAdapter)
except ImportError:
    # Anthropic not available
    pass

try:
    from bead.items.adapters.google import GoogleAdapter

    default_registry.register("google", GoogleAdapter)
except ImportError:
    # Google not available
    pass

try:
    from bead.items.adapters.togetherai import TogetherAIAdapter

    default_registry.register("togetherai", TogetherAIAdapter)
except ImportError:
    # Together AI not available
    pass
