"""Factory registry for structurally typed serving predictors."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

__all__ = ["PredictorFactory", "PredictorRegistry"]

type PredictorFactory[T] = Callable[[str | Path], T]
"""A callable that loads a predictor from its native artifact."""


class PredictorRegistry[T]:
    """Map provider names to native-artifact loading factories.

    The registry is generic in the capability required by its consumer. It
    accepts any callable with the factory signature and performs no inheritance
    or ``issubclass`` check.

    Examples
    --------
    >>> from bead.serving import PredictorRegistry, ServingPredictor
    >>> registry = PredictorRegistry[ServingPredictor]()
    >>> registry.register("external", ExternalPredictor.load)  # doctest: +SKIP
    >>> predictor = registry.load("external", "model.bin")  # doctest: +SKIP
    """

    def __init__(self) -> None:
        self._factories: dict[str, PredictorFactory[T]] = {}

    def register(self, name: str, factory: PredictorFactory[T]) -> None:
        """Register a loading factory under ``name``.

        Parameters
        ----------
        name : str
            Provider name used by :meth:`load`.
        factory : PredictorFactory[T]
            Callable that accepts an artifact path and returns a predictor.

        Raises
        ------
        ValueError
            If ``name`` is empty.
        """
        normalized_name = name.strip()
        if not normalized_name:
            raise ValueError("Predictor factory name cannot be empty")
        self._factories[normalized_name] = factory

    def load(self, name: str, path: str | Path) -> T:
        """Load a predictor through the factory registered under ``name``.

        Parameters
        ----------
        name : str
            Registered provider name.
        path : str | Path
            Native artifact path passed to the provider factory.

        Returns
        -------
        T
            Predictor returned by the registered factory.

        Raises
        ------
        KeyError
            If no factory is registered under ``name``.
        """
        try:
            factory = self._factories[name]
        except KeyError:
            available = ", ".join(sorted(self._factories)) or "none"
            raise KeyError(
                f"Unknown predictor factory {name!r}; available factories: {available}"
            ) from None
        return factory(path)

    def list_factories(self) -> tuple[str, ...]:
        """Return registered factory names in sorted order."""
        return tuple(sorted(self._factories))
