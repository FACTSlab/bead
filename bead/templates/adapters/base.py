"""Base adapter for template filling models.

This module defines the abstract interface for models used in template filling.
These adapters are SEPARATE from judgment prediction model adapters (Stage 6).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class TemplateFillingModelAdapter(ABC):
    """Base adapter for models used in template filling.

    This is SEPARATE from judgment prediction model adapters,
    which are used later in the pipeline for predicting human judgments.

    Parameters
    ----------
    model_name : str
        Model identifier (e.g., "bert-base-uncased")
    device : str
        Computation device ("cpu", "cuda", "mps")
    cache_dir : Path | None
        Directory for caching model files

    Examples
    --------
    >>> from bead.templates.adapters import TemplateFillingModelAdapter
    >>> # Implemented by HuggingFaceMLMAdapter
    >>> adapter = HuggingFaceMLMAdapter("bert-base-uncased", device="cpu")
    >>> adapter.load_model()
    >>> predictions = adapter.predict_masked_token(
    ...     text="The cat [MASK] on the mat",
    ...     mask_position=2,
    ...     top_k=5
    ... )
    >>> adapter.unload_model()
    """

    def __init__(
        self,
        model_name: str,
        device: str = "cpu",
        cache_dir: Path | None = None,
    ) -> None:
        """Initialize model adapter.

        Parameters
        ----------
        model_name : str
            Model identifier
        device : str
            Computation device
        cache_dir : Path | None
            Directory for model cache
        """
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self._model_loaded = False

    @abstractmethod
    def load_model(self) -> None:
        """Load model into memory.

        Raises
        ------
        RuntimeError
            If model loading fails
        """
        pass

    @abstractmethod
    def unload_model(self) -> None:
        """Unload model from memory to free resources."""
        pass

    @abstractmethod
    def predict_masked_token(
        self,
        text: str,
        mask_position: int,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Predict masked token at specified position.

        Parameters
        ----------
        text : str
            Text with mask token (e.g., "The cat [MASK] quickly")
        mask_position : int
            Token position of mask (0-indexed)
        top_k : int
            Number of top predictions to return

        Returns
        -------
        list[tuple[str, float]]
            List of (token, log_probability) tuples, sorted by probability

        Raises
        ------
        RuntimeError
            If model is not loaded
        ValueError
            If mask_position is invalid

        Examples
        --------
        >>> predictions = adapter.predict_masked_token(
        ...     text="The cat [MASK] on the mat",
        ...     mask_position=2,
        ...     top_k=3
        ... )
        >>> predictions
        [("sat", -0.5), ("slept", -1.2), ("jumped", -1.5)]
        """
        pass

    def is_loaded(self) -> bool:
        """Check if model is loaded.

        Returns
        -------
        bool
            True if model is loaded in memory
        """
        return self._model_loaded

    def __enter__(self) -> TemplateFillingModelAdapter:
        """Context manager entry."""
        self.load_model()
        return self

    def __exit__(self, *args: object) -> None:
        """Context manager exit."""
        self.unload_model()
