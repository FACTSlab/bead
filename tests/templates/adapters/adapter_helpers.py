"""Test helpers for template adapter tests."""

from __future__ import annotations

from pathlib import Path

from bead.templates.adapters.base import TemplateFillingModelAdapter


class MockMLMAdapter(TemplateFillingModelAdapter):
    """Mock MLM adapter for testing."""

    def __init__(
        self,
        model_name: str = "mock-model",
        device: str = "cpu",
        cache_dir: Path | None = None,
        predictions: dict[str, list[tuple[str, float]]] | None = None,
    ) -> None:
        """Initialize mock adapter.

        Parameters
        ----------
        model_name : str
            Model name
        device : str
            Device
        cache_dir : Path | None
            Cache directory
        predictions : dict[str, list[tuple[str, float]]] | None
            Predefined predictions for mocking
        """
        super().__init__(model_name, device, cache_dir)
        self.predictions = predictions or {}
        self.load_calls = 0
        self.unload_calls = 0
        self.predict_calls: list[tuple[str, int, int]] = []

    def load_model(self) -> None:
        """Mock load."""
        self.load_calls += 1
        self._model_loaded = True

    def unload_model(self) -> None:
        """Mock unload."""
        self.unload_calls += 1
        self._model_loaded = False

    def predict_masked_token(
        self,
        text: str,
        mask_position: int,
        top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Mock predict.

        Parameters
        ----------
        text : str
            Text with mask
        mask_position : int
            Mask position
        top_k : int
            Top-K

        Returns
        -------
        list[tuple[str, float]]
            Predictions
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded")

        self.predict_calls.append((text, mask_position, top_k))

        # Return predefined predictions if available
        if text in self.predictions:
            return self.predictions[text][:top_k]

        # Default predictions
        return [
            ("the", -0.5),
            ("a", -1.0),
            ("an", -1.5),
        ][:top_k]

    def get_mask_token(self) -> str:
        """Get mask token."""
        return "[MASK]"
