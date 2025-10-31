"""HuggingFace masked language model adapter for template filling."""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from bead.adapters.huggingface import DeviceType, HuggingFaceAdapterMixin
from bead.templates.adapters.base import TemplateFillingModelAdapter


class HuggingFaceMLMAdapter(HuggingFaceAdapterMixin, TemplateFillingModelAdapter):
    """Adapter for HuggingFace masked language models.

    Supports BERT, RoBERTa, ALBERT, and other MLM architectures.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier (e.g., "bert-base-uncased")
    device : DeviceType
        Computation device ("cpu", "cuda", "mps")
    cache_dir : Path | None
        Directory for caching model files

    Examples
    --------
    >>> adapter = HuggingFaceMLMAdapter("bert-base-uncased", device="cpu")
    >>> adapter.load_model()
    >>> predictions = adapter.predict_masked_token(
    ...     text="The cat sat on the mat",
    ...     mask_position=2,
    ...     top_k=5
    ... )
    >>> for token, log_prob in predictions:
    ...     print(f"{token}: {log_prob:.2f}")
    >>> adapter.unload_model()
    """

    def __init__(
        self,
        model_name: str,
        device: DeviceType = "cpu",
        cache_dir: Path | None = None,
    ) -> None:
        """Initialize HuggingFace MLM adapter.

        Parameters
        ----------
        model_name : str
            Model identifier
        device : DeviceType
            Computation device
        cache_dir : Path | None
            Model cache directory
        """
        # Validate device before passing to parent
        validated_device = self._validate_device(device)
        super().__init__(model_name, validated_device, cache_dir)
        self.model: PreTrainedModel | None = None
        self.tokenizer: PreTrainedTokenizer | None = None

    def load_model(self) -> None:
        """Load model and tokenizer from HuggingFace.

        Raises
        ------
        RuntimeError
            If model loading fails
        """
        if self._model_loaded:
            return

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
            )

            # Load model
            self.model = AutoModelForMaskedLM.from_pretrained(
                self.model_name,
                cache_dir=self.cache_dir,
            )

            # Move to device
            self.model.to(self.device)

            # Set to evaluation mode
            self.model.eval()

            self._model_loaded = True

        except Exception as e:
            raise RuntimeError(f"Failed to load model {self.model_name}: {e}") from e

    def unload_model(self) -> None:
        """Unload model from memory."""
        if not self._model_loaded:
            return

        # Move model to CPU and delete
        if self.model is not None:
            self.model.to("cpu")
            del self.model
            self.model = None

        del self.tokenizer
        self.tokenizer = None

        self._model_loaded = False

        # Clear CUDA cache if using GPU
        if self.device == "cuda":
            torch.cuda.empty_cache()

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
            If mask_position is invalid or text has no mask token
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Tokenize input
        inputs = self.tokenizer(text, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Find mask token ID
        mask_token_id = self.tokenizer.mask_token_id
        if mask_token_id is None:
            raise ValueError(f"Model {self.model_name} does not have a mask token")

        # Find mask position in tokenized input
        input_ids = inputs["input_ids"][0]
        mask_positions = (input_ids == mask_token_id).nonzero(as_tuple=True)[0]

        if len(mask_positions) == 0:
            raise ValueError(f"No mask token found in text: {text}")

        if mask_position >= len(mask_positions):
            raise ValueError(
                f"mask_position {mask_position} out of range. "
                f"Found {len(mask_positions)} mask tokens in text."
            )

        # Get actual token index
        mask_idx = mask_positions[mask_position].item()

        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Get predictions for mask position
        mask_logits = logits[0, mask_idx]

        # Convert to log probabilities
        log_probs = torch.log_softmax(mask_logits, dim=0)

        # Get top-k predictions
        top_log_probs, top_indices = torch.topk(log_probs, k=min(top_k, len(log_probs)))

        # Convert to tokens
        predictions: list[tuple[str, float]] = []
        for log_prob, idx in zip(top_log_probs.cpu(), top_indices.cpu(), strict=True):
            token = self.tokenizer.decode([idx], skip_special_tokens=True).strip()
            predictions.append((token, float(log_prob)))

        return predictions

    def get_mask_token(self) -> str:
        """Get the mask token for this model.

        Returns
        -------
        str
            Mask token string (e.g., "[MASK]" for BERT)

        Raises
        ------
        RuntimeError
            If model is not loaded
        ValueError
            If model has no mask token
        """
        if not self._model_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        mask_token = self.tokenizer.mask_token
        if mask_token is None:
            raise ValueError(f"Model {self.model_name} does not have a mask token")

        return mask_token
