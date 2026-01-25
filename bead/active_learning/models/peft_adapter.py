"""PEFT-based LoRA adapters for participant-specific fine-tuning.

This module provides participant-specific LoRA adapters using HuggingFace PEFT
library, replacing the custom LoRA implementation with the well-maintained
PEFT library.
"""

from __future__ import annotations

import copy

import torch.nn as nn
from peft import LoraConfig, get_peft_model
from torch import Tensor


class DecoderWrapper(nn.Module):
    """Wrapper to make a decoder module compatible with PEFT.

    PEFT expects a PreTrainedModel, but we often work with decoder modules.
    This wrapper makes the decoder look like a model for PEFT purposes.
    """

    def __init__(self, decoder: nn.Module) -> None:
        """Initialize decoder wrapper.

        Parameters
        ----------
        decoder : nn.Module
            Decoder module to wrap.
        """
        super().__init__()
        self.decoder = decoder
        # PEFT may check for config attribute
        if hasattr(decoder, "config"):
            self.config = decoder.config

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        """Forward pass through decoder.

        Parameters
        ----------
        input_ids : Tensor
            Input token IDs, shape (batch_size, seq_len).
        attention_mask : Tensor | None
            Attention mask, shape (batch_size, seq_len). If None, no masking.

        Returns
        -------
        Tensor
            Decoder output tensor.
        """
        if attention_mask is not None:
            return self.decoder(input_ids, attention_mask=attention_mask)
        return self.decoder(input_ids)

    def __getattr__(self, name: str) -> nn.Module:
        """Delegate attribute access to decoder."""
        if name in ("decoder", "config"):
            return super().__getattr__(name)  # type: ignore[return-value]
        return getattr(self.decoder, name)


def create_participant_lora_adapter(
    base_decoder: nn.Module,
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.1,
    target_modules: list[str] | None = None,
) -> nn.Module:
    """Create a participant-specific LoRA adapter using HuggingFace PEFT.

    Creates a PEFT model with LoRA adapters for participant-specific
    fine-tuning. This replaces the custom LoRA implementation with
    HuggingFace's well-maintained PEFT library.

    Parameters
    ----------
    base_decoder : nn.Module
        Base decoder module to adapt (e.g., T5 decoder, BART decoder).
    rank : int
        LoRA rank r (default: 8).
    alpha : float
        LoRA scaling factor α (default: 16.0).
    dropout : float
        LoRA dropout probability (default: 0.1).
    target_modules : list[str] | None
        Target modules for LoRA injection (e.g., ["q", "v"]).
        If None, uses default based on model architecture.

    Returns
    -------
    nn.Module
        Decoder module with LoRA adapters applied (compatible with original API).

    Examples
    --------
    >>> from transformers import AutoModelForSeq2SeqLM
    >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    >>> decoder = model.get_decoder()
    >>> peft_decoder = create_participant_lora_adapter(
    ...     decoder,
    ...     rank=8,
    ...     alpha=16.0,
    ...     target_modules=["q", "v"]
    ... )
    >>> # peft_decoder now has LoRA adapters and can be used like the original decoder
    """
    # Determine target modules if not provided
    if target_modules is None:
        # Try to infer from decoder config
        if hasattr(base_decoder, "config"):
            model_type = base_decoder.config.model_type.lower()
            if "t5" in model_type:
                target_modules = ["q", "v"]
            elif "bart" in model_type:
                target_modules = ["q_proj", "v_proj"]
            else:
                # Generic: try common attention projection names
                target_modules = ["query", "value", "q_proj", "v_proj"]
        else:
            # Fallback defaults
            target_modules = ["q", "v", "q_proj", "v_proj"]

    # Create deep copy to avoid modifying original
    decoder_copy = copy.deepcopy(base_decoder)

    # Wrap decoder to make it compatible with PEFT
    wrapped_decoder = DecoderWrapper(decoder_copy)

    # Create LoRA configuration
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",  # Don't train bias parameters
        task_type="FEATURE_EXTRACTION",  # Generic task type
    )

    # Apply PEFT LoRA
    peft_model = get_peft_model(wrapped_decoder, lora_config)

    # Extract the adapted decoder from the PEFT model
    # PEFT wraps the model, so we need to get the decoder back
    if hasattr(peft_model, "base_model"):
        # PEFT v0.6+ structure
        if hasattr(peft_model.base_model, "decoder"):
            return peft_model.base_model.decoder
        return peft_model.base_model
    else:
        # Fallback: return the PEFT model itself (it should work as decoder)
        return peft_model
