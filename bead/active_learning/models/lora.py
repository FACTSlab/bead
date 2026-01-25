"""LoRA (Low-Rank Adaptation) implementation for transformer personalization.

Implements participant-specific low-rank updates to attention layers for
efficient parameter-efficient fine-tuning (PEFT) in the GLMM framework.

References
----------
- Hu et al. (2021): "LoRA: Low-Rank Adaptation of Large Language Models"
  https://arxiv.org/abs/2106.09685
- Microsoft LoRA: https://github.com/microsoft/LoRA
"""

from __future__ import annotations

import copy
import math

import torch
import torch.nn as nn
from torch import Tensor

__all__ = ["LoRALayer", "LoRALinear", "ParticipantLoRAAdapter"]


class LoRALayer(nn.Module):
    """Low-rank adaptation layer for attention projections.

    Implements: ΔW = (α/r) * B @ A
    where:
    - B ∈ ℝ^(in_features × rank)
    - A ∈ ℝ^(rank × out_features)
    - r is the rank (much smaller than in_features, out_features)
    - α is a scaling factor

    This additive update is applied to frozen base weights: W' = W + ΔW

    Parameters
    ----------
    in_features : int
        Input dimension.
    out_features : int
        Output dimension.
    rank : int, default=8
        LoRA rank r. Typical values: 4-16.
    alpha : float, default=16.0
        Scaling factor α. Typically 2*rank.
    dropout : float, default=0.1
        Dropout probability for LoRA path.

    Attributes
    ----------
    lora_A : nn.Parameter
        First low-rank matrix, shape (in_features, rank).
        Initialized with Kaiming uniform.
    lora_B : nn.Parameter
        Second low-rank matrix, shape (rank, out_features).
        Initialized with zeros (so ΔW = 0 initially).
    scaling : float
        Computed as α/r.

    Examples
    --------
    >>> lora = LoRALayer(768, 768, rank=8, alpha=16.0)
    >>> x = torch.randn(2, 10, 768)  # (batch, seq_len, in_features)
    >>> delta = lora(x)  # (batch, seq_len, out_features)
    >>> delta.shape
    torch.Size([2, 10, 768])
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
    ) -> None:
        """Initialize LoRA layer."""
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # Low-rank matrices (trainable)
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.dropout = nn.Dropout(dropout)

        # Initialize A with Kaiming uniform, B with zeros
        # This ensures ΔW = 0 at initialization
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: Tensor) -> Tensor:
        """Apply LoRA: x @ (A @ B) * scaling.

        Parameters
        ----------
        x : Tensor
            Input tensor, shape (batch, seq_len, in_features).

        Returns
        -------
        Tensor
            LoRA output, shape (batch, seq_len, out_features).
        """
        # x @ A: (batch, seq_len, in_features) @ (in_features, rank)
        #        = (batch, seq_len, rank)
        # @ B:   (batch, seq_len, rank) @ (rank, out_features)
        #        = (batch, seq_len, out_features)
        result = self.dropout(x) @ self.lora_A @ self.lora_B
        return result * self.scaling


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation.

    Wraps a frozen linear layer and adds trainable low-rank updates.
    Forward pass: output = base_layer(x) + lora(x)

    Parameters
    ----------
    base_layer : nn.Linear
        The original linear layer to adapt (will be frozen).
    rank : int, default=8
        LoRA rank r.
    alpha : float, default=16.0
        LoRA scaling factor α.
    dropout : float, default=0.1
        Dropout for LoRA path.

    Attributes
    ----------
    base_layer : nn.Linear
        Frozen base linear layer.
    lora : LoRALayer
        Low-rank adaptation layer.

    Examples
    --------
    >>> base = nn.Linear(768, 768)
    >>> lora_linear = LoRALinear(base, rank=8)
    >>> x = torch.randn(2, 10, 768)
    >>> out = lora_linear(x)
    >>> out.shape
    torch.Size([2, 10, 768])
    """

    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.1,
    ) -> None:
        """Initialize LoRA linear layer."""
        super().__init__()
        self.base_layer = base_layer

        # Freeze base layer parameters
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # Add LoRA adaptation
        self.lora = LoRALayer(
            base_layer.in_features,
            base_layer.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: base output + LoRA adaptation.

        Parameters
        ----------
        x : Tensor
            Input tensor, shape (batch, seq_len, in_features).

        Returns
        -------
        Tensor
            Output with LoRA adaptation, shape (batch, seq_len, out_features).
        """
        return self.base_layer(x) + self.lora(x)


class ParticipantLoRAAdapter(nn.Module):
    """Participant-specific LoRA adapters for seq2seq decoder.

    Injects LoRA layers into specified target modules (typically query and value
    projections in attention layers). Used for random slopes mode in GLMM.

    This class wraps a decoder module and applies participant-specific low-rank
    adaptations to attention projections.

    Parameters
    ----------
    decoder : nn.Module
        The decoder module to adapt (e.g., T5 decoder, BART decoder).
    rank : int
        LoRA rank r.
    alpha : float
        LoRA scaling factor α.
    dropout : float
        Dropout for LoRA layers.
    target_modules : list[str]
        Names of modules to inject LoRA into (e.g., ["q_proj", "v_proj"]).

    Attributes
    ----------
    decoder : nn.Module
        The adapted decoder (with LoRA layers injected).
    lora_layers : dict[str, LoRALinear]
        Mapping from module name to LoRA linear layer.

    Examples
    --------
    >>> from transformers import AutoModelForSeq2SeqLM
    >>> model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")  # doctest: +SKIP
    >>> decoder = model.get_decoder()  # doctest: +SKIP
    >>> adapter = ParticipantLoRAAdapter(  # doctest: +SKIP
    ...     decoder,
    ...     rank=8,
    ...     alpha=16.0,
    ...     target_modules=["q", "v"]  # T5 uses "q" and "v"
    ... )
    >>> # adapter.decoder now has LoRA layers injected
    """

    def __init__(
        self,
        decoder: nn.Module,
        rank: int,
        alpha: float,
        dropout: float,
        target_modules: list[str],
    ) -> None:
        """Initialize participant LoRA adapter."""
        super().__init__()
        self.decoder = decoder
        self.rank = rank
        self.alpha = alpha
        self.dropout = dropout
        self.target_modules = target_modules
        self.lora_layers: dict[str, LoRALinear] = {}

        # Inject LoRA into target modules
        self._inject_lora()

    def _inject_lora(self) -> None:
        """Inject LoRA into decoder attention layers.

        Searches for modules matching target_modules (e.g., "q_proj", "v_proj")
        and replaces them with LoRALinear wrappers.
        """
        # Build a mapping of full module paths to modules
        module_dict = dict(self.decoder.named_modules())

        for name, module in list(self.decoder.named_modules()):
            # Check if this module name contains any target module substring
            # e.g., "layer.0.SelfAttention.q" contains "q"
            if any(target in name for target in self.target_modules):
                if isinstance(module, nn.Linear):
                    # Get parent module and attribute name
                    path_parts = name.split(".")
                    if len(path_parts) == 1:
                        # Top-level module
                        parent = self.decoder
                        attr_name = name
                    else:
                        parent_path = ".".join(path_parts[:-1])
                        parent = module_dict[parent_path]
                        attr_name = path_parts[-1]

                    # Create LoRA linear layer
                    lora_layer = LoRALinear(
                        module,
                        rank=self.rank,
                        alpha=self.alpha,
                        dropout=self.dropout,
                    )

                    # Replace original module with LoRA version
                    setattr(parent, attr_name, lora_layer)
                    self.lora_layers[name] = lora_layer

    def forward(
        self, input_ids: Tensor, attention_mask: Tensor | None = None
    ) -> Tensor:
        """Forward pass through decoder with LoRA.

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

    def get_lora_parameters(self) -> list[nn.Parameter]:
        """Get all LoRA parameters for optimization.

        Returns
        -------
        list[nn.Parameter]
            List of all trainable LoRA parameters (A and B matrices).
        """
        params: list[nn.Parameter] = []
        for lora_linear in self.lora_layers.values():
            params.extend(lora_linear.lora.parameters())
        return params


def create_participant_lora_adapter(
    base_decoder: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    target_modules: list[str],
) -> ParticipantLoRAAdapter:
    """Create a participant LoRA adapter.

    Creates a deep copy of the base decoder and injects LoRA layers.

    Parameters
    ----------
    base_decoder : nn.Module
        Base decoder to copy and adapt.
    rank : int
        LoRA rank.
    alpha : float
        LoRA scaling factor.
    dropout : float
        LoRA dropout.
    target_modules : list[str]
        Target modules for LoRA injection.

    Returns
    -------
    ParticipantLoRAAdapter
        New adapter with LoRA injected into copied decoder.
    """
    # Deep copy the base decoder
    decoder_copy = copy.deepcopy(base_decoder)

    # Create adapter with LoRA
    adapter = ParticipantLoRAAdapter(
        decoder_copy,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules,
    )

    return adapter
