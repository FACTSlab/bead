"""Shared utilities for HuggingFace Transformers adapters.

This module provides common functionality for adapters that integrate with
HuggingFace Transformers models, including device validation and shared
utilities.
"""

from __future__ import annotations

import logging
from typing import Literal

import torch.backends.mps
import torch.cuda

logger = logging.getLogger(__name__)

DeviceType = Literal["cpu", "cuda", "mps"]


def _cuda_available() -> bool:
    """Check if CUDA is available."""
    return torch.cuda.is_available()  # pyright: ignore[reportAttributeAccessIssue]


def _mps_available() -> bool:
    """Check if MPS (Apple Silicon) is available."""
    return torch.backends.mps.is_available()  # pyright: ignore[reportAttributeAccessIssue]


class HuggingFaceAdapterMixin:
    """Mixin providing common HuggingFace adapter functionality.

    This mixin provides device validation with automatic fallback.

    Attributes
    ----------
    device : DeviceType
        The validated device (cpu, cuda, or mps).
    """

    def _validate_device(self, device: DeviceType) -> DeviceType:
        """Validate device and fallback if unavailable.

        Parameters
        ----------
        device : DeviceType
            Requested device.

        Returns
        -------
        DeviceType
            Validated device (falls back to CPU if unavailable).
        """
        if device == "cuda" and not _cuda_available():
            logger.warning("CUDA unavailable, using CPU")
            return "cpu"
        if device == "mps" and not _mps_available():
            logger.warning("MPS unavailable, using CPU")
            return "cpu"
        return device
