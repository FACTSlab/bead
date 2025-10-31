"""Model adapters for judgment prediction during item construction.

This module provides adapters for various model types used to compute
constraints during Stage 3 (Item Construction). Each adapter integrates
with the ModelOutputCache (Phase 13) for efficient caching.

These are SEPARATE from template filling adapters (bead.templates.models),
which are used in Stage 2.

Available Adapters
------------------
Local Models:
- HuggingFaceLanguageModel: Causal language models (GPT-2, LLaMA, etc.)
- HuggingFaceMaskedLanguageModel: Masked language models (BERT, RoBERTa, etc.)
- HuggingFaceNLI: Natural language inference models (MNLI, etc.)
- HuggingFaceSentenceTransformer: Sentence embedding models

API Models:
- OpenAIAdapter: OpenAI GPT models via API
- AnthropicAdapter: Anthropic Claude models via API
- GoogleAdapter: Google Gemini models via API
- TogetherAIAdapter: Together AI models via API

Registry:
- ModelAdapterRegistry: Centralized registry for all adapters
- default_registry: Pre-configured registry with all built-in adapters
"""

# API utilities - explicit re-exports for type checkers
from bead.items.adapters.api_utils import (
    RateLimiter as RateLimiter,
)
from bead.items.adapters.api_utils import (
    rate_limit as rate_limit,
)
from bead.items.adapters.api_utils import (
    retry_with_backoff as retry_with_backoff,
)
from bead.items.adapters.base import ModelAdapter as ModelAdapter
from bead.items.adapters.huggingface import (
    HuggingFaceLanguageModel as HuggingFaceLanguageModel,
)
from bead.items.adapters.huggingface import (
    HuggingFaceMaskedLanguageModel as HuggingFaceMaskedLanguageModel,
)
from bead.items.adapters.huggingface import (
    HuggingFaceNLI as HuggingFaceNLI,
)

# Registry - explicit re-exports for type checkers
from bead.items.adapters.registry import (
    ModelAdapterRegistry as ModelAdapterRegistry,
)
from bead.items.adapters.registry import (
    default_registry as default_registry,
)
from bead.items.adapters.sentence_transformers import (
    HuggingFaceSentenceTransformer as HuggingFaceSentenceTransformer,
)

# API adapters (optional, may not be available if dependencies not installed)
try:
    from bead.items.adapters.openai import OpenAIAdapter as OpenAIAdapter
except ImportError:
    pass

try:
    from bead.items.adapters.anthropic import AnthropicAdapter as AnthropicAdapter
except ImportError:
    pass

try:
    from bead.items.adapters.google import GoogleAdapter as GoogleAdapter
except ImportError:
    pass

try:
    from bead.items.adapters.togetherai import TogetherAIAdapter as TogetherAIAdapter
except ImportError:
    pass

__all__ = [
    # Base
    "ModelAdapter",
    # HuggingFace adapters
    "HuggingFaceLanguageModel",
    "HuggingFaceMaskedLanguageModel",
    "HuggingFaceNLI",
    "HuggingFaceSentenceTransformer",
    # API utilities
    "RateLimiter",
    "rate_limit",
    "retry_with_backoff",
    # Registry
    "ModelAdapterRegistry",
    "default_registry",
    # API adapters (conditionally exported based on available dependencies)
    "OpenAIAdapter",
    "AnthropicAdapter",
    "GoogleAdapter",
    "TogetherAIAdapter",
]
