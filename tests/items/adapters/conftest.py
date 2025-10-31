"""Fixtures for model adapter tests."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import torch

from bead.items.cache import ModelOutputCache


@pytest.fixture(autouse=True, scope="function")
def reset_adapter_imports():
    """Reset adapter module imports between tests to ensure clean state.

    This prevents issues where mocked API clients from one test
    interfere with imports in subsequent tests.
    """
    # Modules to clean up - only API client adapters and their dependencies
    # NOTE: We leave bead.items.adapters and huggingface adapters alone to avoid
    # PyO3 reload issues with transformers/safetensors
    modules_to_clean = [
        "bead.items.adapters.openai",
        "bead.items.adapters.anthropic",
        "bead.items.adapters.google",
        "bead.items.adapters.togetherai",
        "openai",
        "anthropic",
        "google.generativeai",
    ]

    # Remove cached modules BEFORE test to ensure clean state
    for module in modules_to_clean:
        if module in sys.modules:
            del sys.modules[module]

    yield

    # Clean up again after test
    for module in modules_to_clean:
        if module in sys.modules:
            del sys.modules[module]


@pytest.fixture
def temp_cache_dir(tmp_path: Path) -> Path:
    """Create temporary cache directory for tests.

    Parameters
    ----------
    tmp_path : Path
        pytest temporary directory fixture.

    Returns
    -------
    Path
        Temporary cache directory.
    """
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    return cache_dir


@pytest.fixture
def mock_cache(temp_cache_dir: Path) -> ModelOutputCache:
    """Create ModelOutputCache with temporary directory.

    Parameters
    ----------
    temp_cache_dir : Path
        Temporary cache directory.

    Returns
    -------
    ModelOutputCache
        Cache instance for testing.
    """
    return ModelOutputCache(cache_dir=temp_cache_dir, backend="filesystem")


@pytest.fixture
def in_memory_cache() -> ModelOutputCache:
    """Create in-memory cache for faster tests.

    Returns
    -------
    ModelOutputCache
        In-memory cache instance.
    """
    return ModelOutputCache(backend="memory")


@pytest.fixture
def sample_texts() -> list[str]:
    """Sample texts for testing.

    Returns
    -------
    list[str]
        List of sample sentences.
    """
    return [
        "The cat sat on the mat.",
        "The dog stood on the rug.",
        "Mary loves reading books.",
        "John enjoys playing tennis.",
        "Python is a programming language.",
    ]


@pytest.fixture
def mock_gpt2_tokenizer() -> MagicMock:
    """Create mock GPT-2 tokenizer.

    Returns
    -------
    MagicMock
        Mock tokenizer with basic functionality.
    """
    tokenizer = MagicMock()
    tokenizer.pad_token = None
    tokenizer.eos_token = "<|endoftext|>"
    tokenizer.eos_token_id = 50256

    def mock_encode(text: str, **kwargs: Any) -> dict[str, torch.Tensor]:
        # Simple mock: create fake tokens based on text length
        words = text.split()
        num_tokens = len(words) + 2  # +2 for special tokens
        input_ids = torch.randint(0, 50000, (1, num_tokens))
        attention_mask = torch.ones(1, num_tokens)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    tokenizer.return_value = mock_encode
    tokenizer.side_effect = mock_encode
    return tokenizer


@pytest.fixture
def mock_gpt2_model() -> MagicMock:
    """Create mock GPT-2 causal language model.

    Returns
    -------
    MagicMock
        Mock model with basic functionality.
    """
    model = MagicMock()

    def mock_forward(*args: Any, **kwargs: Any) -> MagicMock:
        output = MagicMock()
        # Mock loss (negative log-likelihood per token)
        output.loss = torch.tensor(2.5)
        # Mock hidden states
        batch_size = 1
        seq_len = kwargs.get("input_ids", torch.zeros(1, 10)).size(1)
        hidden_size = 768
        hidden = torch.randn(batch_size, seq_len, hidden_size)
        output.hidden_states = (hidden,)  # Tuple of layers
        return output

    model.return_value = mock_forward
    model.side_effect = mock_forward
    model.to = MagicMock(return_value=model)
    model.eval = MagicMock(return_value=model)
    return model


@pytest.fixture
def mock_bert_tokenizer() -> MagicMock:
    """Create mock BERT tokenizer.

    Returns
    -------
    MagicMock
        Mock tokenizer with BERT-specific tokens.
    """
    tokenizer = MagicMock()
    tokenizer.cls_token_id = 101
    tokenizer.sep_token_id = 102
    tokenizer.pad_token_id = 0
    tokenizer.mask_token_id = 103

    def mock_encode(text: str, **kwargs: Any) -> dict[str, torch.Tensor]:
        # Simple mock: create fake tokens based on text length
        words = text.split()
        num_tokens = len(words) + 2  # +2 for [CLS] and [SEP]
        input_ids = torch.randint(104, 30000, (1, num_tokens))
        # Set special tokens
        input_ids[0, 0] = 101  # [CLS]
        input_ids[0, -1] = 102  # [SEP]
        attention_mask = torch.ones(1, num_tokens)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    tokenizer.return_value = mock_encode
    tokenizer.side_effect = mock_encode
    return tokenizer


@pytest.fixture
def mock_bert_model() -> MagicMock:
    """Create mock BERT masked language model.

    Returns
    -------
    MagicMock
        Mock BERT model.
    """
    model = MagicMock()

    def mock_forward(*args: Any, **kwargs: Any) -> MagicMock:
        output = MagicMock()
        # Mock logits for masked token prediction
        batch_size = 1
        seq_len = kwargs.get("input_ids", torch.zeros(1, 10)).size(1)
        vocab_size = 30522
        output.logits = torch.randn(batch_size, seq_len, vocab_size)
        # Mock hidden states
        hidden_size = 768
        hidden = torch.randn(batch_size, seq_len, hidden_size)
        output.hidden_states = (hidden,)  # Tuple of layers
        return output

    model.return_value = mock_forward
    model.side_effect = mock_forward
    model.to = MagicMock(return_value=model)
    model.eval = MagicMock(return_value=model)
    return model


@pytest.fixture
def mock_nli_tokenizer() -> MagicMock:
    """Create mock NLI tokenizer.

    Returns
    -------
    MagicMock
        Mock tokenizer for NLI models.
    """
    tokenizer = MagicMock()

    def mock_encode(
        premise: str, hypothesis: str | None = None, **kwargs: Any
    ) -> dict[str, torch.Tensor]:
        # Mock encoding for premise-hypothesis pair
        total_words = len(premise.split())
        if hypothesis:
            total_words += len(hypothesis.split())
        num_tokens = total_words + 3  # +3 for [CLS], [SEP], [SEP]
        input_ids = torch.randint(0, 30000, (1, num_tokens))
        attention_mask = torch.ones(1, num_tokens)
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    tokenizer.return_value = mock_encode
    tokenizer.side_effect = mock_encode
    return tokenizer


@pytest.fixture
def mock_nli_model() -> MagicMock:
    """Create mock NLI model.

    Returns
    -------
    MagicMock
        Mock NLI model with classification head.
    """
    model = MagicMock()

    # Mock RoBERTa base model for embeddings
    base_model = MagicMock()

    def base_forward(*args: Any, **kwargs: Any) -> MagicMock:
        output = MagicMock()
        batch_size = 1
        seq_len = kwargs.get("input_ids", torch.zeros(1, 10)).size(1)
        hidden_size = 768
        output.last_hidden_state = torch.randn(batch_size, seq_len, hidden_size)
        return output

    base_model.return_value = base_forward
    base_model.side_effect = base_forward
    model.roberta = base_model

    def mock_forward(*args: Any, **kwargs: Any) -> MagicMock:
        output = MagicMock()
        # Mock logits for 3-class classification (entailment, neutral, contradiction)
        output.logits = torch.tensor([[2.0, 0.5, -1.0]])  # Favors entailment
        # Mock hidden states for embedding extraction
        batch_size = 1
        seq_len = kwargs.get("input_ids", torch.zeros(1, 10)).size(1)
        hidden_size = 768
        hidden = torch.randn(batch_size, seq_len, hidden_size)
        output.hidden_states = (hidden,)
        return output

    model.return_value = mock_forward
    model.side_effect = mock_forward
    model.to = MagicMock(return_value=model)
    model.eval = MagicMock(return_value=model)
    return model


@pytest.fixture
def mock_nli_config() -> MagicMock:
    """Create mock NLI model config.

    Returns
    -------
    MagicMock
        Mock config with label mappings.
    """
    config = MagicMock()
    config.id2label = {0: "entailment", 1: "neutral", 2: "contradiction"}
    return config


@pytest.fixture
def mock_sentence_transformer() -> MagicMock:
    """Create mock sentence transformer model.

    Returns
    -------
    MagicMock
        Mock sentence transformer.
    """
    model = MagicMock()

    def mock_encode(
        text: str | list[str], convert_to_numpy: bool = True, **kwargs: Any
    ) -> np.ndarray:
        # Return deterministic embedding based on text
        if isinstance(text, str):
            # Use hash of text to create deterministic but varied embeddings
            np.random.seed(hash(text) % (2**32))
            embedding = np.random.randn(384)  # Common size for all-MiniLM
            if kwargs.get("normalize_embeddings", True):
                embedding = embedding / np.linalg.norm(embedding)
            return embedding
        else:
            # Batch encoding
            return np.array([mock_encode(t, **kwargs) for t in text])

    model.encode = mock_encode
    return model


@pytest.fixture
def expected_log_prob() -> float:
    """Provide expected log probability for mock models.

    Returns
    -------
    float
        Expected log probability value.
    """
    return -25.0  # Negative of (loss * num_tokens) for mock


@pytest.fixture
def expected_perplexity() -> float:
    """Provide expected perplexity for mock models.

    Returns
    -------
    float
        Expected perplexity value.
    """
    return np.exp(2.5)  # exp(loss) for mock


@pytest.fixture
def expected_nli_scores() -> dict[str, float]:
    """Provide expected NLI scores for mock model.

    Returns
    -------
    dict[str, float]
        Expected NLI probability distribution.
    """
    # Based on mock logits [2.0, 0.5, -1.0]
    logits = torch.tensor([2.0, 0.5, -1.0])
    probs = torch.softmax(logits, dim=0).numpy()
    return {
        "entailment": float(probs[0]),
        "neutral": float(probs[1]),
        "contradiction": float(probs[2]),
    }
