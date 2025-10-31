"""Together AI adapter for item construction.

This module provides a ModelAdapter implementation for Together AI's API,
which provides access to various open-source models. Together AI uses an
OpenAI-compatible API, so we use the OpenAI client with a custom base URL.
"""

from __future__ import annotations

import os

import numpy as np

try:
    import openai
except ImportError as e:
    raise ImportError(
        "openai package is required for Together AI adapter. "
        "Install it with: pip install openai"
    ) from e

from bead.items.adapters.api_utils import rate_limit, retry_with_backoff
from bead.items.adapters.base import ModelAdapter
from bead.items.cache import ModelOutputCache


class TogetherAIAdapter(ModelAdapter):
    """Adapter for Together AI models.

    Together AI provides access to various open-source models through an
    OpenAI-compatible API. This adapter uses the OpenAI client with a
    custom base URL.

    Parameters
    ----------
    model_name : str
        Together AI model identifier
        (default: "meta-llama/Llama-3-70b-chat-hf").
    api_key : str | None
        Together AI API key. If None, uses TOGETHER_API_KEY environment variable.
    cache : ModelOutputCache | None
        Cache for model outputs. If None, creates in-memory cache.
    model_version : str
        Model version for cache tracking (default: "latest").

    Attributes
    ----------
    model_name : str
        Together AI model identifier (e.g., "meta-llama/Llama-3-70b-chat-hf").
    client : openai.OpenAI
        OpenAI-compatible client configured for Together AI.

    Raises
    ------
    ValueError
        If no API key is provided and TOGETHER_API_KEY is not set.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-3-70b-chat-hf",
        api_key: str | None = None,
        cache: ModelOutputCache | None = None,
        model_version: str = "latest",
    ) -> None:
        if cache is None:
            cache = ModelOutputCache(backend="memory")

        super().__init__(
            model_name=model_name, cache=cache, model_version=model_version
        )

        # Get API key from parameter or environment
        if api_key is None:
            api_key = os.environ.get("TOGETHER_API_KEY")
            if api_key is None:
                raise ValueError(
                    "Together AI API key must be provided via api_key parameter "
                    "or TOGETHER_API_KEY environment variable"
                )

        # Together AI uses OpenAI-compatible API
        self.client = openai.OpenAI(
            api_key=api_key, base_url="https://api.together.xyz/v1"
        )

    @retry_with_backoff(
        max_retries=3,
        initial_delay=1.0,
        backoff_factor=2.0,
        exceptions=(openai.APIError, openai.APIConnectionError, openai.RateLimitError),
    )
    @rate_limit(calls_per_minute=60)
    def compute_log_probability(self, text: str) -> float:
        """Compute log probability of text using Together AI API.

        Uses the completions API with logprobs to get token-level log probabilities
        and sums them to get the total log probability.

        Parameters
        ----------
        text : str
            Text to compute log probability for.

        Returns
        -------
        float
            Log probability of the text (sum of token log probabilities).
        """
        # Check cache
        cached = self.cache.get(
            model_name=self.model_name, operation="log_probability", text=text
        )
        if cached is not None:
            return float(cached)

        # Call API
        try:
            response = self.client.completions.create(
                model=self.model_name,
                prompt=text,
                max_tokens=0,
                echo=True,
                logprobs=1,
            )

            # Sum token log probabilities
            logprobs = response.choices[0].logprobs
            if logprobs is None or logprobs.token_logprobs is None:
                raise ValueError("API response did not include logprobs")

            # Filter out None values (first token may have None)
            token_logprobs = [lp for lp in logprobs.token_logprobs if lp is not None]
            total_log_prob = sum(token_logprobs)

        except (openai.BadRequestError, AttributeError) as e:
            # Some models may not support completions API, fall back to chat
            raise NotImplementedError(
                f"Log probability computation is not supported for model "
                f"{self.model_name}. This model may not support the "
                "completions API with logprobs."
            ) from e

        # Cache result
        self.cache.set(
            model_name=self.model_name,
            operation="log_probability",
            result=total_log_prob,
            model_version=self.model_version,
            text=text,
        )

        return float(total_log_prob)

    def compute_perplexity(self, text: str) -> float:
        """Compute perplexity of text.

        Perplexity is computed as exp(-log_prob / num_tokens).

        Parameters
        ----------
        text : str
            Text to compute perplexity for.

        Returns
        -------
        float
            Perplexity of the text (must be positive).

        Raises
        ------
        NotImplementedError
            If log probability computation is not supported.
        """
        # Check cache
        cached = self.cache.get(
            model_name=self.model_name, operation="perplexity", text=text
        )
        if cached is not None:
            return float(cached)

        # Get log probability
        log_prob = self.compute_log_probability(text)

        # Estimate number of tokens (rough approximation: 1 token ~ 4 chars)
        num_tokens = max(1, len(text) // 4)

        # Compute perplexity: exp(-log_prob / num_tokens)
        perplexity = np.exp(-log_prob / num_tokens)

        # Cache result
        self.cache.set(
            model_name=self.model_name,
            operation="perplexity",
            result=float(perplexity),
            model_version=self.model_version,
            text=text,
        )

        return float(perplexity)

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text.

        Not supported by Together AI (no embedding-specific models).

        Raises
        ------
        NotImplementedError
            Always raised - Together AI does not provide embeddings.
        """
        raise NotImplementedError(
            "Embedding computation is not supported by Together AI. "
            "Together AI focuses on text generation models. "
            "Consider using OpenAI's text-embedding models or sentence transformers."
        )

    @retry_with_backoff(
        max_retries=3,
        initial_delay=1.0,
        backoff_factor=2.0,
        exceptions=(openai.APIError, openai.APIConnectionError, openai.RateLimitError),
    )
    @rate_limit(calls_per_minute=60)
    def compute_nli(self, premise: str, hypothesis: str) -> dict[str, float]:
        """Compute natural language inference scores via prompting.

        Uses chat completions API with a prompt to classify the relationship
        between premise and hypothesis.

        Parameters
        ----------
        premise : str
            Premise text.
        hypothesis : str
            Hypothesis text.

        Returns
        -------
        dict[str, float]
            Dictionary with keys "entailment", "neutral", "contradiction"
            mapping to probability scores.
        """
        # Check cache
        cached = self.cache.get(
            model_name=self.model_name,
            operation="nli",
            premise=premise,
            hypothesis=hypothesis,
        )
        if cached is not None:
            return dict(cached)

        # Construct prompt
        prompt = (
            "Given the following premise and hypothesis, "
            "determine the relationship between them.\n\n"
            f"Premise: {premise}\n"
            f"Hypothesis: {hypothesis}\n\n"
            "Choose one of the following:\n"
            "- entailment: The hypothesis is definitely true given the premise\n"
            "- neutral: The hypothesis might be true given the premise\n"
            "- contradiction: The hypothesis is definitely false given the premise\n\n"
            "Respond with only one word: entailment, neutral, or contradiction."
        )

        # Call API
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10,
        )

        # Parse response
        answer = response.choices[0].message.content
        if answer is None:
            raise ValueError("API response did not include content")

        answer = answer.strip().lower()

        # Map to scores
        scores: dict[str, float] = {
            "entailment": 0.0,
            "neutral": 0.0,
            "contradiction": 0.0,
        }

        if "entailment" in answer:
            scores["entailment"] = 1.0
        elif "neutral" in answer:
            scores["neutral"] = 1.0
        elif "contradiction" in answer:
            scores["contradiction"] = 1.0
        else:
            # Default to neutral if unclear
            scores["neutral"] = 1.0

        # Cache result
        self.cache.set(
            model_name=self.model_name,
            operation="nli",
            result=scores,
            model_version=self.model_version,
            premise=premise,
            hypothesis=hypothesis,
        )

        return scores
