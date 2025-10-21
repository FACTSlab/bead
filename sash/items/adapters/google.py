"""Google Generative AI adapter for item construction.

This module provides a ModelAdapter implementation for Google's Generative AI
models (Gemini), supporting natural language inference via prompting and
embeddings. Note that Gemini API does not provide direct access to log
probabilities.
"""

from __future__ import annotations

import os

import numpy as np

try:
    import google.generativeai as genai
except ImportError as e:
    raise ImportError(
        "google-generativeai package is required for Google adapter. "
        "Install it with: pip install google-generativeai"
    ) from e

from sash.items.adapters.api_utils import rate_limit, retry_with_backoff
from sash.items.adapters.base import ModelAdapter
from sash.items.cache import ModelOutputCache


class GoogleAdapter(ModelAdapter):
    """Adapter for Google Generative AI models (Gemini).

    Provides access to Gemini models for natural language inference and
    embeddings. Note that Gemini API does not support log probability
    computation.

    Attributes
    ----------
    model_name : str
        Gemini model identifier (e.g., "gemini-pro").
    model : genai.GenerativeModel
        Google Generative AI model instance.
    embedding_model : str
        Model to use for embeddings (default: "models/embedding-001").
    """

    def __init__(
        self,
        model_name: str = "gemini-pro",
        api_key: str | None = None,
        cache: ModelOutputCache | None = None,
        model_version: str = "latest",
        embedding_model: str = "models/embedding-001",
    ) -> None:
        """Initialize Google adapter.

        Parameters
        ----------
        model_name : str
            Gemini model identifier (default: "gemini-pro").
        api_key : str | None
            Google API key. If None, uses GOOGLE_API_KEY environment variable.
        cache : ModelOutputCache | None
            Cache for model outputs. If None, creates in-memory cache.
        model_version : str
            Model version for cache tracking (default: "latest").
        embedding_model : str
            Model to use for embeddings (default: "models/embedding-001").

        Raises
        ------
        ValueError
            If no API key is provided and GOOGLE_API_KEY is not set.
        """
        if cache is None:
            from sash.items.cache import ModelOutputCache

            cache = ModelOutputCache(backend="memory")

        super().__init__(
            model_name=model_name, cache=cache, model_version=model_version
        )

        # Get API key from parameter or environment
        if api_key is None:
            api_key = os.environ.get("GOOGLE_API_KEY")
            if api_key is None:
                raise ValueError(
                    "Google API key must be provided via api_key parameter "
                    "or GOOGLE_API_KEY environment variable"
                )

        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        self.embedding_model = embedding_model

    def compute_log_probability(self, text: str) -> float:
        """Compute log probability of text.

        Not supported by Google Generative AI API.

        Raises
        ------
        NotImplementedError
            Always raised - Gemini API does not provide log probabilities.
        """
        raise NotImplementedError(
            "Log probability computation is not supported by Google Generative AI. "
            "Gemini does not provide access to token-level probabilities."
        )

    def compute_perplexity(self, text: str) -> float:
        """Compute perplexity of text.

        Not supported by Google Generative AI API (requires log probabilities).

        Raises
        ------
        NotImplementedError
            Always raised - requires log probability support.
        """
        raise NotImplementedError(
            "Perplexity computation is not supported by Google Generative AI. "
            "This operation requires log probabilities, which Gemini does not provide."
        )

    @retry_with_backoff(
        max_retries=3,
        initial_delay=1.0,
        backoff_factor=2.0,
        exceptions=(Exception,),  # Google API uses generic exceptions
    )
    @rate_limit(calls_per_minute=60)
    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding vector for text using Google's embedding model.

        Parameters
        ----------
        text : str
            Text to embed.

        Returns
        -------
        np.ndarray
            Embedding vector for the text.
        """
        # Check cache
        cached = self.cache.get(
            model_name=self.embedding_model, operation="embedding", text=text
        )
        if cached is not None:
            return np.array(cached)

        # Call API
        result = genai.embed_content(
            model=self.embedding_model,
            content=text,
            task_type="retrieval_document",
        )

        embedding = np.array(result["embedding"])

        # Cache result
        self.cache.set(
            model_name=self.embedding_model,
            operation="embedding",
            result=embedding.tolist(),
            model_version=self.model_version,
            text=text,
        )

        return embedding

    @retry_with_backoff(
        max_retries=3,
        initial_delay=1.0,
        backoff_factor=2.0,
        exceptions=(Exception,),  # Google API uses generic exceptions
    )
    @rate_limit(calls_per_minute=60)
    def compute_nli(self, premise: str, hypothesis: str) -> dict[str, float]:
        """Compute natural language inference scores via prompting.

        Uses Gemini's generation API with a prompt to classify the relationship
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
        response = self.model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.0,
                max_output_tokens=10,
            ),
        )

        # Parse response
        if not response.text:
            raise ValueError("API response did not include text")

        answer = response.text.strip().lower()

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
