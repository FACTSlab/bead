"""Ordinal scale model for ordered rating scales (Likert, sliders, etc.).

Implements truncated normal distribution for bounded continuous responses on [0, 1].
Supports GLMM with participant-level random effects (intercepts and slopes).
"""

from __future__ import annotations

import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal
from transformers import AutoModel, AutoTokenizer

from bead.active_learning.config import MixedEffectsConfig, VarianceComponents
from bead.active_learning.models.base import ActiveLearningModel, ModelPrediction
from bead.active_learning.models.random_effects import RandomEffectsManager
from bead.config.active_learning import OrdinalScaleModelConfig
from bead.items.item import Item
from bead.items.item_template import ItemTemplate, TaskType

__all__ = ["OrdinalScaleModel"]


class OrdinalScaleModel(ActiveLearningModel):
    """Model for ordinal_scale tasks with bounded continuous responses.

    Uses truncated normal distribution on [scale_min, scale_max] to model
    slider/Likert responses while properly handling endpoints (0 and 1).
    Supports three modes: fixed effects, random intercepts, random slopes.

    Parameters
    ----------
    config : OrdinalScaleModelConfig
        Configuration object containing all model parameters.

    Attributes
    ----------
    config : OrdinalScaleModelConfig
        Model configuration.
    tokenizer : AutoTokenizer
        Transformer tokenizer.
    encoder : AutoModel
        Transformer encoder model.
    regression_head : nn.Sequential
        Regression head (fixed effects head) - outputs continuous μ.
    random_effects : RandomEffectsManager
        Manager for participant-level random effects.
    variance_history : list[VarianceComponents]
        Variance component estimates over training (for diagnostics).
    _is_fitted : bool
        Whether model has been trained.

    Examples
    --------
    >>> from uuid import uuid4
    >>> from bead.items.item import Item
    >>> from bead.config.active_learning import OrdinalScaleModelConfig
    >>> items = [
    ...     Item(
    ...         item_template_id=uuid4(),
    ...         rendered_elements={"text": f"Sentence {i}"}
    ...     )
    ...     for i in range(10)
    ... ]
    >>> labels = ["0.3", "0.7"] * 5  # Continuous values as strings
    >>> config = OrdinalScaleModelConfig(  # doctest: +SKIP
    ...     num_epochs=1, batch_size=2, device="cpu"
    ... )
    >>> model = OrdinalScaleModel(config=config)  # doctest: +SKIP
    >>> metrics = model.train(items, labels, participant_ids=None)  # doctest: +SKIP
    >>> predictions = model.predict(items[:3], participant_ids=None)  # doctest: +SKIP
    """

    def __init__(
        self,
        config: OrdinalScaleModelConfig | None = None,
    ) -> None:
        """Initialize ordinal scale model.

        Parameters
        ----------
        config : OrdinalScaleModelConfig | None
            Configuration object. If None, uses default configuration.
        """
        self.config = config or OrdinalScaleModelConfig()

        # Validate mixed_effects configuration
        super().__init__(self.config)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.encoder = AutoModel.from_pretrained(self.config.model_name)

        self.regression_head: nn.Sequential | None = None
        self._is_fitted = False

        # Initialize random effects manager
        self.random_effects: RandomEffectsManager | None = None
        self.variance_history: list[VarianceComponents] = []

        self.encoder.to(self.config.device)

    @property
    def supported_task_types(self) -> list[TaskType]:
        """Get supported task types.

        Returns
        -------
        list[TaskType]
            List containing "ordinal_scale".
        """
        return ["ordinal_scale"]

    def validate_item_compatibility(
        self, item: Item, item_template: ItemTemplate
    ) -> None:
        """Validate item is compatible with ordinal scale model.

        Parameters
        ----------
        item : Item
            Item to validate.
        item_template : ItemTemplate
            Template the item was constructed from.

        Raises
        ------
        ValueError
            If task_type is not "ordinal_scale".
        """
        if item_template.task_type != "ordinal_scale":
            raise ValueError(
                f"Expected task_type 'ordinal_scale', got '{item_template.task_type}'"
            )

    def _initialize_regression_head(self) -> None:
        """Initialize regression head for continuous output μ."""
        hidden_size = self.encoder.config.hidden_size

        # Single output for location parameter μ
        self.regression_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),  # Output μ (location parameter)
        )
        self.regression_head.to(self.config.device)

    def _encode_texts(self, texts: list[str]) -> torch.Tensor:
        """Encode texts using transformer.

        Parameters
        ----------
        texts : list[str]
            Texts to encode.

        Returns
        -------
        torch.Tensor
            Encoded representations of shape (batch_size, hidden_size).
        """
        encodings = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        )
        encodings = {k: v.to(self.config.device) for k, v in encodings.items()}

        outputs = self.encoder(**encodings)
        return outputs.last_hidden_state[:, 0, :]

    def _prepare_inputs(self, items: list[Item]) -> torch.Tensor:
        """Prepare inputs for encoding.

        For ordinal scale tasks, concatenates all rendered elements.

        Parameters
        ----------
        items : list[Item]
            Items to encode.

        Returns
        -------
        torch.Tensor
            Encoded representations.
        """
        texts = []
        for item in items:
            # Concatenate all rendered elements
            all_text = " ".join(item.rendered_elements.values())
            texts.append(all_text)
        return self._encode_texts(texts)

    def _truncated_normal_log_prob(
        self, y: torch.Tensor, mu: torch.Tensor, sigma: float
    ) -> torch.Tensor:
        """Compute log probability of truncated normal distribution.

        Uses truncated normal on [scale_min, scale_max] to properly handle
        endpoint responses (0.0 and 1.0) without arbitrary nudging.

        Parameters
        ----------
        y : torch.Tensor
            Observed values, shape (batch,).
        mu : torch.Tensor
            Location parameters (before truncation), shape (batch,).
        sigma : float
            Scale parameter (standard deviation).

        Returns
        -------
        torch.Tensor
            Log probabilities, shape (batch,).
        """
        base_dist = Normal(mu.squeeze(), sigma)

        # Unnormalized log prob
        log_prob_unnorm = base_dist.log_prob(y)

        # Normalizer: log(Φ((high-μ)/σ) - Φ((low-μ)/σ))
        alpha = (self.config.scale_min - mu.squeeze()) / sigma
        beta = (self.config.scale_max - mu.squeeze()) / sigma
        normalizer = base_dist.cdf(beta) - base_dist.cdf(alpha)

        # Clamp to avoid log(0)
        normalizer = torch.clamp(normalizer, min=1e-8)
        log_normalizer = torch.log(normalizer)

        return log_prob_unnorm - log_normalizer

    def train(
        self,
        items: list[Item],
        labels: list[str],
        participant_ids: list[str] | None = None,
        validation_items: list[Item] | None = None,
        validation_labels: list[str] | None = None,
    ) -> dict[str, float]:
        """Train model on ordinal scale data with participant-level random effects.

        Parameters
        ----------
        items : list[Item]
            Training items.
        labels : list[str]
            Training labels (continuous values as strings, e.g., "0.5", "0.75").
        participant_ids : list[str] | None
            Participant identifier for each item.
            - For fixed effects (mode='fixed'): Pass None.
            - For mixed effects: Must provide list[str] with same length as items.
        validation_items : list[Item] | None
            Optional validation items.
        validation_labels : list[str] | None
            Optional validation labels.

        Returns
        -------
        dict[str, float]
            Training metrics including:
            - "train_mse": Final training MSE
            - "train_loss": Final training negative log-likelihood
            - "val_mse": Validation MSE (if validation data provided)
            - "participant_variance": σ²_u (if estimate_variance_components=True)
            - "n_participants": Number of unique participants

        Raises
        ------
        ValueError
            If participant_ids is None when mode is 'random_intercepts'
            or 'random_slopes'.
        ValueError
            If items and labels have different lengths.
        ValueError
            If items and participant_ids have different lengths.
        ValueError
            If participant_ids contains empty strings.
        ValueError
            If labels contain invalid values (non-numeric or out of bounds).
        ValueError
            If validation data is incomplete.
        """
        # Validate and normalize participant_ids
        if participant_ids is None:
            if self.config.mixed_effects.mode != "fixed":
                raise ValueError(
                    f"participant_ids is required when "
                    f"mode='{self.config.mixed_effects.mode}'. "
                    f"For fixed effects, set mode='fixed' in config. "
                    f"For mixed effects, provide participant_ids as list[str]."
                )
            participant_ids = ["_fixed_"] * len(items)
        elif self.config.mixed_effects.mode == "fixed":
            warnings.warn(
                "participant_ids provided but mode='fixed'. "
                "Participant IDs will be ignored.",
                UserWarning,
                stacklevel=2,
            )
            participant_ids = ["_fixed_"] * len(items)

        # Validate input lengths
        if len(items) != len(labels):
            raise ValueError(
                f"Number of items ({len(items)}) must match "
                f"number of labels ({len(labels)})"
            )

        if len(items) != len(participant_ids):
            raise ValueError(
                f"Length mismatch: {len(items)} items != {len(participant_ids)} "
                f"participant_ids. participant_ids must have same length as items."
            )

        if any(not pid for pid in participant_ids):
            raise ValueError(
                "participant_ids cannot contain empty strings. "
                "Ensure all participants have valid identifiers."
            )

        if (validation_items is None) != (validation_labels is None):
            raise ValueError(
                "Both validation_items and validation_labels must be "
                "provided, or neither"
            )

        # Parse labels to floats and validate bounds
        try:
            y_values = [float(label) for label in labels]
        except ValueError as e:
            raise ValueError(
                f"Labels must be numeric strings (e.g., '0.5', '0.75'). "
                f"Got error: {e}"
            ) from e

        # Validate all values are within bounds
        for i, val in enumerate(y_values):
            if not (self.config.scale_min <= val <= self.config.scale_max):
                raise ValueError(
                    f"Label at index {i} ({val}) is outside bounds "
                    f"[{self.config.scale_min}, {self.config.scale_max}]"
                )

        y = torch.tensor(y_values, dtype=torch.float, device=self.config.device)

        self._initialize_regression_head()

        # Initialize random effects manager
        self.random_effects = RandomEffectsManager(
            self.config.mixed_effects, n_classes=1  # Scalar bias for μ
        )

        # Register participants for adaptive regularization
        participant_counts = Counter(participant_ids)
        for pid, count in participant_counts.items():
            self.random_effects.register_participant(pid, count)

        # Build optimizer parameters based on mode
        params_to_optimize = list(self.encoder.parameters()) + list(
            self.regression_head.parameters()
        )

        # Add random effects parameters
        if self.config.mixed_effects.mode == "random_intercepts":
            for param_dict in self.random_effects.intercepts.values():
                params_to_optimize.extend(param_dict.values())
        elif self.config.mixed_effects.mode == "random_slopes":
            for head in self.random_effects.slopes.values():
                params_to_optimize.extend(head.parameters())

        optimizer = torch.optim.AdamW(params_to_optimize, lr=self.config.learning_rate)

        self.encoder.train()
        self.regression_head.train()

        for _epoch in range(self.config.num_epochs):
            n_batches = (
                len(items) + self.config.batch_size - 1
            ) // self.config.batch_size
            epoch_loss = 0.0
            epoch_mse = 0.0

            for i in range(n_batches):
                start_idx = i * self.config.batch_size
                end_idx = min(start_idx + self.config.batch_size, len(items))

                batch_items = items[start_idx:end_idx]
                batch_labels = y[start_idx:end_idx]
                batch_participant_ids = participant_ids[start_idx:end_idx]

                embeddings = self._prepare_inputs(batch_items)

                # Forward pass depends on mixed effects mode
                if self.config.mixed_effects.mode == "fixed":
                    # Standard forward pass
                    mu = self.regression_head(embeddings).squeeze(1)  # (batch,)

                elif self.config.mixed_effects.mode == "random_intercepts":
                    # Fixed head + per-participant scalar bias
                    mu = self.regression_head(embeddings).squeeze(1)  # (batch,)
                    for j, pid in enumerate(batch_participant_ids):
                        # Scalar bias for location parameter
                        bias = self.random_effects.get_intercepts(
                            pid, n_classes=1, param_name="mu", create_if_missing=True
                        )
                        mu[j] = mu[j] + bias.item()

                elif self.config.mixed_effects.mode == "random_slopes":
                    # Per-participant head
                    mu_list = []
                    for j, pid in enumerate(batch_participant_ids):
                        participant_head = self.random_effects.get_slopes(
                            pid,
                            fixed_head=self.regression_head,
                            create_if_missing=True,
                        )
                        mu_j = participant_head(embeddings[j : j + 1]).squeeze()
                        mu_list.append(mu_j)
                    mu = torch.stack(mu_list)

                # Negative log-likelihood of truncated normal
                log_probs = self._truncated_normal_log_prob(
                    batch_labels, mu, self.config.sigma
                )
                loss_nll = -log_probs.mean()

                # Add prior regularization
                loss_prior = self.random_effects.compute_prior_loss()
                loss = loss_nll + loss_prior

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                # Also track MSE for interpretability
                mse = ((mu - batch_labels) ** 2).mean().item()
                epoch_mse += mse

            epoch_loss = epoch_loss / n_batches
            epoch_mse = epoch_mse / n_batches

        self._is_fitted = True

        metrics: dict[str, float] = {
            "train_loss": epoch_loss,
            "train_mse": epoch_mse,
        }

        # Estimate variance components
        if self.config.mixed_effects.estimate_variance_components:
            var_comps = self.random_effects.estimate_variance_components()
            if var_comps:
                var_comp = var_comps.get("mu") or var_comps.get("slopes")
                if var_comp:
                    self.variance_history.append(var_comp)
                    metrics["participant_variance"] = var_comp.variance
                    metrics["n_participants"] = var_comp.n_groups

        if validation_items is not None and validation_labels is not None:
            if len(validation_items) != len(validation_labels):
                raise ValueError(
                    f"Number of validation items ({len(validation_items)}) "
                    f"must match number of validation labels ({len(validation_labels)})"
                )

            # Validation
            if self.config.mixed_effects.mode == "fixed":
                val_predictions = self.predict(validation_items, participant_ids=None)
            else:
                val_participant_ids = ["_validation_"] * len(validation_items)
                val_predictions = self.predict(
                    validation_items, participant_ids=val_participant_ids
                )

            val_pred_values = [float(p.predicted_class) for p in val_predictions]
            val_true_values = [float(label) for label in validation_labels]
            val_mse = np.mean([
                (pred - true) ** 2
                for pred, true in zip(val_pred_values, val_true_values, strict=True)
            ])
            metrics["val_mse"] = val_mse

        return metrics

    def predict(
        self, items: list[Item], participant_ids: list[str] | None = None
    ) -> list[ModelPrediction]:
        """Predict continuous values for items with participant-specific random effects.

        Parameters
        ----------
        items : list[Item]
            Items to predict.
        participant_ids : list[str] | None
            Participant identifier for each item.
            - For fixed effects (mode='fixed'): Pass None.
            - For mixed effects: Must provide list[str] with same length as items.

        Returns
        -------
        list[ModelPrediction]
            Predictions with predicted_class as string representation of value.

        Raises
        ------
        ValueError
            If model has not been trained yet.
        ValueError
            If participant_ids is None when mode requires mixed effects.
        ValueError
            If items and participant_ids have different lengths.
        ValueError
            If participant_ids contains empty strings.
        """
        if not self._is_fitted:
            raise ValueError("Model not trained. Call train() before predict().")

        # Validate and normalize participant_ids
        if participant_ids is None:
            if self.config.mixed_effects.mode != "fixed":
                raise ValueError(
                    f"participant_ids is required when "
                    f"mode='{self.config.mixed_effects.mode}'. "
                    f"For fixed effects, set mode='fixed' in config. "
                    f"For mixed effects, provide participant_ids as list[str]."
                )
            participant_ids = ["_fixed_"] * len(items)
        elif self.config.mixed_effects.mode == "fixed":
            warnings.warn(
                "participant_ids provided but mode='fixed'. "
                "Participant IDs will be ignored.",
                UserWarning,
                stacklevel=2,
            )
            participant_ids = ["_fixed_"] * len(items)

        if len(items) != len(participant_ids):
            raise ValueError(
                f"Length mismatch: {len(items)} items != {len(participant_ids)} "
                f"participant_ids"
            )

        if any(not pid for pid in participant_ids):
            raise ValueError(
                "participant_ids cannot contain empty strings. "
                "Ensure all participants have valid identifiers."
            )

        self.encoder.eval()
        self.regression_head.eval()

        with torch.no_grad():
            embeddings = self._prepare_inputs(items)

            # Forward pass depends on mixed effects mode
            if self.config.mixed_effects.mode == "fixed":
                mu = self.regression_head(embeddings).squeeze(1)

            elif self.config.mixed_effects.mode == "random_intercepts":
                mu = self.regression_head(embeddings).squeeze(1)
                for i, pid in enumerate(participant_ids):
                    # Unknown participants: use prior mean (zero bias)
                    bias = self.random_effects.get_intercepts(
                        pid, n_classes=1, param_name="mu", create_if_missing=False
                    )
                    mu[i] = mu[i] + bias.item()

            elif self.config.mixed_effects.mode == "random_slopes":
                mu_list = []
                for i, pid in enumerate(participant_ids):
                    # Unknown participants: use fixed head
                    participant_head = self.random_effects.get_slopes(
                        pid, fixed_head=self.regression_head, create_if_missing=False
                    )
                    mu_i = participant_head(embeddings[i : i + 1]).squeeze()
                    mu_list.append(mu_i)
                mu = torch.stack(mu_list)

            # Clamp predictions to bounds
            mu = torch.clamp(mu, self.config.scale_min, self.config.scale_max)
            predictions_array = mu.cpu().numpy()

        predictions = []
        for i, item in enumerate(items):
            pred_value = float(predictions_array[i])
            predictions.append(
                ModelPrediction(
                    item_id=str(item.id),
                    probabilities={},  # Not applicable for regression
                    predicted_class=str(pred_value),  # Continuous value as string
                    confidence=1.0,  # Not applicable for regression
                )
            )

        return predictions

    def predict_proba(
        self, items: list[Item], participant_ids: list[str] | None = None
    ) -> np.ndarray:
        """Predict probabilities (not applicable for ordinal scale regression).

        For ordinal scale regression, returns μ values directly.

        Parameters
        ----------
        items : list[Item]
            Items to predict.
        participant_ids : list[str] | None
            Participant identifiers.

        Returns
        -------
        np.ndarray
            Array of shape (n_items, 1) with predicted μ values.
        """
        predictions = self.predict(items, participant_ids)
        return np.array([[float(p.predicted_class)] for p in predictions])

    def save(self, path: str) -> None:
        """Save model to disk including random effects and variance history.

        Parameters
        ----------
        path : str
            Directory path to save the model.

        Raises
        ------
        ValueError
            If model has not been trained yet.
        """
        if not self._is_fitted:
            raise ValueError("Model not trained. Call train() before save().")

        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        self.encoder.save_pretrained(save_path / "encoder")
        self.tokenizer.save_pretrained(save_path / "encoder")

        torch.save(
            self.regression_head.state_dict(),
            save_path / "regression_head.pt",
        )

        # Save random effects
        if self.random_effects is not None:
            self.random_effects.save(save_path / "random_effects")

        # Save config
        config_dict = self.config.model_dump()

        with open(save_path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

    def load(self, path: str) -> None:
        """Load model from disk including random effects and variance history.

        Parameters
        ----------
        path : str
            Directory path to load the model from.

        Raises
        ------
        FileNotFoundError
            If model directory does not exist.
        """
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Model directory not found: {path}")

        with open(load_path / "config.json") as f:
            config_dict = json.load(f)

        # Reconstruct configuration
        if "mixed_effects" in config_dict and isinstance(
            config_dict["mixed_effects"], dict
        ):
            config_dict["mixed_effects"] = MixedEffectsConfig(
                **config_dict["mixed_effects"]
            )

        self.config = OrdinalScaleModelConfig(**config_dict)

        self.encoder = AutoModel.from_pretrained(load_path / "encoder")
        self.tokenizer = AutoTokenizer.from_pretrained(load_path / "encoder")

        self._initialize_regression_head()
        self.regression_head.load_state_dict(
            torch.load(
                load_path / "regression_head.pt", map_location=self.config.device
            )
        )

        # Initialize and load random effects
        self.random_effects = RandomEffectsManager(
            self.config.mixed_effects, n_classes=1
        )
        random_effects_path = load_path / "random_effects"
        if random_effects_path.exists():
            self.random_effects.load(
                random_effects_path, fixed_head=self.regression_head
            )
            if self.random_effects.variance_history:
                self.variance_history = self.random_effects.variance_history.copy()

        self.encoder.to(self.config.device)
        self.regression_head.to(self.config.device)
        self._is_fitted = True
