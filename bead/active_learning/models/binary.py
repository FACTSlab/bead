"""Binary model for yes/no or true/false judgments.

Expected architecture: Binary classification with 2-class output.
Different from 2AFC in semantics - represents absolute judgment rather than choice.
"""

from __future__ import annotations

import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from bead.active_learning.config import MixedEffectsConfig, VarianceComponents
from bead.active_learning.models.base import ActiveLearningModel, ModelPrediction
from bead.active_learning.models.random_effects import RandomEffectsManager
from bead.config.active_learning import BinaryModelConfig
from bead.items.item import Item
from bead.items.item_template import ItemTemplate, TaskType

__all__ = ["BinaryModel"]


class BinaryModel(ActiveLearningModel):
    """Model for binary tasks (yes/no, true/false judgments).

    Uses true binary classification with a single output unit and sigmoid
    activation (logistic regression). This is more efficient than using
    2-class softmax, as we only need to output P(y=1) and compute
    P(y=0) = 1 - P(y=1).

    Parameters
    ----------
    config : BinaryModelConfig
        Configuration object containing all model parameters.

    Attributes
    ----------
    config : BinaryModelConfig
        Model configuration.
    tokenizer : AutoTokenizer
        Transformer tokenizer.
    encoder : AutoModel
        Transformer encoder model.
    classifier_head : nn.Sequential
        Classification head (fixed effects head) - outputs single logit.
    num_classes : int
        Number of output units (always 1 for binary classification).
    label_names : list[str] | None
        Label names (e.g., ["no", "yes"] sorted alphabetically).
    positive_class : str | None
        Which label corresponds to y=1 (second alphabetically).
    random_effects : RandomEffectsManager
        Manager for participant-level random effects (scalar biases).
    variance_history : list[VarianceComponents]
        Variance component estimates over training (for diagnostics).
    _is_fitted : bool
        Whether model has been trained.

    Examples
    --------
    >>> from uuid import uuid4
    >>> from bead.items.item import Item
    >>> from bead.config.active_learning import BinaryModelConfig
    >>> items = [
    ...     Item(
    ...         item_template_id=uuid4(),
    ...         rendered_elements={"text": f"Sentence {i}"}
    ...     )
    ...     for i in range(10)
    ... ]
    >>> labels = ["yes"] * 5 + ["no"] * 5
    >>> config = BinaryModelConfig(  # doctest: +SKIP
    ...     num_epochs=1, batch_size=2, device="cpu"
    ... )
    >>> model = BinaryModel(config=config)  # doctest: +SKIP
    >>> metrics = model.train(items, labels, participant_ids=None)  # doctest: +SKIP
    >>> predictions = model.predict(items[:3], participant_ids=None)  # doctest: +SKIP

    Notes
    -----
    This model uses BCEWithLogitsLoss instead of CrossEntropyLoss, and applies
    sigmoid activation to get probabilities. Random intercepts are scalar values
    (1-dimensional) that shift the logit for each participant.
    """

    def __init__(
        self,
        config: BinaryModelConfig | None = None,
    ) -> None:
        """Initialize binary model.

        Parameters
        ----------
        config : BinaryModelConfig | None
            Configuration object. If None, uses default configuration.
        """
        self.config = config or BinaryModelConfig()

        # Validate mixed_effects configuration
        super().__init__(self.config)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.encoder = AutoModel.from_pretrained(self.config.model_name)

        self.num_classes: int = 1  # Single output unit for binary classification
        self.label_names: list[str] | None = None
        self.positive_class: str | None = None  # Which label corresponds to 1
        self.classifier_head: nn.Sequential | None = None
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
            List containing "binary".
        """
        return ["binary"]

    def validate_item_compatibility(
        self, item: Item, item_template: ItemTemplate
    ) -> None:
        """Validate item is compatible with binary model.

        Parameters
        ----------
        item : Item
            Item to validate.
        item_template : ItemTemplate
            Template the item was constructed from.

        Raises
        ------
        ValueError
            If task_type is not "binary".
        """
        if item_template.task_type != "binary":
            raise ValueError(
                f"Expected task_type 'binary', got '{item_template.task_type}'"
            )

    def _initialize_classifier(self) -> None:
        """Initialize classification head for binary classification.

        Outputs a single value (logit) for sigmoid activation.
        """
        hidden_size = self.encoder.config.hidden_size

        # Single output unit for binary classification
        if self.config.encoder_mode == "dual_encoder":
            input_size = hidden_size * 2
        else:
            input_size = hidden_size

        self.classifier_head = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),  # Single output unit
        )
        self.classifier_head.to(self.config.device)

    def _encode_texts(self, texts: list[str]) -> torch.Tensor:
        """Encode texts using single encoder.

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

        For binary tasks, concatenates all rendered elements.

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

    def _validate_labels(self, labels: list[str]) -> None:
        """Validate that all labels are valid.

        Parameters
        ----------
        labels : list[str]
            Labels to validate.

        Raises
        ------
        ValueError
            If any label is not in label_names.
        """
        if self.label_names is None:
            raise ValueError("label_names not initialized")

        valid_labels = set(self.label_names)
        invalid = [label for label in labels if label not in valid_labels]
        if invalid:
            raise ValueError(
                f"Invalid labels found: {set(invalid)}. "
                f"Labels must be one of {valid_labels}."
            )

    def train(
        self,
        items: list[Item],
        labels: list[str],
        participant_ids: list[str] | None = None,
        validation_items: list[Item] | None = None,
        validation_labels: list[str] | None = None,
    ) -> dict[str, float]:
        """Train model on binary data with participant-level random effects.

        Parameters
        ----------
        items : list[Item]
            Training items.
        labels : list[str]
            Training labels (binary values like "yes"/"no" or "true"/"false").
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
            - "train_accuracy": Final training accuracy
            - "train_loss": Final training loss
            - "val_accuracy": Validation accuracy (if validation data provided)
            - "participant_variance": σ²_u (if estimate_variance_components=True)
            - "n_participants": Number of unique participants

        Raises
        ------
        ValueError
            If participant_ids is None when mode is 'random_intercepts' or 'random_slopes'.
        ValueError
            If items and labels have different lengths.
        ValueError
            If items and participant_ids have different lengths.
        ValueError
            If participant_ids contains empty strings.
        ValueError
            If labels contain invalid values.
        ValueError
            If validation data is incomplete.
        """
        # Validate and normalize participant_ids
        if participant_ids is None:
            if self.config.mixed_effects.mode != "fixed":
                raise ValueError(
                    f"participant_ids is required when mode='{self.config.mixed_effects.mode}'. "
                    f"For fixed effects, set mode='fixed' in config. "
                    f"For mixed effects, provide participant_ids as list[str]."
                )
            participant_ids = ["_fixed_"] * len(items)
        elif self.config.mixed_effects.mode == "fixed":
            warnings.warn(
                f"participant_ids provided but mode='fixed'. Participant IDs will be ignored.",
                UserWarning,
                stacklevel=2
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

        # Initialize label names
        unique_labels = sorted(set(labels))
        if len(unique_labels) != 2:
            raise ValueError(
                f"Binary classification requires exactly 2 classes, "
                f"got {len(unique_labels)}: {unique_labels}"
            )
        self.label_names = unique_labels
        # Positive class is the second one alphabetically (index 1)
        self.positive_class = unique_labels[1]

        self._validate_labels(labels)
        self._initialize_classifier()

        # Initialize random effects manager with n_classes=1 (scalar bias)
        self.random_effects = RandomEffectsManager(
            self.config.mixed_effects, n_classes=self.num_classes
        )

        # Register participants for adaptive regularization
        participant_counts = Counter(participant_ids)
        for pid, count in participant_counts.items():
            self.random_effects.register_participant(pid, count)

        # Convert labels to binary (0/1) floats
        # Positive class (second alphabetically) = 1, negative = 0
        y = torch.tensor(
            [1.0 if label == self.positive_class else 0.0 for label in labels],
            dtype=torch.float,
            device=self.config.device,
        )

        # Build optimizer parameters based on mode
        params_to_optimize = list(self.encoder.parameters()) + list(
            self.classifier_head.parameters()
        )

        # Add random effects parameters
        if self.config.mixed_effects.mode == "random_intercepts":
            for param_dict in self.random_effects.intercepts.values():
                params_to_optimize.extend(param_dict.values())
        elif self.config.mixed_effects.mode == "random_slopes":
            for head in self.random_effects.slopes.values():
                params_to_optimize.extend(head.parameters())

        optimizer = torch.optim.AdamW(params_to_optimize, lr=self.config.learning_rate)
        # BCE with Logits Loss for binary classification (applies sigmoid internally)
        criterion = nn.BCEWithLogitsLoss()

        self.encoder.train()
        self.classifier_head.train()

        for _epoch in range(self.config.num_epochs):
            n_batches = (
                len(items) + self.config.batch_size - 1
            ) // self.config.batch_size
            epoch_loss = 0.0
            epoch_correct = 0

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
                    logits = self.classifier_head(embeddings).squeeze(1)  # (batch,)

                elif self.config.mixed_effects.mode == "random_intercepts":
                    # Fixed head + per-participant scalar bias
                    logits = self.classifier_head(embeddings).squeeze(1)  # (batch,)
                    for j, pid in enumerate(batch_participant_ids):
                        # bias is now scalar (n_classes=1)
                        bias = self.random_effects.get_intercepts(
                            pid,
                            n_classes=self.num_classes,
                            param_name="mu",
                            create_if_missing=True,
                        )
                        logits[j] = logits[j] + bias.item()

                elif self.config.mixed_effects.mode == "random_slopes":
                    # Per-participant head
                    logits_list = []
                    for j, pid in enumerate(batch_participant_ids):
                        participant_head = self.random_effects.get_slopes(
                            pid,
                            fixed_head=self.classifier_head,
                            create_if_missing=True,
                        )
                        logits_j = participant_head(embeddings[j : j + 1]).squeeze()
                        logits_list.append(logits_j)
                    logits = torch.stack(logits_list)

                # Data loss + prior regularization
                loss_bce = criterion(logits, batch_labels)
                loss_prior = self.random_effects.compute_prior_loss()
                loss = loss_bce + loss_prior

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                # Predictions: threshold at 0.5 on sigmoid(logits)
                predictions = (torch.sigmoid(logits) > 0.5).float()
                epoch_correct += (predictions == batch_labels).sum().item()

            epoch_acc = epoch_correct / len(items)
            epoch_loss = epoch_loss / n_batches

        self._is_fitted = True

        metrics: dict[str, float] = {
            "train_accuracy": epoch_acc,
            "train_loss": epoch_loss,
        }

        # Estimate variance components
        if self.config.mixed_effects.estimate_variance_components:
            var_comps = self.random_effects.estimate_variance_components()
            if var_comps:
                # Extract the "mu" variance component (single-parameter model)
                var_comp = var_comps.get("mu") or var_comps.get("slopes")
                if var_comp:
                    self.variance_history.append(var_comp)
                    metrics["participant_variance"] = var_comp.variance
                    metrics["n_participants"] = var_comp.n_groups

        if validation_items is not None and validation_labels is not None:
            self._validate_labels(validation_labels)

            if len(validation_items) != len(validation_labels):
                raise ValueError(
                    f"Number of validation items ({len(validation_items)}) "
                    f"must match number of validation labels ({len(validation_labels)})"
                )

            # Validation with placeholder participant_ids for mixed effects
            if self.config.mixed_effects.mode == "fixed":
                val_predictions = self.predict(validation_items, participant_ids=None)
            else:
                val_participant_ids = ["_validation_"] * len(validation_items)
                val_predictions = self.predict(validation_items, participant_ids=val_participant_ids)
            val_pred_labels = [p.predicted_class for p in val_predictions]
            val_acc = sum(
                pred == true
                for pred, true in zip(val_pred_labels, validation_labels, strict=True)
            ) / len(validation_labels)
            metrics["val_accuracy"] = val_acc

        return metrics

    def predict(
        self, items: list[Item], participant_ids: list[str] | None = None
    ) -> list[ModelPrediction]:
        """Predict class labels for items with participant-specific random effects.

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
            Predictions with probabilities and predicted class.

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
                    f"participant_ids is required when mode='{self.config.mixed_effects.mode}'. "
                    f"For fixed effects, set mode='fixed' in config. "
                    f"For mixed effects, provide participant_ids as list[str]."
                )
            participant_ids = ["_fixed_"] * len(items)
        elif self.config.mixed_effects.mode == "fixed":
            warnings.warn(
                f"participant_ids provided but mode='fixed'. Participant IDs will be ignored.",
                UserWarning,
                stacklevel=2
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
        self.classifier_head.eval()

        with torch.no_grad():
            embeddings = self._prepare_inputs(items)

            # Forward pass depends on mixed effects mode
            if self.config.mixed_effects.mode == "fixed":
                logits = self.classifier_head(embeddings).squeeze(1)  # (n_items,)

            elif self.config.mixed_effects.mode == "random_intercepts":
                logits = self.classifier_head(embeddings).squeeze(1)  # (n_items,)
                for i, pid in enumerate(participant_ids):
                    # Unknown participants: use prior mean (zero bias)
                    bias = self.random_effects.get_intercepts(
                        pid,
                        n_classes=self.num_classes,
                        param_name="mu",
                        create_if_missing=False,
                    )
                    logits[i] = logits[i] + bias.item()

            elif self.config.mixed_effects.mode == "random_slopes":
                logits_list = []
                for i, pid in enumerate(participant_ids):
                    # Unknown participants: use fixed head
                    participant_head = self.random_effects.get_slopes(
                        pid, fixed_head=self.classifier_head, create_if_missing=False
                    )
                    logits_i = participant_head(embeddings[i : i + 1]).squeeze()
                    logits_list.append(logits_i)
                logits = torch.stack(logits_list)

            # Compute probabilities using sigmoid
            proba_positive = torch.sigmoid(logits).cpu().numpy()  # P(y=1)
            pred_is_positive = proba_positive > 0.5

        predictions = []
        for i, item in enumerate(items):
            # Determine predicted class
            if pred_is_positive[i]:
                pred_label = self.positive_class
            else:
                pred_label = self.label_names[0]

            # Build probability dict: {negative_class: p0, positive_class: p1}
            p1 = float(proba_positive[i])
            p0 = 1.0 - p1
            prob_dict = {
                self.label_names[0]: p0,  # Negative class (first alphabetically)
                self.positive_class: p1,   # Positive class (second alphabetically)
            }

            predictions.append(
                ModelPrediction(
                    item_id=str(item.id),
                    probabilities=prob_dict,
                    predicted_class=pred_label,
                    confidence=max(p0, p1),
                )
            )

        return predictions

    def predict_proba(
        self, items: list[Item], participant_ids: list[str] | None = None
    ) -> np.ndarray:
        """Predict class probabilities for items with random effects.

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
        np.ndarray
            Array of shape (n_items, 2) with probabilities.

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
            raise ValueError("Model not trained. Call train() before predict_proba().")

        # Validate and normalize participant_ids
        if participant_ids is None:
            if self.config.mixed_effects.mode != "fixed":
                raise ValueError(
                    f"participant_ids is required when mode='{self.config.mixed_effects.mode}'. "
                    f"For fixed effects, set mode='fixed' in config. "
                    f"For mixed effects, provide participant_ids as list[str]."
                )
            participant_ids = ["_fixed_"] * len(items)
        elif self.config.mixed_effects.mode == "fixed":
            warnings.warn(
                f"participant_ids provided but mode='fixed'. Participant IDs will be ignored.",
                UserWarning,
                stacklevel=2
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
        self.classifier_head.eval()

        with torch.no_grad():
            embeddings = self._prepare_inputs(items)

            # Forward pass depends on mixed effects mode
            if self.config.mixed_effects.mode == "fixed":
                logits = self.classifier_head(embeddings).squeeze(1)

            elif self.config.mixed_effects.mode == "random_intercepts":
                logits = self.classifier_head(embeddings).squeeze(1)
                for i, pid in enumerate(participant_ids):
                    bias = self.random_effects.get_intercepts(
                        pid,
                        n_classes=self.num_classes,
                        param_name="mu",
                        create_if_missing=False,
                    )
                    logits[i] = logits[i] + bias.item()

            elif self.config.mixed_effects.mode == "random_slopes":
                logits_list = []
                for i, pid in enumerate(participant_ids):
                    participant_head = self.random_effects.get_slopes(
                        pid, fixed_head=self.classifier_head, create_if_missing=False
                    )
                    logits_i = participant_head(embeddings[i : i + 1]).squeeze()
                    logits_list.append(logits_i)
                logits = torch.stack(logits_list)

            # Compute probabilities using sigmoid
            proba_positive = torch.sigmoid(logits).cpu().numpy()  # P(y=1)

        # Return (n_items, 2) array: [P(negative), P(positive)]
        proba = np.stack([1.0 - proba_positive, proba_positive], axis=1)

        return proba

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
            self.classifier_head.state_dict(),
            save_path / "classifier_head.pt",
        )

        # Save random effects (includes variance history)
        if self.random_effects is not None:
            self.random_effects.save(save_path / "random_effects")

        # Save both config and training state
        config_dict = self.config.model_dump()
        config_dict["num_classes"] = self.num_classes
        config_dict["label_names"] = self.label_names
        config_dict["positive_class"] = self.positive_class

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

        # Extract training state
        self.num_classes = config_dict.pop("num_classes")
        self.label_names = config_dict.pop("label_names")
        self.positive_class = config_dict.pop("positive_class")

        # Reconstruct configuration
        # Handle mixed_effects reconstruction
        if "mixed_effects" in config_dict and isinstance(
            config_dict["mixed_effects"], dict
        ):
            config_dict["mixed_effects"] = MixedEffectsConfig(
                **config_dict["mixed_effects"]
            )

        self.config = BinaryModelConfig(**config_dict)

        self.encoder = AutoModel.from_pretrained(load_path / "encoder")
        self.tokenizer = AutoTokenizer.from_pretrained(load_path / "encoder")

        self._initialize_classifier()
        self.classifier_head.load_state_dict(
            torch.load(
                load_path / "classifier_head.pt", map_location=self.config.device
            )
        )

        # Initialize and load random effects
        self.random_effects = RandomEffectsManager(
            self.config.mixed_effects, n_classes=self.num_classes
        )
        random_effects_path = load_path / "random_effects"
        if random_effects_path.exists():
            self.random_effects.load(
                random_effects_path, fixed_head=self.classifier_head
            )
            # Restore variance history
            if self.random_effects.variance_history:
                self.variance_history = self.random_effects.variance_history.copy()

        self.encoder.to(self.config.device)
        self.classifier_head.to(self.config.device)
        self._is_fitted = True
