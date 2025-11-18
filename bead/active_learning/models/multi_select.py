"""Multi-select model for selecting multiple options.

Expected architecture: Multi-label classification with sigmoid output per option.
Each option can be independently selected or not selected.
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
from bead.config.active_learning import MultiSelectModelConfig
from bead.items.item import Item
from bead.items.item_template import ItemTemplate, TaskType

__all__ = ["MultiSelectModel"]


class MultiSelectModel(ActiveLearningModel):
    """Model for multi_select tasks with N selectable options.

    Uses multi-label classification where each option can be independently
    selected or not selected. Applies sigmoid activation to each option's
    logit and uses BCEWithLogitsLoss for training.

    Parameters
    ----------
    config : MultiSelectModelConfig
        Configuration object containing all model parameters.

    Attributes
    ----------
    config : MultiSelectModelConfig
        Model configuration.
    tokenizer : AutoTokenizer
        Transformer tokenizer.
    encoder : AutoModel
        Transformer encoder model.
    classifier_head : nn.Sequential
        Classification head (fixed effects head) - outputs N logits.
    num_options : int | None
        Number of selectable options (inferred from training data).
    option_names : list[str] | None
        Option names (e.g., ["option_a", "option_b", "option_c"]).
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
    >>> from bead.config.active_learning import MultiSelectModelConfig
    >>> items = [
    ...     Item(
    ...         item_template_id=uuid4(),
    ...         rendered_elements={
    ...             "option_a": "First option",
    ...             "option_b": "Second option",
    ...             "option_c": "Third option"
    ...         }
    ...     )
    ...     for _ in range(10)
    ... ]
    >>> # Labels as lists of selected options
    >>> labels_list = [["option_a", "option_b"], ["option_c"], ["option_a"]]
    >>> labels = labels_list * 3 + [["option_b"]]
    >>> config = MultiSelectModelConfig(  # doctest: +SKIP
    ...     num_epochs=1, batch_size=2, device="cpu"
    ... )
    >>> model = MultiSelectModel(config=config)  # doctest: +SKIP
    >>> # Convert labels to serialized format for train()
    >>> label_strs = [json.dumps(sorted(lbls)) for lbls in labels]  # doctest: +SKIP
    >>> metrics = model.train(items, label_strs, participant_ids=None)  # doctest: +SKIP

    Notes
    -----
    This model uses BCEWithLogitsLoss (not CrossEntropyLoss) and applies
    sigmoid activation to get independent probabilities for each option.
    Random intercepts are bias vectors (one per option) that shift logits
    independently for each participant.
    """

    def __init__(
        self,
        config: MultiSelectModelConfig | None = None,
    ) -> None:
        """Initialize multi-select model.

        Parameters
        ----------
        config : MultiSelectModelConfig | None
            Configuration object. If None, uses default configuration.
        """
        self.config = config or MultiSelectModelConfig()

        # Validate mixed_effects configuration
        super().__init__(self.config)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.encoder = AutoModel.from_pretrained(self.config.model_name)

        self.num_options: int | None = None
        self.option_names: list[str] | None = None
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
            List containing "multi_select".
        """
        return ["multi_select"]

    def validate_item_compatibility(
        self, item: Item, item_template: ItemTemplate
    ) -> None:
        """Validate item is compatible with multi-select model.

        Parameters
        ----------
        item : Item
            Item to validate.
        item_template : ItemTemplate
            Template the item was constructed from.

        Raises
        ------
        ValueError
            If task_type is not "multi_select".
        """
        if item_template.task_type != "multi_select":
            raise ValueError(
                f"Expected task_type 'multi_select', got '{item_template.task_type}'"
            )

    def _initialize_classifier(self, num_options: int) -> None:
        """Initialize classification head for given number of options.

        Parameters
        ----------
        num_options : int
            Number of selectable options (output units).
        """
        hidden_size = self.encoder.config.hidden_size

        if self.config.encoder_mode == "dual_encoder":
            input_size = hidden_size * num_options
        else:
            input_size = hidden_size

        self.classifier_head = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_options),  # N independent outputs
        )
        self.classifier_head.to(self.config.device)

    def _encode_single(self, texts: list[str]) -> torch.Tensor:
        """Encode texts using single encoder strategy.

        Concatenates all option texts with [SEP] tokens and encodes once.

        Parameters
        ----------
        texts : list[str]
            List of concatenated option texts for each item.

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

    def _encode_dual(self, options_per_item: list[list[str]]) -> torch.Tensor:
        """Encode texts using dual encoder strategy.

        Encodes each option separately and concatenates embeddings.

        Parameters
        ----------
        options_per_item : list[list[str]]
            List of option lists. Each inner list contains option texts for one item.

        Returns
        -------
        torch.Tensor
            Concatenated encodings of shape (batch_size, hidden_size * num_options).
        """
        all_embeddings = []

        for options in options_per_item:
            option_embeddings = []
            for option_text in options:
                encodings = self.tokenizer(
                    [option_text],
                    padding=True,
                    truncation=True,
                    max_length=self.config.max_length,
                    return_tensors="pt",
                )
                encodings = {k: v.to(self.config.device) for k, v in encodings.items()}

                outputs = self.encoder(**encodings)
                cls_embedding = outputs.last_hidden_state[0, 0, :]
                option_embeddings.append(cls_embedding)

            concatenated = torch.cat(option_embeddings, dim=0)
            all_embeddings.append(concatenated)

        return torch.stack(all_embeddings)

    def _prepare_inputs(self, items: list[Item]) -> torch.Tensor:
        """Prepare inputs for encoding based on encoder mode.

        For multi-select tasks, uses all options from rendered_elements.

        Parameters
        ----------
        items : list[Item]
            Items to encode.

        Returns
        -------
        torch.Tensor
            Encoded representations.
        """
        if self.option_names is None:
            raise ValueError("Model not initialized. Call train() first.")

        if self.config.encoder_mode == "single_encoder":
            texts = []
            for item in items:
                option_texts = [
                    item.rendered_elements.get(opt, "") for opt in self.option_names
                ]
                concatenated = " [SEP] ".join(option_texts)
                texts.append(concatenated)
            return self._encode_single(texts)
        else:
            options_per_item = []
            for item in items:
                option_texts = [
                    item.rendered_elements.get(opt, "") for opt in self.option_names
                ]
                options_per_item.append(option_texts)
            return self._encode_dual(options_per_item)

    def _parse_multi_select_labels(self, label_str: str) -> list[str]:
        """Parse multi-select label from JSON string.

        Parameters
        ----------
        label_str : str
            JSON-serialized list of selected options.

        Returns
        -------
        list[str]
            List of selected option names.
        """
        try:
            selected = json.loads(label_str)
            if not isinstance(selected, list):
                raise ValueError(
                    f"Label must be JSON list of option names, got {type(selected)}"
                )
            return selected
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Label must be valid JSON list of selected options. "
                f"Got: {label_str!r}. Error: {e}"
            ) from e

    def train(
        self,
        items: list[Item],
        labels: list[str],
        participant_ids: list[str] | None = None,
        validation_items: list[Item] | None = None,
        validation_labels: list[str] | None = None,
    ) -> dict[str, float]:
        """Train model on multi-select data with participant-level random effects.

        Parameters
        ----------
        items : list[Item]
            Training items.
        labels : list[str]
            Training labels. Each label is a JSON-serialized list of selected
            option names, e.g., '["option_a", "option_c"]'.
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
            - "train_accuracy": Final training accuracy (Hamming score)
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

        # Infer option names from first item
        if not items:
            raise ValueError("Cannot train with empty items list")

        self.option_names = sorted(items[0].rendered_elements.keys())
        self.num_options = len(self.option_names)
        option_to_idx = {opt: idx for idx, opt in enumerate(self.option_names)}

        # Parse labels and convert to binary matrix
        y = torch.zeros(
            (len(items), self.num_options), dtype=torch.float, device=self.config.device
        )
        for i, label_str in enumerate(labels):
            selected_options = self._parse_multi_select_labels(label_str)
            for opt in selected_options:
                if opt not in option_to_idx:
                    raise ValueError(
                        f"Invalid option {opt!r} in label. "
                        f"Valid options: {self.option_names}"
                    )
                y[i, option_to_idx[opt]] = 1.0

        self._initialize_classifier(self.num_options)

        # Initialize random effects manager
        self.random_effects = RandomEffectsManager(
            self.config.mixed_effects, n_classes=self.num_options
        )

        # Register participants for adaptive regularization
        participant_counts = Counter(participant_ids)
        for pid, count in participant_counts.items():
            self.random_effects.register_participant(pid, count)

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
        # BCE with Logits Loss for multi-label classification
        criterion = nn.BCEWithLogitsLoss()

        self.encoder.train()
        self.classifier_head.train()

        for _epoch in range(self.config.num_epochs):
            n_batches = (
                len(items) + self.config.batch_size - 1
            ) // self.config.batch_size
            epoch_loss = 0.0
            epoch_correct_predictions = 0
            epoch_total_predictions = 0

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
                    logits = self.classifier_head(embeddings)

                elif self.config.mixed_effects.mode == "random_intercepts":
                    # Fixed head + per-participant bias (independent per option)
                    logits = self.classifier_head(embeddings)
                    for j, pid in enumerate(batch_participant_ids):
                        bias = self.random_effects.get_intercepts(
                            pid, n_classes=self.num_options, param_name="mu", create_if_missing=True
                        )
                        logits[j] = logits[j] + bias

                elif self.config.mixed_effects.mode == "random_slopes":
                    # Per-participant head
                    logits_list = []
                    for j, pid in enumerate(batch_participant_ids):
                        participant_head = self.random_effects.get_slopes(
                            pid,
                            fixed_head=self.classifier_head,
                            create_if_missing=True,
                        )
                        logits_j = participant_head(embeddings[j : j + 1])
                        logits_list.append(logits_j)
                    logits = torch.cat(logits_list, dim=0)

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
                # Hamming accuracy: fraction of correct predictions (per option)
                batch_correct = (predictions == batch_labels).sum().item()
                batch_total = batch_labels.numel()
                epoch_correct_predictions += batch_correct
                epoch_total_predictions += batch_total

            # Hamming accuracy: average over all (item, option) pairs
            epoch_acc = epoch_correct_predictions / epoch_total_predictions
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

            # Validation with placeholder participant_ids for mixed effects
            if self.config.mixed_effects.mode == "fixed":
                val_predictions = self.predict(validation_items, participant_ids=None)
            else:
                val_participant_ids = ["_validation_"] * len(validation_items)
                val_predictions = self.predict(validation_items, participant_ids=val_participant_ids)

            # Parse validation labels
            val_labels_parsed = [
                set(self._parse_multi_select_labels(lbl))
                for lbl in validation_labels
            ]

            # Compute Hamming accuracy
            val_correct = 0
            val_total = 0
            for pred, true_set in zip(
                val_predictions, val_labels_parsed, strict=True
            ):
                # pred.predicted_class is JSON string of selected options
                pred_set = set(json.loads(pred.predicted_class))
                for opt in self.option_names:
                    if (opt in pred_set) == (opt in true_set):
                        val_correct += 1
                    val_total += 1

            val_acc = val_correct / val_total
            metrics["val_accuracy"] = val_acc

        return metrics

    def predict(
        self, items: list[Item], participant_ids: list[str] | None = None
    ) -> list[ModelPrediction]:
        """Predict selected options for items with participant-specific random effects.

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
            Predictions with probabilities and predicted selected options.
            predicted_class is a JSON string of selected option names.

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
                logits = self.classifier_head(embeddings)

            elif self.config.mixed_effects.mode == "random_intercepts":
                logits = self.classifier_head(embeddings)
                for i, pid in enumerate(participant_ids):
                    # Unknown participants: use prior mean (zero bias)
                    bias = self.random_effects.get_intercepts(
                        pid, n_classes=self.num_options, param_name="mu", create_if_missing=False
                    )
                    logits[i] = logits[i] + bias

            elif self.config.mixed_effects.mode == "random_slopes":
                logits_list = []
                for i, pid in enumerate(participant_ids):
                    # Unknown participants: use fixed head
                    participant_head = self.random_effects.get_slopes(
                        pid, fixed_head=self.classifier_head, create_if_missing=False
                    )
                    logits_i = participant_head(embeddings[i : i + 1])
                    logits_list.append(logits_i)
                logits = torch.cat(logits_list, dim=0)

            # Compute probabilities using sigmoid
            proba = torch.sigmoid(logits).cpu().numpy()  # (n_items, n_options)
            pred_binary = proba > 0.5  # Threshold at 0.5

        predictions = []
        for i, item in enumerate(items):
            # Determine selected options
            selected_options = [
                self.option_names[j]
                for j in range(self.num_options)
                if pred_binary[i, j]
            ]

            # Build probability dict: {option: probability}
            prob_dict = {
                opt: float(proba[i, idx])
                for idx, opt in enumerate(self.option_names)
            }

            # Confidence: average probability of selected options (or 0.5 if none)
            if selected_options:
                option_probs = [
                    proba[i, self.option_names.index(opt)]
                    for opt in selected_options
                ]
                confidence = float(np.mean(option_probs))
            else:
                confidence = 0.5  # Neutral confidence when nothing selected

            predictions.append(
                ModelPrediction(
                    item_id=str(item.id),
                    probabilities=prob_dict,
                    predicted_class=json.dumps(sorted(selected_options)),
                    confidence=confidence,
                )
            )

        return predictions

    def predict_proba(
        self, items: list[Item], participant_ids: list[str] | None = None
    ) -> np.ndarray:
        """Predict selection probabilities for items with random effects.

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
            Array of shape (n_items, n_options) with probabilities.

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
                logits = self.classifier_head(embeddings)

            elif self.config.mixed_effects.mode == "random_intercepts":
                logits = self.classifier_head(embeddings)
                for i, pid in enumerate(participant_ids):
                    bias = self.random_effects.get_intercepts(
                        pid, n_classes=self.num_options, param_name="mu", create_if_missing=False
                    )
                    logits[i] = logits[i] + bias

            elif self.config.mixed_effects.mode == "random_slopes":
                logits_list = []
                for i, pid in enumerate(participant_ids):
                    participant_head = self.random_effects.get_slopes(
                        pid, fixed_head=self.classifier_head, create_if_missing=False
                    )
                    logits_i = participant_head(embeddings[i : i + 1])
                    logits_list.append(logits_i)
                logits = torch.cat(logits_list, dim=0)

            # Compute probabilities using sigmoid
            proba = torch.sigmoid(logits).cpu().numpy()

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
        config_dict["num_options"] = self.num_options
        config_dict["option_names"] = self.option_names

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
        self.num_options = config_dict.pop("num_options")
        self.option_names = config_dict.pop("option_names")

        # Reconstruct configuration
        # Handle mixed_effects reconstruction
        if "mixed_effects" in config_dict and isinstance(
            config_dict["mixed_effects"], dict
        ):
            config_dict["mixed_effects"] = MixedEffectsConfig(
                **config_dict["mixed_effects"]
            )

        self.config = MultiSelectModelConfig(**config_dict)

        self.encoder = AutoModel.from_pretrained(load_path / "encoder")
        self.tokenizer = AutoTokenizer.from_pretrained(load_path / "encoder")

        self._initialize_classifier(self.num_options)
        self.classifier_head.load_state_dict(
            torch.load(
                load_path / "classifier_head.pt", map_location=self.config.device
            )
        )

        # Initialize and load random effects
        self.random_effects = RandomEffectsManager(
            self.config.mixed_effects, n_classes=self.num_options
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
