"""Cloze model for fill-in-the-blank tasks with GLMM support.

Implements masked language modeling with participant-level random effects for
predicting tokens at unfilled slots in partially-filled templates. Supports
three modes: fixed effects, random intercepts, random slopes.

Architecture: Masked LM (BERT/RoBERTa) for token prediction
"""

from __future__ import annotations

import copy
import json
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from bead.active_learning.config import MixedEffectsConfig, VarianceComponents
from bead.active_learning.models.base import ActiveLearningModel, ModelPrediction
from bead.active_learning.models.random_effects import RandomEffectsManager
from bead.config.active_learning import ClozeModelConfig
from bead.items.item import Item
from bead.items.item_template import ItemTemplate, TaskType

__all__ = ["ClozeModel"]


class ClozeModel(ActiveLearningModel):
    """Model for cloze tasks with participant-level random effects.

    Uses masked language modeling (BERT/RoBERTa) to predict tokens at unfilled
    slots in partially-filled templates. Supports three GLMM modes:
    - Fixed effects: Standard MLM
    - Random intercepts: Participant-specific bias on output logits
    - Random slopes: Participant-specific MLM heads

    Parameters
    ----------
    config : ClozeModelConfig
        Configuration object containing all model parameters.

    Attributes
    ----------
    config : ClozeModelConfig
        Model configuration.
    tokenizer : AutoTokenizer
        Masked LM tokenizer.
    model : AutoModelForMaskedLM
        Masked language model (BERT or RoBERTa).
    encoder : nn.Module
        Encoder module from the model.
    mlm_head : nn.Module
        MLM prediction head.
    random_effects : RandomEffectsManager
        Manager for participant-level random effects.
    variance_history : list[VarianceComponents]
        Variance component estimates over training.
    _is_fitted : bool
        Whether model has been trained.

    Examples
    --------
    >>> from uuid import uuid4
    >>> from bead.items.item import Item, UnfilledSlot
    >>> from bead.config.active_learning import ClozeModelConfig
    >>> items = [
    ...     Item(
    ...         item_template_id=uuid4(),
    ...         rendered_elements={"text": "The cat ___."},
    ...         unfilled_slots=[
    ...             UnfilledSlot(slot_name="verb", position=2, constraint_ids=[])
    ...         ]
    ...     )
    ...     for _ in range(6)
    ... ]
    >>> labels = [["ran"], ["jumped"], ["slept"]] * 2  # One token per unfilled slot
    >>> config = ClozeModelConfig(  # doctest: +SKIP
    ...     num_epochs=1, batch_size=2, device="cpu"
    ... )
    >>> model = ClozeModel(config=config)  # doctest: +SKIP
    >>> metrics = model.train(items, labels, participant_ids=None)  # doctest: +SKIP
    """

    def __init__(
        self,
        config: ClozeModelConfig | None = None,
    ) -> None:
        """Initialize cloze model.

        Parameters
        ----------
        config : ClozeModelConfig | None
            Configuration object. If None, uses default configuration.
        """
        self.config = config or ClozeModelConfig()

        # Validate mixed_effects configuration
        super().__init__(self.config)

        # Load tokenizer and masked LM model
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(self.config.model_name)

        # Extract encoder and MLM head
        # BERT-style models use model.bert and model.cls
        # RoBERTa-style models use model.roberta and model.lm_head
        if hasattr(self.model, "bert"):
            self.encoder = self.model.bert
            self.mlm_head = self.model.cls
        elif hasattr(self.model, "roberta"):
            self.encoder = self.model.roberta
            self.mlm_head = self.model.lm_head
        else:
            # Fallback: try to use the base model attribute
            self.encoder = self.model.base_model
            self.mlm_head = self.model.lm_head

        self._is_fitted = False

        # Initialize random effects manager (created during training)
        self.random_effects: RandomEffectsManager | None = None
        self.variance_history: list[VarianceComponents] = []

        self.model.to(self.config.device)

    @property
    def supported_task_types(self) -> list[TaskType]:
        """Get supported task types.

        Returns
        -------
        list[TaskType]
            List containing "cloze".
        """
        return ["cloze"]

    def validate_item_compatibility(
        self, item: Item, item_template: ItemTemplate
    ) -> None:
        """Validate item is compatible with cloze model.

        Parameters
        ----------
        item : Item
            Item to validate.
        item_template : ItemTemplate
            Template the item was constructed from.

        Raises
        ------
        ValueError
            If task_type is not "cloze".
        ValueError
            If item has no unfilled_slots.
        """
        if item_template.task_type != "cloze":
            raise ValueError(
                f"Expected task_type 'cloze', got '{item_template.task_type}'"
            )

        if not item.unfilled_slots:
            raise ValueError(
                "Cloze items must have at least one unfilled slot. "
                f"Item {item.id} has no unfilled_slots."
            )

    def _prepare_inputs_and_masks(
        self, items: list[Item]
    ) -> tuple[dict[str, torch.Tensor], list[list[int]]]:
        """Prepare tokenized inputs with masked positions.

        Extracts text from items, tokenizes, and replaces tokens at unfilled_slots
        positions with [MASK] token.

        Parameters
        ----------
        items : list[Item]
            Items to prepare.

        Returns
        -------
        tuple[dict[str, torch.Tensor], list[list[int]]]
            - Tokenized inputs (input_ids, attention_mask)
            - List of masked token positions per item (token-level indices)
        """
        texts = []
        n_slots_per_item = []

        for item in items:
            # Get rendered text
            text = item.rendered_elements.get("text", "")
            texts.append(text)
            n_slots_per_item.append(len(item.unfilled_slots))

        # Tokenize all texts
        tokenized = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
        ).to(self.config.device)

        mask_token_id = self.tokenizer.mask_token_id

        # Find and replace "___" placeholders with [MASK]
        # Track ONE position per unfilled slot (even if "___" spans multiple tokens)
        token_masked_positions = []
        for i, text in enumerate(texts):
            # Tokenize individually to find "___" positions
            tokens = self.tokenizer.tokenize(text)
            masked_indices = []

            # Track which tokens are part of "___" to avoid duplicates
            in_blank = False
            for j, token in enumerate(tokens):
                # Check if this token is part of a "___" placeholder
                if "_" in token and not in_blank:
                    # Start of a new blank - record this position
                    token_idx = j + 1  # Add 1 for [CLS] token
                    masked_indices.append(token_idx)
                    in_blank = True
                    # Replace with [MASK]
                    if token_idx < tokenized["input_ids"].shape[1]:
                        tokenized["input_ids"][i, token_idx] = mask_token_id
                elif "_" in token and in_blank:
                    # Continuation of current blank - also mask but don't record
                    token_idx = j + 1
                    if token_idx < tokenized["input_ids"].shape[1]:
                        tokenized["input_ids"][i, token_idx] = mask_token_id
                else:
                    # Not a blank token - reset in_blank
                    in_blank = False

            # Verify we found the expected number of masked positions
            expected_slots = n_slots_per_item[i]
            if len(masked_indices) != expected_slots:
                raise ValueError(
                    f"Mismatch between masked positions and unfilled_slots "
                    f"for item {i}: found {len(masked_indices)} '___' "
                    f"placeholders in text but item has {expected_slots} "
                    f"unfilled_slots. Ensure rendered text uses exactly one "
                    f"'___' per unfilled_slot. Text: '{text}'"
                )

            token_masked_positions.append(masked_indices)

        return tokenized, token_masked_positions

    def train(
        self,
        items: list[Item],
        labels: list[list[str]],
        participant_ids: list[str] | None = None,
        validation_items: list[Item] | None = None,
        validation_labels: list[list[str]] | None = None,
    ) -> dict[str, float]:
        """Train model on cloze data with participant-level random effects.

        Parameters
        ----------
        items : list[Item]
            Training items with unfilled_slots.
        labels : list[list[str]]
            Training labels as list of lists. Each inner list contains one
            token per unfilled slot in that item.
            Example: [["ran"], ["The", "dog"], ["jumped"]].
        participant_ids : list[str] | None
            Participant identifier for each item.
            - For fixed effects (mode='fixed'): Pass None.
            - For mixed effects: Must provide list[str] with same length as items.
        validation_items : list[Item] | None
            Optional validation items.
        validation_labels : list[list[str]] | None
            Optional validation labels.

        Returns
        -------
        dict[str, float]
            Training metrics including:
            - "train_loss": Final training negative log-likelihood
            - "train_accuracy": Token-level accuracy on training set
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
            If labels[i] length doesn't match items[i].unfilled_slots length.
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

        # Validate labels format: each label must be a list
        # matching unfilled_slots length
        for i, (item, label) in enumerate(zip(items, labels, strict=True)):
            if len(label) != len(item.unfilled_slots):
                raise ValueError(
                    f"Label length mismatch for item {i}: "
                    f"expected {len(item.unfilled_slots)} tokens "
                    f"(matching unfilled_slots), got {len(label)} tokens. "
                    f"Ensure each label is a list with one token per unfilled slot."
                )

        if (validation_items is None) != (validation_labels is None):
            raise ValueError(
                "Both validation_items and validation_labels must be "
                "provided, or neither"
            )

        # Initialize random effects manager
        vocab_size = self.tokenizer.vocab_size
        self.random_effects = RandomEffectsManager(
            self.config.mixed_effects,
            vocab_size=vocab_size,  # For random intercepts (bias on logits)
        )

        # Register participants for adaptive regularization
        participant_counts = Counter(participant_ids)
        for pid, count in participant_counts.items():
            self.random_effects.register_participant(pid, count)

        # Build optimizer parameters based on mode
        params_to_optimize = list(self.model.parameters())

        # Add random effects parameters
        if self.config.mixed_effects.mode == "random_intercepts":
            for param_dict in self.random_effects.intercepts.values():
                params_to_optimize.extend(param_dict.values())
        elif self.config.mixed_effects.mode == "random_slopes":
            for head in self.random_effects.slopes.values():
                params_to_optimize.extend(head.parameters())

        optimizer = torch.optim.AdamW(params_to_optimize, lr=self.config.learning_rate)

        self.model.train()

        for _epoch in range(self.config.num_epochs):
            n_batches = (
                len(items) + self.config.batch_size - 1
            ) // self.config.batch_size
            epoch_loss = 0.0

            for i in range(n_batches):
                start_idx = i * self.config.batch_size
                end_idx = min(start_idx + self.config.batch_size, len(items))

                batch_items = items[start_idx:end_idx]
                batch_labels = labels[start_idx:end_idx]
                batch_participant_ids = participant_ids[start_idx:end_idx]

                # Prepare inputs with masking
                tokenized, masked_positions = self._prepare_inputs_and_masks(
                    batch_items
                )

                # Tokenize labels to get target token IDs
                target_token_ids = []
                for label_list in batch_labels:
                    # Tokenize each token in the label
                    token_ids = []
                    for token in label_list:
                        # Get token ID for this token
                        tid = self.tokenizer.encode(token, add_special_tokens=False)[0]
                        token_ids.append(tid)
                    target_token_ids.append(token_ids)

                # Forward pass depends on mixed effects mode
                if self.config.mixed_effects.mode == "fixed":
                    # Standard MLM training
                    outputs = self.model(**tokenized)
                    logits = outputs.logits  # (batch, seq_len, vocab_size)

                elif self.config.mixed_effects.mode == "random_intercepts":
                    # Get encoder outputs
                    encoder_outputs = self.encoder(**tokenized)

                    # Get MLM logits
                    logits = self.mlm_head(encoder_outputs.last_hidden_state)

                    # Add participant-specific bias to masked position logits
                    for j, pid in enumerate(batch_participant_ids):
                        bias = self.random_effects.get_intercepts(
                            pid,
                            n_classes=vocab_size,
                            param_name="mu",
                            create_if_missing=True,
                        )
                        # bias shape: (vocab_size,)
                        # Add to all masked positions
                        for pos in masked_positions[j]:
                            if pos < logits.shape[1]:
                                logits[j, pos] = logits[j, pos] + bias

                elif self.config.mixed_effects.mode == "random_slopes":
                    # Use participant-specific MLM heads
                    # Need to process each participant separately
                    all_logits = []
                    for j, pid in enumerate(batch_participant_ids):
                        # Get participant-specific MLM head
                        participant_head = self.random_effects.get_slopes(
                            pid,
                            fixed_head=copy.deepcopy(self.mlm_head),
                            create_if_missing=True,
                        )

                        # Get encoder outputs for this item
                        item_inputs = {k: v[j : j + 1] for k, v in tokenized.items()}
                        encoder_outputs_j = self.encoder(**item_inputs)

                        # Run participant-specific MLM head
                        logits_j = participant_head(encoder_outputs_j.last_hidden_state)
                        all_logits.append(logits_j)

                    logits = torch.cat(all_logits, dim=0)

                # Compute loss only on masked positions
                losses = []
                for j, (masked_pos, target_ids) in enumerate(
                    zip(masked_positions, target_token_ids, strict=True)
                ):
                    for pos, target_id in zip(masked_pos, target_ids, strict=True):
                        if pos < logits.shape[1]:
                            # Cross-entropy loss for this position
                            loss_j = torch.nn.functional.cross_entropy(
                                logits[j, pos : pos + 1],
                                torch.tensor([target_id], device=self.config.device),
                            )
                            losses.append(loss_j)

                if losses:
                    loss_nll = torch.stack(losses).mean()
                else:
                    loss_nll = torch.tensor(0.0, device=self.config.device)

                # Add prior regularization
                loss_prior = self.random_effects.compute_prior_loss()
                loss = loss_nll + loss_prior

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            epoch_loss = epoch_loss / n_batches

        self._is_fitted = True

        metrics: dict[str, float] = {
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

        # Compute training accuracy
        train_predictions = self.predict(items, participant_ids)
        correct = 0
        total = 0
        for pred, label in zip(train_predictions, labels, strict=True):
            # pred.predicted_class is comma-separated tokens
            pred_tokens = pred.predicted_class.split(", ")
            for pt, lt in zip(pred_tokens, label, strict=True):
                if pt.lower() == lt.lower():
                    correct += 1
                total += 1
        if total > 0:
            metrics["train_accuracy"] = correct / total

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

            val_correct = 0
            val_total = 0
            for pred, label in zip(val_predictions, validation_labels, strict=True):
                pred_tokens = pred.predicted_class.split(", ")
                for pt, lt in zip(pred_tokens, label, strict=True):
                    if pt.lower() == lt.lower():
                        val_correct += 1
                    val_total += 1
            if val_total > 0:
                metrics["val_accuracy"] = val_correct / val_total

        return metrics

    def predict(
        self, items: list[Item], participant_ids: list[str] | None = None
    ) -> list[ModelPrediction]:
        """Predict tokens for masked positions with participant-specific random effects.

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
            Predictions with predicted_class as comma-separated tokens.

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

        self.model.eval()

        # Prepare inputs with masking
        tokenized, masked_positions = self._prepare_inputs_and_masks(items)

        with torch.no_grad():
            if self.config.mixed_effects.mode == "fixed":
                # Standard MLM prediction
                outputs = self.model(**tokenized)
                logits = outputs.logits

            elif self.config.mixed_effects.mode == "random_intercepts":
                # Get encoder outputs
                encoder_outputs = self.encoder(**tokenized)
                logits = self.mlm_head(encoder_outputs.last_hidden_state)

                # Add participant-specific bias
                vocab_size = self.tokenizer.vocab_size
                for j, pid in enumerate(participant_ids):
                    bias = self.random_effects.get_intercepts(
                        pid,
                        n_classes=vocab_size,
                        param_name="mu",
                        create_if_missing=False,
                    )
                    # Add to all masked positions
                    for pos in masked_positions[j]:
                        if pos < logits.shape[1]:
                            logits[j, pos] = logits[j, pos] + bias

            elif self.config.mixed_effects.mode == "random_slopes":
                # Use participant-specific MLM heads
                all_logits = []
                for j, pid in enumerate(participant_ids):
                    # Get participant-specific MLM head
                    participant_head = self.random_effects.get_slopes(
                        pid,
                        fixed_head=copy.deepcopy(self.mlm_head),
                        create_if_missing=False,
                    )

                    # Get encoder outputs
                    item_inputs = {k: v[j : j + 1] for k, v in tokenized.items()}
                    encoder_outputs_j = self.encoder(**item_inputs)

                    # Run participant-specific MLM head
                    logits_j = participant_head(encoder_outputs_j.last_hidden_state)
                    all_logits.append(logits_j)

                logits = torch.cat(all_logits, dim=0)

            # Get argmax at masked positions
            predictions = []
            for i, masked_pos in enumerate(masked_positions):
                predicted_tokens = []
                for pos in masked_pos:
                    if pos < logits.shape[1]:
                        # Get token ID with highest probability
                        token_id = torch.argmax(logits[i, pos]).item()
                        # Decode token
                        token = self.tokenizer.decode([token_id])
                        predicted_tokens.append(token.strip())

                # Join with comma for multi-slot items
                predicted_class = ", ".join(predicted_tokens)

                predictions.append(
                    ModelPrediction(
                        item_id=str(items[i].id),
                        probabilities={},  # Not applicable for generation
                        predicted_class=predicted_class,
                        confidence=1.0,  # Not applicable for generation
                    )
                )

        return predictions

    def predict_proba(
        self, items: list[Item], participant_ids: list[str] | None = None
    ) -> np.ndarray:
        """Predict probabilities at masked positions.

        For cloze tasks, returns empty array as probabilities are not typically
        used for evaluation.

        Parameters
        ----------
        items : list[Item]
            Items to predict.
        participant_ids : list[str] | None
            Participant identifiers.

        Returns
        -------
        np.ndarray
            Empty array of shape (n_items, 0).
        """
        return np.zeros((len(items), 0))

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

        # Save model
        self.model.save_pretrained(save_path / "model")
        self.tokenizer.save_pretrained(save_path / "model")

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

        self.config = ClozeModelConfig(**config_dict)

        # Load model
        self.model = AutoModelForMaskedLM.from_pretrained(load_path / "model")
        self.tokenizer = AutoTokenizer.from_pretrained(load_path / "model")

        # Re-extract components
        if hasattr(self.model, "bert"):
            self.encoder = self.model.bert
            self.mlm_head = self.model.cls
        elif hasattr(self.model, "roberta"):
            self.encoder = self.model.roberta
            self.mlm_head = self.model.lm_head
        else:
            self.encoder = self.model.base_model
            self.mlm_head = self.model.lm_head

        # Initialize and load random effects
        vocab_size = self.tokenizer.vocab_size
        self.random_effects = RandomEffectsManager(
            self.config.mixed_effects,
            vocab_size=vocab_size,
        )
        random_effects_path = load_path / "random_effects"
        if random_effects_path.exists():
            # For random_slopes, need to provide a template head
            if self.config.mixed_effects.mode == "random_slopes":
                template_head = copy.deepcopy(self.mlm_head)
                self.random_effects.load(random_effects_path, fixed_head=template_head)
            else:
                self.random_effects.load(random_effects_path)

            if self.random_effects.variance_history:
                self.variance_history = self.random_effects.variance_history.copy()

        self.model.to(self.config.device)
        self._is_fitted = True
