"""Model for forced choice tasks (2AFC, 3AFC, 4AFC, nAFC)."""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    AutoModel,
    AutoTokenizer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)

from bead.active_learning.config import MixedEffectsConfig, VarianceComponents
from bead.active_learning.models.base import ActiveLearningModel, ModelPrediction
from bead.active_learning.models.random_effects import RandomEffectsManager
from bead.active_learning.trainers.data_collator import MixedEffectsDataCollator
from bead.active_learning.trainers.dataset_utils import items_to_dataset
from bead.active_learning.trainers.model_wrapper import EncoderClassifierWrapper
from bead.config.active_learning import ForcedChoiceModelConfig
from bead.items.item import Item
from bead.items.item_template import ItemTemplate, TaskType

__all__ = ["ForcedChoiceModel"]


class _DevEarlyStopping(TrainerCallback):
    """Early-stop forced-choice training on a held-out dev set.

    The shared mixed-effects trainer's evaluation emits no usable loss or
    accuracy for forced choice, so the dev set is scored through the model's own
    predict path at the end of each epoch. Training stops after ``patience``
    consecutive epochs without dev cross-entropy improvement, and the best
    encoder and classifier head seen so far are restored.
    """

    def __init__(
        self,
        *,
        predict_proba: Callable[[list[Item], list[str]], np.ndarray],
        encoder: nn.Module,
        classifier_head: nn.Module,
        dev_items: list[Item],
        dev_participant_ids: list[str],
        dev_y: list[int],
        patience: int,
    ) -> None:
        self._predict_proba = predict_proba
        self._encoder = encoder
        self._classifier_head = classifier_head
        self._dev_items = dev_items
        self._dev_participant_ids = dev_participant_ids
        self._dev_y = np.asarray(dev_y)
        self._patience = patience
        self._best_loss = float("inf")
        self._best_state: (
            tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]] | None
        ) = None
        self._wait = 0

    def _dev_cross_entropy(self) -> float:
        proba = self._predict_proba(self._dev_items, self._dev_participant_ids)
        chosen = np.clip(proba[np.arange(len(self._dev_y)), self._dev_y], 1e-9, 1.0)
        return float(-np.mean(np.log(chosen)))

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        loss = self._dev_cross_entropy()
        if loss < self._best_loss - 1e-4:
            self._best_loss = loss
            self._best_state = (
                {
                    k: v.detach().cpu().clone()
                    for k, v in self._encoder.state_dict().items()
                },
                {
                    k: v.detach().cpu().clone()
                    for k, v in self._classifier_head.state_dict().items()
                },
            )
            self._wait = 0
        else:
            self._wait += 1
            if self._wait >= self._patience:
                control.should_training_stop = True

    def on_train_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: object,
    ) -> None:
        if self._best_state is not None:
            encoder_state, head_state = self._best_state
            self._encoder.load_state_dict(encoder_state)
            self._classifier_head.load_state_dict(head_state)


class ForcedChoiceModel(ActiveLearningModel):
    """Model for forced_choice tasks with n alternatives.

    Supports 2AFC, 3AFC, 4AFC, and general nAFC tasks using any
    HuggingFace transformer model. Provides two encoding strategies:
    single encoder (concatenate options) or dual encoder (separate embeddings).
    With ``scoring_mode="independent"``, the dual encoder retains the option
    dimension and one shared head assigns a scalar to each option. Softmax over
    these scores gives choice probabilities. All rendered elements are options
    in this mode; their names must match across training and prediction items.

    Parameters
    ----------
    config : ForcedChoiceModelConfig
        Configuration object containing all model parameters.

    Attributes
    ----------
    config : ForcedChoiceModelConfig
        Model configuration.
    tokenizer : AutoTokenizer
        Transformer tokenizer.
    encoder : AutoModel
        Transformer encoder model.
    classifier_head : nn.Sequential
        Classification head (fixed effects head).
    num_classes : int | None
        Number of classes (inferred from training data).
    option_names : list[str] | None
        Option names (e.g., ["option_a", "option_b"]).
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
    >>> from bead.config.active_learning import ForcedChoiceModelConfig
    >>> items = [
    ...     Item(
    ...         item_template_id=uuid4(),
    ...         rendered_elements={"option_a": "sentence A", "option_b": "sentence B"}
    ...     )
    ...     for _ in range(10)
    ... ]
    >>> labels = ["option_a"] * 5 + ["option_b"] * 5
    >>> config = ForcedChoiceModelConfig(  # doctest: +SKIP
    ...     num_epochs=1, batch_size=2, device="cpu"
    ... )
    >>> model = ForcedChoiceModel(config=config)  # doctest: +SKIP
    >>> metrics = model.train(items, labels)  # doctest: +SKIP
    >>> predictions = model.predict(items[:3])  # doctest: +SKIP
    """

    def __init__(
        self,
        config: ForcedChoiceModelConfig | None = None,
    ) -> None:
        """Initialize forced choice model.

        Parameters
        ----------
        config : ForcedChoiceModelConfig | None
            Configuration object. If None, uses default configuration.
        """
        self.config = config or ForcedChoiceModelConfig()

        # Validate mixed_effects configuration
        super().__init__(self.config)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.encoder = AutoModel.from_pretrained(self.config.model_name)

        self.num_classes: int | None = None
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
            List containing "forced_choice".
        """
        return ["forced_choice"]

    def validate_item_compatibility(
        self, item: Item, item_template: ItemTemplate
    ) -> None:
        """Validate item is compatible with forced choice model.

        Parameters
        ----------
        item : Item
            Item to validate.
        item_template : ItemTemplate
            Template the item was constructed from.

        Raises
        ------
        ValueError
            If task_type is not "forced_choice".
        ValueError
            If task_spec.options is not defined.
        ValueError
            If item is missing required rendered_elements.
        """
        if item_template.task_type != "forced_choice":
            raise ValueError(
                f"Expected task_type 'forced_choice', got '{item_template.task_type}'"
            )

        if item_template.task_spec.options is None:
            raise ValueError(
                "task_spec.options must be defined for forced_choice tasks"
            )

        for option_name in item_template.task_spec.options:
            if option_name not in item.rendered_elements:
                raise ValueError(
                    f"Item missing required element '{option_name}' "
                    f"from rendered_elements"
                )

    def _initialize_classifier(self, num_classes: int) -> None:
        """Initialize classification head for given number of classes.

        Parameters
        ----------
        num_classes : int
            Number of output classes.
        """
        hidden_size = self.encoder.config.hidden_size

        if self.config.scoring_mode == "independent":
            input_size = hidden_size
        elif self.config.encoder_mode == "dual_encoder":
            input_size = hidden_size * num_classes
        else:
            input_size = hidden_size

        self.classifier_head = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(
                256, 1 if self.config.scoring_mode == "independent" else num_classes
            ),
        )
        self.classifier_head.to(self.config.device)

    def _encode_single(self, texts: list[str]) -> torch.Tensor:
        """Encode a batch of texts with the shared transformer.

        Return the first-token representation of each text. Callers provide
        either concatenated alternatives or independently encoded options.

        Parameters
        ----------
        texts : list[str]
            Texts to encode, one per tokenized sequence.

        Returns
        -------
        torch.Tensor
            Encoded representations of shape (len(texts), hidden_size).
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

        Encode options independently with shared transformer weights.
        Shared scalar scoring retains the option dimension; joint scoring
        concatenates the option embeddings.

        Parameters
        ----------
        options_per_item : list[list[str]]
            List of option lists. Each inner list contains option texts for one item.

        Returns
        -------
        torch.Tensor
            Encodings of shape (batch_size, num_options, hidden_size) for shared
            scalar scoring, or (batch_size, num_options * hidden_size) otherwise.
        """
        texts = [text for options in options_per_item for text in options]
        embeddings = self._encode_single(texts)
        if self.config.scoring_mode == "independent":
            return embeddings.reshape(len(options_per_item), -1, embeddings.shape[-1])
        return embeddings.reshape(len(options_per_item), -1)

    def _prepare_inputs(self, items: list[Item]) -> torch.Tensor:
        """Prepare inputs for encoding based on encoder mode.

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

        if self.config.scoring_mode == "independent":
            for item in items:
                if set(item.rendered_elements) != set(self.option_names):
                    raise ValueError(
                        "independent items must have exactly the fitted option "
                        f"names {self.option_names} in rendered_elements"
                    )

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

    def _validate_labels(self, labels: list[str]) -> None:
        """Validate that all labels are valid option names.

        Parameters
        ----------
        labels : list[str]
            Labels to validate.

        Raises
        ------
        ValueError
            If any label is not in option_names.
        """
        if self.option_names is None:
            raise ValueError("option_names not initialized")

        valid_labels = set(self.option_names)
        invalid = [label for label in labels if label not in valid_labels]
        if invalid:
            raise ValueError(
                f"Invalid labels found: {set(invalid)}. "
                f"Labels must be one of {valid_labels}."
            )

    def _prepare_training_data(
        self,
        items: list[Item],
        labels: list[str],
        participant_ids: list[str],
        validation_items: list[Item] | None,
        validation_labels: list[str] | None,
    ) -> tuple[list[Item], list[int], list[str], list[Item] | None, list[int] | None]:
        """Prepare training data for forced choice model.

        Parameters
        ----------
        items : list[Item]
            Training items.
        labels : list[str]
            Training labels (option names).
        participant_ids : list[str]
            Normalized participant IDs.
        validation_items : list[Item] | None
            Validation items.
        validation_labels : list[str] | None
            Validation labels.

        Returns
        -------
        tuple[list[Item], list[int], list[str], list[Item] | None, list[int] | None]
            Prepared items, numeric labels, participant_ids, validation_items,
            numeric validation_labels.
        """
        if self.config.scoring_mode == "independent":
            if not items:
                raise ValueError("independent scoring requires nonempty training items")
            # include options that were never selected in the training responses
            unique_labels = sorted(items[0].rendered_elements)
            if len(unique_labels) < 2:
                raise ValueError("independent scoring requires at least two options")
            for item in items + (validation_items or []):
                if sorted(item.rendered_elements) != unique_labels:
                    raise ValueError(
                        "independent items must have identical option names "
                        "in rendered_elements"
                    )
        else:
            unique_labels = sorted(set(labels))
        self.num_classes = len(unique_labels)
        self.option_names = unique_labels

        self._validate_labels(labels)
        self._initialize_classifier(self.num_classes)

        label_to_idx = {label: idx for idx, label in enumerate(self.option_names)}
        y_numeric = [label_to_idx[label] for label in labels]

        # Convert validation labels if provided
        val_y_numeric = None
        if validation_items is not None and validation_labels is not None:
            self._validate_labels(validation_labels)
            if len(validation_items) != len(validation_labels):
                raise ValueError(
                    f"Number of validation items ({len(validation_items)}) "
                    f"must match number of validation labels ({len(validation_labels)})"
                )
            val_y_numeric = [label_to_idx[label] for label in validation_labels]

        return items, y_numeric, participant_ids, validation_items, val_y_numeric

    def _initialize_random_effects(self, n_classes: int) -> None:
        """Initialize random effects manager.

        Parameters
        ----------
        n_classes : int
            Number of classes.
        """
        self.random_effects = RandomEffectsManager(
            self.config.mixed_effects, n_classes=n_classes
        )

    def _do_training(
        self,
        items: list[Item],
        labels_numeric: list[int],
        participant_ids: list[str],
        validation_items: list[Item] | None,
        validation_labels_numeric: list[int] | None,
    ) -> dict[str, float]:
        """Perform forced choice model training.

        Parameters
        ----------
        items : list[Item]
            Training items.
        labels_numeric : list[int]
            Numeric labels (class indices).
        participant_ids : list[str]
            Participant IDs.
        validation_items : list[Item] | None
            Validation items.
        validation_labels_numeric : list[int] | None
            Numeric validation labels.

        Returns
        -------
        dict[str, float]
            Training metrics.
        """
        # Convert validation_labels_numeric back to string labels for validation metrics
        validation_labels = None
        if validation_items is not None and validation_labels_numeric is not None:
            validation_labels = [
                self.option_names[label_idx] for label_idx in validation_labels_numeric
            ]

        # the generic Trainer wrapper only handles one embedding per item
        if self.config.encoder_mode == "single_encoder" and (
            self.config.mixed_effects.mode in ("fixed", "random_intercepts")
        ):
            metrics = self._train_with_huggingface_trainer(
                items=items,
                y_numeric=labels_numeric,
                participant_ids=participant_ids,
                validation_items=validation_items,
                validation_labels=validation_labels,
            )
        else:
            # dual encoders and participant heads use the prediction encoding path
            metrics = self._train_with_custom_loop(
                items=items,
                y_numeric=labels_numeric,
                participant_ids=participant_ids,
                validation_items=validation_items,
                validation_labels=validation_labels,
            )

        # Add validation accuracy if validation data provided and not already computed
        if (
            validation_items is not None
            and validation_labels is not None
            and "val_accuracy" not in metrics
        ):
            # Validation with placeholder participant_ids for mixed effects
            if self.config.mixed_effects.mode == "fixed":
                val_participant_ids = ["_fixed_"] * len(validation_items)
            else:
                val_participant_ids = ["_validation_"] * len(validation_items)
            val_predictions = self._do_predict(validation_items, val_participant_ids)
            val_pred_labels = [p.predicted_class for p in val_predictions]
            val_acc = sum(
                pred == true
                for pred, true in zip(val_pred_labels, validation_labels, strict=True)
            ) / len(validation_labels)
            metrics["val_accuracy"] = val_acc

        return metrics

    def _train_with_huggingface_trainer(
        self,
        items: list[Item],
        y_numeric: list[int],
        participant_ids: list[str],
        validation_items: list[Item] | None,
        validation_labels: list[str] | None,
    ) -> dict[str, float]:
        """Train using HuggingFace Trainer with mixed effects support.

        Parameters
        ----------
        items : list[Item]
            Training items.
        y_numeric : list[int]
            Numeric labels (class indices).
        participant_ids : list[str]
            Participant IDs.
        validation_items : list[Item] | None
            Validation items.
        validation_labels : list[str] | None
            Validation labels.

        Returns
        -------
        dict[str, float]
            Training metrics.
        """
        # Convert items to HuggingFace Dataset
        train_dataset = items_to_dataset(
            items=items,
            labels=y_numeric,
            participant_ids=participant_ids,
            tokenizer=self.tokenizer,
            max_length=self.config.max_length,
        )

        # Prepare a held-out dev set for early stopping, if provided
        has_dev = validation_items is not None and validation_labels is not None
        val_y_numeric: list[int] = []
        val_participant_ids: list[str] = []
        if has_dev:
            assert validation_items is not None and validation_labels is not None
            label_to_idx = {label: idx for idx, label in enumerate(self.option_names)}
            val_y_numeric = [label_to_idx[label] for label in validation_labels]
            val_participant_ids = (
                ["_validation_"] * len(validation_items)
                if self.config.mixed_effects.mode != "fixed"
                else ["_fixed_"] * len(validation_items)
            )

        # Create wrapper model for Trainer
        wrapped_model = EncoderClassifierWrapper(
            encoder=self.encoder, classifier_head=self.classifier_head
        )

        # Create data collator
        data_collator = MixedEffectsDataCollator(tokenizer=self.tokenizer)

        # Create training arguments with checkpointing
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir) / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            training_args = TrainingArguments(
                output_dir=str(checkpoint_dir),
                num_train_epochs=self.config.num_epochs,
                per_device_train_batch_size=self.config.batch_size,
                per_device_eval_batch_size=self.config.batch_size,
                learning_rate=self.config.learning_rate,
                logging_steps=10,
                eval_strategy="no",
                save_strategy="no",
                report_to="none",
                remove_unused_columns=False,
                use_cpu=self.config.device == "cpu",
            )

            # Import here to avoid circular import
            from bead.active_learning.trainers.mixed_effects import (  # noqa: PLC0415
                MixedEffectsTrainer,
            )

            # Create trainer
            trainer = MixedEffectsTrainer(
                model=wrapped_model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator,
                tokenizer=self.tokenizer,
                random_effects_manager=self.random_effects,
            )
            # Early-stop on the dev set, scored through the model's own predict
            # path (the shared trainer's evaluate emits no usable loss or
            # accuracy for forced choice); the best encoder and head are restored.
            if has_dev:
                assert validation_items is not None
                trainer.add_callback(
                    _DevEarlyStopping(
                        predict_proba=self._do_predict_proba,
                        encoder=self.encoder,
                        classifier_head=self.classifier_head,
                        dev_items=validation_items,
                        dev_participant_ids=val_participant_ids,
                        dev_y=val_y_numeric,
                        patience=self.config.early_stopping_patience,
                    )
                )

            train_result = trainer.train()

        # Score train (and dev) metrics through the model's own predict path,
        # which applies the fitted random effects. trainer.evaluate cannot score
        # accuracy here (its shared prediction step only labels cloze batches).
        train_scores = self._classification_metrics(items, participant_ids, y_numeric)
        metrics: dict[str, float] = {
            "train_loss": float(train_result.training_loss),
            "train_accuracy": train_scores["accuracy"],
            "train_precision": train_scores["precision"],
            "train_recall": train_scores["recall"],
            "train_f1": train_scores["f1"],
        }
        if has_dev:
            assert validation_items is not None
            val_scores = self._classification_metrics(
                validation_items, val_participant_ids, val_y_numeric
            )
            metrics.update(
                {
                    "val_accuracy": val_scores["accuracy"],
                    "val_precision": val_scores["precision"],
                    "val_recall": val_scores["recall"],
                    "val_f1": val_scores["f1"],
                }
            )

        # Estimate variance components
        if self.config.mixed_effects.estimate_variance_components:
            var_comps = self.random_effects.estimate_variance_components()
            if var_comps:
                var_comp = var_comps.get("mu") or var_comps.get("slopes")
                if var_comp:
                    self.variance_history.append(var_comp)
                    metrics["participant_variance"] = var_comp.variance
                    metrics["n_participants"] = var_comp.n_groups

        return metrics

    def _classification_metrics(
        self,
        items: list[Item],
        participant_ids: list[str],
        y_numeric: list[int],
    ) -> dict[str, float]:
        """Accuracy, precision, recall, and F1 from the model's own predictions.

        Predictions come from the model's predict path (which applies the fitted
        random effects), so the reported metrics reflect the deployed model
        rather than the shared trainer's cloze-only evaluation.

        Parameters
        ----------
        items : list[Item]
            Items to score.
        participant_ids : list[str]
            Participant id per item.
        y_numeric : list[int]
            True class indices.

        Returns
        -------
        dict[str, float]
            Keys ``accuracy``, ``precision``, ``recall``, ``f1``.
        """
        proba = self._do_predict_proba(items, participant_ids)
        predictions = np.argmax(proba, axis=1)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_numeric, predictions, average="macro", zero_division=0
        )
        return {
            "accuracy": float(accuracy_score(y_numeric, predictions)),
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        }

    def _head_logits(self, head: nn.Module, embeddings: torch.Tensor) -> torch.Tensor:
        """Apply a head, removing only the scalar output dimension when present."""
        logits = head(embeddings)
        if self.config.scoring_mode == "independent":
            return logits.squeeze(-1)
        return logits

    def _compute_logits(
        self,
        embeddings: torch.Tensor,
        participant_ids: list[str],
        *,
        training: bool = False,
    ) -> torch.Tensor:
        """Compute option logits with the configured participant effects."""
        assert self.classifier_head is not None
        if self.config.mixed_effects.mode == "fixed":
            return self._head_logits(self.classifier_head, embeddings)

        assert self.random_effects is not None
        assert self.num_classes is not None
        if self.config.mixed_effects.mode == "random_intercepts":
            logits = self._head_logits(self.classifier_head, embeddings)
            biases = torch.stack(
                [
                    self.random_effects.get_intercepts(
                        pid,
                        n_classes=self.num_classes,
                        param_name="mu",
                        create_if_missing=training,
                    ).to(embeddings.device)
                    for pid in participant_ids
                ]
            )
            return logits + biases

        logits_list: list[torch.Tensor] = []
        for i, pid in enumerate(participant_ids):
            head = self.random_effects.get_slopes(
                pid, fixed_head=self.classifier_head, create_if_missing=training
            ).to(embeddings.device)
            head.train(training)
            logits_list.append(self._head_logits(head, embeddings[i : i + 1]))
        return torch.cat(logits_list, dim=0)

    def _train_with_custom_loop(
        self,
        items: list[Item],
        y_numeric: list[int],
        participant_ids: list[str],
        validation_items: list[Item] | None,
        validation_labels: list[str] | None,
    ) -> dict[str, float]:
        """Train dual encoders or participant heads through the prediction encoder.

        Parameters
        ----------
        items : list[Item]
            Training items.
        y_numeric : list[int]
            Numeric labels (class indices).
        participant_ids : list[str]
            Participant IDs.
        validation_items : list[Item] | None
            Held-out items for early stopping.
        validation_labels : list[str] | None
            Held-out option names.

        Returns
        -------
        dict[str, float]
            Training metrics and, when provided, validation metrics.
        """
        assert self.classifier_head is not None
        assert self.random_effects is not None
        assert self.num_classes is not None
        assert self.option_names is not None
        y = torch.tensor(y_numeric, dtype=torch.long, device=self.config.device)

        # create participant parameters before collecting optimizer parameters
        params_to_optimize = list(self.encoder.parameters()) + list(
            self.classifier_head.parameters()
        )
        for pid in dict.fromkeys(participant_ids):
            if self.config.mixed_effects.mode == "random_intercepts":
                bias = self.random_effects.get_intercepts(
                    pid, n_classes=self.num_classes, param_name="mu"
                )
                bias = bias.to(self.config.device).detach().requires_grad_()
                self.random_effects.intercepts["mu"][pid] = bias
                params_to_optimize.append(bias)
            elif self.config.mixed_effects.mode == "random_slopes":
                head = self.random_effects.get_slopes(
                    pid, fixed_head=self.classifier_head
                ).to(self.config.device)
                params_to_optimize.extend(head.parameters())

        optimizer = torch.optim.AdamW(params_to_optimize, lr=self.config.learning_rate)
        criterion = nn.CrossEntropyLoss()
        val_ids = ["_validation_"] * len(validation_items or [])
        val_y = [self.option_names.index(label) for label in validation_labels or []]
        best_loss = float("inf")
        best_encoder: dict[str, torch.Tensor] | None = None
        best_head: dict[str, torch.Tensor] | None = None
        best_effects: RandomEffectsManager | None = None
        wait = 0
        epoch_loss = 0.0

        for _epoch in range(self.config.num_epochs):
            # validation switches modules to eval mode at the end of each epoch
            self.encoder.train()
            self.classifier_head.train()
            epoch_loss = 0.0
            for start_idx in range(0, len(items), self.config.batch_size):
                end_idx = min(start_idx + self.config.batch_size, len(items))
                embeddings = self._prepare_inputs(items[start_idx:end_idx])
                logits = self._compute_logits(
                    embeddings, participant_ids[start_idx:end_idx], training=True
                )
                loss_ce = criterion(logits, y[start_idx:end_idx])
                loss_prior = self.random_effects.compute_prior_loss().to(logits.device)
                loss = loss_ce + loss_prior
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * (end_idx - start_idx)
            epoch_loss /= len(items)

            if validation_items:
                proba = self._do_predict_proba(validation_items, val_ids)
                chosen = np.clip(proba[np.arange(len(val_y)), val_y], 1e-9, 1.0)
                val_loss = float(-np.mean(np.log(chosen)))
                if val_loss < best_loss - 1e-4:
                    best_loss = val_loss
                    best_encoder = deepcopy(self.encoder.state_dict())
                    best_head = deepcopy(self.classifier_head.state_dict())
                    best_effects = deepcopy(self.random_effects)
                    wait = 0
                else:
                    wait += 1
                    if wait >= self.config.early_stopping_patience:
                        break

        if best_encoder is not None and best_head is not None:
            self.encoder.load_state_dict(best_encoder)
            self.classifier_head.load_state_dict(best_head)
            assert best_effects is not None
            self.random_effects = best_effects

        train_scores = self._classification_metrics(items, participant_ids, y_numeric)
        metrics = {f"train_{key}": value for key, value in train_scores.items()}
        metrics["train_loss"] = epoch_loss
        if validation_items:
            val_scores = self._classification_metrics(validation_items, val_ids, val_y)
            metrics.update({f"val_{key}": value for key, value in val_scores.items()})

        if self.config.mixed_effects.estimate_variance_components:
            var_comps = self.random_effects.estimate_variance_components()
            if var_comps:
                var_comp = var_comps.get("mu") or var_comps.get("slopes")
                if var_comp:
                    self.variance_history.append(var_comp)
                    metrics["participant_variance"] = var_comp.variance
                    metrics["n_participants"] = var_comp.n_groups
        return metrics

    def _do_predict(
        self, items: list[Item], participant_ids: list[str]
    ) -> list[ModelPrediction]:
        """Perform forced choice model prediction.

        Parameters
        ----------
        items : list[Item]
            Items to predict.
        participant_ids : list[str]
            Normalized participant IDs.

        Returns
        -------
        list[ModelPrediction]
            Predictions.
        """
        proba = self._do_predict_proba(items, participant_ids)
        pred_classes = np.argmax(proba, axis=1)

        predictions = []
        for i, item in enumerate(items):
            pred_label = self.option_names[pred_classes[i]]
            prob_dict = {
                opt: float(proba[i, idx]) for idx, opt in enumerate(self.option_names)
            }
            predictions.append(
                ModelPrediction(
                    item_id=str(item.id),
                    probabilities=prob_dict,
                    predicted_class=pred_label,
                    confidence=float(proba[i, pred_classes[i]]),
                )
            )

        return predictions

    def _do_predict_proba(
        self, items: list[Item], participant_ids: list[str]
    ) -> np.ndarray:
        """Perform forced choice model probability prediction.

        Parameters
        ----------
        items : list[Item]
            Items to predict.
        participant_ids : list[str]
            Normalized participant IDs.

        Returns
        -------
        np.ndarray
            Probability array of shape (n_items, n_classes).
        """
        assert self.classifier_head is not None
        assert self.num_classes is not None
        self.encoder.eval()
        self.classifier_head.eval()
        batches: list[np.ndarray] = []
        with torch.no_grad():
            for start in range(0, len(items), self.config.batch_size):
                end = start + self.config.batch_size
                embeddings = self._prepare_inputs(items[start:end])
                logits = self._compute_logits(embeddings, participant_ids[start:end])
                batches.append(torch.softmax(logits, dim=1).cpu().numpy())
        if not batches:
            return np.empty((0, self.num_classes))
        return np.concatenate(batches, axis=0)

    def _save_model_components(self, save_path: Path) -> None:
        """Save model-specific components.

        Parameters
        ----------
        save_path : Path
            Directory to save to.
        """
        self.encoder.save_pretrained(save_path / "encoder")
        self.tokenizer.save_pretrained(save_path / "encoder")

        torch.save(
            self.classifier_head.state_dict(),
            save_path / "classifier_head.pt",
        )

    def _get_save_state(self) -> dict[str, object]:
        """Get model-specific state to save.

        Returns
        -------
        dict[str, object]
            State dictionary.
        """
        return {
            "num_classes": self.num_classes,
            "option_names": self.option_names,
        }

    def _restore_training_state(self, config_dict: dict[str, object]) -> None:
        """Restore model-specific training state.

        Parameters
        ----------
        config_dict : dict[str, object]
            Configuration dictionary with training state.
        """
        self.num_classes = config_dict.pop("num_classes")
        self.option_names = config_dict.pop("option_names")

    def _load_model_components(
        self, load_path: Path, config_dict: dict[str, object]
    ) -> None:
        """Load model-specific components.

        Parameters
        ----------
        load_path : Path
            Directory to load from.
        config_dict : dict[str, object]
            Schema-only config dict.
        """
        # Reconstruct MixedEffectsConfig if needed
        if "mixed_effects" in config_dict and isinstance(
            config_dict["mixed_effects"], dict
        ):
            config_dict["mixed_effects"] = MixedEffectsConfig(
                **config_dict["mixed_effects"]
            )

        self.config = ForcedChoiceModelConfig(**config_dict)

        self.encoder = AutoModel.from_pretrained(load_path / "encoder")
        self.tokenizer = AutoTokenizer.from_pretrained(load_path / "encoder")

        self._initialize_classifier(self.num_classes)
        self.classifier_head.load_state_dict(
            torch.load(
                load_path / "classifier_head.pt", map_location=self.config.device
            )
        )

        self.encoder.to(self.config.device)
        self.classifier_head.to(self.config.device)

    def _get_n_classes_for_random_effects(self) -> int:
        """Get the number of classes for initializing RandomEffectsManager.

        Returns
        -------
        int
            Number of classes.
        """
        return self.num_classes

    def _get_random_effects_fixed_head(self) -> torch.nn.Module | None:
        """Get the fixed head for random effects.

        Returns
        -------
        torch.nn.Module | None
            The classifier head, or None if not applicable.
        """
        return self.classifier_head
