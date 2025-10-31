"""Model for forced choice tasks (2AFC, 3AFC, 4AFC, nAFC)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from bead.active_learning.models.base import ActiveLearningModel, ModelPrediction
from bead.config.models import ForcedChoiceModelConfig
from bead.items.models import Item, ItemTemplate, TaskType

__all__ = ["ForcedChoiceModel"]


class ForcedChoiceModel(ActiveLearningModel):
    """Model for forced_choice tasks with n alternatives.

    Supports 2AFC, 3AFC, 4AFC, and general nAFC tasks using any
    HuggingFace transformer model. Provides two encoding strategies:
    single encoder (concatenate options) or dual encoder (separate embeddings).

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
        Classification head.
    num_classes : int | None
        Number of classes (inferred from training data).
    option_names : list[str] | None
        Option names (e.g., ["option_a", "option_b"]).
    _is_fitted : bool
        Whether model has been trained.

    Examples
    --------
    >>> from uuid import uuid4
    >>> from bead.items.models import Item
    >>> from bead.config.models import ForcedChoiceModelConfig
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

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.encoder = AutoModel.from_pretrained(self.config.model_name)

        self.num_classes: int | None = None
        self.option_names: list[str] | None = None
        self.classifier_head: nn.Sequential | None = None
        self._is_fitted = False

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

        if self.config.encoder_mode == "dual_encoder":
            input_size = hidden_size * num_classes
        else:
            input_size = hidden_size

        self.classifier_head = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_classes),
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

    def train(
        self,
        items: list[Item],
        labels: list[str],
        validation_items: list[Item] | None = None,
        validation_labels: list[str] | None = None,
    ) -> dict[str, float]:
        """Train model on forced choice data.

        Parameters
        ----------
        items : list[Item]
            Training items.
        labels : list[str]
            Training labels (option names like "option_a", "option_b").
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

        Raises
        ------
        ValueError
            If items and labels have different lengths.
        ValueError
            If labels contain invalid values.
        ValueError
            If validation data is incomplete.
        """
        if len(items) != len(labels):
            raise ValueError(
                f"Number of items ({len(items)}) must match "
                f"number of labels ({len(labels)})"
            )

        if (validation_items is None) != (validation_labels is None):
            raise ValueError(
                "Both validation_items and validation_labels must be "
                "provided, or neither"
            )

        unique_labels = sorted(set(labels))
        self.num_classes = len(unique_labels)
        self.option_names = unique_labels

        self._validate_labels(labels)
        self._initialize_classifier(self.num_classes)

        label_to_idx = {label: idx for idx, label in enumerate(self.option_names)}
        y = torch.tensor(
            [label_to_idx[label] for label in labels],
            dtype=torch.long,
            device=self.config.device,
        )

        optimizer = torch.optim.AdamW(
            list(self.encoder.parameters()) + list(self.classifier_head.parameters()),
            lr=self.config.learning_rate,
        )
        criterion = nn.CrossEntropyLoss()

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

                embeddings = self._prepare_inputs(batch_items)
                logits = self.classifier_head(embeddings)

                loss = criterion(logits, batch_labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                predictions = torch.argmax(logits, dim=1)
                epoch_correct += (predictions == batch_labels).sum().item()

            epoch_acc = epoch_correct / len(items)
            epoch_loss = epoch_loss / n_batches

        self._is_fitted = True

        metrics: dict[str, float] = {
            "train_accuracy": epoch_acc,
            "train_loss": epoch_loss,
        }

        if validation_items is not None and validation_labels is not None:
            self._validate_labels(validation_labels)

            if len(validation_items) != len(validation_labels):
                raise ValueError(
                    f"Number of validation items ({len(validation_items)}) "
                    f"must match number of validation labels ({len(validation_labels)})"
                )

            val_predictions = self.predict(validation_items)
            val_pred_labels = [p.predicted_class for p in val_predictions]
            val_acc = sum(
                pred == true
                for pred, true in zip(val_pred_labels, validation_labels, strict=True)
            ) / len(validation_labels)
            metrics["val_accuracy"] = val_acc

        return metrics

    def predict(self, items: list[Item]) -> list[ModelPrediction]:
        """Predict class labels for items.

        Parameters
        ----------
        items : list[Item]
            Items to predict.

        Returns
        -------
        list[ModelPrediction]
            Predictions with probabilities and predicted class.

        Raises
        ------
        ValueError
            If model has not been trained yet.
        """
        if not self._is_fitted:
            raise ValueError("Model not trained. Call train() before predict().")

        self.encoder.eval()
        self.classifier_head.eval()

        with torch.no_grad():
            embeddings = self._prepare_inputs(items)
            logits = self.classifier_head(embeddings)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
            pred_classes = torch.argmax(logits, dim=1).cpu().numpy()

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

    def predict_proba(self, items: list[Item]) -> np.ndarray:
        """Predict class probabilities for items.

        Parameters
        ----------
        items : list[Item]
            Items to predict.

        Returns
        -------
        np.ndarray
            Array of shape (n_items, n_classes) with probabilities.

        Raises
        ------
        ValueError
            If model has not been trained yet.
        """
        if not self._is_fitted:
            raise ValueError("Model not trained. Call train() before predict_proba().")

        self.encoder.eval()
        self.classifier_head.eval()

        with torch.no_grad():
            embeddings = self._prepare_inputs(items)
            logits = self.classifier_head(embeddings)
            proba = torch.softmax(logits, dim=1).cpu().numpy()

        return proba

    def save(self, path: str) -> None:
        """Save model to disk.

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

        # Save both config and training state
        config_dict = self.config.model_dump()
        config_dict["num_classes"] = self.num_classes
        config_dict["option_names"] = self.option_names

        with open(save_path / "config.json", "w") as f:
            json.dump(config_dict, f, indent=2)

    def load(self, path: str) -> None:
        """Load model from disk.

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
        self.option_names = config_dict.pop("option_names")

        # Reconstruct configuration
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
        self._is_fitted = True
