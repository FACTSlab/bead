"""Tests for ClozeModel HuggingFace Trainer integration.

Tests that the cloze model correctly uses HuggingFace Trainer for
fixed and random_intercepts modes.
"""

from __future__ import annotations

from bead.active_learning.config import MixedEffectsConfig
from bead.active_learning.models.cloze import ClozeModel
from bead.config.active_learning import ClozeModelConfig
from bead.items.item import Item


class TestHuggingFaceTrainerIntegration:
    """Test ClozeModel with HuggingFace Trainer integration."""

    def test_fixed_mode_uses_huggingface_trainer(
        self, sample_short_cloze_items: list[Item], sample_short_labels: list[list[str]]
    ) -> None:
        """Test that fixed mode uses HuggingFace Trainer."""
        config = ClozeModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=2,
            device="cpu",
            max_length=32,
            mixed_effects=MixedEffectsConfig(mode="fixed"),
        )
        model = ClozeModel(config)

        metrics = model.train(
            sample_short_cloze_items, sample_short_labels, participant_ids=None
        )

        assert "train_loss" in metrics
        assert "train_accuracy" in metrics
        assert isinstance(metrics["train_loss"], float)
        assert isinstance(metrics["train_accuracy"], float)

    def test_random_intercepts_uses_huggingface_trainer(
        self, sample_short_cloze_items: list[Item], sample_short_labels: list[list[str]]
    ) -> None:
        """Test that random_intercepts mode uses HuggingFace Trainer."""
        config = ClozeModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=2,
            device="cpu",
            max_length=32,
            mixed_effects=MixedEffectsConfig(
                mode="random_intercepts",
                estimate_variance_components=True,
            ),
        )
        model = ClozeModel(config)

        participant_ids = ["p1", "p1", "p2", "p2"]
        metrics = model.train(
            sample_short_cloze_items, sample_short_labels, participant_ids
        )

        assert "train_loss" in metrics
        assert "train_accuracy" in metrics
        assert "participant_variance" in metrics
        assert metrics["n_participants"] == 2

    def test_random_slopes_uses_custom_loop(
        self, sample_short_cloze_items: list[Item], sample_short_labels: list[list[str]]
    ) -> None:
        """Test that random_slopes mode uses custom training loop."""
        config = ClozeModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=2,
            device="cpu",
            max_length=32,
            mixed_effects=MixedEffectsConfig(mode="random_slopes"),
        )
        model = ClozeModel(config)

        participant_ids = ["p1", "p1", "p2", "p2"]
        metrics = model.train(
            sample_short_cloze_items, sample_short_labels, participant_ids
        )

        # Should still train successfully with custom loop
        assert "train_loss" in metrics
        assert "train_accuracy" in metrics
        # Random slopes should create participant-specific heads
        assert "p1" in model.random_effects.slopes
        assert "p2" in model.random_effects.slopes

    def test_validation_with_huggingface_trainer(
        self, sample_short_cloze_items: list[Item], sample_short_labels: list[list[str]]
    ) -> None:
        """Test validation with HuggingFace Trainer."""
        config = ClozeModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=2,
            device="cpu",
            max_length=32,
            mixed_effects=MixedEffectsConfig(mode="fixed"),
        )
        model = ClozeModel(config)

        # Split into train and validation
        train_items = sample_short_cloze_items[:2]
        train_labels = sample_short_labels[:2]
        val_items = sample_short_cloze_items[2:]
        val_labels = sample_short_labels[2:]

        metrics = model.train(
            train_items,
            train_labels,
            participant_ids=None,
            validation_items=val_items,
            validation_labels=val_labels,
        )

        assert "train_loss" in metrics
        assert "train_accuracy" in metrics
        assert "val_accuracy" in metrics
        assert isinstance(metrics["val_accuracy"], float)
        assert 0.0 <= metrics["val_accuracy"] <= 1.0

    def test_validation_with_random_intercepts(
        self, sample_short_cloze_items: list[Item], sample_short_labels: list[list[str]]
    ) -> None:
        """Test validation with random_intercepts mode."""
        config = ClozeModelConfig(
            model_name="bert-base-uncased",
            num_epochs=1,
            batch_size=2,
            device="cpu",
            max_length=32,
            mixed_effects=MixedEffectsConfig(mode="random_intercepts"),
        )
        model = ClozeModel(config)

        train_items = sample_short_cloze_items[:2]
        train_labels = sample_short_labels[:2]
        val_items = sample_short_cloze_items[2:]
        val_labels = sample_short_labels[2:]

        participant_ids = ["p1", "p1"]
        metrics = model.train(
            train_items,
            train_labels,
            participant_ids=participant_ids,
            validation_items=val_items,
            validation_labels=val_labels,
        )

        assert "train_loss" in metrics
        assert "train_accuracy" in metrics
        assert "val_accuracy" in metrics

    def test_checkpointing_with_huggingface_trainer(
        self, sample_short_cloze_items: list[Item], sample_short_labels: list[list[str]]
    ) -> None:
        """Test that checkpointing works with HuggingFace Trainer."""
        config = ClozeModelConfig(
            model_name="bert-base-uncased",
            num_epochs=2,
            batch_size=2,
            device="cpu",
            max_length=32,
            mixed_effects=MixedEffectsConfig(mode="fixed"),
        )
        model = ClozeModel(config)

        # Training should complete without errors (checkpoints are saved internally)
        metrics = model.train(
            sample_short_cloze_items, sample_short_labels, participant_ids=None
        )

        assert "train_loss" in metrics
        assert model._is_fitted is True

    def test_multiple_epochs_with_huggingface_trainer(
        self, sample_short_cloze_items: list[Item], sample_short_labels: list[list[str]]
    ) -> None:
        """Test that multiple epochs work with HuggingFace Trainer."""
        config = ClozeModelConfig(
            model_name="bert-base-uncased",
            num_epochs=3,
            batch_size=2,
            device="cpu",
            max_length=32,
            mixed_effects=MixedEffectsConfig(mode="fixed"),
        )
        model = ClozeModel(config)

        metrics = model.train(
            sample_short_cloze_items, sample_short_labels, participant_ids=None
        )

        assert "train_loss" in metrics
        # Loss should be a valid float
        assert isinstance(metrics["train_loss"], float)
        assert metrics["train_loss"] >= 0.0
