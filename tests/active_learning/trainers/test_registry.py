"""Tests for trainer registry."""

from __future__ import annotations

from pathlib import Path

import pytest

from bead.active_learning.trainers.base import BaseTrainer, ModelMetadata
from bead.active_learning.trainers.huggingface import HuggingFaceTrainer
from bead.active_learning.trainers.lightning import PyTorchLightningTrainer
from bead.active_learning.trainers.registry import (
    get_trainer,
    list_trainers,
    register_trainer,
)


class TestTrainerRegistry:
    """Test suite for trainer registry."""

    def test_list_trainers_includes_built_ins(self) -> None:
        """Test list_trainers includes built-in trainers."""
        trainers = list_trainers()
        assert "huggingface" in trainers
        assert "pytorch_lightning" in trainers

    def test_get_trainer_huggingface(self) -> None:
        """Test get_trainer returns HuggingFaceTrainer."""
        trainer_class = get_trainer("huggingface")
        assert trainer_class == HuggingFaceTrainer

    def test_get_trainer_pytorch_lightning(self) -> None:
        """Test get_trainer returns PyTorchLightningTrainer."""
        trainer_class = get_trainer("pytorch_lightning")
        assert trainer_class == PyTorchLightningTrainer

    def test_get_trainer_raises_on_unknown(self) -> None:
        """Test get_trainer raises ValueError for unknown trainer."""
        with pytest.raises(ValueError, match="Unknown trainer: unknown"):
            get_trainer("unknown")

    def test_get_trainer_error_lists_available(self) -> None:
        """Test get_trainer error message lists available trainers."""
        with pytest.raises(ValueError, match="Available trainers:"):
            get_trainer("nonexistent")

    def test_register_trainer(self) -> None:
        """Test register_trainer adds new trainer."""

        class CustomTrainer(BaseTrainer):
            def train(self, train_data, eval_data=None):
                return ModelMetadata(
                    model_name="custom",
                    framework="custom",
                    training_config={},
                    training_data_path=Path("train.json"),
                    metrics={},
                    training_time=0.0,
                    training_timestamp="2025-01-17T00:00:00+00:00",
                )

            def save_model(self, output_dir, metadata):
                pass

            def load_model(self, model_dir):
                return None

        register_trainer("custom", CustomTrainer)

        assert "custom" in list_trainers()
        assert get_trainer("custom") == CustomTrainer

    def test_register_trainer_overwrite(self) -> None:
        """Test register_trainer can overwrite existing trainer."""

        class NewHuggingFaceTrainer(BaseTrainer):
            def train(self, train_data, eval_data=None):
                return ModelMetadata(
                    model_name="new",
                    framework="huggingface",
                    training_config={},
                    training_data_path=Path("train.json"),
                    metrics={},
                    training_time=0.0,
                    training_timestamp="2025-01-17T00:00:00+00:00",
                )

            def save_model(self, output_dir, metadata):
                pass

            def load_model(self, model_dir):
                return None

        original = get_trainer("huggingface")
        register_trainer("huggingface", NewHuggingFaceTrainer)

        assert get_trainer("huggingface") == NewHuggingFaceTrainer

        # Restore original for other tests
        register_trainer("huggingface", original)

    def test_get_trainer_returns_class_not_instance(self) -> None:
        """Test get_trainer returns class, not instance."""
        trainer_class = get_trainer("huggingface")

        # Should be able to instantiate it
        config = {"model_name": "bert-base-uncased"}
        trainer = trainer_class(config)

        assert isinstance(trainer, HuggingFaceTrainer)
        assert trainer.config == config

    def test_list_trainers_returns_list(self) -> None:
        """Test list_trainers returns a list."""
        trainers = list_trainers()
        assert isinstance(trainers, list)
        assert len(trainers) >= 2

    def test_list_trainers_returns_strings(self) -> None:
        """Test list_trainers returns list of strings."""
        trainers = list_trainers()
        assert all(isinstance(name, str) for name in trainers)

    def test_register_multiple_trainers(self) -> None:
        """Test registering multiple custom trainers."""

        class Trainer1(BaseTrainer):
            def train(self, train_data, eval_data=None):
                return ModelMetadata(
                    model_name="t1",
                    framework="custom1",
                    training_config={},
                    training_data_path=Path("train.json"),
                    metrics={},
                    training_time=0.0,
                    training_timestamp="2025-01-17T00:00:00+00:00",
                )

            def save_model(self, output_dir, metadata):
                pass

            def load_model(self, model_dir):
                return None

        class Trainer2(BaseTrainer):
            def train(self, train_data, eval_data=None):
                return ModelMetadata(
                    model_name="t2",
                    framework="custom2",
                    training_config={},
                    training_data_path=Path("train.json"),
                    metrics={},
                    training_time=0.0,
                    training_timestamp="2025-01-17T00:00:00+00:00",
                )

            def save_model(self, output_dir, metadata):
                pass

            def load_model(self, model_dir):
                return None

        register_trainer("trainer1", Trainer1)
        register_trainer("trainer2", Trainer2)

        assert "trainer1" in list_trainers()
        assert "trainer2" in list_trainers()
        assert get_trainer("trainer1") == Trainer1
        assert get_trainer("trainer2") == Trainer2
