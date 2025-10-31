"""Tests for base trainer interface."""

from __future__ import annotations

from pathlib import Path

import pytest

from bead.active_learning.trainers.base import BaseTrainer, ModelMetadata


class TestModelMetadata:
    """Test suite for ModelMetadata."""

    def test_model_metadata_creation(self) -> None:
        """Test ModelMetadata creation and fields."""
        metadata = ModelMetadata(
            model_name="bert-base-uncased",
            framework="huggingface",
            training_config={"epochs": 3},
            training_data_path=Path("train.json"),
            metrics={"accuracy": 0.95},
            training_time=120.5,
            training_timestamp="2025-01-17T00:00:00+00:00",
        )

        assert metadata.model_name == "bert-base-uncased"
        assert metadata.framework == "huggingface"
        assert metadata.training_config == {"epochs": 3}
        assert metadata.training_data_path == Path("train.json")
        assert metadata.eval_data_path is None
        assert metadata.metrics == {"accuracy": 0.95}
        assert metadata.best_checkpoint is None
        assert metadata.training_time == 120.5
        assert metadata.training_timestamp == "2025-01-17T00:00:00+00:00"

    def test_model_metadata_with_optional_fields(self) -> None:
        """Test ModelMetadata with optional fields."""
        metadata = ModelMetadata(
            model_name="bert-base-uncased",
            framework="huggingface",
            training_config={"epochs": 3},
            training_data_path=Path("train.json"),
            eval_data_path=Path("eval.json"),
            metrics={"accuracy": 0.95},
            best_checkpoint=Path("/tmp/checkpoint"),
            training_time=120.5,
            training_timestamp="2025-01-17T00:00:00+00:00",
        )

        assert metadata.eval_data_path == Path("eval.json")
        assert metadata.best_checkpoint == Path("/tmp/checkpoint")

    def test_model_metadata_serialization(self) -> None:
        """Test ModelMetadata can be serialized to dict."""
        metadata = ModelMetadata(
            model_name="bert-base-uncased",
            framework="huggingface",
            training_config={"epochs": 3},
            training_data_path=Path("train.json"),
            metrics={"accuracy": 0.95},
            training_time=120.5,
            training_timestamp="2025-01-17T00:00:00+00:00",
        )

        metadata_dict = metadata.model_dump()
        assert isinstance(metadata_dict, dict)
        assert metadata_dict["model_name"] == "bert-base-uncased"
        assert metadata_dict["framework"] == "huggingface"

    def test_model_metadata_inherits_base_fields(self) -> None:
        """Test ModelMetadata inherits BeadBaseModel fields."""
        metadata = ModelMetadata(
            model_name="bert-base-uncased",
            framework="huggingface",
            training_config={"epochs": 3},
            training_data_path=Path("train.json"),
            metrics={"accuracy": 0.95},
            training_time=120.5,
            training_timestamp="2025-01-17T00:00:00+00:00",
        )

        # Should have inherited fields from BeadBaseModel
        assert hasattr(metadata, "id")
        assert hasattr(metadata, "created_at")
        assert hasattr(metadata, "modified_at")
        assert hasattr(metadata, "version")
        assert hasattr(metadata, "metadata")


class TestBaseTrainer:
    """Test suite for BaseTrainer."""

    def test_base_trainer_cannot_be_instantiated(self) -> None:
        """Test BaseTrainer is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseTrainer({})  # type: ignore

    def test_base_trainer_requires_implementation(self) -> None:
        """Test BaseTrainer requires abstract methods to be implemented."""

        # Missing implementations should raise TypeError
        class IncompleteTrainer(BaseTrainer):
            pass

        with pytest.raises(TypeError):
            IncompleteTrainer({})  # type: ignore

    def test_base_trainer_can_be_subclassed(self) -> None:
        """Test BaseTrainer can be properly subclassed."""

        class ConcreteTrainer(BaseTrainer):
            def train(self, train_data, eval_data=None):
                return ModelMetadata(
                    model_name="test",
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

        config = {"test": "config"}
        trainer = ConcreteTrainer(config)
        assert trainer.config == config

    def test_base_trainer_train_signature(self) -> None:
        """Test BaseTrainer train method signature."""

        class ConcreteTrainer(BaseTrainer):
            def train(self, train_data, eval_data=None):
                return ModelMetadata(
                    model_name="test",
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

        trainer = ConcreteTrainer({})
        metadata = trainer.train("train_data")
        assert isinstance(metadata, ModelMetadata)

        metadata = trainer.train("train_data", "eval_data")
        assert isinstance(metadata, ModelMetadata)

    def test_base_trainer_save_model_signature(self) -> None:
        """Test BaseTrainer save_model method signature."""

        class ConcreteTrainer(BaseTrainer):
            def train(self, train_data, eval_data=None):
                return ModelMetadata(
                    model_name="test",
                    framework="custom",
                    training_config={},
                    training_data_path=Path("train.json"),
                    metrics={},
                    training_time=0.0,
                    training_timestamp="2025-01-17T00:00:00+00:00",
                )

            def save_model(self, output_dir, metadata):
                self.saved_dir = output_dir
                self.saved_metadata = metadata

            def load_model(self, model_dir):
                return None

        trainer = ConcreteTrainer({})
        metadata = ModelMetadata(
            model_name="test",
            framework="custom",
            training_config={},
            training_data_path=Path("train.json"),
            metrics={},
            training_time=0.0,
            training_timestamp="2025-01-17T00:00:00+00:00",
        )

        trainer.save_model(Path("/tmp/model"), metadata)
        assert trainer.saved_dir == Path("/tmp/model")  # type: ignore
        assert trainer.saved_metadata == metadata  # type: ignore

    def test_base_trainer_load_model_signature(self) -> None:
        """Test BaseTrainer load_model method signature."""

        class ConcreteTrainer(BaseTrainer):
            def train(self, train_data, eval_data=None):
                return ModelMetadata(
                    model_name="test",
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
                return f"loaded from {model_dir}"

        trainer = ConcreteTrainer({})
        result = trainer.load_model(Path("/tmp/model"))
        assert result == "loaded from /tmp/model"
