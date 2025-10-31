"""Tests for HuggingFace trainer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest

from bead.active_learning.trainers.base import ModelMetadata
from bead.active_learning.trainers.huggingface import HuggingFaceTrainer


class TestHuggingFaceTrainer:
    """Test suite for HuggingFaceTrainer."""

    def test_initialization(self, training_config: dict) -> None:
        """Test trainer initialization."""
        trainer = HuggingFaceTrainer(training_config)
        assert trainer.config == training_config
        assert trainer.model is None
        assert trainer.tokenizer is None

    def test_train_with_mock(
        self,
        training_config: dict,
        mock_transformers: dict,
        mock_dataset,
    ) -> None:
        """Test training with mocked transformers."""
        trainer = HuggingFaceTrainer(training_config)
        metadata = trainer.train(mock_dataset, mock_dataset)

        # Verify training was called
        assert mock_transformers["trainer_instance"].train.called

        # Verify metadata
        assert metadata.framework == "huggingface"
        assert metadata.model_name == "bert-base-uncased"
        assert "eval_loss" in metadata.metrics
        assert metadata.metrics["eval_loss"] == 0.5
        assert metadata.training_time > 0

    def test_train_without_eval_data(
        self,
        training_config: dict,
        mock_transformers: dict,
        mock_dataset,
    ) -> None:
        """Test training without evaluation data."""
        trainer = HuggingFaceTrainer(training_config)
        metadata = trainer.train(mock_dataset, None)

        # Should not call evaluate
        assert not mock_transformers["trainer_instance"].evaluate.called

        # Metrics should be empty
        assert metadata.metrics == {}
        assert metadata.eval_data_path is None

    def test_train_with_unsupported_task_type(
        self, training_config: dict, mock_transformers: dict, mock_dataset: Mock
    ) -> None:
        """Test training with unsupported task type raises error."""
        training_config["task_type"] = "unsupported"
        trainer = HuggingFaceTrainer(training_config)

        with pytest.raises(ValueError, match="Task type not supported"):
            trainer.train(mock_dataset)

    def test_save_model(
        self, training_config: dict, tmp_path: Path, mock_transformers: dict
    ) -> None:
        """Test model saving."""
        trainer = HuggingFaceTrainer(training_config)
        trainer.model = mock_transformers["model"]
        trainer.tokenizer = mock_transformers["tokenizer"]

        metadata = ModelMetadata(
            model_name="bert-base-uncased",
            framework="huggingface",
            training_config={},
            training_data_path=Path("train.json"),
            metrics={},
            training_time=10.0,
            training_timestamp="2025-01-17T00:00:00+00:00",
        )

        output_dir = tmp_path / "saved_model"
        trainer.save_model(output_dir, metadata)

        # Verify directory was created
        assert output_dir.exists()

        # Verify metadata file exists
        assert (output_dir / "metadata.json").exists()

        # Verify model and tokenizer save_pretrained were called
        assert mock_transformers["model"].save_pretrained.called
        assert mock_transformers["tokenizer"].save_pretrained.called

    def test_save_model_creates_directory(
        self, training_config: dict, tmp_path: Path, mock_transformers: dict
    ) -> None:
        """Test save_model creates output directory if it doesn't exist."""
        trainer = HuggingFaceTrainer(training_config)
        trainer.model = mock_transformers["model"]
        trainer.tokenizer = mock_transformers["tokenizer"]

        metadata = ModelMetadata(
            model_name="bert-base-uncased",
            framework="huggingface",
            training_config={},
            training_data_path=Path("train.json"),
            metrics={},
            training_time=10.0,
            training_timestamp="2025-01-17T00:00:00+00:00",
        )

        output_dir = tmp_path / "nested" / "output" / "dir"
        trainer.save_model(output_dir, metadata)

        assert output_dir.exists()

    def test_load_model(
        self, training_config: dict, tmp_path: Path, mock_transformers: dict
    ) -> None:
        """Test model loading."""
        trainer = HuggingFaceTrainer(training_config)

        model_dir = tmp_path / "model"
        model_dir.mkdir(parents=True)

        loaded_model = trainer.load_model(model_dir)

        # Verify from_pretrained was called
        assert mock_transformers["auto_model"].from_pretrained.called
        assert mock_transformers["auto_tokenizer"].from_pretrained.called

        # Verify model is set
        assert trainer.model is not None
        assert trainer.tokenizer is not None
        assert loaded_model == trainer.model

    def test_get_config_value_from_dict(self, tmp_path: Path) -> None:
        """Test _get_config_value with dict config."""
        config = {"model_name": "bert", "num_epochs": 5}
        trainer = HuggingFaceTrainer(config)

        assert trainer._get_config_value("model_name") == "bert"
        assert trainer._get_config_value("num_epochs") == 5
        assert trainer._get_config_value("missing", "default") == "default"

    def test_get_config_value_from_object(self, tmp_path: Path) -> None:
        """Test _get_config_value with object config."""

        class Config:
            model_name = "bert"
            num_epochs = 5

        config = Config()
        trainer = HuggingFaceTrainer(config)

        assert trainer._get_config_value("model_name") == "bert"
        assert trainer._get_config_value("num_epochs") == 5
        assert trainer._get_config_value("missing", "default") == "default"

    def test_train_updates_trainer_state(
        self,
        training_config: dict,
        mock_transformers: dict,
        mock_dataset,
    ) -> None:
        """Test that train updates trainer model and tokenizer."""
        trainer = HuggingFaceTrainer(training_config)

        assert trainer.model is None
        assert trainer.tokenizer is None

        trainer.train(mock_dataset, mock_dataset)

        # After training, model and tokenizer should be set
        assert trainer.model is not None
        assert trainer.tokenizer is not None

    def test_train_with_best_checkpoint(
        self,
        training_config: dict,
        mock_transformers: dict,
        mock_dataset,
    ) -> None:
        """Test training with best checkpoint saved."""
        # Set best checkpoint
        mock_transformers[
            "trainer_instance"
        ].state.best_model_checkpoint = "/tmp/checkpoint-123"

        trainer = HuggingFaceTrainer(training_config)
        metadata = trainer.train(mock_dataset, mock_dataset)

        assert metadata.best_checkpoint == Path("/tmp/checkpoint-123")

    def test_save_model_handles_none_model(
        self, training_config: dict, tmp_path: Path
    ) -> None:
        """Test save_model handles None model gracefully."""
        trainer = HuggingFaceTrainer(training_config)

        metadata = ModelMetadata(
            model_name="bert-base-uncased",
            framework="huggingface",
            training_config={},
            training_data_path=Path("train.json"),
            metrics={},
            training_time=10.0,
            training_timestamp="2025-01-17T00:00:00+00:00",
        )

        output_dir = tmp_path / "saved_model"
        trainer.save_model(output_dir, metadata)

        # Should still create directory and save metadata
        assert output_dir.exists()
        assert (output_dir / "metadata.json").exists()
