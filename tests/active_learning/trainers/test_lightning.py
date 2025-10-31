"""Tests for PyTorch Lightning trainer."""

from __future__ import annotations

from pathlib import Path

from bead.active_learning.trainers.base import ModelMetadata
from bead.active_learning.trainers.lightning import PyTorchLightningTrainer


class TestPyTorchLightningTrainer:
    """Test suite for PyTorchLightningTrainer."""

    def test_initialization(self, training_config: dict) -> None:
        """Test trainer initialization."""
        trainer = PyTorchLightningTrainer(training_config)
        assert trainer.config == training_config
        assert trainer.lightning_module is None

    def test_train_with_mock(
        self,
        training_config: dict,
        mock_lightning: dict,
        mock_dataset,
    ) -> None:
        """Test training with mocked Lightning."""
        trainer = PyTorchLightningTrainer(training_config)
        metadata = trainer.train(mock_dataset, mock_dataset)

        # Verify training was called
        assert mock_lightning["trainer_instance"].fit.called

        # Verify metadata
        assert metadata.framework == "pytorch_lightning"
        assert metadata.model_name == "bert-base-uncased"
        assert "val_loss" in metadata.metrics
        assert metadata.metrics["val_loss"] == 0.5
        assert metadata.training_time > 0

    def test_train_without_eval_data(
        self,
        training_config: dict,
        mock_lightning: dict,
        mock_dataset,
        mocker,
    ) -> None:
        """Test training without evaluation data."""
        mock_lightning["trainer_instance"].validate = mocker.Mock(return_value=None)

        trainer = PyTorchLightningTrainer(training_config)
        metadata = trainer.train(mock_dataset, None)

        # Metrics should be empty when no eval data
        assert metadata.metrics == {}
        assert metadata.eval_data_path is None

    def test_train_with_best_checkpoint(
        self,
        training_config: dict,
        mock_lightning: dict,
        mock_dataset,
    ) -> None:
        """Test training saves best checkpoint."""
        trainer = PyTorchLightningTrainer(training_config)
        metadata = trainer.train(mock_dataset, mock_dataset)

        # Should have best checkpoint from mock
        assert metadata.best_checkpoint == Path("/tmp/best.ckpt")

    def test_train_with_logging_dir(
        self,
        training_config: dict,
        mock_lightning: dict,
        mock_dataset,
        tmp_path: Path,
    ) -> None:
        """Test training with logging directory."""
        training_config["logging_dir"] = tmp_path / "logs"

        trainer = PyTorchLightningTrainer(training_config)
        _metadata = trainer.train(mock_dataset, mock_dataset)

        # Should create trainer with logger
        trainer_call_kwargs = mock_lightning["trainer_cls"].call_args[1]
        assert trainer_call_kwargs["logger"] is not None

    def test_train_without_logging_dir(
        self,
        training_config: dict,
        mock_lightning: dict,
        mock_dataset,
    ) -> None:
        """Test training without logging directory."""
        training_config["logging_dir"] = None

        trainer = PyTorchLightningTrainer(training_config)
        _metadata = trainer.train(mock_dataset, mock_dataset)

        # Should create trainer without logger
        trainer_call_kwargs = mock_lightning["trainer_cls"].call_args[1]
        assert trainer_call_kwargs["logger"] is None

    def test_save_model(
        self, training_config: dict, tmp_path: Path, mock_lightning: dict, mocker
    ) -> None:
        """Test model saving."""
        trainer = PyTorchLightningTrainer(training_config)

        # Create a mock lightning module
        mock_module = mocker.Mock()
        mock_module.state_dict = mocker.Mock(return_value={"weight": "value"})
        trainer.lightning_module = mock_module

        metadata = ModelMetadata(
            model_name="bert-base-uncased",
            framework="pytorch_lightning",
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

        # Verify torch.save was called
        assert mock_lightning["torch"].save.called

    def test_save_model_creates_directory(
        self, training_config: dict, tmp_path: Path, mock_lightning: dict, mocker
    ) -> None:
        """Test save_model creates output directory if it doesn't exist."""
        trainer = PyTorchLightningTrainer(training_config)

        mock_module = mocker.Mock()
        mock_module.state_dict = mocker.Mock(return_value={})
        trainer.lightning_module = mock_module

        metadata = ModelMetadata(
            model_name="bert-base-uncased",
            framework="pytorch_lightning",
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
        self, training_config: dict, tmp_path: Path, mock_lightning: dict, mocker
    ) -> None:
        """Test model loading."""
        trainer = PyTorchLightningTrainer(training_config)

        # Set up lightning module
        mock_module = mocker.Mock()
        mock_module.load_state_dict = mocker.Mock()
        trainer.lightning_module = mock_module

        model_dir = tmp_path / "model"
        model_dir.mkdir(parents=True)

        loaded_model = trainer.load_model(model_dir)

        # Verify torch.load was called
        assert mock_lightning["torch"].load.called

        # Verify load_state_dict was called
        assert mock_module.load_state_dict.called

        # Verify model is returned
        assert loaded_model == trainer.lightning_module

    def test_get_config_value_from_dict(self, tmp_path: Path) -> None:
        """Test _get_config_value with dict config."""
        config = {"model_name": "bert", "num_epochs": 5}
        trainer = PyTorchLightningTrainer(config)

        assert trainer._get_config_value("model_name") == "bert"
        assert trainer._get_config_value("num_epochs") == 5
        assert trainer._get_config_value("missing", "default") == "default"

    def test_get_config_value_from_object(self, tmp_path: Path) -> None:
        """Test _get_config_value with object config."""

        class Config:
            model_name = "bert"
            num_epochs = 5

        config = Config()
        trainer = PyTorchLightningTrainer(config)

        assert trainer._get_config_value("model_name") == "bert"
        assert trainer._get_config_value("num_epochs") == 5
        assert trainer._get_config_value("missing", "default") == "default"

    def test_train_updates_lightning_module(
        self,
        training_config: dict,
        mock_lightning: dict,
        mock_dataset,
    ) -> None:
        """Test that train updates lightning_module."""
        trainer = PyTorchLightningTrainer(training_config)

        assert trainer.lightning_module is None

        trainer.train(mock_dataset, mock_dataset)

        # After training, lightning_module should be set
        assert trainer.lightning_module is not None

    def test_train_creates_callbacks(
        self,
        training_config: dict,
        mock_lightning: dict,
        mock_dataset,
    ) -> None:
        """Test that train creates callbacks."""
        trainer = PyTorchLightningTrainer(training_config)
        trainer.train(mock_dataset, mock_dataset)

        # Verify trainer was created with callbacks
        trainer_call_kwargs = mock_lightning["trainer_cls"].call_args[1]
        assert "callbacks" in trainer_call_kwargs
        assert len(trainer_call_kwargs["callbacks"]) == 2

    def test_save_model_handles_none_module(
        self, training_config: dict, tmp_path: Path, mock_lightning: dict
    ) -> None:
        """Test save_model handles None lightning_module gracefully."""
        trainer = PyTorchLightningTrainer(training_config)

        metadata = ModelMetadata(
            model_name="bert-base-uncased",
            framework="pytorch_lightning",
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

    def test_load_model_returns_none_when_no_module(
        self, training_config: dict, tmp_path: Path, mock_lightning: dict
    ) -> None:
        """Test load_model returns None when no lightning_module."""
        trainer = PyTorchLightningTrainer(training_config)

        model_dir = tmp_path / "model"
        model_dir.mkdir(parents=True)

        loaded_model = trainer.load_model(model_dir)

        assert loaded_model is None
