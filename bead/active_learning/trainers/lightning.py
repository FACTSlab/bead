"""PyTorch Lightning trainer implementation.

This module provides a trainer that uses PyTorch Lightning for model training
with callbacks for checkpointing and early stopping.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from bead.active_learning.trainers.base import BaseTrainer, ModelMetadata
from bead.data.timestamps import format_iso8601, now_iso8601

if TYPE_CHECKING:
    import pytorch_lightning as pl
    from torch.nn import Module
    from torch.utils.data import DataLoader


def create_lightning_module(
    model: Module, learning_rate: float = 2e-5
) -> pl.LightningModule:
    """Create a PyTorch Lightning module.

    Parameters
    ----------
    model : Any
        The model to wrap.
    learning_rate : float
        Learning rate for optimizer.

    Returns
    -------
    Any
        Lightning module instance.
    """
    import pytorch_lightning as pl  # noqa: PLC0415
    import torch  # noqa: PLC0415

    class _LightningModule(pl.LightningModule):
        def __init__(self) -> None:
            super().__init__()
            self.model = model
            self.learning_rate = learning_rate

        def forward(self, **inputs: Any) -> Any:
            return self.model(**inputs)

        def training_step(self, batch: Any, batch_idx: int) -> Any:
            outputs = self(**batch)
            loss = outputs.loss
            self.log("train_loss", loss)
            return loss

        def validation_step(self, batch: Any, batch_idx: int) -> Any:
            outputs = self(**batch)
            loss = outputs.loss
            self.log("val_loss", loss)
            return loss

        def configure_optimizers(self) -> Any:
            optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
            return optimizer

    return _LightningModule()


class PyTorchLightningTrainer(BaseTrainer):
    """Trainer using PyTorch Lightning.

    This trainer uses PyTorch Lightning to train models with support for
    callbacks, logging, and advanced training features.

    Parameters
    ----------
    config : Any
        Training configuration with the following expected fields:
        - model_name: str - Base model name/path
        - num_labels: int | None - Number of labels
        - num_epochs: int - Number of training epochs
        - learning_rate: float - Learning rate
        - output_dir: Path - Directory for outputs
        - logging_dir: Path | None - Logging directory

    Attributes
    ----------
    config : Any
        Training configuration.
    lightning_module : Any | None
        The Lightning module wrapper.

    Examples
    --------
    >>> from pathlib import Path
    >>> config = {
    ...     "model_name": "bert-base-uncased",
    ...     "num_labels": 2,
    ...     "num_epochs": 3,
    ...     "learning_rate": 2e-5,
    ...     "output_dir": Path("output"),
    ...     "logging_dir": None
    ... }
    >>> trainer = PyTorchLightningTrainer(config)
    >>> trainer.lightning_module is None
    True
    """

    def __init__(
        self, config: dict[str, int | str | float | bool | Path] | object
    ) -> None:
        super().__init__(config)
        self.lightning_module: pl.LightningModule | None = None

    def _get_config_value(
        self, key: str, default: int | str | float | bool | Path | None = None
    ) -> int | str | float | bool | Path | None:
        """Get configuration value with fallback to default.

        Parameters
        ----------
        key : str
            Configuration key.
        default : Any
            Default value if key not found.

        Returns
        -------
        Any
            Configuration value.
        """
        if hasattr(self.config, key):
            return getattr(self.config, key)
        if isinstance(self.config, dict):
            return self.config.get(key, default)
        return default

    def train(
        self, train_data: DataLoader, eval_data: DataLoader | None = None
    ) -> ModelMetadata:
        """Train using PyTorch Lightning.

        Parameters
        ----------
        train_data : Any
            Training dataloader.
        eval_data : Any | None
            Evaluation dataloader.

        Returns
        -------
        ModelMetadata
            Training metadata.

        Examples
        --------
        >>> config = {"model_name": "bert-base-uncased"}  # doctest: +SKIP
        >>> trainer = PyTorchLightningTrainer(config)  # doctest: +SKIP
        >>> metadata = trainer.train(train_loader)  # doctest: +SKIP
        >>> metadata.framework  # doctest: +SKIP
        'pytorch_lightning'
        """
        import pytorch_lightning as pl  # noqa: PLC0415
        from transformers import AutoModelForSequenceClassification  # noqa: PLC0415

        start_time = time.time()

        # Get config values
        model_name = self._get_config_value("model_name", "bert-base-uncased")
        num_labels = self._get_config_value("num_labels", 2)
        num_epochs = self._get_config_value("num_epochs", 3)
        learning_rate = self._get_config_value("learning_rate", 2e-5)
        output_dir = self._get_config_value("output_dir", Path("output"))
        logging_dir = self._get_config_value("logging_dir", None)

        # Load model
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

        # Create Lightning module
        self.lightning_module = create_lightning_module(model, learning_rate)

        # Create callbacks
        callbacks = [
            pl.callbacks.ModelCheckpoint(
                monitor="val_loss",
                dirpath=output_dir,
                filename="best-{epoch:02d}-{val_loss:.2f}",
            ),
            pl.callbacks.EarlyStopping(monitor="val_loss", patience=3),
        ]

        # Create logger
        logger = None
        if logging_dir:
            logger = pl.loggers.TensorBoardLogger(str(logging_dir))

        # Create trainer
        trainer = pl.Trainer(
            max_epochs=num_epochs,
            accelerator="auto",
            devices="auto",
            logger=logger,
            callbacks=callbacks,
        )

        # Train
        trainer.fit(
            self.lightning_module,
            train_dataloaders=train_data,
            val_dataloaders=eval_data,
        )

        # Evaluate
        metrics: dict[str, float] = {}
        if eval_data is not None:
            eval_results = trainer.validate(
                self.lightning_module, dataloaders=eval_data
            )
            if eval_results:
                metrics = {k: float(v) for k, v in eval_results[0].items()}

        training_time = time.time() - start_time

        # Get best checkpoint path
        best_checkpoint = None
        if hasattr(trainer.checkpoint_callback, "best_model_path"):
            best_checkpoint_str = trainer.checkpoint_callback.best_model_path
            if best_checkpoint_str:
                best_checkpoint = Path(best_checkpoint_str)

        # Create metadata
        config_dict = (
            self.config
            if isinstance(self.config, dict)
            else (
                self.config.model_dump() if hasattr(self.config, "model_dump") else {}
            )
        )

        metadata = ModelMetadata(
            model_name=model_name,
            framework="pytorch_lightning",
            training_config=config_dict,
            training_data_path=Path("train.json"),
            eval_data_path=Path("eval.json") if eval_data else None,
            metrics=metrics,
            best_checkpoint=best_checkpoint,
            training_time=training_time,
            training_timestamp=format_iso8601(now_iso8601()),
        )

        return metadata

    def save_model(self, output_dir: Path, metadata: ModelMetadata) -> None:
        """Save model.

        Parameters
        ----------
        output_dir : Path
            Directory to save model and metadata.
        metadata : ModelMetadata
            Training metadata to save.

        Examples
        --------
        >>> trainer = PyTorchLightningTrainer({})  # doctest: +SKIP
        >>> trainer.save_model(Path("output"), metadata)  # doctest: +SKIP
        """
        import torch  # noqa: PLC0415

        output_dir.mkdir(parents=True, exist_ok=True)

        # Save Lightning checkpoint
        if self.lightning_module is not None:
            torch.save(
                self.lightning_module.state_dict(),
                output_dir / "lightning_model.pt",
            )

        # Save metadata
        with open(output_dir / "metadata.json", "w") as f:
            metadata_dict = metadata.model_dump()
            json.dump(metadata_dict, f, indent=2, default=str)

    def load_model(self, model_dir: Path) -> pl.LightningModule | None:
        """Load model.

        Parameters
        ----------
        model_dir : Path
            Directory containing saved model.

        Returns
        -------
        Any
            Loaded Lightning module.

        Examples
        --------
        >>> trainer = PyTorchLightningTrainer({})  # doctest: +SKIP
        >>> model = trainer.load_model(Path("saved_model"))  # doctest: +SKIP
        """
        import torch  # noqa: PLC0415

        if self.lightning_module is not None:
            self.lightning_module.load_state_dict(
                torch.load(model_dir / "lightning_model.pt")
            )
        return self.lightning_module
