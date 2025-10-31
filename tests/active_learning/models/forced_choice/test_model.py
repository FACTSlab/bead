"""Comprehensive tests for ForcedChoiceModel transformer model."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch

from bead.active_learning.models.base import ModelPrediction
from bead.active_learning.models.forced_choice import ForcedChoiceModel
from bead.config.models import ForcedChoiceModelConfig


def test_forced_choice_initialization():
    """Test forced choice model initializes correctly."""
    config = ForcedChoiceModelConfig(
        model_name="bert-base-uncased",
        max_length=64,
        learning_rate=2e-5,
        batch_size=8,
        num_epochs=2,
        device="cpu",
    )
    model = ForcedChoiceModel(config=config)

    assert model.config.model_name == "bert-base-uncased"
    assert model.config.max_length == 64
    assert model.config.learning_rate == 2e-5
    assert model.config.batch_size == 8
    assert model.config.num_epochs == 2
    assert model.config.device == "cpu"
    assert model._is_fitted is False


def test_forced_choice_default_initialization():
    """Test forced choice initializes with defaults."""
    model = ForcedChoiceModel()

    assert model.config.model_name == "bert-base-uncased"
    assert model.config.max_length == 128
    assert model.config.num_epochs == 3
    assert model._is_fitted is False


def test_forced_choice_tokenize_items(test_items):
    """Test tokenization produces correct format."""
    items = test_items(n=3)
    labels = ["option_a"] * 2 + ["option_b"]
    config = ForcedChoiceModelConfig(num_epochs=1, device="cpu")
    model = ForcedChoiceModel(config=config)
    model.train(items, labels)

    # Test that preparing inputs works
    embeddings = model._prepare_inputs(items)

    # Check embeddings shape
    assert embeddings.shape[0] == 3
    assert embeddings.device.type == "cpu"


def test_forced_choice_train_small_dataset(test_items):
    """Test training on small dataset."""
    items = test_items(n=10)
    labels = ["option_a"] * 5 + ["option_b"] * 5

    config = ForcedChoiceModelConfig(num_epochs=1, batch_size=2, device="cpu")
    model = ForcedChoiceModel(config=config)
    metrics = model.train(items, labels)

    assert "train_accuracy" in metrics
    assert "train_loss" in metrics
    assert 0.0 <= metrics["train_accuracy"] <= 1.0
    assert metrics["train_loss"] >= 0.0
    assert model._is_fitted is True


def test_forced_choice_train_with_varied_items(varied_test_items):
    """Test training on varied dataset."""
    items = varied_test_items(n=20)
    labels = ["option_a" if i < 10 else "option_b" for i in range(20)]

    config = ForcedChoiceModelConfig(num_epochs=2, batch_size=4, device="cpu")
    model = ForcedChoiceModel(config=config)
    metrics = model.train(items, labels)

    assert model._is_fitted is True
    assert metrics["train_accuracy"] >= 0.0


def test_forced_choice_predict_before_training_raises(test_items):
    """Test that predict before training raises error."""
    config = ForcedChoiceModelConfig(device="cpu")
    model = ForcedChoiceModel(config=config)
    items = test_items(n=5)

    with pytest.raises(ValueError, match="Model not trained"):
        model.predict(items)


def test_forced_choice_predict(test_items):
    """Test forced choice predictions."""
    items = test_items(n=10)
    labels = ["option_a"] * 5 + ["option_b"] * 5

    config = ForcedChoiceModelConfig(num_epochs=1, batch_size=2, device="cpu")
    model = ForcedChoiceModel(config=config)
    model.train(items, labels)

    predictions = model.predict(items)

    assert len(predictions) == len(items)
    assert all(isinstance(p, ModelPrediction) for p in predictions)
    assert all(p.predicted_class in ["option_a", "option_b"] for p in predictions)
    assert all(0.0 <= p.confidence <= 1.0 for p in predictions)


def test_forced_choice_predict_proba(test_items):
    """Test forced choice probability predictions."""
    items = test_items(n=10)
    labels = ["option_a"] * 5 + ["option_b"] * 5

    config = ForcedChoiceModelConfig(num_epochs=1, batch_size=2, device="cpu")
    model = ForcedChoiceModel(config=config)
    model.train(items, labels)

    proba = model.predict_proba(items)

    assert proba.shape == (len(items), 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-5)
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)


def test_forced_choice_save_before_training_raises():
    """Test that save before training raises error."""
    config = ForcedChoiceModelConfig(device="cpu")
    model = ForcedChoiceModel(config=config)

    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="Model not trained"):
            model.save(str(Path(tmpdir) / "model"))


def test_forced_choice_save_and_load(test_items):
    """Test forced choice save/load."""
    items = test_items(n=10)
    labels = ["option_a"] * 5 + ["option_b"] * 5

    config = ForcedChoiceModelConfig(num_epochs=1, batch_size=2, device="cpu")
    model1 = ForcedChoiceModel(config=config)
    model1.train(items, labels)

    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "forced_choice_model"
        model1.save(str(save_path))

        model2 = ForcedChoiceModel()
        model2.load(str(save_path))

        # Check predictions are similar
        pred1 = model1.predict(items)
        pred2 = model2.predict(items)

        # Predictions should match
        assert all(
            p1.predicted_class == p2.predicted_class for p1, p2 in zip(pred1, pred2)
        )
        assert model2._is_fitted is True


def test_forced_choice_load_nonexistent_raises():
    """Test loading nonexistent model raises error."""
    model = ForcedChoiceModel()

    with pytest.raises(FileNotFoundError):
        model.load("/nonexistent/path")


def test_forced_choice_with_validation(test_items):
    """Test training with validation set."""
    train_items = test_items(n=20)
    train_labels = ["option_a"] * 10 + ["option_b"] * 10

    val_items = test_items(n=10)
    val_labels = ["option_a"] * 5 + ["option_b"] * 5

    config = ForcedChoiceModelConfig(num_epochs=1, batch_size=4, device="cpu")
    model = ForcedChoiceModel(config=config)
    metrics = model.train(
        train_items,
        train_labels,
        validation_items=val_items,
        validation_labels=val_labels,
    )

    assert "train_accuracy" in metrics
    assert "val_accuracy" in metrics
    assert 0.0 <= metrics["val_accuracy"] <= 1.0


def test_forced_choice_items_labels_mismatch_raises(test_items):
    """Test mismatched items/labels raises error."""
    items = test_items(n=10)
    labels = ["option_a"] * 5  # Wrong length

    model = ForcedChoiceModel()

    with pytest.raises(ValueError, match="Number of items .* must match"):
        model.train(items, labels)


def test_forced_choice_3afc_support(test_items):
    """Test that model supports 3AFC (and n-AFC generally)."""
    items = test_items(n=12)
    labels = ["option_a"] * 4 + ["option_b"] * 4 + ["option_c"] * 4

    config = ForcedChoiceModelConfig(num_epochs=1, batch_size=3, device="cpu")
    model = ForcedChoiceModel(config=config)
    metrics = model.train(items, labels)

    # Should handle 3 classes
    assert model._is_fitted is True
    assert model.num_classes == 3
    assert model.option_names == ["option_a", "option_b", "option_c"]
    assert "train_accuracy" in metrics


def test_forced_choice_different_batch_sizes(test_items):
    """Test different batch sizes."""
    items = test_items(n=12)
    labels = ["option_a"] * 6 + ["option_b"] * 6

    # Batch size 2
    config2 = ForcedChoiceModelConfig(num_epochs=1, batch_size=2, device="cpu")
    model2 = ForcedChoiceModel(config=config2)
    metrics2 = model2.train(items, labels)

    # Batch size 4
    config4 = ForcedChoiceModelConfig(num_epochs=1, batch_size=4, device="cpu")
    model4 = ForcedChoiceModel(config=config4)
    metrics4 = model4.train(items, labels)

    assert model2._is_fitted is True
    assert model4._is_fitted is True
    assert "train_accuracy" in metrics2
    assert "train_accuracy" in metrics4


def test_forced_choice_different_num_epochs(test_items):
    """Test different numbers of epochs."""
    items = test_items(n=15)
    labels = ["option_a"] * 8 + ["option_b"] * 7

    # 1 epoch
    config1 = ForcedChoiceModelConfig(num_epochs=1, batch_size=3, device="cpu")
    model1 = ForcedChoiceModel(config=config1)
    model1.train(items, labels)

    # 2 epochs
    config2 = ForcedChoiceModelConfig(num_epochs=2, batch_size=3, device="cpu")
    model2 = ForcedChoiceModel(config=config2)
    model2.train(items, labels)

    assert model1._is_fitted is True
    assert model2._is_fitted is True


def test_forced_choice_empty_rendered_elements():
    """Test handling of empty rendered elements."""
    from uuid import uuid4

    from bead.items.models import Item

    items = [
        Item(item_template_id=uuid4(), rendered_elements={}),
        Item(item_template_id=uuid4(), rendered_elements={"option_a": "text"}),
    ]
    labels = ["option_a", "option_b"]

    config = ForcedChoiceModelConfig(num_epochs=1, batch_size=2, device="cpu")
    model = ForcedChoiceModel(config=config)
    metrics = model.train(items, labels)

    # Should handle gracefully
    assert model._is_fitted is True


def test_forced_choice_prediction_consistency(test_items):
    """Test that predictions are consistent."""
    items = test_items(n=10)
    labels = ["option_a"] * 5 + ["option_b"] * 5

    config = ForcedChoiceModelConfig(num_epochs=1, batch_size=2, device="cpu")
    model = ForcedChoiceModel(config=config)
    model.train(items, labels)

    # Set to eval mode for consistency
    pred1 = model.predict(items)
    pred2 = model.predict(items)

    # Should be identical in eval mode
    for p1, p2 in zip(pred1, pred2):
        assert p1.predicted_class == p2.predicted_class
        assert abs(p1.confidence - p2.confidence) < 1e-5


def test_forced_choice_predict_and_predict_proba_agreement(test_items):
    """Test that predict and predict_proba agree."""
    items = test_items(n=10)
    labels = ["option_a"] * 5 + ["option_b"] * 5

    config = ForcedChoiceModelConfig(num_epochs=1, batch_size=2, device="cpu")
    model = ForcedChoiceModel(config=config)
    model.train(items, labels)

    predictions = model.predict(items)
    proba = model.predict_proba(items)

    for i, pred in enumerate(predictions):
        # Check predicted class matches max probability
        if pred.predicted_class == "option_a":
            assert proba[i, 0] >= proba[i, 1]
        else:
            assert proba[i, 1] >= proba[i, 0]

        # Check confidence matches probability
        expected_confidence = max(proba[i, 0], proba[i, 1])
        assert abs(pred.confidence - expected_confidence) < 1e-5


def test_forced_choice_max_length_truncation(test_items):
    """Test that long sequences are truncated."""
    from uuid import uuid4

    from bead.items.models import Item

    # Create item with very long text
    long_text = " ".join(["word"] * 200)
    items = [
        Item(
            item_template_id=uuid4(),
            rendered_elements={"option_a": long_text, "option_b": long_text},
        )
    ]
    labels = ["option_a"]

    config = ForcedChoiceModelConfig(
        max_length=32, num_epochs=1, batch_size=1, device="cpu"
    )
    model = ForcedChoiceModel(config=config)
    metrics = model.train(items, labels)

    # Should handle without error
    assert model._is_fitted is True


def test_forced_choice_incomplete_validation_data_raises(test_items):
    """Test incomplete validation data raises error."""
    train_items = test_items(n=10)
    train_labels = ["option_a"] * 5 + ["option_b"] * 5
    val_items = test_items(n=5)

    model = ForcedChoiceModel()

    with pytest.raises(ValueError, match="Both validation_items and validation_labels"):
        model.train(
            train_items,
            train_labels,
            validation_items=val_items,
            validation_labels=None,
        )


def test_forced_choice_validation_mismatch_raises(test_items):
    """Test validation items/labels mismatch raises error."""
    train_items = test_items(n=10)
    train_labels = ["option_a"] * 5 + ["option_b"] * 5

    val_items = test_items(n=10)
    val_labels = ["option_a"] * 5  # Wrong length

    model = ForcedChoiceModel()

    with pytest.raises(ValueError, match="Number of validation items"):
        model.train(
            train_items,
            train_labels,
            validation_items=val_items,
            validation_labels=val_labels,
        )


def test_forced_choice_small_batch_with_odd_size(test_items):
    """Test batch processing with odd-sized dataset."""
    items = test_items(n=7)  # Not divisible by batch size
    labels = ["option_a"] * 4 + ["option_b"] * 3

    config = ForcedChoiceModelConfig(num_epochs=1, batch_size=3, device="cpu")
    model = ForcedChoiceModel(config=config)
    metrics = model.train(items, labels)

    # Should handle correctly
    assert model._is_fitted is True
    predictions = model.predict(items)
    assert len(predictions) == 7


def test_forced_choice_tokenizer_special_tokens(test_items):
    """Test that tokenizer properly handles special tokens."""
    items = test_items(n=2)
    labels = ["option_a", "option_b"]
    config = ForcedChoiceModelConfig(num_epochs=1, device="cpu")
    model = ForcedChoiceModel(config=config)
    model.train(items, labels)

    embeddings = model._prepare_inputs(items)

    # Check that embeddings are produced correctly
    assert embeddings.shape[0] == 2
    assert embeddings.shape[1] > 0  # Has embedding dimension


def test_forced_choice_gradients_update_parameters(test_items):
    """Test that training actually updates model parameters."""
    items = test_items(n=10)
    labels = ["option_a"] * 5 + ["option_b"] * 5

    config = ForcedChoiceModelConfig(num_epochs=1, batch_size=5, device="cpu")
    model = ForcedChoiceModel(config=config)

    # Train first to initialize classifier
    model.train(items, labels)

    # Get initial parameters after first training
    initial_params = [p.clone().detach() for p in model.classifier_head.parameters()]

    # Train again
    model.train(items, labels)

    # Get final parameters
    final_params = list(model.classifier_head.parameters())

    # At least some parameters should have changed
    params_changed = any(
        not torch.allclose(p1.detach(), p2.detach())
        for p1, p2 in zip(initial_params, final_params)
    )
    assert params_changed, "Training should update model parameters"
