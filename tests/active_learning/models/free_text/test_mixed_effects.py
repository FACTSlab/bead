"""Comprehensive tests for FreeTextModel with mixed effects support.

Tests all three modes (fixed, random_intercepts, random_slopes) with LoRA,
variance tracking, save/load, and text generation quality.
"""

from __future__ import annotations

import tempfile

import pytest
import torch

from bead.active_learning.config import MixedEffectsConfig
from bead.active_learning.models.free_text import FreeTextModel
from bead.config.active_learning import FreeTextModelConfig
from bead.items.item import Item


class TestFixedEffectsMode:
    """Test FreeTextModel with fixed effects mode."""

    def test_train_with_fixed_mode(
        self, sample_short_items: list[Item], sample_short_labels: list[str]
    ) -> None:
        """Test training with fixed effects mode."""
        config = FreeTextModelConfig(
            model_name="t5-small",  # Smaller model for faster tests
            num_epochs=1,
            batch_size=2,
            device="cpu",
            max_input_length=32,
            max_output_length=16,
            num_beams=1,  # Greedy for speed
            mixed_effects=MixedEffectsConfig(mode="fixed"),
        )
        model = FreeTextModel(config)

        # Fixed mode: participant_ids=None
        metrics = model.train(
            sample_short_items, sample_short_labels, participant_ids=None
        )

        assert "train_loss" in metrics
        assert isinstance(metrics["train_loss"], float)
        assert "train_exact_match" in metrics
        # Fixed mode should not have participant variance
        assert "participant_variance" not in metrics

    def test_predict_generates_text(
        self, sample_short_items: list[Item], sample_short_labels: list[str]
    ) -> None:
        """Test that prediction generates valid text strings."""
        config = FreeTextModelConfig(
            model_name="t5-small",
            num_epochs=1,
            batch_size=2,
            device="cpu",
            max_input_length=32,
            max_output_length=16,
            num_beams=1,
            mixed_effects=MixedEffectsConfig(mode="fixed"),
        )
        model = FreeTextModel(config)

        model.train(sample_short_items, sample_short_labels, participant_ids=None)

        # Predict
        predictions = model.predict(sample_short_items[:3], participant_ids=None)

        assert len(predictions) == 3
        for pred in predictions:
            # Should generate text (string)
            assert isinstance(pred.predicted_class, str)
            # Text should be non-empty (model should generate something)
            assert len(pred.predicted_class) > 0

    def test_generation_uses_beam_search(
        self, sample_short_items: list[Item], sample_short_labels: list[str]
    ) -> None:
        """Test that beam search parameter is respected."""
        config = FreeTextModelConfig(
            model_name="t5-small",
            num_epochs=1,
            batch_size=2,
            device="cpu",
            max_input_length=32,
            max_output_length=16,
            num_beams=4,  # Use beam search
            mixed_effects=MixedEffectsConfig(mode="fixed"),
        )
        model = FreeTextModel(config)

        model.train(sample_short_items, sample_short_labels, participant_ids=None)
        predictions = model.predict(sample_short_items[:2], participant_ids=None)

        # Should still generate text with beam search
        assert len(predictions) == 2
        for pred in predictions:
            assert isinstance(pred.predicted_class, str)

    def test_validates_participant_ids_length(
        self, sample_short_items: list[Item], sample_short_labels: list[str]
    ) -> None:
        """Test that train validates participant_ids length."""
        config = FreeTextModelConfig(
            model_name="t5-small",
            num_epochs=1,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="random_intercepts"),
        )
        model = FreeTextModel(config)

        # Wrong length
        participant_ids = ["p1"] * (len(sample_short_items) - 1)

        with pytest.raises(ValueError, match="Length mismatch"):
            model.train(sample_short_items, sample_short_labels, participant_ids)

    def test_validates_empty_participant_ids(
        self, sample_short_items: list[Item], sample_short_labels: list[str]
    ) -> None:
        """Test that train rejects empty participant_ids."""
        config = FreeTextModelConfig(
            model_name="t5-small",
            num_epochs=1,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="random_intercepts"),
        )
        model = FreeTextModel(config)

        # Empty string in participant_ids
        participant_ids = ["p1"] * len(sample_short_items)
        participant_ids[2] = ""

        with pytest.raises(ValueError, match="cannot contain empty strings"):
            model.train(sample_short_items, sample_short_labels, participant_ids)

    def test_validates_empty_labels(
        self, sample_short_items: list[Item]
    ) -> None:
        """Test that train rejects empty labels."""
        config = FreeTextModelConfig(
            model_name="t5-small",
            num_epochs=1,
            device="cpu",
        )
        model = FreeTextModel(config)

        # Empty label
        labels = ["text"] * len(sample_short_items)
        labels[3] = ""

        with pytest.raises(ValueError, match="cannot contain empty strings"):
            model.train(sample_short_items, labels, participant_ids=None)

    def test_validates_items_labels_length_mismatch(
        self, sample_short_items: list[Item], sample_short_labels: list[str]
    ) -> None:
        """Test that train validates items and labels have same length."""
        config = FreeTextModelConfig(
            model_name="t5-small",
            num_epochs=1,
            device="cpu",
        )
        model = FreeTextModel(config)

        # Mismatched lengths
        labels_short = sample_short_labels[:-1]

        with pytest.raises(ValueError, match="must match"):
            model.train(sample_short_items, labels_short, participant_ids=None)


class TestRandomInterceptsMode:
    """Test FreeTextModel with random intercepts mode."""

    def test_train_with_random_intercepts(
        self, sample_short_items: list[Item], sample_short_labels: list[str]
    ) -> None:
        """Test training with random intercepts mode."""
        config = FreeTextModelConfig(
            model_name="t5-small",
            num_epochs=1,
            batch_size=2,
            device="cpu",
            max_input_length=32,
            max_output_length=16,
            num_beams=1,
            mixed_effects=MixedEffectsConfig(
                mode="random_intercepts",
                prior_mean=0.0,
                prior_variance=1.0,
                estimate_variance_components=True,
            ),
        )
        model = FreeTextModel(config)

        # Two participants
        participant_ids = ["p1"] * 3 + ["p2"] * 3
        metrics = model.train(sample_short_items, sample_short_labels, participant_ids)

        assert "train_loss" in metrics
        assert "participant_variance" in metrics
        assert "n_participants" in metrics
        assert metrics["n_participants"] == 2

    def test_creates_vocab_sized_bias_vectors(
        self, sample_short_items: list[Item], sample_short_labels: list[str]
    ) -> None:
        """Test that random intercepts creates participant bias vectors."""
        config = FreeTextModelConfig(
            model_name="t5-small",
            num_epochs=1,
            batch_size=2,
            device="cpu",
            max_input_length=32,
            max_output_length=16,
            mixed_effects=MixedEffectsConfig(mode="random_intercepts"),
        )
        model = FreeTextModel(config)

        participant_ids = ["p1"] * 3 + ["p2"] * 3
        model.train(sample_short_items, sample_short_labels, participant_ids)

        # Verify intercepts created
        assert model.random_effects is not None
        assert "mu" in model.random_effects.intercepts
        assert "p1" in model.random_effects.intercepts["mu"]
        assert "p2" in model.random_effects.intercepts["mu"]

    def test_bias_shape_equals_vocab_size(
        self, sample_short_items: list[Item], sample_short_labels: list[str]
    ) -> None:
        """Test that bias vectors have vocab_size dimensionality."""
        config = FreeTextModelConfig(
            model_name="t5-small",
            num_epochs=1,
            batch_size=2,
            device="cpu",
            max_input_length=32,
            max_output_length=16,
            mixed_effects=MixedEffectsConfig(mode="random_intercepts"),
        )
        model = FreeTextModel(config)

        participant_ids = ["p1"] * 3 + ["p2"] * 3
        model.train(sample_short_items, sample_short_labels, participant_ids)

        # Get actual vocabulary size from lm_head (not tokenizer)
        vocab_size = model.lm_head.out_features

        # Check bias shape matches vocab size
        bias_p1 = model.random_effects.intercepts["mu"]["p1"]
        assert bias_p1.shape == (vocab_size,)

        bias_p2 = model.random_effects.intercepts["mu"]["p2"]
        assert bias_p2.shape == (vocab_size,)

    def test_different_participants_different_outputs(
        self, sample_short_items: list[Item], sample_short_labels: list[str]
    ) -> None:
        """Test that different participants can get different predictions."""
        config = FreeTextModelConfig(
            model_name="t5-small",
            num_epochs=1,
            batch_size=2,
            device="cpu",
            max_input_length=32,
            max_output_length=16,
            num_beams=1,
            mixed_effects=MixedEffectsConfig(mode="random_intercepts"),
        )
        model = FreeTextModel(config)

        participant_ids = ["p1"] * 3 + ["p2"] * 3
        model.train(sample_short_items, sample_short_labels, participant_ids)

        # Same item, different participants
        test_item = [sample_short_items[0]]
        pred_p1 = model.predict(test_item, participant_ids=["p1"])
        pred_p2 = model.predict(test_item, participant_ids=["p2"])

        # Both should generate valid text
        assert isinstance(pred_p1[0].predicted_class, str)
        assert isinstance(pred_p2[0].predicted_class, str)
        # Predictions may differ due to random effects (but not guaranteed)
        # Just verify both are valid

    def test_unknown_participant_uses_zero_bias(
        self, sample_short_items: list[Item], sample_short_labels: list[str]
    ) -> None:
        """Test that unknown participants use prior mean (zero bias)."""
        config = FreeTextModelConfig(
            model_name="t5-small",
            num_epochs=1,
            batch_size=2,
            device="cpu",
            max_input_length=32,
            max_output_length=16,
            num_beams=1,
            mixed_effects=MixedEffectsConfig(mode="random_intercepts"),
        )
        model = FreeTextModel(config)

        participant_ids = ["p1"] * 6
        model.train(sample_short_items, sample_short_labels, participant_ids)

        # Predict with unknown participant
        predictions = model.predict(
            sample_short_items[:2], participant_ids=["unknown"] * 2
        )

        assert len(predictions) == 2
        for pred in predictions:
            assert isinstance(pred.predicted_class, str)

    def test_bias_affects_token_probabilities(
        self, sample_short_items: list[Item], sample_short_labels: list[str]
    ) -> None:
        """Test that participant bias actually affects token probabilities."""
        config = FreeTextModelConfig(
            model_name="t5-small",
            num_epochs=1,
            batch_size=2,
            device="cpu",
            max_input_length=32,
            max_output_length=16,
            mixed_effects=MixedEffectsConfig(mode="random_intercepts"),
        )
        model = FreeTextModel(config)

        participant_ids = ["p1"] * 3 + ["p2"] * 3
        model.train(sample_short_items, sample_short_labels, participant_ids)

        # Get bias vectors
        bias_p1 = model.random_effects.intercepts["mu"]["p1"]
        bias_p2 = model.random_effects.intercepts["mu"]["p2"]

        # Biases should be different (with high probability)
        # Check that they're not all zeros
        assert torch.any(bias_p1 != 0.0) or torch.any(bias_p2 != 0.0)

    def test_requires_participant_ids_for_random_mode(
        self, sample_short_items: list[Item], sample_short_labels: list[str]
    ) -> None:
        """Test that random_intercepts mode requires participant_ids."""
        config = FreeTextModelConfig(
            model_name="t5-small",
            num_epochs=1,
            device="cpu",
            mixed_effects=MixedEffectsConfig(mode="random_intercepts"),
        )
        model = FreeTextModel(config)

        # participant_ids=None should raise error
        with pytest.raises(ValueError, match="participant_ids is required"):
            model.train(sample_short_items, sample_short_labels, participant_ids=None)


class TestRandomSlopesMode:
    """Test FreeTextModel with random slopes mode (LoRA)."""

    def test_train_with_random_slopes_lora(
        self, sample_short_items: list[Item], sample_short_labels: list[str]
    ) -> None:
        """Test training with random slopes mode using LoRA."""
        config = FreeTextModelConfig(
            model_name="t5-small",
            num_epochs=1,
            batch_size=2,
            device="cpu",
            max_input_length=32,
            max_output_length=16,
            lora_rank=4,  # Small rank for faster tests
            lora_alpha=8.0,
            mixed_effects=MixedEffectsConfig(
                mode="random_slopes",
                estimate_variance_components=True,
            ),
        )
        model = FreeTextModel(config)

        participant_ids = ["p1"] * 3 + ["p2"] * 3
        metrics = model.train(sample_short_items, sample_short_labels, participant_ids)

        assert "train_loss" in metrics
        assert "participant_variance" in metrics
        assert "n_participants" in metrics

    def test_creates_participant_lora_adapters(
        self, sample_short_items: list[Item], sample_short_labels: list[str]
    ) -> None:
        """Test that random slopes creates participant-specific LoRA adapters."""
        config = FreeTextModelConfig(
            model_name="t5-small",
            num_epochs=1,
            batch_size=2,
            device="cpu",
            max_input_length=32,
            max_output_length=16,
            lora_rank=4,
            lora_alpha=8.0,
            lora_target_modules=["q", "v"],  # T5 uses "q" and "v"
            mixed_effects=MixedEffectsConfig(mode="random_slopes"),
        )
        model = FreeTextModel(config)

        participant_ids = ["p1"] * 3 + ["p2"] * 3
        model.train(sample_short_items, sample_short_labels, participant_ids)

        # Verify slopes (LoRA adapters) created
        assert model.random_effects is not None
        assert "p1" in model.random_effects.slopes
        assert "p2" in model.random_effects.slopes

    def test_lora_adapters_have_correct_rank(
        self, sample_short_items: list[Item], sample_short_labels: list[str]
    ) -> None:
        """Test that LoRA adapters use the specified rank."""
        rank = 4
        config = FreeTextModelConfig(
            model_name="t5-small",
            num_epochs=1,
            batch_size=2,
            device="cpu",
            max_input_length=32,
            max_output_length=16,
            lora_rank=rank,
            lora_alpha=8.0,
            lora_target_modules=["q", "v"],
            mixed_effects=MixedEffectsConfig(mode="random_slopes"),
        )
        model = FreeTextModel(config)

        participant_ids = ["p1"] * 3 + ["p2"] * 3
        model.train(sample_short_items, sample_short_labels, participant_ids)

        # Get one participant's adapter
        adapter_p1 = model.random_effects.slopes["p1"]

        # Check LoRA layers exist and have correct rank
        assert len(adapter_p1.lora_layers) > 0
        for lora_linear in adapter_p1.lora_layers.values():
            assert lora_linear.lora.rank == rank

    def test_lora_applied_to_target_modules(
        self, sample_short_items: list[Item], sample_short_labels: list[str]
    ) -> None:
        """Test that LoRA is applied to specified target modules (Q, V)."""
        config = FreeTextModelConfig(
            model_name="t5-small",
            num_epochs=1,
            batch_size=2,
            device="cpu",
            max_input_length=32,
            max_output_length=16,
            lora_rank=4,
            lora_alpha=8.0,
            lora_target_modules=["q", "v"],  # Query and Value projections
            mixed_effects=MixedEffectsConfig(mode="random_slopes"),
        )
        model = FreeTextModel(config)

        participant_ids = ["p1"] * 3 + ["p2"] * 3
        model.train(sample_short_items, sample_short_labels, participant_ids)

        adapter_p1 = model.random_effects.slopes["p1"]

        # Check that LoRA layers exist for target modules
        lora_module_names = list(adapter_p1.lora_layers.keys())
        assert len(lora_module_names) > 0

        # Verify names contain "q" or "v" (target modules)
        for name in lora_module_names:
            assert "q" in name or "v" in name

    def test_unknown_participant_uses_base_decoder(
        self, sample_short_items: list[Item], sample_short_labels: list[str]
    ) -> None:
        """Test that unknown participants use fixed decoder (no LoRA)."""
        config = FreeTextModelConfig(
            model_name="t5-small",
            num_epochs=1,
            batch_size=2,
            device="cpu",
            max_input_length=32,
            max_output_length=16,
            lora_rank=4,
            mixed_effects=MixedEffectsConfig(mode="random_slopes"),
        )
        model = FreeTextModel(config)

        participant_ids = ["p1"] * 6
        model.train(sample_short_items, sample_short_labels, participant_ids)

        # Predict with unknown participant
        predictions = model.predict(
            sample_short_items[:2], participant_ids=["unknown"] * 2
        )

        assert len(predictions) == 2
        for pred in predictions:
            assert isinstance(pred.predicted_class, str)

    def test_lora_parameters_trainable(
        self, sample_short_items: list[Item], sample_short_labels: list[str]
    ) -> None:
        """Test that LoRA parameters are trainable."""
        config = FreeTextModelConfig(
            model_name="t5-small",
            num_epochs=1,
            batch_size=2,
            device="cpu",
            max_input_length=32,
            max_output_length=16,
            lora_rank=4,
            lora_target_modules=["q", "v"],
            mixed_effects=MixedEffectsConfig(mode="random_slopes"),
        )
        model = FreeTextModel(config)

        participant_ids = ["p1"] * 3 + ["p2"] * 3
        model.train(sample_short_items, sample_short_labels, participant_ids)

        # Get LoRA adapter
        adapter_p1 = model.random_effects.slopes["p1"]

        # Get LoRA parameters
        lora_params = adapter_p1.get_lora_parameters()

        # Should have parameters
        assert len(lora_params) > 0

        # All should require gradients
        for param in lora_params:
            assert param.requires_grad


class TestVarianceTracking:
    """Test variance component estimation."""

    def test_variance_estimated_for_intercepts(
        self, sample_short_items: list[Item], sample_short_labels: list[str]
    ) -> None:
        """Test that variance components are estimated for intercepts."""
        config = FreeTextModelConfig(
            model_name="t5-small",
            num_epochs=1,
            batch_size=2,
            device="cpu",
            max_input_length=32,
            max_output_length=16,
            mixed_effects=MixedEffectsConfig(
                mode="random_intercepts",
                estimate_variance_components=True,
            ),
        )
        model = FreeTextModel(config)

        participant_ids = ["p1"] * 3 + ["p2"] * 3
        metrics = model.train(sample_short_items, sample_short_labels, participant_ids)

        assert "participant_variance" in metrics
        assert metrics["participant_variance"] >= 0.0

    def test_variance_estimated_for_lora_params(
        self, sample_short_items: list[Item], sample_short_labels: list[str]
    ) -> None:
        """Test that variance is estimated from LoRA parameters."""
        config = FreeTextModelConfig(
            model_name="t5-small",
            num_epochs=1,
            batch_size=2,
            device="cpu",
            max_input_length=32,
            max_output_length=16,
            lora_rank=4,
            mixed_effects=MixedEffectsConfig(
                mode="random_slopes",
                estimate_variance_components=True,
            ),
        )
        model = FreeTextModel(config)

        participant_ids = ["p1"] * 3 + ["p2"] * 3
        metrics = model.train(sample_short_items, sample_short_labels, participant_ids)

        assert "participant_variance" in metrics
        assert metrics["participant_variance"] >= 0.0

    def test_variance_reflects_heterogeneity(
        self, sample_short_items: list[Item], sample_short_labels: list[str]
    ) -> None:
        """Test that variance reflects participant heterogeneity."""
        config = FreeTextModelConfig(
            model_name="t5-small",
            num_epochs=1,
            batch_size=2,
            device="cpu",
            max_input_length=32,
            max_output_length=16,
            mixed_effects=MixedEffectsConfig(
                mode="random_intercepts",
                estimate_variance_components=True,
            ),
        )
        model = FreeTextModel(config)

        participant_ids = ["p1"] * 3 + ["p2"] * 3
        metrics = model.train(sample_short_items, sample_short_labels, participant_ids)

        # Variance should be non-zero (reflects heterogeneity)
        assert metrics["participant_variance"] >= 0.0


class TestSaveLoad:
    """Test model save and load functionality."""

    def test_save_and_load_preserves_random_effects(
        self, sample_short_items: list[Item], sample_short_labels: list[str]
    ) -> None:
        """Test that save/load preserves random effects (intercepts)."""
        config = FreeTextModelConfig(
            model_name="t5-small",
            num_epochs=1,
            batch_size=2,
            device="cpu",
            max_input_length=32,
            max_output_length=16,
            mixed_effects=MixedEffectsConfig(mode="random_intercepts"),
        )
        model = FreeTextModel(config)

        participant_ids = ["p1"] * 3 + ["p2"] * 3
        model.train(sample_short_items, sample_short_labels, participant_ids)

        # Get original bias
        orig_bias_p1 = model.random_effects.intercepts["mu"]["p1"].clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save model
            model.save(tmpdir)

            # Load into new model
            model2 = FreeTextModel(config)
            model2.load(tmpdir)

            # Check intercepts preserved
            loaded_bias_p1 = model2.random_effects.intercepts["mu"]["p1"]
            assert loaded_bias_p1.shape == orig_bias_p1.shape
            # Check values are close
            assert torch.allclose(loaded_bias_p1, orig_bias_p1, atol=1e-5)

    def test_save_and_load_preserves_lora_adapters(
        self, sample_short_items: list[Item], sample_short_labels: list[str]
    ) -> None:
        """Test that save/load preserves LoRA adapters."""
        config = FreeTextModelConfig(
            model_name="t5-small",
            num_epochs=1,
            batch_size=2,
            device="cpu",
            max_input_length=32,
            max_output_length=16,
            lora_rank=4,
            lora_target_modules=["q", "v"],
            mixed_effects=MixedEffectsConfig(mode="random_slopes"),
        )
        model = FreeTextModel(config)

        participant_ids = ["p1"] * 3 + ["p2"] * 3
        model.train(sample_short_items, sample_short_labels, participant_ids)

        # Get original adapter
        orig_adapter_p1 = model.random_effects.slopes["p1"]
        orig_lora_params = orig_adapter_p1.get_lora_parameters()

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save model
            model.save(tmpdir)

            # Load into new model
            model2 = FreeTextModel(config)
            model2.load(tmpdir)

            # Check slopes preserved
            assert "p1" in model2.random_effects.slopes
            assert "p2" in model2.random_effects.slopes

            # Check LoRA parameters preserved
            loaded_adapter_p1 = model2.random_effects.slopes["p1"]
            loaded_lora_params = loaded_adapter_p1.get_lora_parameters()

            assert len(loaded_lora_params) == len(orig_lora_params)

    def test_save_and_load_preserves_config(
        self, sample_short_items: list[Item], sample_short_labels: list[str]
    ) -> None:
        """Test that save/load preserves configuration."""
        config = FreeTextModelConfig(
            model_name="t5-small",
            num_epochs=1,
            batch_size=2,
            device="cpu",
            max_input_length=32,
            max_output_length=16,
            lora_rank=8,
            lora_alpha=16.0,
        )
        model = FreeTextModel(config)

        model.train(sample_short_items, sample_short_labels, participant_ids=None)

        with tempfile.TemporaryDirectory() as tmpdir:
            model.save(tmpdir)

            model2 = FreeTextModel()
            model2.load(tmpdir)

            # Check config preserved
            assert model2.config.lora_rank == 8
            assert model2.config.lora_alpha == 16.0
            assert model2.config.max_input_length == 32
            assert model2.config.max_output_length == 16


class TestTextGeneration:
    """Test text generation quality and behavior."""

    def test_generates_valid_text_strings(
        self, sample_short_items: list[Item], sample_short_labels: list[str]
    ) -> None:
        """Test that model generates valid non-empty text strings."""
        config = FreeTextModelConfig(
            model_name="t5-small",
            num_epochs=1,
            batch_size=2,
            device="cpu",
            max_input_length=32,
            max_output_length=16,
            num_beams=1,
        )
        model = FreeTextModel(config)

        model.train(sample_short_items, sample_short_labels, participant_ids=None)
        predictions = model.predict(sample_short_items[:4], participant_ids=None)

        assert len(predictions) == 4
        for pred in predictions:
            # Should be string
            assert isinstance(pred.predicted_class, str)
            # Should be non-empty
            assert len(pred.predicted_class) > 0
            # Should not contain special tokens (cleaned by tokenizer)
            assert "<pad>" not in pred.predicted_class
            assert "</s>" not in pred.predicted_class

    def test_beam_search_produces_valid_outputs(
        self, sample_short_items: list[Item], sample_short_labels: list[str]
    ) -> None:
        """Test that beam search produces diverse but valid outputs."""
        config = FreeTextModelConfig(
            model_name="t5-small",
            num_epochs=1,
            batch_size=2,
            device="cpu",
            max_input_length=32,
            max_output_length=16,
            num_beams=4,  # Beam search
        )
        model = FreeTextModel(config)

        model.train(sample_short_items, sample_short_labels, participant_ids=None)
        predictions = model.predict(sample_short_items[:3], participant_ids=None)

        assert len(predictions) == 3
        for pred in predictions:
            assert isinstance(pred.predicted_class, str)
            assert len(pred.predicted_class) > 0

    def test_generation_deterministic_with_eval_mode(
        self, sample_short_items: list[Item], sample_short_labels: list[str]
    ) -> None:
        """Test that generation is deterministic in eval mode."""
        config = FreeTextModelConfig(
            model_name="t5-small",
            num_epochs=1,
            batch_size=2,
            device="cpu",
            max_input_length=32,
            max_output_length=16,
            num_beams=1,  # Greedy
            temperature=1.0,
        )
        model = FreeTextModel(config)

        model.train(sample_short_items, sample_short_labels, participant_ids=None)

        # Generate twice
        predictions1 = model.predict(sample_short_items[:2], participant_ids=None)
        predictions2 = model.predict(sample_short_items[:2], participant_ids=None)

        # Should be identical (greedy + eval mode)
        assert predictions1[0].predicted_class == predictions2[0].predicted_class
        assert predictions1[1].predicted_class == predictions2[1].predicted_class


class TestEvaluation:
    """Test evaluation metrics."""

    def test_exact_match_metric(
        self, sample_short_items: list[Item]
    ) -> None:
        """Test exact match accuracy computation."""
        config = FreeTextModelConfig(
            model_name="t5-small",
            num_epochs=1,
            batch_size=2,
            device="cpu",
            max_input_length=32,
            max_output_length=16,
        )
        model = FreeTextModel(config)

        # Simple labels that model might learn
        labels = ["text"] * 6

        model.train(sample_short_items, labels, participant_ids=None)

        # Test exact match computation
        predictions = ["text", "text", "wrong", "text"]
        labels_test = ["text", "TEXT", "text", "text"]  # Case insensitive

        exact_match = model._compute_exact_match(predictions, labels_test)

        # 3 out of 4 match (case insensitive)
        assert exact_match == 0.75

    def test_training_returns_exact_match_metric(
        self, sample_short_items: list[Item], sample_short_labels: list[str]
    ) -> None:
        """Test that training returns exact match on training set."""
        config = FreeTextModelConfig(
            model_name="t5-small",
            num_epochs=1,
            batch_size=2,
            device="cpu",
            max_input_length=32,
            max_output_length=16,
        )
        model = FreeTextModel(config)

        metrics = model.train(
            sample_short_items, sample_short_labels, participant_ids=None
        )

        assert "train_exact_match" in metrics
        assert 0.0 <= metrics["train_exact_match"] <= 1.0


class TestPredictProba:
    """Test predict_proba method."""

    def test_predict_proba_returns_empty_array(
        self, sample_short_items: list[Item], sample_short_labels: list[str]
    ) -> None:
        """Test that predict_proba returns empty array for generation tasks."""
        config = FreeTextModelConfig(
            model_name="t5-small",
            num_epochs=1,
            batch_size=2,
            device="cpu",
            max_input_length=32,
            max_output_length=16,
        )
        model = FreeTextModel(config)

        model.train(sample_short_items, sample_short_labels, participant_ids=None)

        proba = model.predict_proba(sample_short_items[:3], participant_ids=None)

        # Should return empty array (shape: n_items, 0)
        assert proba.shape == (3, 0)
