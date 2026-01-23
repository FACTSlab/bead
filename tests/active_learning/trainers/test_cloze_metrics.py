"""Tests for compute_cloze_metrics function.

This module tests the cloze metrics computation including accuracy,
top-k accuracy, and perplexity at masked positions.
"""

from __future__ import annotations

import numpy as np
import pytest
from transformers import EvalPrediction

from bead.active_learning.trainers.metrics import compute_cloze_metrics


class TestComputeClozeMetrics:
    """Tests for compute_cloze_metrics function."""

    @pytest.fixture
    def mock_tokenizer(self, mocker):
        """Create a mock tokenizer for tests."""
        tokenizer = mocker.Mock()
        tokenizer.vocab_size = 30522
        return tokenizer

    def test_perfect_predictions(self, mock_tokenizer):
        """Test accuracy=1.0 when all predictions are correct."""
        # 2 samples, 5 seq_len, vocab_size=100
        predictions = np.zeros((2, 5, 100))
        labels = np.full((2, 5), -100)

        # Set up correct predictions
        predictions[0, 2, 42] = 10.0  # High logit for token 42 at pos 2
        labels[0, 2] = 42
        predictions[1, 1, 17] = 10.0  # High logit for token 17 at pos 1
        labels[1, 1] = 17

        eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)
        metrics = compute_cloze_metrics(eval_pred, mock_tokenizer)

        assert metrics["accuracy"] == 1.0
        assert metrics["top_3_accuracy"] == 1.0
        assert metrics["top_5_accuracy"] == 1.0

    def test_zero_accuracy(self, mock_tokenizer):
        """Test accuracy=0.0 when all predictions are wrong."""
        predictions = np.zeros((2, 5, 100))
        labels = np.full((2, 5), -100)

        # Set up wrong predictions
        predictions[0, 2, 99] = 10.0  # Predicts 99
        labels[0, 2] = 42  # Correct is 42

        eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)
        metrics = compute_cloze_metrics(eval_pred, mock_tokenizer)

        assert metrics["accuracy"] == 0.0

    def test_topk_accuracy_in_top3(self, mock_tokenizer):
        """Test top-k accuracy when correct token is in top-k but not top-1."""
        predictions = np.zeros((1, 5, 100))
        labels = np.full((1, 5), -100)

        # Set up: correct token (42) is 3rd highest
        predictions[0, 2, 10] = 10.0  # Highest
        predictions[0, 2, 20] = 9.0  # 2nd
        predictions[0, 2, 42] = 8.0  # 3rd (correct)
        labels[0, 2] = 42

        eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)
        metrics = compute_cloze_metrics(eval_pred, mock_tokenizer)

        assert metrics["accuracy"] == 0.0  # Not top-1
        assert metrics["top_3_accuracy"] == 1.0  # In top-3
        assert metrics["top_5_accuracy"] == 1.0  # In top-5

    def test_topk_accuracy_in_top5_not_top3(self, mock_tokenizer):
        """Test top-5 accuracy when correct token is 4th or 5th."""
        predictions = np.zeros((1, 5, 100))
        labels = np.full((1, 5), -100)

        # Set up: correct token (42) is 5th highest
        predictions[0, 2, 10] = 10.0  # 1st
        predictions[0, 2, 20] = 9.0  # 2nd
        predictions[0, 2, 30] = 8.0  # 3rd
        predictions[0, 2, 40] = 7.0  # 4th
        predictions[0, 2, 42] = 6.0  # 5th (correct)
        labels[0, 2] = 42

        eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)
        metrics = compute_cloze_metrics(eval_pred, mock_tokenizer)

        assert metrics["accuracy"] == 0.0  # Not top-1
        assert metrics["top_3_accuracy"] == 0.0  # Not in top-3
        assert metrics["top_5_accuracy"] == 1.0  # In top-5

    def test_perplexity_low_for_confident_correct(self, mock_tokenizer):
        """Test perplexity is low when model is confident and correct."""
        predictions = np.zeros((1, 5, 100))
        labels = np.full((1, 5), -100)

        # Very confident correct prediction
        predictions[0, 2, 42] = 100.0  # Very high logit
        labels[0, 2] = 42

        eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)
        metrics = compute_cloze_metrics(eval_pred, mock_tokenizer)

        # Perplexity should be close to 1 (perfect)
        assert 1.0 <= metrics["perplexity"] < 2.0

    def test_perplexity_high_for_confident_wrong(self, mock_tokenizer):
        """Test perplexity is high when model is confident but wrong."""
        predictions = np.zeros((1, 5, 100))
        labels = np.full((1, 5), -100)

        # Confident but wrong prediction
        predictions[0, 2, 99] = 100.0  # High logit for wrong token
        labels[0, 2] = 42  # Correct token

        eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)
        metrics = compute_cloze_metrics(eval_pred, mock_tokenizer)

        # Perplexity should be very high
        assert metrics["perplexity"] > 10.0

    def test_no_masked_positions(self, mock_tokenizer):
        """Test graceful handling when no positions are masked."""
        predictions = np.random.randn(2, 5, 100)
        labels = np.full((2, 5), -100)  # All -100, no masks

        eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)
        metrics = compute_cloze_metrics(eval_pred, mock_tokenizer)

        assert metrics["accuracy"] == 0.0
        assert metrics["top_3_accuracy"] == 0.0
        assert metrics["top_5_accuracy"] == 0.0
        assert metrics["perplexity"] == float("inf")

    def test_multiple_masks_per_sample(self, mock_tokenizer):
        """Test handling of multiple masked positions per sample."""
        predictions = np.zeros((1, 5, 100))
        labels = np.full((1, 5), -100)

        # Two masked positions in one sample
        predictions[0, 1, 10] = 10.0  # Correct at pos 1
        labels[0, 1] = 10
        predictions[0, 3, 99] = 10.0  # Wrong at pos 3 (predicts 99)
        labels[0, 3] = 20  # Correct is 20

        eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)
        metrics = compute_cloze_metrics(eval_pred, mock_tokenizer)

        # 1 correct out of 2
        assert metrics["accuracy"] == 0.5

    def test_none_predictions(self, mock_tokenizer):
        """Test handling of None predictions."""
        eval_pred = EvalPrediction(predictions=None, label_ids=None)
        metrics = compute_cloze_metrics(eval_pred, mock_tokenizer)

        assert metrics["accuracy"] == 0.0
        assert metrics["perplexity"] == float("inf")

    def test_none_labels(self, mock_tokenizer):
        """Test handling of None labels."""
        predictions = np.random.randn(2, 5, 100)
        eval_pred = EvalPrediction(predictions=predictions, label_ids=None)
        metrics = compute_cloze_metrics(eval_pred, mock_tokenizer)

        assert metrics["accuracy"] == 0.0
        assert metrics["perplexity"] == float("inf")

    def test_wrong_predictions_shape(self, mock_tokenizer):
        """Test handling of predictions with wrong dimensions."""
        predictions = np.random.randn(10)  # 1D instead of 3D
        labels = np.full((2, 5), -100)

        eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)
        metrics = compute_cloze_metrics(eval_pred, mock_tokenizer)

        assert metrics["accuracy"] == 0.0
        assert metrics["perplexity"] == float("inf")

    def test_shape_mismatch(self, mock_tokenizer):
        """Test handling of shape mismatch between predictions and labels."""
        predictions = np.random.randn(2, 5, 100)
        labels = np.full((3, 6), -100)  # Different shape

        eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)
        metrics = compute_cloze_metrics(eval_pred, mock_tokenizer)

        assert metrics["accuracy"] == 0.0
        assert metrics["perplexity"] == float("inf")

    def test_mixed_correct_incorrect(self, mock_tokenizer):
        """Test with a mix of correct and incorrect predictions."""
        predictions = np.zeros((3, 5, 100))
        labels = np.full((3, 5), -100)

        # Sample 0, position 1: correct
        predictions[0, 1, 42] = 10.0
        labels[0, 1] = 42

        # Sample 1, position 2: incorrect
        predictions[1, 2, 99] = 10.0
        labels[1, 2] = 42

        # Sample 2, position 3: correct
        predictions[2, 3, 50] = 10.0
        labels[2, 3] = 50

        eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)
        metrics = compute_cloze_metrics(eval_pred, mock_tokenizer)

        # 2 out of 3 correct
        assert abs(metrics["accuracy"] - 2 / 3) < 1e-6

    def test_returns_all_metrics(self, mock_tokenizer):
        """Test that all expected metrics are returned."""
        predictions = np.zeros((1, 5, 100))
        labels = np.full((1, 5), -100)
        predictions[0, 2, 42] = 10.0
        labels[0, 2] = 42

        eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)
        metrics = compute_cloze_metrics(eval_pred, mock_tokenizer)

        assert "accuracy" in metrics
        assert "top_3_accuracy" in metrics
        assert "top_5_accuracy" in metrics
        assert "perplexity" in metrics

    def test_small_vocab_topk(self, mock_tokenizer):
        """Test top-k accuracy with vocab smaller than k."""
        # Vocab size 2 (smaller than top-5)
        predictions = np.zeros((1, 5, 2))
        labels = np.full((1, 5), -100)

        predictions[0, 2, 1] = 10.0  # Predict token 1
        labels[0, 2] = 0  # But correct is token 0

        eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)
        metrics = compute_cloze_metrics(eval_pred, mock_tokenizer)

        # With only 2 tokens, top-5 should still find the correct one
        assert metrics["accuracy"] == 0.0  # Not top-1
        assert metrics["top_5_accuracy"] == 1.0  # In top-5 (since vocab < 5)

    def test_uniform_predictions(self, mock_tokenizer):
        """Test with uniform (random) predictions."""
        np.random.seed(42)
        predictions = np.random.randn(2, 5, 100)
        labels = np.full((2, 5), -100)

        # Set up some masked positions
        labels[0, 2] = 42
        labels[1, 3] = 17

        eval_pred = EvalPrediction(predictions=predictions, label_ids=labels)
        metrics = compute_cloze_metrics(eval_pred, mock_tokenizer)

        # With random predictions, accuracy is low but metrics are valid floats
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert 0.0 <= metrics["top_3_accuracy"] <= 1.0
        assert 0.0 <= metrics["top_5_accuracy"] <= 1.0
        assert metrics["perplexity"] > 0  # Positive perplexity
