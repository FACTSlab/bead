"""Tests for ModelMetrics class."""

from __future__ import annotations

import numpy as np
import pytest

from bead.evaluation.model_metrics import ModelMetrics


class TestAccuracy:
    """Test accuracy calculation."""

    def test_perfect_predictions(self, perfect_predictions):
        """Test with perfect predictions."""
        accuracy = ModelMetrics.accuracy(
            perfect_predictions["y_true"], perfect_predictions["y_pred"]
        )
        assert accuracy == 1.0

    def test_all_wrong_predictions(self):
        """Test with all wrong predictions."""
        y_true = [0, 0, 0, 0, 0]
        y_pred = [1, 1, 1, 1, 1]

        accuracy = ModelMetrics.accuracy(y_true, y_pred)
        assert accuracy == 0.0

    def test_partial_accuracy(self, binary_predictions):
        """Test with partial accuracy."""
        accuracy = ModelMetrics.accuracy(
            binary_predictions["y_true"], binary_predictions["y_pred"]
        )

        # y_true = [0, 1, 1, 0, 1, 0, 1, 1, 0, 0]
        # y_pred = [0, 1, 0, 0, 1, 1, 1, 1, 0, 0]
        # Correct: 0, 1, _, 0, 1, _, 1, 1, 0, 0 = 8/10
        assert accuracy == 0.8

    def test_multiclass_accuracy(self, multi_class_predictions):
        """Test with multi-class classification."""
        accuracy = ModelMetrics.accuracy(
            multi_class_predictions["y_true"], multi_class_predictions["y_pred"]
        )

        # y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
        # y_pred = [0, 1, 2, 0, 1, 1, 0, 2, 2, 0]
        # Correct: 0, 1, 2, 0, 1, _, 0, _, 2, 0 = 8/10
        assert accuracy == 0.8

    def test_mismatched_lengths(self):
        """Test with mismatched lengths."""
        y_true = [0, 1, 1]
        y_pred = [0, 1]

        with pytest.raises(ValueError, match="must have same length"):
            ModelMetrics.accuracy(y_true, y_pred)

    def test_empty_lists(self):
        """Test with empty lists."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ModelMetrics.accuracy([], [])


class TestConfusionMatrix:
    """Test confusion matrix calculation."""

    def test_binary_confusion_matrix(self, binary_predictions):
        """Test binary classification confusion matrix."""
        cm = ModelMetrics.confusion_matrix(
            binary_predictions["y_true"], binary_predictions["y_pred"]
        )

        # Verify shape
        assert cm.shape == (2, 2)

        # y_true = [0, 1, 1, 0, 1, 0, 1, 1, 0, 0]
        # y_pred = [0, 1, 0, 0, 1, 1, 1, 1, 0, 0]

        # TN (0,0): predicted 0, true 0 → positions: 0, 3, 8, 9 = 4
        assert cm[0, 0] == 4

        # FP (0,1): predicted 1, true 0 → position: 5 = 1
        assert cm[0, 1] == 1

        # FN (1,0): predicted 0, true 1 → position: 2 = 1
        assert cm[1, 0] == 1

        # TP (1,1): predicted 1, true 1 → positions: 1, 4, 6, 7 = 4
        assert cm[1, 1] == 4

        # Verify diagonal = correct predictions
        assert np.trace(cm) == 8

    def test_multiclass_confusion_matrix(self, multi_class_predictions):
        """Test multi-class confusion matrix."""
        cm = ModelMetrics.confusion_matrix(
            multi_class_predictions["y_true"], multi_class_predictions["y_pred"]
        )

        # Verify shape
        assert cm.shape == (3, 3)

        # y_true = [0, 1, 2, 0, 1, 2, 0, 1, 2, 0]
        # y_pred = [0, 1, 2, 0, 1, 1, 0, 2, 2, 0]

        # Check diagonal (correct predictions)
        # Class 0: positions 0, 3, 6, 9 = 4
        assert cm[0, 0] == 4
        # Class 1: positions 1, 4 = 2
        assert cm[1, 1] == 2
        # Class 2: position 2, 8 = 2
        assert cm[2, 2] == 2

        # Total diagonal should be 8
        assert np.trace(cm) == 8

    def test_perfect_predictions_confusion_matrix(self, perfect_predictions):
        """Test confusion matrix with perfect predictions."""
        cm = ModelMetrics.confusion_matrix(
            perfect_predictions["y_true"], perfect_predictions["y_pred"]
        )

        # All correct predictions should be on diagonal
        assert np.all(cm - np.diag(np.diag(cm)) == 0)

    def test_with_explicit_labels(self):
        """Test with explicit label ordering."""
        y_true = [1, 2, 1, 2]
        y_pred = [1, 2, 1, 2]

        # Specify labels in reverse order
        cm = ModelMetrics.confusion_matrix(y_true, y_pred, labels=[2, 1])

        assert cm.shape == (2, 2)
        # Label 2 is now at index 0, label 1 at index 1
        assert cm[0, 0] == 2  # Class 2 correct
        assert cm[1, 1] == 2  # Class 1 correct

    def test_missing_labels(self):
        """Test with labels specified that don't appear in data."""
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 0, 1]

        # Include label 2 which doesn't appear
        cm = ModelMetrics.confusion_matrix(y_true, y_pred, labels=[0, 1, 2])

        assert cm.shape == (3, 3)
        # Class 2 row and column should be all zeros
        assert np.all(cm[2, :] == 0)
        assert np.all(cm[:, 2] == 0)

    def test_empty_lists(self):
        """Test with empty lists."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ModelMetrics.confusion_matrix([], [])


class TestPrecisionRecallF1:
    """Test precision, recall, and F1 score."""

    def test_perfect_predictions(self, perfect_predictions):
        """Test with perfect predictions."""
        metrics = ModelMetrics.precision_recall_f1(
            perfect_predictions["y_true"], perfect_predictions["y_pred"]
        )

        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_binary_classification(self, binary_predictions):
        """Test with binary classification."""
        metrics = ModelMetrics.precision_recall_f1(
            binary_predictions["y_true"],
            binary_predictions["y_pred"],
            average="macro",
        )

        # All metrics should be between 0 and 1
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["f1"] <= 1.0

        # F1 should be harmonic mean of precision and recall
        if metrics["precision"] + metrics["recall"] > 0:
            expected_f1 = (
                2
                * metrics["precision"]
                * metrics["recall"]
                / (metrics["precision"] + metrics["recall"])
            )
            assert metrics["f1"] == pytest.approx(expected_f1, abs=0.001)

    def test_macro_average(self):
        """Test macro averaging."""
        # Balanced classes
        y_true = [0, 0, 1, 1, 2, 2]
        y_pred = [0, 0, 1, 1, 2, 2]

        metrics = ModelMetrics.precision_recall_f1(y_true, y_pred, average="macro")

        # Perfect predictions → all 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_micro_average(self):
        """Test micro averaging."""
        # Imbalanced classes
        y_true = [0, 0, 0, 1, 1, 2]
        y_pred = [0, 0, 0, 1, 1, 2]

        metrics = ModelMetrics.precision_recall_f1(y_true, y_pred, average="micro")

        # Micro average of perfect predictions is 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_weighted_average(self):
        """Test weighted averaging."""
        # Imbalanced classes: 3 of class 0, 2 of class 1, 1 of class 2
        y_true = [0, 0, 0, 1, 1, 2]
        y_pred = [0, 0, 1, 1, 1, 2]  # One error in class 0

        metrics = ModelMetrics.precision_recall_f1(y_true, y_pred, average="weighted")

        # Should weight by class support
        assert 0.8 < metrics["precision"] <= 1.0
        assert 0.8 < metrics["recall"] <= 1.0
        assert metrics["support"] == 6.0

    def test_edge_case_all_one_class(self):
        """Test when all predictions are one class."""
        y_true = [0, 1, 0, 1]
        y_pred = [0, 0, 0, 0]  # All predicted as class 0

        metrics = ModelMetrics.precision_recall_f1(y_true, y_pred, average="macro")

        # Class 0: precision = 2/4 = 0.5, recall = 2/2 = 1.0
        # Class 1: precision = 0/0 = 0.0, recall = 0/2 = 0.0
        # Macro avg: (0.5 + 0.0) / 2 = 0.25, (1.0 + 0.0) / 2 = 0.5
        assert metrics["precision"] == pytest.approx(0.25, abs=0.01)
        assert metrics["recall"] == pytest.approx(0.5, abs=0.01)

    def test_invalid_average(self):
        """Test with invalid averaging strategy."""
        y_true = [0, 1, 0, 1]
        y_pred = [0, 1, 0, 1]

        with pytest.raises(ValueError, match="Unknown average strategy"):
            ModelMetrics.precision_recall_f1(y_true, y_pred, average="invalid")

    def test_mismatched_lengths(self):
        """Test with mismatched lengths."""
        with pytest.raises(ValueError, match="must have same length"):
            ModelMetrics.precision_recall_f1([0, 1, 0], [0, 1])

    def test_empty_lists(self):
        """Test with empty lists."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ModelMetrics.precision_recall_f1([], [])


class TestClassificationReport:
    """Test classification report generation."""

    def test_basic_report(self, binary_predictions):
        """Test basic report structure."""
        report = ModelMetrics.classification_report(
            binary_predictions["y_true"], binary_predictions["y_pred"]
        )

        # Check required keys
        assert "0" in report
        assert "1" in report
        assert "macro_avg" in report
        assert "weighted_avg" in report
        assert "accuracy" in report

        # Check per-class metrics
        for class_key in ["0", "1"]:
            assert "precision" in report[class_key]
            assert "recall" in report[class_key]
            assert "f1" in report[class_key]
            assert "support" in report[class_key]

    def test_all_classes_included(self, multi_class_predictions):
        """Test that all classes are included."""
        report = ModelMetrics.classification_report(
            multi_class_predictions["y_true"], multi_class_predictions["y_pred"]
        )

        # All three classes should be present
        assert "0" in report
        assert "1" in report
        assert "2" in report

    def test_macro_and_weighted_averages(self):
        """Test macro and weighted averages in report."""
        y_true = [0, 0, 0, 1, 1, 2]
        y_pred = [0, 0, 0, 1, 1, 2]

        report = ModelMetrics.classification_report(y_true, y_pred)

        # Macro average
        assert "macro_avg" in report
        assert report["macro_avg"]["precision"] == 1.0
        assert report["macro_avg"]["recall"] == 1.0
        assert report["macro_avg"]["f1"] == 1.0

        # Weighted average
        assert "weighted_avg" in report
        assert report["weighted_avg"]["precision"] == 1.0
        assert report["weighted_avg"]["recall"] == 1.0
        assert report["weighted_avg"]["f1"] == 1.0

    def test_accuracy_in_report(self, binary_predictions):
        """Test that accuracy is included correctly."""
        report = ModelMetrics.classification_report(
            binary_predictions["y_true"], binary_predictions["y_pred"]
        )

        assert "accuracy" in report
        assert "value" in report["accuracy"]

        # Verify accuracy matches standalone calculation
        expected_accuracy = ModelMetrics.accuracy(
            binary_predictions["y_true"], binary_predictions["y_pred"]
        )
        assert report["accuracy"]["value"] == expected_accuracy

    def test_support_values(self):
        """Test support values are correct."""
        y_true = [0, 0, 0, 1, 1, 2]
        y_pred = [0, 0, 0, 1, 1, 2]

        report = ModelMetrics.classification_report(y_true, y_pred)

        # Check support counts
        assert report["0"]["support"] == 3.0
        assert report["1"]["support"] == 2.0
        assert report["2"]["support"] == 1.0

    def test_with_explicit_labels(self):
        """Test report with explicit labels."""
        y_true = [1, 2, 1, 2]
        y_pred = [1, 2, 1, 2]

        report = ModelMetrics.classification_report(y_true, y_pred, labels=[2, 1])

        # Both labels should be in report
        assert "2" in report
        assert "1" in report

    def test_empty_lists(self):
        """Test with empty lists."""
        with pytest.raises(ValueError, match="cannot be empty"):
            ModelMetrics.classification_report([], [])


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_single_class(self):
        """Test with single class."""
        y_true = [0, 0, 0, 0]
        y_pred = [0, 0, 0, 0]

        accuracy = ModelMetrics.accuracy(y_true, y_pred)
        assert accuracy == 1.0

        metrics = ModelMetrics.precision_recall_f1(y_true, y_pred)
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_with_strings(self):
        """Test with string labels."""
        y_true = ["cat", "dog", "cat", "dog"]
        y_pred = ["cat", "dog", "dog", "dog"]

        accuracy = ModelMetrics.accuracy(y_true, y_pred)
        assert accuracy == 0.75

        cm = ModelMetrics.confusion_matrix(y_true, y_pred)
        assert cm.shape == (2, 2)

    def test_with_mixed_types(self):
        """Test with mixed label types."""
        y_true = [0, 1, "A", "B"]
        y_pred = [0, 1, "A", "B"]

        accuracy = ModelMetrics.accuracy(y_true, y_pred)
        assert accuracy == 1.0

    def test_large_number_of_classes(self):
        """Test with many classes."""
        n_classes = 50
        y_true = list(range(n_classes))
        y_pred = list(range(n_classes))

        cm = ModelMetrics.confusion_matrix(y_true, y_pred)
        assert cm.shape == (n_classes, n_classes)
        assert np.trace(cm) == n_classes

    def test_zero_division_handling(self):
        """Test handling of zero division in metrics."""
        # All predicted as one class, but true labels are different
        y_true = [0, 1, 2]
        y_pred = [0, 0, 0]

        metrics = ModelMetrics.precision_recall_f1(y_true, y_pred, average="macro")

        # Should not raise error, should return 0.0 for undefined metrics
        assert isinstance(metrics["precision"], float)
        assert isinstance(metrics["recall"], float)
        assert isinstance(metrics["f1"], float)

    def test_imbalanced_classes(self):
        """Test with highly imbalanced classes."""
        # 95% class 0, 5% class 1
        y_true = [0] * 95 + [1] * 5
        y_pred = [0] * 95 + [1] * 5

        metrics_macro = ModelMetrics.precision_recall_f1(
            y_true, y_pred, average="macro"
        )
        metrics_weighted = ModelMetrics.precision_recall_f1(
            y_true, y_pred, average="weighted"
        )

        # Both should be 1.0 for perfect predictions
        assert metrics_macro["f1"] == 1.0
        assert metrics_weighted["f1"] == 1.0

        # But they would differ if predictions were imperfect
        y_pred_imperfect = [0] * 98 + [1] * 2  # Misclassify some class 1

        metrics_macro_imp = ModelMetrics.precision_recall_f1(
            y_true, y_pred_imperfect, average="macro"
        )
        metrics_weighted_imp = ModelMetrics.precision_recall_f1(
            y_true, y_pred_imperfect, average="weighted"
        )

        # Weighted should be higher (dominated by majority class)
        assert metrics_weighted_imp["f1"] > metrics_macro_imp["f1"]

    def test_verify_f1_formula(self):
        """Verify F1 is harmonic mean of precision and recall."""
        y_true = [0, 1, 1, 0, 1, 0]
        y_pred = [0, 1, 0, 0, 1, 1]

        metrics = ModelMetrics.precision_recall_f1(y_true, y_pred, average="macro")

        # Manually calculate harmonic mean
        p = metrics["precision"]
        r = metrics["recall"]

        if p + r > 0:
            expected_f1 = 2 * p * r / (p + r)
            assert metrics["f1"] == pytest.approx(expected_f1, abs=0.001)


class TestComparisonWithSklearn:
    """Test against sklearn.metrics when available."""

    def test_accuracy_matches_sklearn(self, binary_predictions):
        """Test accuracy matches sklearn (if available)."""
        pytest.importorskip("sklearn")
        from sklearn.metrics import accuracy_score

        our_accuracy = ModelMetrics.accuracy(
            binary_predictions["y_true"], binary_predictions["y_pred"]
        )
        sklearn_accuracy = accuracy_score(
            binary_predictions["y_true"], binary_predictions["y_pred"]
        )

        assert our_accuracy == pytest.approx(sklearn_accuracy, abs=0.001)

    def test_confusion_matrix_matches_sklearn(self, binary_predictions):
        """Test confusion matrix matches sklearn (if available)."""
        pytest.importorskip("sklearn")
        from sklearn.metrics import confusion_matrix

        our_cm = ModelMetrics.confusion_matrix(
            binary_predictions["y_true"], binary_predictions["y_pred"]
        )
        sklearn_cm = confusion_matrix(
            binary_predictions["y_true"], binary_predictions["y_pred"]
        )

        assert np.array_equal(our_cm, sklearn_cm)

    def test_precision_recall_f1_matches_sklearn(self, binary_predictions):
        """Test precision/recall/F1 matches sklearn (if available)."""
        pytest.importorskip("sklearn")
        from sklearn.metrics import precision_recall_fscore_support

        our_metrics = ModelMetrics.precision_recall_f1(
            binary_predictions["y_true"],
            binary_predictions["y_pred"],
            average="macro",
        )

        sklearn_p, sklearn_r, sklearn_f1, _ = precision_recall_fscore_support(
            binary_predictions["y_true"],
            binary_predictions["y_pred"],
            average="macro",
        )

        assert our_metrics["precision"] == pytest.approx(sklearn_p, abs=0.001)
        assert our_metrics["recall"] == pytest.approx(sklearn_r, abs=0.001)
        assert our_metrics["f1"] == pytest.approx(sklearn_f1, abs=0.001)
