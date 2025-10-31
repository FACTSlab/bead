"""Tests for CrossValidator class."""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import numpy as np
import pytest

from bead.evaluation.cross_validation import CrossValidator
from bead.items.models import Item


class TestCrossValidatorInitialization:
    """Test CrossValidator initialization."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        cv = CrossValidator()
        assert cv.k == 5
        assert cv.shuffle is True
        assert cv.random_seed is None

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        cv = CrossValidator(k=3, shuffle=False, random_seed=42)
        assert cv.k == 3
        assert cv.shuffle is False
        assert cv.random_seed == 42

    def test_invalid_k(self):
        """Test that k < 2 raises ValueError."""
        with pytest.raises(ValueError, match="k must be >= 2"):
            CrossValidator(k=1)

        with pytest.raises(ValueError, match="k must be >= 2"):
            CrossValidator(k=0)

        with pytest.raises(ValueError, match="k must be >= 2"):
            CrossValidator(k=-1)


class TestKFoldSplit:
    """Test K-fold split functionality."""

    def test_k_fold_split_basic(self, sample_items):
        """Verify k folds created and all items used exactly once as test."""
        cv = CrossValidator(k=5, random_seed=42)
        folds = cv.k_fold_split(sample_items)

        # Verify k folds created
        assert len(folds) == 5

        # Verify all items used exactly once as test
        all_test_items = []
        for _, test in folds:
            all_test_items.extend(test)

        # Check all items appear exactly once
        assert len(all_test_items) == len(sample_items)
        assert set(item.id for item in all_test_items) == set(
            item.id for item in sample_items
        )

        # Verify train + test sizes sum to total
        for train, test in folds:
            assert len(train) + len(test) == len(sample_items)

    def test_k_fold_split_sizes(self):
        """Test fold sizes are balanced."""
        cv = CrossValidator(k=3, random_seed=42)
        items = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={},
                item_metadata={"idx": i},
            )
            for i in range(10)
        ]

        folds = cv.k_fold_split(items)

        # 10 items / 3 folds = 3, 3, 4
        test_sizes = [len(test) for _, test in folds]
        assert sorted(test_sizes) == [3, 3, 4]

    def test_empty_items(self):
        """Test with empty item list."""
        cv = CrossValidator(k=5)
        folds = cv.k_fold_split([])
        assert folds == []

    def test_k_equals_two(self):
        """Test minimum k value (k=2)."""
        cv = CrossValidator(k=2, random_seed=42)
        items = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={},
                item_metadata={"idx": i},
            )
            for i in range(10)
        ]

        folds = cv.k_fold_split(items)
        assert len(folds) == 2

        # Verify no overlap between test sets
        test1_ids = {item.id for item in folds[0][1]}
        test2_ids = {item.id for item in folds[1][1]}
        assert len(test1_ids & test2_ids) == 0

    def test_k_greater_than_n(self):
        """Test k > n_items (should handle gracefully)."""
        cv = CrossValidator(k=10, random_seed=42)
        items = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={},
                item_metadata={"idx": i},
            )
            for i in range(5)
        ]

        folds = cv.k_fold_split(items)

        # Should create k folds, some with empty test sets
        assert len(folds) == 10

        # Count non-empty folds
        non_empty_folds = [fold for fold in folds if len(fold[1]) > 0]
        assert len(non_empty_folds) == 5

    def test_reproducibility(self, sample_items):
        """Same random_seed produces same splits."""
        cv1 = CrossValidator(k=5, random_seed=42)
        folds1 = cv1.k_fold_split(sample_items)

        # Create new instance with same seed
        cv2 = CrossValidator(k=5, random_seed=42)
        folds2 = cv2.k_fold_split(sample_items)

        # Compare test sets (convert to sets since order within fold doesn't matter)
        for (_, test1), (_, test2) in zip(folds1, folds2):
            test1_ids = set(item.id for item in test1)
            test2_ids = set(item.id for item in test2)
            assert test1_ids == test2_ids


class TestStratifiedSplit:
    """Test stratified K-fold split."""

    def test_stratified_split(self, sample_items):
        """Verify stratification preserves class distribution."""
        cv = CrossValidator(k=5, random_seed=42)
        folds = cv.k_fold_split(sample_items, stratify_by="item_metadata.label")

        # Original distribution: labels 0, 1, 2 with ~33% each
        original_dist = [item.item_metadata["label"] for item in sample_items]
        original_counts = {
            0: original_dist.count(0),
            1: original_dist.count(1),
            2: original_dist.count(2),
        }

        # Check each fold maintains distribution
        for train, test in folds:
            # Test set distribution
            test_labels = [item.item_metadata["label"] for item in test]
            test_counts = {
                0: test_labels.count(0),
                1: test_labels.count(1),
                2: test_labels.count(2),
            }

            # Each class should be represented approximately equally
            # With 100 items and 3 classes, each fold has ~20 items
            # So each class should have ~7 items per fold (20/3)
            for label in [0, 1, 2]:
                expected_proportion = original_counts[label] / len(sample_items)
                actual_proportion = test_counts[label] / len(test)
                # Allow some tolerance due to rounding
                assert abs(expected_proportion - actual_proportion) < 0.15

    def test_stratified_with_different_variables(self):
        """Test stratification with different metadata variables."""
        items = []
        for i in range(100):
            items.append(
                Item(
                    item_template_id=uuid4(),
                    rendered_elements={},
                    item_metadata={
                        "label": i % 3,
                        "category": "A" if i < 50 else "B",
                    },
                )
            )

        cv = CrossValidator(k=5, random_seed=42)

        # Stratify by label
        folds_by_label = cv.k_fold_split(items, stratify_by="item_metadata.label")

        # Stratify by category
        folds_by_category = cv.k_fold_split(items, stratify_by="item_metadata.category")

        # Both should work and produce different splits
        assert len(folds_by_label) == 5
        assert len(folds_by_category) == 5

        # Verify category stratification
        for _, test in folds_by_category:
            categories = [item.item_metadata["category"] for item in test]
            # Should have roughly 50-50 split
            a_count = categories.count("A")
            b_count = categories.count("B")
            assert abs(a_count - b_count) <= 2  # Allow small imbalance

    def test_single_class_stratification(self):
        """Test stratification with single class."""
        items = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={},
                item_metadata={"label": 1},
            )
            for _ in range(50)
        ]

        cv = CrossValidator(k=5, random_seed=42)
        folds = cv.k_fold_split(items, stratify_by="item_metadata.label")

        # Should work like regular k-fold
        assert len(folds) == 5

        # All items should be used
        all_test = []
        for _, test in folds:
            all_test.extend(test)
        assert len(all_test) == len(items)


class TestPropertyExtraction:
    """Test property value extraction."""

    def test_get_property_value_dict(self):
        """Test extracting values from dict-like metadata."""
        cv = CrossValidator()
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={},
            item_metadata={"label": 5, "nested": {"value": 10}},
        )

        # Direct metadata access
        value = cv._get_property_value(item, "item_metadata.label")
        assert value == 5

        # Nested metadata access
        nested_value = cv._get_property_value(item, "item_metadata.nested.value")
        assert nested_value == 10

    def test_get_property_value_attribute(self):
        """Test extracting attribute values that are valid stratification types."""
        cv = CrossValidator()
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={},
            item_metadata={"category": "test", "score": 42},
        )

        # Access metadata values (valid stratification types)
        category = cv._get_property_value(item, "item_metadata.category")
        assert category == "test"

        score = cv._get_property_value(item, "item_metadata.score")
        assert score == 42

    def test_invalid_property_path(self):
        """Test invalid property path raises AttributeError."""
        cv = CrossValidator()
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={},
            item_metadata={},
        )

        with pytest.raises(AttributeError):
            cv._get_property_value(item, "nonexistent.property")


class TestEvaluateFold:
    """Test fold evaluation."""

    def test_evaluate_fold_basic(self):
        """Test fold evaluation with mock trainer."""
        cv = CrossValidator()

        # Create mock trainer
        trainer = MagicMock()
        trainer.train.return_value = MagicMock(metrics={"accuracy": 0.85, "f1": 0.82})

        # Create sample items
        train_items = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={},
                item_metadata={"label": i % 2},
            )
            for i in range(80)
        ]
        test_items = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={},
                item_metadata={"label": i % 2},
            )
            for i in range(20)
        ]

        # Evaluate fold
        metrics = cv.evaluate_fold(trainer, train_items, test_items)

        # Verify trainer was called
        trainer.train.assert_called_once_with(train_items, eval_data=test_items)

        # Verify metrics returned
        assert metrics == {"accuracy": 0.85, "f1": 0.82}


class TestAggregateResults:
    """Test result aggregation."""

    def test_aggregate_results_basic(self):
        """Test aggregation with multiple folds."""
        fold_results = [
            {"accuracy": 0.85, "f1": 0.82},
            {"accuracy": 0.87, "f1": 0.84},
            {"accuracy": 0.83, "f1": 0.80},
        ]

        agg = CrossValidator.aggregate_results(fold_results)

        # Check structure
        assert "mean" in agg
        assert "std" in agg
        assert "fold_results" in agg
        assert "n_folds" in agg

        # Check values
        assert agg["n_folds"] == 3
        assert agg["fold_results"] == fold_results

        # Check mean
        assert agg["mean"]["accuracy"] == pytest.approx(0.85, abs=0.01)
        assert agg["mean"]["f1"] == pytest.approx(0.82, abs=0.01)

        # Check std
        expected_std_acc = np.std([0.85, 0.87, 0.83])
        assert agg["std"]["accuracy"] == pytest.approx(expected_std_acc, abs=0.001)

    def test_aggregate_results_empty(self):
        """Test aggregation with empty results."""
        agg = CrossValidator.aggregate_results([])

        assert agg == {
            "mean": {},
            "std": {},
            "fold_results": [],
            "n_folds": 0,
        }

    def test_aggregate_results_single_fold(self):
        """Test aggregation with single fold."""
        fold_results = [{"accuracy": 0.85, "f1": 0.82}]

        agg = CrossValidator.aggregate_results(fold_results)

        assert agg["n_folds"] == 1
        assert agg["mean"]["accuracy"] == 0.85
        assert agg["std"]["accuracy"] == 0.0  # No variance with single value

    def test_aggregate_results_multiple_metrics(self):
        """Test aggregation with many metrics."""
        fold_results = [
            {"accuracy": 0.85, "precision": 0.83, "recall": 0.87, "f1": 0.85},
            {"accuracy": 0.87, "precision": 0.85, "recall": 0.89, "f1": 0.87},
            {"accuracy": 0.83, "precision": 0.81, "recall": 0.85, "f1": 0.83},
        ]

        agg = CrossValidator.aggregate_results(fold_results)

        # All metrics should be aggregated
        assert set(agg["mean"].keys()) == {"accuracy", "precision", "recall", "f1"}
        assert set(agg["std"].keys()) == {"accuracy", "precision", "recall", "f1"}


class TestCrossValidate:
    """Test full cross-validation workflow."""

    def test_cross_validate_basic(self, sample_items):
        """Test full cross-validation with mock trainer."""
        cv = CrossValidator(k=3, random_seed=42)

        # Create mock trainer
        trainer = MagicMock()
        trainer.train.return_value = MagicMock(metrics={"accuracy": 0.85, "f1": 0.82})

        # Run cross-validation
        results = cv.cross_validate(trainer, sample_items)

        # Verify structure
        assert "mean" in results
        assert "std" in results
        assert "fold_results" in results
        assert "n_folds" in results

        # Verify trainer called k times
        assert trainer.train.call_count == 3

        # Verify results
        assert results["n_folds"] == 3
        assert len(results["fold_results"]) == 3

    def test_cross_validate_with_stratification(self, sample_items):
        """Test cross-validation with stratification."""
        cv = CrossValidator(k=5, random_seed=42)

        # Create mock trainer
        trainer = MagicMock()
        trainer.train.return_value = MagicMock(metrics={"accuracy": 0.85, "f1": 0.82})

        # Run with stratification
        results = cv.cross_validate(
            trainer, sample_items, stratify_by="item_metadata.label"
        )

        # Verify results
        assert results["n_folds"] == 5
        assert trainer.train.call_count == 5


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_no_shuffle(self):
        """Test that shuffle=False maintains order."""
        items = list(range(10))
        cv = CrossValidator(k=2, shuffle=False)

        folds = cv.k_fold_split(items)

        # First fold test should be first items
        _, test1 = folds[0]
        assert test1[:3] == [0, 1, 2]  # First 5 items

    def test_large_k(self):
        """Test with k close to n."""
        items = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={},
                item_metadata={"idx": i},
            )
            for i in range(10)
        ]

        cv = CrossValidator(k=10, random_seed=42)
        folds = cv.k_fold_split(items)

        # Each fold should have 1 test item
        non_empty = [fold for fold in folds if len(fold[1]) > 0]
        assert len(non_empty) == 10

        for _, test in non_empty:
            assert len(test) == 1

    def test_odd_number_items(self):
        """Test with odd number of items."""
        items = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={},
                item_metadata={"idx": i},
            )
            for i in range(7)
        ]

        cv = CrossValidator(k=3, random_seed=42)
        folds = cv.k_fold_split(items)

        # 7 items / 3 folds = 2, 2, 3
        test_sizes = sorted([len(test) for _, test in folds])
        assert test_sizes == [2, 2, 3]

        # All items should still be used
        all_test = []
        for _, test in folds:
            all_test.extend(test)
        assert len(all_test) == 7
