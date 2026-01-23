"""Tests for InterAnnotatorMetrics class."""

from __future__ import annotations

import numpy as np
import pytest

from bead.evaluation.interannotator import InterAnnotatorMetrics


class TestPercentageAgreement:
    """Test percentage agreement calculation."""

    def test_perfect_agreement(self, perfect_agreement_data):
        """Test perfect agreement (100%)."""
        agreement = InterAnnotatorMetrics.percentage_agreement(
            perfect_agreement_data["rater1"], perfect_agreement_data["rater2"]
        )
        assert agreement == 1.0

    def test_no_agreement(self):
        """Test no agreement (0%)."""
        rater1 = [0, 0, 0, 0, 0]
        rater2 = [1, 1, 1, 1, 1]
        agreement = InterAnnotatorMetrics.percentage_agreement(rater1, rater2)
        assert agreement == 0.0

    def test_partial_agreement(self):
        """Test partial agreement."""
        rater1 = [1, 2, 3, 1, 2]
        rater2 = [1, 2, 2, 1, 2]
        agreement = InterAnnotatorMetrics.percentage_agreement(rater1, rater2)
        assert agreement == 0.8  # 4 out of 5 agree

    def test_mismatched_lengths(self):
        """Test that mismatched lengths raise ValueError."""
        rater1 = [1, 2, 3]
        rater2 = [1, 2]

        with pytest.raises(ValueError, match="must have same length"):
            InterAnnotatorMetrics.percentage_agreement(rater1, rater2)

    def test_empty_lists(self):
        """Test with empty lists."""
        agreement = InterAnnotatorMetrics.percentage_agreement([], [])
        assert agreement == 1.0  # Empty lists trivially agree


class TestCohensKappa:
    """Test Cohen's kappa calculation."""

    def test_perfect_agreement(self, perfect_agreement_data):
        """Test perfect agreement (κ=1.0)."""
        kappa = InterAnnotatorMetrics.cohens_kappa(
            perfect_agreement_data["rater1"], perfect_agreement_data["rater2"]
        )
        assert kappa == pytest.approx(1.0, abs=0.01)

    def test_complete_disagreement(self):
        """Test complete disagreement (κ=-1.0)."""
        # Systematic disagreement: rater1=0 when rater2=1 and vice versa
        rater1 = [0, 0, 1, 1, 0, 0, 1, 1]
        rater2 = [1, 1, 0, 0, 1, 1, 0, 0]

        kappa = InterAnnotatorMetrics.cohens_kappa(rater1, rater2)
        assert kappa == pytest.approx(-1.0, abs=0.01)

    def test_no_agreement_beyond_chance(self):
        """Test no agreement beyond chance (κ≈0.0)."""
        # Create random-ish ratings
        np.random.seed(42)
        rater1 = np.random.randint(0, 3, size=100).tolist()
        rater2 = np.random.randint(0, 3, size=100).tolist()

        kappa = InterAnnotatorMetrics.cohens_kappa(rater1, rater2)

        # Should be close to 0 (within range for random data)
        assert -0.2 < kappa < 0.2

    def test_known_kappa_value(self):
        """Test against known kappa value from literature."""
        # Example from Fleiss (1971)
        rater1 = [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
        rater2 = [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]

        kappa = InterAnnotatorMetrics.cohens_kappa(rater1, rater2)

        # Expected kappa for this data is 0.0 (half agree by chance)
        assert kappa == pytest.approx(0.0, abs=0.01)

    def test_mismatched_lengths(self):
        """Test that mismatched lengths raise ValueError."""
        rater1 = [1, 2, 3]
        rater2 = [1, 2]

        with pytest.raises(ValueError, match="must have same length"):
            InterAnnotatorMetrics.cohens_kappa(rater1, rater2)

    def test_empty_lists(self):
        """Test with empty lists."""
        with pytest.raises(ValueError, match="cannot be empty"):
            InterAnnotatorMetrics.cohens_kappa([], [])

    def test_all_same_category(self):
        """Test when all items are in one category."""
        rater1 = [1, 1, 1, 1, 1]
        rater2 = [1, 1, 1, 1, 1]

        kappa = InterAnnotatorMetrics.cohens_kappa(rater1, rater2)
        assert kappa == 1.0

    def test_with_strings(self):
        """Test with string labels."""
        rater1 = ["A", "B", "A", "B", "C"]
        rater2 = ["A", "B", "A", "C", "C"]

        kappa = InterAnnotatorMetrics.cohens_kappa(rater1, rater2)
        assert 0.0 < kappa < 1.0  # Some agreement but not perfect


class TestFleissKappa:
    """Test Fleiss' kappa calculation."""

    def test_perfect_agreement(self):
        """Test perfect agreement among all raters."""
        # 5 items, 3 categories, 4 raters
        # All raters agree on category for each item
        ratings = np.array(
            [
                [4, 0, 0],  # Item 1: all chose category 0
                [0, 4, 0],  # Item 2: all chose category 1
                [0, 0, 4],  # Item 3: all chose category 2
                [4, 0, 0],  # Item 4: all chose category 0
                [0, 4, 0],  # Item 5: all chose category 1
            ]
        )

        kappa = InterAnnotatorMetrics.fleiss_kappa(ratings)
        assert kappa == pytest.approx(1.0, abs=0.01)

    def test_moderate_agreement(self):
        """Test moderate agreement among raters."""
        # 4 items, 3 categories, 5 raters each (with reasonable agreement)
        ratings = np.array(
            [
                [4, 1, 0],  # Item 1: strong agreement on cat 0
                [0, 4, 1],  # Item 2: strong agreement on cat 1
                [1, 0, 4],  # Item 3: strong agreement on cat 2
                [3, 2, 0],  # Item 4: moderate agreement on cat 0
            ]
        )

        kappa = InterAnnotatorMetrics.fleiss_kappa(ratings)

        # Should have positive agreement (can be small with mixed patterns)
        assert 0.1 < kappa < 1.0

    def test_no_agreement(self):
        """Test no agreement beyond chance."""
        # Completely random assignment
        ratings = np.array(
            [
                [2, 2, 1],  # Item 1
                [1, 2, 2],  # Item 2
                [2, 1, 2],  # Item 3
                [2, 2, 1],  # Item 4
            ]
        )

        kappa = InterAnnotatorMetrics.fleiss_kappa(ratings)

        # Should be close to 0
        assert -0.3 < kappa < 0.3

    def test_empty_matrix(self):
        """Test with empty matrix."""
        with pytest.raises(ValueError, match="cannot be empty"):
            InterAnnotatorMetrics.fleiss_kappa(np.array([]))

    def test_invalid_shape(self):
        """Test with invalid matrix shape."""
        with pytest.raises(ValueError, match="cannot be empty"):
            InterAnnotatorMetrics.fleiss_kappa(np.array([[]]))

    def test_unequal_rater_counts(self):
        """Test with unequal number of raters per item."""
        # Different total raters per item
        ratings = np.array(
            [
                [3, 2, 0],  # 5 raters
                [2, 2, 0],  # 4 raters
            ]
        )

        with pytest.raises(ValueError, match="same number of raters"):
            InterAnnotatorMetrics.fleiss_kappa(ratings)

    def test_varying_rater_counts(self):
        """Test Fleiss kappa with varying numbers of raters."""
        # 3 items, 3 categories, 3 raters each
        ratings = np.array(
            [
                [2, 1, 0],  # Item 1
                [0, 2, 1],  # Item 2
                [1, 0, 2],  # Item 3
            ]
        )

        kappa = InterAnnotatorMetrics.fleiss_kappa(ratings)
        assert isinstance(kappa, float)


class TestKrippendorffAlpha:
    """Test Krippendorff's alpha calculation."""

    def test_perfect_agreement_nominal(self, perfect_agreement_data):
        """Test perfect agreement with nominal metric."""
        alpha = InterAnnotatorMetrics.krippendorff_alpha(
            perfect_agreement_data, metric="nominal"
        )
        assert alpha == pytest.approx(1.0, abs=0.01)

    def test_with_missing_data(self, ratings_with_missing):
        """Test with missing data (None values)."""
        alpha = InterAnnotatorMetrics.krippendorff_alpha(
            ratings_with_missing, metric="nominal"
        )

        # Should handle missing data gracefully
        assert isinstance(alpha, float)
        assert -1.0 <= alpha <= 1.0

    def test_ordinal_metric(self):
        """Test with ordinal metric."""
        data = {
            "rater1": [1, 2, 3, 4, 5],
            "rater2": [1, 2, 3, 4, 5],
            "rater3": [1, 2, 3, 5, 5],  # Small disagreement
        }

        alpha = InterAnnotatorMetrics.krippendorff_alpha(data, metric="ordinal")

        # Should have high agreement
        assert alpha > 0.7

    def test_interval_metric(self):
        """Test with interval metric."""
        data = {
            "rater1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "rater2": [1.1, 2.1, 3.0, 3.9, 5.0],
            "rater3": [1.0, 2.0, 3.1, 4.0, 5.1],
        }

        alpha = InterAnnotatorMetrics.krippendorff_alpha(data, metric="interval")

        # Should have high agreement (small numerical differences)
        assert alpha > 0.8

    def test_ratio_metric(self):
        """Test with ratio metric."""
        data = {
            "rater1": [10, 20, 30, 40, 50],
            "rater2": [10, 20, 30, 40, 50],
            "rater3": [10, 20, 30, 40, 51],  # Small disagreement
        }

        alpha = InterAnnotatorMetrics.krippendorff_alpha(data, metric="ratio")

        # Should have high agreement
        assert alpha > 0.9

    def test_empty_data(self):
        """Test with empty data."""
        with pytest.raises(ValueError, match="cannot be empty"):
            InterAnnotatorMetrics.krippendorff_alpha({})

    def test_mismatched_lengths(self):
        """Test with mismatched rating lengths."""
        data = {"rater1": [1, 2, 3], "rater2": [1, 2]}

        with pytest.raises(ValueError, match="same number of items"):
            InterAnnotatorMetrics.krippendorff_alpha(data)

    def test_all_missing_data(self):
        """Test with all missing data."""
        data = {
            "rater1": [None, None, None],
            "rater2": [None, None, None],
        }

        alpha = InterAnnotatorMetrics.krippendorff_alpha(data)
        assert alpha == 0.0  # No comparisons possible

    def test_single_value(self):
        """Test with only one non-missing value."""
        data = {
            "rater1": [1, None, None],
            "rater2": [None, None, None],
        }

        alpha = InterAnnotatorMetrics.krippendorff_alpha(data)
        assert alpha == 0.0  # Not enough data

    def test_all_same_value(self):
        """Test when all values are identical."""
        data = {
            "rater1": [1, 1, 1, 1, 1],
            "rater2": [1, 1, 1, 1, 1],
            "rater3": [1, 1, 1, 1, 1],
        }

        alpha = InterAnnotatorMetrics.krippendorff_alpha(data)
        assert alpha == 1.0  # Perfect agreement

    def test_invalid_metric(self):
        """Test with invalid metric type."""
        data = {"rater1": [1, 2, 3], "rater2": [1, 2, 3]}

        with pytest.raises(ValueError, match="Unknown metric"):
            InterAnnotatorMetrics.krippendorff_alpha(data, metric="invalid")

    def test_systematic_disagreement(self):
        """Test systematic disagreement (negative alpha)."""
        data = {
            "rater1": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "rater2": [3, 3, 3, 2, 2, 2, 1, 1, 1],  # Inverse pattern
        }

        alpha = InterAnnotatorMetrics.krippendorff_alpha(data, metric="ordinal")

        # Should have negative alpha (systematic disagreement)
        assert alpha < 0.0

    def test_published_example(self):
        """Test against published example from Krippendorff."""
        # Example from Krippendorff (2011)
        data = {
            "rater1": [1, 2, 3, 3, 2, 1, 4, 1, 2, None],
            "rater2": [1, 2, 3, 3, 2, 2, 4, 1, 2, 5],
            "rater3": [None, 3, 3, 3, 2, 3, 4, 2, 2, 5],
            "rater4": [1, 2, 3, 3, 2, 4, 4, 1, 2, 5],
        }

        alpha = InterAnnotatorMetrics.krippendorff_alpha(data, metric="nominal")

        # Should have moderate agreement (published value ≈ 0.743)
        assert 0.6 < alpha < 0.85


class TestPairwiseAgreement:
    """Test pairwise agreement calculation."""

    def test_pairwise_agreement_basic(self, multi_rater_data):
        """Test pairwise agreement computation."""
        result = InterAnnotatorMetrics.pairwise_agreement(multi_rater_data)

        # Check structure
        assert "percentage_agreement" in result
        assert "cohens_kappa" in result

        # Check all pairs computed
        list(multi_rater_data.keys())
        expected_pairs = [
            ("rater1", "rater2"),
            ("rater1", "rater3"),
            ("rater2", "rater3"),
        ]

        assert len(result["percentage_agreement"]) == 3
        assert len(result["cohens_kappa"]) == 3

        for pair in expected_pairs:
            assert pair in result["percentage_agreement"]
            assert pair in result["cohens_kappa"]

    def test_pairwise_with_perfect_agreement(self, perfect_agreement_data):
        """Test pairwise with perfect agreement."""
        result = InterAnnotatorMetrics.pairwise_agreement(perfect_agreement_data)

        # Single pair
        pair = ("rater1", "rater2")
        assert result["percentage_agreement"][pair] == 1.0
        assert result["cohens_kappa"][pair] == pytest.approx(1.0, abs=0.01)

    def test_single_rater(self):
        """Test with single rater (should return empty)."""
        data = {"rater1": [1, 2, 3]}

        result = InterAnnotatorMetrics.pairwise_agreement(data)

        assert result == {
            "percentage_agreement": {},
            "cohens_kappa": {},
        }

    def test_many_raters(self):
        """Test with many raters."""
        data = {f"rater{i}": [1, 2, 3, 1, 2] for i in range(5)}

        result = InterAnnotatorMetrics.pairwise_agreement(data)

        # Should have C(5, 2) = 10 pairs
        assert len(result["percentage_agreement"]) == 10
        assert len(result["cohens_kappa"]) == 10

        # All should have perfect agreement
        for pair_agreement in result["percentage_agreement"].values():
            assert pair_agreement == 1.0

    def test_verify_values(self):
        """Test that percentage and kappa values are correct."""
        data = {
            "rater1": [0, 1, 0, 1, 1],
            "rater2": [0, 1, 1, 1, 1],
        }

        result = InterAnnotatorMetrics.pairwise_agreement(data)

        pair = ("rater1", "rater2")

        # Manually verify percentage agreement: 4/5 = 0.8
        assert result["percentage_agreement"][pair] == 0.8

        # Manually verify kappa
        expected_kappa = InterAnnotatorMetrics.cohens_kappa(
            data["rater1"], data["rater2"]
        )
        assert result["cohens_kappa"][pair] == pytest.approx(expected_kappa, abs=0.001)


class TestEdgeCases:
    """Test edge cases and special scenarios."""

    def test_binary_classification(self):
        """Test with binary classification."""
        rater1 = [0, 1, 0, 1, 0, 1, 0, 1]
        rater2 = [0, 1, 0, 1, 0, 1, 0, 1]

        # Percentage agreement
        perc = InterAnnotatorMetrics.percentage_agreement(rater1, rater2)
        assert perc == 1.0

        # Cohen's kappa
        kappa = InterAnnotatorMetrics.cohens_kappa(rater1, rater2)
        assert kappa == 1.0

    def test_many_categories(self):
        """Test with many categories."""
        rater1 = list(range(20))
        rater2 = list(range(20))

        kappa = InterAnnotatorMetrics.cohens_kappa(rater1, rater2)
        assert kappa == 1.0

    def test_rare_category(self):
        """Test with rare category."""
        # Category 2 appears only once
        rater1 = [0, 0, 0, 0, 1, 1, 1, 1, 2]
        rater2 = [0, 0, 0, 0, 1, 1, 1, 1, 2]

        kappa = InterAnnotatorMetrics.cohens_kappa(rater1, rater2)
        assert kappa == 1.0

    def test_numerical_stability(self):
        """Test numerical stability with edge case values."""
        # Very small differences
        data = {
            "rater1": [1.0000001, 2.0000002, 3.0000001],
            "rater2": [1.0000002, 2.0000001, 3.0000002],
        }

        alpha = InterAnnotatorMetrics.krippendorff_alpha(data, metric="interval")

        # Should have very high agreement
        assert alpha > 0.99

    def test_mixed_data_types_nominal(self):
        """Test with mixed data types in nominal metric."""
        data = {
            "rater1": ["A", "B", "C", 1, 2],
            "rater2": ["A", "B", "C", 1, 2],
        }

        alpha = InterAnnotatorMetrics.krippendorff_alpha(data, metric="nominal")
        assert alpha == 1.0
