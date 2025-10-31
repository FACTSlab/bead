"""Tests for ConvergenceDetector class."""

from __future__ import annotations

import pytest

from bead.evaluation.convergence import ConvergenceDetector


class TestInitialization:
    """Test ConvergenceDetector initialization."""

    def test_default_initialization(self):
        """Test initialization with default parameters."""
        detector = ConvergenceDetector()

        assert detector.human_agreement_metric == "krippendorff_alpha"
        assert detector.convergence_threshold == 0.05
        assert detector.min_iterations == 3
        assert detector.statistical_test is True
        assert detector.alpha == 0.05
        assert detector.human_baseline is None

    def test_custom_initialization(self):
        """Test initialization with custom parameters."""
        detector = ConvergenceDetector(
            human_agreement_metric="cohens_kappa",
            convergence_threshold=0.10,
            min_iterations=5,
            statistical_test=False,
            alpha=0.01,
        )

        assert detector.human_agreement_metric == "cohens_kappa"
        assert detector.convergence_threshold == 0.10
        assert detector.min_iterations == 5
        assert detector.statistical_test is False
        assert detector.alpha == 0.01

    def test_invalid_metric(self):
        """Test that invalid metric raises ValueError."""
        with pytest.raises(ValueError, match="human_agreement_metric must be one of"):
            ConvergenceDetector(human_agreement_metric="invalid_metric")

    def test_invalid_threshold(self):
        """Test that invalid threshold raises ValueError."""
        with pytest.raises(ValueError, match="convergence_threshold must be in"):
            ConvergenceDetector(convergence_threshold=-0.1)

        with pytest.raises(ValueError, match="convergence_threshold must be in"):
            ConvergenceDetector(convergence_threshold=1.5)

    def test_invalid_min_iterations(self):
        """Test that invalid min_iterations raises ValueError."""
        with pytest.raises(ValueError, match="min_iterations must be >= 1"):
            ConvergenceDetector(min_iterations=0)

        with pytest.raises(ValueError, match="min_iterations must be >= 1"):
            ConvergenceDetector(min_iterations=-1)

    def test_invalid_alpha(self):
        """Test that invalid alpha raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be in"):
            ConvergenceDetector(alpha=0.0)

        with pytest.raises(ValueError, match="alpha must be in"):
            ConvergenceDetector(alpha=1.0)

        with pytest.raises(ValueError, match="alpha must be in"):
            ConvergenceDetector(alpha=1.5)


class TestComputeHumanBaseline:
    """Test human baseline computation."""

    def test_with_krippendorff_alpha(self, multi_rater_data):
        """Test baseline computation with Krippendorff's alpha."""
        detector = ConvergenceDetector(human_agreement_metric="krippendorff_alpha")

        baseline = detector.compute_human_baseline(multi_rater_data, metric="nominal")

        # Should return a valid agreement score
        assert isinstance(baseline, float)
        assert -1.0 <= baseline <= 1.0

        # Should store in detector
        assert detector.human_baseline == baseline

    def test_with_percentage_agreement(self, multi_rater_data):
        """Test baseline with percentage agreement."""
        detector = ConvergenceDetector(human_agreement_metric="percentage_agreement")

        baseline = detector.compute_human_baseline(multi_rater_data)

        # Should be average of pairwise agreements
        assert isinstance(baseline, float)
        assert 0.0 <= baseline <= 1.0

    def test_with_cohens_kappa(self, perfect_agreement_data):
        """Test baseline with Cohen's kappa."""
        detector = ConvergenceDetector(human_agreement_metric="cohens_kappa")

        baseline = detector.compute_human_baseline(perfect_agreement_data)

        # Perfect agreement should give kappa ≈ 1.0
        assert baseline == pytest.approx(1.0, abs=0.01)

    def test_cohens_kappa_requires_two_raters(self, multi_rater_data):
        """Test that Cohen's kappa requires exactly 2 raters."""
        detector = ConvergenceDetector(human_agreement_metric="cohens_kappa")

        # multi_rater_data has 3 raters
        with pytest.raises(ValueError, match="requires exactly 2 raters"):
            detector.compute_human_baseline(multi_rater_data)

    def test_fleiss_kappa_not_implemented(self, multi_rater_data):
        """Test that Fleiss' kappa raises NotImplementedError."""
        detector = ConvergenceDetector(human_agreement_metric="fleiss_kappa")

        with pytest.raises(
            NotImplementedError, match="fleiss_kappa not yet implemented"
        ):
            detector.compute_human_baseline(multi_rater_data)

    def test_empty_ratings(self):
        """Test with empty ratings."""
        detector = ConvergenceDetector()

        with pytest.raises(ValueError, match="cannot be empty"):
            detector.compute_human_baseline({})

    def test_single_rater(self):
        """Test with single rater."""
        detector = ConvergenceDetector()

        with pytest.raises(ValueError, match="at least 2 raters"):
            detector.compute_human_baseline({"rater1": [1, 2, 3]})

    def test_high_agreement(self, perfect_agreement_data):
        """Test with high agreement data."""
        detector = ConvergenceDetector()

        baseline = detector.compute_human_baseline(
            perfect_agreement_data, metric="nominal"
        )

        # Perfect agreement should give high score
        assert baseline > 0.95

    def test_low_agreement(self, no_agreement_data):
        """Test with low agreement data."""
        detector = ConvergenceDetector()

        baseline = detector.compute_human_baseline(no_agreement_data, metric="nominal")

        # Random agreement should give low score
        assert baseline < 0.3


class TestCheckConvergence:
    """Test convergence checking."""

    def test_before_min_iterations(self):
        """Test that convergence returns False before min_iterations."""
        detector = ConvergenceDetector(min_iterations=3, convergence_threshold=0.05)
        detector.human_baseline = 0.80

        # Even if accuracy is good, should return False before min_iterations
        converged = detector.check_convergence(model_accuracy=0.85, iteration=1)
        assert converged is False

        converged = detector.check_convergence(model_accuracy=0.85, iteration=2)
        assert converged is False

    def test_after_min_iterations_converged(self):
        """Test convergence after min_iterations when criteria met."""
        detector = ConvergenceDetector(min_iterations=3, convergence_threshold=0.05)
        detector.human_baseline = 0.80

        # 0.77 >= 0.80 - 0.05 = 0.75
        converged = detector.check_convergence(model_accuracy=0.77, iteration=3)
        assert converged is True

    def test_after_min_iterations_not_converged(self):
        """Test no convergence after min_iterations when criteria not met."""
        detector = ConvergenceDetector(min_iterations=3, convergence_threshold=0.05)
        detector.human_baseline = 0.80

        # 0.74 < 0.80 - 0.05 = 0.75
        converged = detector.check_convergence(model_accuracy=0.74, iteration=3)
        assert converged is False

    def test_exact_threshold(self):
        """Test convergence exactly at threshold."""
        detector = ConvergenceDetector(min_iterations=1, convergence_threshold=0.05)
        detector.human_baseline = 0.80

        # Exactly at required accuracy: 0.75 == 0.80 - 0.05
        converged = detector.check_convergence(model_accuracy=0.75, iteration=1)
        assert converged is True

    def test_exceeds_human_baseline(self):
        """Test when model exceeds human baseline."""
        detector = ConvergenceDetector(min_iterations=1, convergence_threshold=0.05)
        detector.human_baseline = 0.80

        # Model better than humans
        converged = detector.check_convergence(model_accuracy=0.85, iteration=1)
        assert converged is True

    def test_with_explicit_human_agreement(self):
        """Test with explicit human_agreement parameter."""
        detector = ConvergenceDetector(min_iterations=1, convergence_threshold=0.05)

        # Don't set baseline, pass it explicitly
        converged = detector.check_convergence(
            model_accuracy=0.77, iteration=1, human_agreement=0.80
        )
        assert converged is True

    def test_without_baseline_raises_error(self):
        """Test that missing baseline raises ValueError."""
        detector = ConvergenceDetector(min_iterations=1)

        # No baseline set, no explicit human_agreement
        with pytest.raises(ValueError, match="human_baseline not set"):
            detector.check_convergence(model_accuracy=0.80, iteration=1)

    def test_zero_threshold(self):
        """Test with zero threshold (must match exactly)."""
        detector = ConvergenceDetector(min_iterations=1, convergence_threshold=0.0)
        detector.human_baseline = 0.80

        # Must be >= 0.80
        assert detector.check_convergence(0.80, 1) is True
        assert detector.check_convergence(0.79, 1) is False

    def test_large_threshold(self):
        """Test with large threshold."""
        detector = ConvergenceDetector(min_iterations=1, convergence_threshold=0.20)
        detector.human_baseline = 0.80

        # Need >= 0.60 (use slightly above to avoid floating point issues)
        assert detector.check_convergence(0.601, 1) is True
        assert detector.check_convergence(0.59, 1) is False


class TestStatisticalTest:
    """Test statistical significance testing."""

    def test_mcnemar_test_identical_predictions(self):
        """Test McNemar test with identical predictions."""
        detector = ConvergenceDetector()

        model_preds = [1, 1, 0, 1, 0]
        human_consensus = [1, 1, 0, 1, 0]

        result = detector.compute_statistical_test(model_preds, human_consensus)

        assert "statistic" in result
        assert "p_value" in result

        # Perfect match should have high p-value (no significant difference)
        assert result["p_value"] > 0.05
        assert result["statistic"] == 1.0  # 100% accuracy

    def test_mcnemar_test_different_predictions(self):
        """Test McNemar test with different predictions."""
        detector = ConvergenceDetector()

        model_preds = [1, 1, 0, 0, 0]
        human_consensus = [1, 0, 1, 0, 1]

        result = detector.compute_statistical_test(model_preds, human_consensus)

        # Model accuracy = 2/5 = 0.4
        assert result["statistic"] == pytest.approx(0.4, abs=0.01)

        # Should have low p-value (significant difference from 1.0)
        assert result["p_value"] < 0.05

    def test_ttest_not_implemented(self):
        """Test that t-test raises NotImplementedError."""
        detector = ConvergenceDetector()

        with pytest.raises(
            NotImplementedError, match="ttest not yet fully implemented"
        ):
            detector.compute_statistical_test([1, 1, 0], [1, 0, 0], test_type="ttest")

    def test_invalid_test_type(self):
        """Test with invalid test type."""
        detector = ConvergenceDetector()

        with pytest.raises(ValueError, match="Unknown test_type"):
            detector.compute_statistical_test([1, 1, 0], [1, 0, 0], test_type="invalid")

    def test_mismatched_lengths(self):
        """Test with mismatched prediction lengths."""
        detector = ConvergenceDetector()

        with pytest.raises(ValueError, match="must have same length"):
            detector.compute_statistical_test([1, 1, 0], [1, 0])


class TestConvergenceReport:
    """Test convergence report generation."""

    def test_basic_report(self):
        """Test basic report generation."""
        detector = ConvergenceDetector(min_iterations=3, convergence_threshold=0.05)
        detector.human_baseline = 0.80

        report = detector.get_convergence_report(model_accuracy=0.77, iteration=5)

        # Check all required keys
        assert "converged" in report
        assert "model_accuracy" in report
        assert "human_agreement" in report
        assert "gap" in report
        assert "required_accuracy" in report
        assert "threshold" in report
        assert "iteration" in report
        assert "meets_min_iterations" in report
        assert "min_iterations_required" in report

    def test_converged_report(self):
        """Test report when converged."""
        detector = ConvergenceDetector(min_iterations=2, convergence_threshold=0.05)
        detector.human_baseline = 0.80

        report = detector.get_convergence_report(model_accuracy=0.77, iteration=5)

        assert report["converged"] is True
        assert report["model_accuracy"] == 0.77
        assert report["human_agreement"] == 0.80
        assert report["gap"] == pytest.approx(0.03, abs=0.001)
        assert report["required_accuracy"] == pytest.approx(0.75, abs=0.001)
        assert report["meets_min_iterations"] is True

    def test_not_converged_report(self):
        """Test report when not converged."""
        detector = ConvergenceDetector(min_iterations=3, convergence_threshold=0.05)
        detector.human_baseline = 0.80

        report = detector.get_convergence_report(model_accuracy=0.70, iteration=5)

        assert report["converged"] is False
        assert report["gap"] == pytest.approx(0.10, abs=0.001)

    def test_not_met_min_iterations(self):
        """Test report when min_iterations not met."""
        detector = ConvergenceDetector(min_iterations=5, convergence_threshold=0.05)
        detector.human_baseline = 0.80

        report = detector.get_convergence_report(model_accuracy=0.85, iteration=3)

        assert report["converged"] is False
        assert report["meets_min_iterations"] is False
        assert report["iteration"] == 3
        assert report["min_iterations_required"] == 5

    def test_with_explicit_human_agreement(self):
        """Test report with explicit human_agreement."""
        detector = ConvergenceDetector(min_iterations=1, convergence_threshold=0.05)

        report = detector.get_convergence_report(
            model_accuracy=0.77, iteration=1, human_agreement=0.80
        )

        assert report["human_agreement"] == 0.80
        assert report["converged"] is True

    def test_without_baseline_raises_error(self):
        """Test that missing baseline raises ValueError."""
        detector = ConvergenceDetector()

        with pytest.raises(ValueError, match="human_baseline not set"):
            detector.get_convergence_report(model_accuracy=0.80, iteration=1)

    def test_gap_calculation(self):
        """Test gap calculation in report."""
        detector = ConvergenceDetector(min_iterations=1)
        detector.human_baseline = 0.85

        report = detector.get_convergence_report(model_accuracy=0.75, iteration=1)

        # Gap = human - model = 0.85 - 0.75 = 0.10
        assert report["gap"] == pytest.approx(0.10, abs=0.001)

    def test_negative_gap(self):
        """Test when model exceeds human (negative gap)."""
        detector = ConvergenceDetector(min_iterations=1)
        detector.human_baseline = 0.75

        report = detector.get_convergence_report(model_accuracy=0.85, iteration=1)

        # Gap = 0.75 - 0.85 = -0.10
        assert report["gap"] == pytest.approx(-0.10, abs=0.001)
        assert report["converged"] is True


class TestIntegration:
    """Test full convergence detection workflow."""

    def test_full_workflow(self, multi_rater_data):
        """Test complete workflow: compute baseline → check convergence → get report."""
        detector = ConvergenceDetector(
            human_agreement_metric="krippendorff_alpha",
            convergence_threshold=0.05,
            min_iterations=3,
        )

        # Step 1: Compute human baseline
        baseline = detector.compute_human_baseline(multi_rater_data, metric="nominal")
        assert detector.human_baseline is not None
        assert baseline == detector.human_baseline

        # Step 2: Check convergence (early iterations)
        assert detector.check_convergence(0.80, iteration=1) is False
        assert detector.check_convergence(0.80, iteration=2) is False

        # Step 3: Check convergence (after min_iterations)
        # Assuming baseline is around 0.70-0.80
        if baseline > 0.60:
            converged = detector.check_convergence(baseline - 0.03, iteration=3)
            assert converged is True

        # Step 4: Get full report
        report = detector.get_convergence_report(baseline - 0.03, iteration=3)
        assert "converged" in report
        assert "gap" in report

    def test_convergence_progression(self):
        """Test convergence progression over iterations."""
        detector = ConvergenceDetector(min_iterations=3, convergence_threshold=0.05)
        detector.human_baseline = 0.80

        # Simulate improving model accuracy
        accuracies = [0.60, 0.65, 0.70, 0.75, 0.78]

        results = []
        for iteration, accuracy in enumerate(accuracies, start=1):
            converged = detector.check_convergence(accuracy, iteration)
            results.append(converged)

        # Should not converge until accuracy >= 0.75 and iteration >= 3
        assert results[0] is False  # 0.60 < 0.75, iteration 1
        assert results[1] is False  # 0.65 < 0.75, iteration 2
        assert results[2] is False  # 0.70 < 0.75, iteration 3
        assert results[3] is True  # 0.75 >= 0.75, iteration 4
        assert results[4] is True  # 0.78 >= 0.75, iteration 5

    def test_different_metrics_same_data(self, perfect_agreement_data):
        """Test different metrics on same data."""
        data = perfect_agreement_data

        # Krippendorff's alpha
        detector1 = ConvergenceDetector(human_agreement_metric="krippendorff_alpha")
        baseline1 = detector1.compute_human_baseline(data, metric="nominal")

        # Percentage agreement
        detector2 = ConvergenceDetector(human_agreement_metric="percentage_agreement")
        baseline2 = detector2.compute_human_baseline(data)

        # Cohen's kappa
        detector3 = ConvergenceDetector(human_agreement_metric="cohens_kappa")
        baseline3 = detector3.compute_human_baseline(data)

        # All should give high agreement for perfect data
        assert baseline1 > 0.95
        assert baseline2 == 1.0  # Percentage agreement should be 1.0
        assert baseline3 > 0.95


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_zero_min_iterations_rejected(self):
        """Test that zero min_iterations is rejected."""
        with pytest.raises(ValueError):
            ConvergenceDetector(min_iterations=0)

    def test_one_min_iteration(self):
        """Test with min_iterations=1."""
        detector = ConvergenceDetector(min_iterations=1, convergence_threshold=0.05)
        detector.human_baseline = 0.80

        # Should converge immediately if criteria met
        assert detector.check_convergence(0.76, iteration=1) is True

    def test_very_high_threshold(self):
        """Test with very high threshold."""
        detector = ConvergenceDetector(min_iterations=1, convergence_threshold=0.50)
        detector.human_baseline = 0.80

        # Required accuracy = 0.80 - 0.50 = 0.30 (use slightly above to avoid floating point issues)
        assert detector.check_convergence(0.301, iteration=1) is True
        assert detector.check_convergence(0.29, iteration=1) is False

    def test_human_baseline_zero(self):
        """Test with zero human baseline."""
        detector = ConvergenceDetector(min_iterations=1, convergence_threshold=0.05)
        detector.human_baseline = 0.0

        # Required accuracy = 0.0 - 0.05 = -0.05
        # Any model accuracy should converge
        assert detector.check_convergence(0.0, iteration=1) is True
        assert detector.check_convergence(0.5, iteration=1) is True

    def test_human_baseline_one(self):
        """Test with perfect human baseline."""
        detector = ConvergenceDetector(min_iterations=1, convergence_threshold=0.05)
        detector.human_baseline = 1.0

        # Required accuracy = 1.0 - 0.05 = 0.95
        assert detector.check_convergence(0.95, iteration=1) is True
        assert detector.check_convergence(0.94, iteration=1) is False

    def test_model_accuracy_bounds(self):
        """Test with model accuracy at boundaries."""
        detector = ConvergenceDetector(min_iterations=1, convergence_threshold=0.05)
        detector.human_baseline = 0.50

        # Test with 0.0 and 1.0 accuracy
        assert detector.check_convergence(1.0, iteration=1) is True
        assert detector.check_convergence(0.0, iteration=1) is False

    def test_with_missing_data_in_ratings(self, ratings_with_missing):
        """Test baseline computation with missing data."""
        detector = ConvergenceDetector(human_agreement_metric="krippendorff_alpha")

        # Krippendorff's alpha handles missing data
        baseline = detector.compute_human_baseline(
            ratings_with_missing, metric="nominal"
        )

        assert isinstance(baseline, float)
        assert -1.0 <= baseline <= 1.0
