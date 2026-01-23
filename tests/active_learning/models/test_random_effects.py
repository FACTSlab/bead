"""Tests for RandomEffectsManager with GLMM variance estimation."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from bead.active_learning.config import MixedEffectsConfig, VarianceComponents
from bead.active_learning.models.random_effects import RandomEffectsManager


class TestRandomEffectsManagerInit:
    """Test RandomEffectsManager initialization."""

    def test_init_with_fixed_mode(self) -> None:
        """Test initialization with fixed mode."""
        config = MixedEffectsConfig(mode="fixed")
        manager = RandomEffectsManager(config)

        assert manager.config.mode == "fixed"
        assert len(manager.intercepts) == 0
        assert len(manager.slopes) == 0
        assert len(manager.participant_sample_counts) == 0
        assert manager.variance_components is None
        assert len(manager.variance_history) == 0

    def test_init_with_random_intercepts_mode(self) -> None:
        """Test initialization with random_intercepts mode."""
        config = MixedEffectsConfig(mode="random_intercepts", prior_variance=0.5)
        manager = RandomEffectsManager(config, n_classes=3)

        assert manager.config.mode == "random_intercepts"
        assert manager.config.prior_variance == 0.5
        assert manager.creation_kwargs["n_classes"] == 3

    def test_init_with_random_slopes_mode(self) -> None:
        """Test initialization with random_slopes mode."""
        config = MixedEffectsConfig(mode="random_slopes", adaptive_regularization=True)
        manager = RandomEffectsManager(config, hidden_dim=768, n_classes=2)

        assert manager.config.mode == "random_slopes"
        assert manager.config.adaptive_regularization is True
        assert manager.creation_kwargs["hidden_dim"] == 768


class TestRegisterParticipant:
    """Test participant registration and sample tracking."""

    def test_register_single_participant(self) -> None:
        """Test registering a single participant."""
        config = MixedEffectsConfig(mode="random_intercepts")
        manager = RandomEffectsManager(config)

        manager.register_participant("alice", n_samples=10)

        assert manager.participant_sample_counts["alice"] == 10

    def test_register_multiple_participants(self) -> None:
        """Test registering multiple participants."""
        config = MixedEffectsConfig(mode="random_intercepts")
        manager = RandomEffectsManager(config)

        manager.register_participant("alice", n_samples=10)
        manager.register_participant("bob", n_samples=15)
        manager.register_participant("charlie", n_samples=8)

        assert len(manager.participant_sample_counts) == 3
        assert manager.participant_sample_counts["alice"] == 10
        assert manager.participant_sample_counts["bob"] == 15
        assert manager.participant_sample_counts["charlie"] == 8

    def test_register_participant_accumulates_samples(self) -> None:
        """Test that registering same participant accumulates samples."""
        config = MixedEffectsConfig(mode="random_intercepts")
        manager = RandomEffectsManager(config)

        manager.register_participant("alice", n_samples=10)
        manager.register_participant("alice", n_samples=5)
        manager.register_participant("alice", n_samples=3)

        assert manager.participant_sample_counts["alice"] == 18

    def test_register_participant_empty_id_raises(self) -> None:
        """Test that empty participant_id raises ValueError."""
        config = MixedEffectsConfig(mode="random_intercepts")
        manager = RandomEffectsManager(config)

        with pytest.raises(ValueError, match="participant_id cannot be empty"):
            manager.register_participant("", n_samples=10)

    def test_register_participant_zero_samples_raises(self) -> None:
        """Test that n_samples=0 raises ValueError."""
        config = MixedEffectsConfig(mode="random_intercepts")
        manager = RandomEffectsManager(config)

        with pytest.raises(ValueError, match="n_samples must be positive"):
            manager.register_participant("alice", n_samples=0)

    def test_register_participant_negative_samples_raises(self) -> None:
        """Test that negative n_samples raises ValueError."""
        config = MixedEffectsConfig(mode="random_intercepts")
        manager = RandomEffectsManager(config)

        with pytest.raises(ValueError, match="n_samples must be positive"):
            manager.register_participant("alice", n_samples=-5)


class TestGetIntercepts:
    """Test random intercepts retrieval and creation."""

    def test_get_intercepts_creates_if_missing(self) -> None:
        """Test that get_intercepts creates intercepts for unknown participant."""
        config = MixedEffectsConfig(mode="random_intercepts", prior_variance=1.0)
        manager = RandomEffectsManager(config)

        bias = manager.get_intercepts(
            "alice", n_classes=3, param_name="bias", create_if_missing=True
        )

        assert bias.shape == (3,)
        assert bias.requires_grad
        assert "alice" in manager.intercepts.get("bias", {})

    def test_get_intercepts_returns_existing(self) -> None:
        """Test that get_intercepts returns existing intercepts."""
        config = MixedEffectsConfig(mode="random_intercepts")
        manager = RandomEffectsManager(config)

        # First call creates
        bias1 = manager.get_intercepts(
            "alice", n_classes=3, param_name="bias", create_if_missing=True
        )
        # Second call returns same
        bias2 = manager.get_intercepts(
            "alice", n_classes=3, param_name="bias", create_if_missing=True
        )

        assert torch.equal(bias1, bias2)

    def test_get_intercepts_not_create_if_missing_returns_prior(self) -> None:
        """Test that create_if_missing=False returns prior mean for unknown."""
        config = MixedEffectsConfig(mode="random_intercepts", prior_mean=0.0)
        manager = RandomEffectsManager(config)

        bias = manager.get_intercepts(
            "unknown", n_classes=3, param_name="bias", create_if_missing=False
        )

        assert bias.shape == (3,)
        assert torch.allclose(bias, torch.zeros(3))
        assert "unknown" not in manager.intercepts.get("bias", {})

    def test_get_intercepts_uses_prior_mean(self) -> None:
        """Test that intercepts use prior_mean when initialized."""
        config = MixedEffectsConfig(
            mode="random_intercepts", prior_mean=2.0, prior_variance=0.0
        )
        manager = RandomEffectsManager(config)

        bias = manager.get_intercepts(
            "alice", n_classes=3, param_name="bias", create_if_missing=True
        )

        # With zero variance, should equal prior mean exactly
        assert torch.allclose(bias, torch.full((3,), 2.0))

    def test_get_intercepts_wrong_mode_raises(self) -> None:
        """Test that get_intercepts raises if mode is not random_intercepts."""
        config = MixedEffectsConfig(mode="fixed")
        manager = RandomEffectsManager(config)

        with pytest.raises(ValueError, match="expected 'random_intercepts'"):
            manager.get_intercepts("alice", n_classes=3, param_name="bias")

    def test_get_intercepts_random_slopes_mode_raises(self) -> None:
        """Test that get_intercepts raises if mode is random_slopes."""
        config = MixedEffectsConfig(mode="random_slopes")
        manager = RandomEffectsManager(config)

        with pytest.raises(ValueError, match="expected 'random_intercepts'"):
            manager.get_intercepts("alice", n_classes=3, param_name="bias")


class TestGetInterceptsWithShrinkage:
    """Test Empirical Bayes shrinkage for intercepts."""

    def test_shrinkage_for_small_sample_participant(self) -> None:
        """Test that small sample participants are shrunk toward prior mean."""
        config = MixedEffectsConfig(
            mode="random_intercepts",
            prior_mean=0.0,
            prior_variance=1.0,
            min_samples_for_random_effects=10,
        )
        manager = RandomEffectsManager(config)

        # Create participant with strong deviation
        manager.register_participant("alice", n_samples=2)
        manager.intercepts.setdefault("bias", {})["alice"] = torch.tensor(
            [5.0, 5.0, 5.0], requires_grad=True
        )

        bias_shrunk = manager.get_intercepts_with_shrinkage("alice", n_classes=3)

        # With 2 samples and k=10, λ = 2/(2+10) = 1/6
        # Shrunk value should be closer to 0 than to 5
        assert torch.all(bias_shrunk < torch.tensor([5.0, 5.0, 5.0]))
        assert torch.all(bias_shrunk > torch.tensor([0.0, 0.0, 0.0]))

    def test_shrinkage_for_large_sample_participant(self) -> None:
        """Test that large sample participants use their specific estimate."""
        config = MixedEffectsConfig(
            mode="random_intercepts", prior_mean=0.0, min_samples_for_random_effects=10
        )
        manager = RandomEffectsManager(config)

        # Large sample participant
        manager.register_participant("bob", n_samples=100)
        manager.intercepts.setdefault("bias", {})["bob"] = torch.tensor(
            [5.0, 5.0, 5.0], requires_grad=True
        )

        bias_shrunk = manager.get_intercepts_with_shrinkage("bob", n_classes=3)

        # With 100 samples and k=10, λ = 100/(100+10) ≈ 0.91
        # Shrunk value should be very close to original
        assert torch.allclose(bias_shrunk, torch.tensor([5.0, 5.0, 5.0]), atol=1.0)

    def test_shrinkage_for_unknown_participant(self) -> None:
        """Test that unknown participants return prior mean (no shrinkage)."""
        config = MixedEffectsConfig(mode="random_intercepts", prior_mean=0.0)
        manager = RandomEffectsManager(config)

        bias_shrunk = manager.get_intercepts_with_shrinkage("unknown", n_classes=3)

        assert torch.allclose(bias_shrunk, torch.zeros(3))

    def test_shrinkage_uses_variance_components_if_available(self) -> None:
        """Test that shrinkage uses variance components if estimated."""
        config = MixedEffectsConfig(mode="random_intercepts", prior_mean=0.0)
        manager = RandomEffectsManager(config)

        # Set variance components
        manager.variance_components = VarianceComponents(
            grouping_factor="participant",
            effect_type="intercept",
            variance=2.0,  # Large variance → less shrinkage
            n_groups=10,
            n_observations_per_group={"alice": 5},
        )

        manager.register_participant("alice", n_samples=5)
        manager.intercepts.setdefault("bias", {})["alice"] = torch.tensor(
            [5.0, 5.0, 5.0], requires_grad=True
        )

        bias_shrunk = manager.get_intercepts_with_shrinkage("alice", n_classes=3)

        # With variance=2.0, k = 1.0/2.0 = 0.5, λ = 5/(5+0.5) ≈ 0.91
        # Should be close to original
        assert torch.allclose(bias_shrunk, torch.tensor([5.0, 5.0, 5.0]), atol=1.0)

    def test_shrinkage_wrong_mode_raises(self) -> None:
        """Test that shrinkage raises if mode is not random_intercepts."""
        config = MixedEffectsConfig(mode="random_slopes")
        manager = RandomEffectsManager(config)

        with pytest.raises(
            ValueError, match="Shrinkage only for random_intercepts mode"
        ):
            manager.get_intercepts_with_shrinkage("alice", n_classes=3)


class TestGetSlopes:
    """Test random slopes retrieval and creation."""

    def test_get_slopes_creates_if_missing(self) -> None:
        """Test that get_slopes creates slopes for unknown participant."""
        config = MixedEffectsConfig(mode="random_slopes", prior_variance=0.1)
        manager = RandomEffectsManager(config)

        fixed_head = nn.Linear(10, 3)
        head = manager.get_slopes("alice", fixed_head, create_if_missing=True)

        assert isinstance(head, nn.Linear)
        assert "alice" in manager.slopes

    def test_get_slopes_returns_existing(self) -> None:
        """Test that get_slopes returns existing slopes."""
        config = MixedEffectsConfig(mode="random_slopes")
        manager = RandomEffectsManager(config)

        fixed_head = nn.Linear(10, 3)
        head1 = manager.get_slopes("alice", fixed_head, create_if_missing=True)
        head2 = manager.get_slopes("alice", fixed_head, create_if_missing=True)

        assert head1 is head2

    def test_get_slopes_not_create_if_missing_returns_clone(self) -> None:
        """Test that create_if_missing=False returns clone of fixed head."""
        config = MixedEffectsConfig(mode="random_slopes")
        manager = RandomEffectsManager(config)

        fixed_head = nn.Linear(10, 3)
        head = manager.get_slopes("unknown", fixed_head, create_if_missing=False)

        assert isinstance(head, nn.Linear)
        assert "unknown" not in manager.slopes
        assert head is not fixed_head  # Should be a clone

    def test_get_slopes_adds_noise_to_parameters(self) -> None:
        """Test that new slopes have noise added to parameters."""
        config = MixedEffectsConfig(mode="random_slopes", prior_variance=0.1)
        manager = RandomEffectsManager(config)

        fixed_head = nn.Linear(10, 3)
        with torch.no_grad():
            fixed_head.weight.fill_(1.0)
            fixed_head.bias.fill_(0.0)

        head = manager.get_slopes("alice", fixed_head, create_if_missing=True)

        # Parameters should be different from fixed head due to noise
        assert not torch.allclose(head.weight, fixed_head.weight)
        assert not torch.allclose(head.bias, fixed_head.bias)

    def test_get_slopes_wrong_mode_raises(self) -> None:
        """Test that get_slopes raises if mode is not random_slopes."""
        config = MixedEffectsConfig(mode="fixed")
        manager = RandomEffectsManager(config)

        fixed_head = nn.Linear(10, 3)
        with pytest.raises(ValueError, match="expected 'random_slopes'"):
            manager.get_slopes("alice", fixed_head)


class TestEstimateVarianceComponents:
    """Test variance component estimation (G matrix)."""

    def test_estimate_variance_fixed_mode_returns_none(self) -> None:
        """Test that fixed mode returns None for variance components."""
        config = MixedEffectsConfig(mode="fixed")
        manager = RandomEffectsManager(config)

        var_comp = manager.estimate_variance_components()

        assert var_comp is None

    def test_estimate_variance_intercepts_empty_returns_none(self) -> None:
        """Test that empty intercepts returns None."""
        config = MixedEffectsConfig(mode="random_intercepts")
        manager = RandomEffectsManager(config)

        var_comp = manager.estimate_variance_components()

        assert var_comp is None

    def test_estimate_variance_intercepts_single_participant(self) -> None:
        """Test variance estimation with single participant."""
        config = MixedEffectsConfig(mode="random_intercepts")
        manager = RandomEffectsManager(config)

        manager.register_participant("alice", n_samples=10)
        manager.intercepts.setdefault("bias", {})["alice"] = torch.tensor(
            [1.0, 2.0, 3.0]
        )

        var_comps = manager.estimate_variance_components()

        assert var_comps is not None
        assert "bias" in var_comps
        var_comp = var_comps["bias"]
        assert var_comp.grouping_factor == "participant"
        assert var_comp.effect_type == "intercept"
        assert var_comp.n_groups == 1
        assert var_comp.variance >= 0.0

    def test_estimate_variance_intercepts_multiple_participants(self) -> None:
        """Test variance estimation with multiple participants."""
        config = MixedEffectsConfig(mode="random_intercepts")
        manager = RandomEffectsManager(config)

        manager.register_participant("alice", n_samples=10)
        manager.register_participant("bob", n_samples=15)
        manager.register_participant("charlie", n_samples=8)

        # Create intercepts with different values
        manager.intercepts.setdefault("bias", {})["alice"] = torch.tensor(
            [0.0, 0.0, 0.0]
        )
        manager.intercepts.setdefault("bias", {})["bob"] = torch.tensor([1.0, 1.0, 1.0])
        manager.intercepts.setdefault("bias", {})["charlie"] = torch.tensor(
            [2.0, 2.0, 2.0]
        )

        var_comps = manager.estimate_variance_components()

        assert var_comps is not None
        assert "bias" in var_comps
        var_comp = var_comps["bias"]
        assert var_comp.n_groups == 3
        assert var_comp.variance > 0.0  # Should have variance across participants

    def test_estimate_variance_intercepts_updates_history(self) -> None:
        """Test that variance estimation updates history."""
        config = MixedEffectsConfig(mode="random_intercepts")
        manager = RandomEffectsManager(config)

        manager.register_participant("alice", n_samples=10)
        manager.intercepts.setdefault("bias", {})["alice"] = torch.tensor(
            [1.0, 2.0, 3.0]
        )

        # First estimation
        manager.estimate_variance_components()
        assert len(manager.variance_history) == 1

        # Second estimation
        manager.register_participant("bob", n_samples=15)
        manager.intercepts.setdefault("bias", {})["bob"] = torch.tensor([4.0, 5.0, 6.0])
        var_comp2 = manager.estimate_variance_components()

        assert len(manager.variance_history) == 2
        assert manager.variance_components == var_comp2

    def test_estimate_variance_slopes_empty_returns_none(self) -> None:
        """Test that empty slopes returns None."""
        config = MixedEffectsConfig(mode="random_slopes")
        manager = RandomEffectsManager(config)

        var_comp = manager.estimate_variance_components()

        assert var_comp is None

    def test_estimate_variance_slopes_multiple_participants(self) -> None:
        """Test variance estimation with random slopes."""
        config = MixedEffectsConfig(mode="random_slopes")
        manager = RandomEffectsManager(config)

        fixed_head = nn.Linear(10, 3)

        manager.register_participant("alice", n_samples=10)
        manager.register_participant("bob", n_samples=15)

        # Create participant-specific heads
        manager.get_slopes("alice", fixed_head, create_if_missing=True)
        manager.get_slopes("bob", fixed_head, create_if_missing=True)

        var_comps = manager.estimate_variance_components()

        assert var_comps is not None
        # Slopes return a single variance component (not per-param)
        assert "slopes" in var_comps
        var_comp = var_comps["slopes"]
        assert var_comp.grouping_factor == "participant"
        assert var_comp.effect_type == "slope"
        assert var_comp.n_groups == 2
        assert var_comp.variance >= 0.0


class TestComputePriorLoss:
    """Test prior regularization loss computation."""

    def test_prior_loss_fixed_mode_returns_zero(self) -> None:
        """Test that fixed mode returns zero loss."""
        config = MixedEffectsConfig(mode="fixed")
        manager = RandomEffectsManager(config)

        loss = manager.compute_prior_loss()

        assert torch.allclose(loss, torch.tensor(0.0))

    def test_prior_loss_intercepts_no_participants_returns_zero(self) -> None:
        """Test that no participants returns zero loss."""
        config = MixedEffectsConfig(mode="random_intercepts")
        manager = RandomEffectsManager(config)

        loss = manager.compute_prior_loss()

        assert torch.allclose(loss, torch.tensor(0.0))

    def test_prior_loss_intercepts_single_participant(self) -> None:
        """Test prior loss with single participant."""
        config = MixedEffectsConfig(
            mode="random_intercepts",
            prior_mean=0.0,
            regularization_strength=0.1,
            adaptive_regularization=False,
        )
        manager = RandomEffectsManager(config)

        manager.intercepts.setdefault("bias", {})["alice"] = torch.tensor(
            [1.0, 2.0, 3.0], requires_grad=True
        )

        loss = manager.compute_prior_loss()

        # Loss = 0.1 * (1² + 2² + 3²) = 0.1 * 14 = 1.4
        expected = 0.1 * (1.0**2 + 2.0**2 + 3.0**2)
        assert torch.allclose(loss, torch.tensor(expected), atol=1e-5)

    def test_prior_loss_intercepts_multiple_participants(self) -> None:
        """Test prior loss with multiple participants."""
        config = MixedEffectsConfig(
            mode="random_intercepts",
            prior_mean=0.0,
            regularization_strength=0.1,
            adaptive_regularization=False,
        )
        manager = RandomEffectsManager(config)

        manager.intercepts.setdefault("bias", {})["alice"] = torch.tensor(
            [1.0, 0.0, 0.0], requires_grad=True
        )
        manager.intercepts.setdefault("bias", {})["bob"] = torch.tensor(
            [0.0, 1.0, 0.0], requires_grad=True
        )

        loss = manager.compute_prior_loss()

        # Loss = 0.1 * (1² + 0² + 0² + 0² + 1² + 0²) = 0.1 * 2 = 0.2
        expected = 0.1 * 2.0
        assert torch.allclose(loss, torch.tensor(expected), atol=1e-5)

    def test_prior_loss_adaptive_regularization(self) -> None:
        """Test adaptive regularization gives stronger weight to small samples."""
        config = MixedEffectsConfig(
            mode="random_intercepts",
            prior_mean=0.0,
            regularization_strength=1.0,
            adaptive_regularization=True,
            min_samples_for_random_effects=5,
        )
        manager = RandomEffectsManager(config)

        # Alice has 2 samples → weight = 1/5 = 0.2
        manager.register_participant("alice", n_samples=2)
        manager.intercepts.setdefault("bias", {})["alice"] = torch.tensor(
            [10.0, 0.0, 0.0], requires_grad=True
        )

        # Bob has 20 samples → weight = 1/20 = 0.05
        manager.register_participant("bob", n_samples=20)
        manager.intercepts.setdefault("bias", {})["bob"] = torch.tensor(
            [10.0, 0.0, 0.0], requires_grad=True
        )

        loss = manager.compute_prior_loss()

        # Alice contribution: 0.2 * 100 = 20
        # Bob contribution: 0.05 * 100 = 5
        # Total: 1.0 * (20 + 5) = 25
        expected = 25.0
        assert torch.allclose(loss, torch.tensor(expected), atol=1e-5)

    def test_prior_loss_slopes_single_participant(self) -> None:
        """Test prior loss with random slopes."""
        config = MixedEffectsConfig(
            mode="random_slopes",
            regularization_strength=0.1,
            adaptive_regularization=False,
        )
        manager = RandomEffectsManager(config)

        # Simple linear layer
        head = nn.Linear(2, 2)
        with torch.no_grad():
            head.weight.fill_(1.0)
            head.bias.fill_(0.0)

        manager.slopes["alice"] = head

        loss = manager.compute_prior_loss()

        # Loss = 0.1 * (4 * 1² + 2 * 0²) = 0.1 * 4 = 0.4
        assert loss.item() > 0.0


class TestSaveLoad:
    """Test saving and loading random effects."""

    def test_save_load_intercepts(self) -> None:
        """Test saving and loading random intercepts."""
        config = MixedEffectsConfig(mode="random_intercepts")
        manager = RandomEffectsManager(config)

        manager.register_participant("alice", n_samples=10)
        manager.register_participant("bob", n_samples=15)
        manager.intercepts.setdefault("bias", {})["alice"] = torch.tensor(
            [1.0, 2.0, 3.0]
        )
        manager.intercepts.setdefault("bias", {})["bob"] = torch.tensor([4.0, 5.0, 6.0])

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "random_effects"
            manager.save(path)

            # Load into new manager
            new_manager = RandomEffectsManager(config)
            new_manager.load(path)

            assert len(new_manager.intercepts.get("bias", {})) == 2
            assert torch.equal(
                new_manager.intercepts["bias"]["alice"], torch.tensor([1.0, 2.0, 3.0])
            )
            assert torch.equal(
                new_manager.intercepts["bias"]["bob"], torch.tensor([4.0, 5.0, 6.0])
            )
            assert new_manager.participant_sample_counts["alice"] == 10
            assert new_manager.participant_sample_counts["bob"] == 15

    def test_save_load_slopes(self) -> None:
        """Test saving and loading random slopes."""
        config = MixedEffectsConfig(mode="random_slopes")
        manager = RandomEffectsManager(config)

        fixed_head = nn.Linear(10, 3)

        manager.register_participant("alice", n_samples=10)
        head_alice = manager.get_slopes("alice", fixed_head, create_if_missing=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "random_effects"
            manager.save(path)

            # Load into new manager
            new_manager = RandomEffectsManager(config)
            new_manager.load(path, fixed_head=fixed_head)

            assert len(new_manager.slopes) == 1
            assert "alice" in new_manager.slopes

            # Check parameters match
            for p1, p2 in zip(
                head_alice.parameters(),
                new_manager.slopes["alice"].parameters(),
                strict=False,
            ):
                assert torch.equal(p1, p2)

    def test_save_load_variance_history(self) -> None:
        """Test saving and loading variance component history."""
        config = MixedEffectsConfig(mode="random_intercepts")
        manager = RandomEffectsManager(config)

        manager.register_participant("alice", n_samples=10)
        manager.intercepts.setdefault("bias", {})["alice"] = torch.tensor(
            [1.0, 2.0, 3.0]
        )

        # Estimate variance multiple times
        manager.estimate_variance_components()
        manager.register_participant("bob", n_samples=15)
        manager.intercepts.setdefault("bias", {})["bob"] = torch.tensor([4.0, 5.0, 6.0])
        manager.estimate_variance_components()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "random_effects"
            manager.save(path)

            # Load into new manager
            new_manager = RandomEffectsManager(config)
            new_manager.load(path)

            assert len(new_manager.variance_history) == 2
            assert new_manager.variance_components is not None
            # variance_components is now a dict; check the bias key matches last history
            assert (
                new_manager.variance_components["bias"]
                == new_manager.variance_history[-1]
            )

    def test_load_nonexistent_path_raises(self) -> None:
        """Test that loading from nonexistent path raises FileNotFoundError."""
        config = MixedEffectsConfig(mode="random_intercepts")
        manager = RandomEffectsManager(config)

        with pytest.raises(
            FileNotFoundError, match="Random effects directory not found"
        ):
            manager.load(Path("/nonexistent/path"))

    def test_load_slopes_without_fixed_head_raises(self) -> None:
        """Test that loading slopes without fixed_head raises ValueError."""
        config = MixedEffectsConfig(mode="random_slopes")
        manager = RandomEffectsManager(config)

        fixed_head = nn.Linear(10, 3)
        manager.register_participant("alice", n_samples=10)
        manager.get_slopes("alice", fixed_head, create_if_missing=True)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "random_effects"
            manager.save(path)

            # Try to load without fixed_head
            new_manager = RandomEffectsManager(config)
            with pytest.raises(ValueError, match="fixed_head is required"):
                new_manager.load(path, fixed_head=None)


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_variance_with_zero_variance_intercepts(self) -> None:
        """Test variance estimation when all intercepts are identical."""
        config = MixedEffectsConfig(mode="random_intercepts")
        manager = RandomEffectsManager(config)

        # All participants have identical intercepts
        manager.register_participant("alice", n_samples=10)
        manager.register_participant("bob", n_samples=15)
        manager.intercepts.setdefault("bias", {})["alice"] = torch.tensor(
            [1.0, 1.0, 1.0]
        )
        manager.intercepts.setdefault("bias", {})["bob"] = torch.tensor([1.0, 1.0, 1.0])

        var_comps = manager.estimate_variance_components()

        # Variance should be very close to zero
        assert var_comps is not None
        assert "bias" in var_comps
        assert var_comps["bias"].variance < 1e-5

    def test_many_participants(self) -> None:
        """Test with many participants (stress test)."""
        config = MixedEffectsConfig(mode="random_intercepts")
        manager = RandomEffectsManager(config)

        # Register 100 participants
        for i in range(100):
            manager.register_participant(f"participant_{i}", n_samples=i + 1)
            manager.intercepts.setdefault("bias", {})[f"participant_{i}"] = torch.randn(
                3
            )

        var_comps = manager.estimate_variance_components()

        assert var_comps is not None
        assert "bias" in var_comps
        var_comp = var_comps["bias"]
        assert var_comp.n_groups == 100
        assert len(var_comp.n_observations_per_group) == 100

    def test_prior_loss_with_non_zero_prior_mean(self) -> None:
        """Test prior loss when prior_mean != 0."""
        config = MixedEffectsConfig(
            mode="random_intercepts",
            prior_mean=5.0,
            regularization_strength=0.1,
            adaptive_regularization=False,
        )
        manager = RandomEffectsManager(config)

        # Intercept at prior mean should have zero loss
        manager.intercepts.setdefault("bias", {})["alice"] = torch.tensor(
            [5.0, 5.0, 5.0], requires_grad=True
        )

        loss = manager.compute_prior_loss()

        # Deviation is zero, so loss should be zero
        assert torch.allclose(loss, torch.tensor(0.0), atol=1e-5)
