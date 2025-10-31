"""Integration tests for active learning loop with TwoAFCModel."""

from bead.active_learning.loop import ActiveLearningLoop
from bead.active_learning.models.forced_choice import ForcedChoiceModel
from bead.active_learning.selection import UncertaintySampler
from bead.config.models import (
    ActiveLearningLoopConfig,
    ForcedChoiceModelConfig,
    UncertaintySamplerConfig,
)


def test_active_learning_loop_with_default_model(test_items, forced_choice_template):
    """Test active learning loop with explicit model."""
    initial_items = test_items(n=20)
    unlabeled_pool = test_items(n=50)

    human_ratings = {
        str(item.id): "option_a" if i < 10 else "option_b"
        for i, item in enumerate(initial_items)
    }

    model_config = ForcedChoiceModelConfig(num_epochs=1, batch_size=4, device="cpu")
    model = ForcedChoiceModel(config=model_config)
    selector_config = UncertaintySamplerConfig(method="entropy")
    selector = UncertaintySampler(config=selector_config)
    loop_config = ActiveLearningLoopConfig(max_iterations=3, budget_per_iteration=10)
    loop = ActiveLearningLoop(item_selector=selector, config=loop_config)

    results = loop.run(
        initial_items=initial_items,
        initial_model=model,
        item_template=forced_choice_template,
        unlabeled_pool=unlabeled_pool,
        human_ratings=human_ratings,
    )

    assert len(results) == 3  # 3 iterations
    assert all("accuracy" in r.metrics for r in results)
    assert all(r.metrics["accuracy"] >= 0.0 for r in results)


def test_active_learning_with_custom_model(test_items, forced_choice_template):
    """Test loop works with custom model instance."""
    items = test_items(n=20)
    pool = test_items(n=30)
    ratings = {
        str(item.id): "option_a" if i < 10 else "option_b"
        for i, item in enumerate(items)
    }

    # Use pre-configured model
    model_config = ForcedChoiceModelConfig(num_epochs=1, batch_size=4, device="cpu")
    model = ForcedChoiceModel(config=model_config)

    selector_config = UncertaintySamplerConfig(method="entropy")
    loop_config = ActiveLearningLoopConfig(max_iterations=2, budget_per_iteration=5)
    loop = ActiveLearningLoop(
        item_selector=UncertaintySampler(config=selector_config),
        config=loop_config,
    )

    results = loop.run(
        initial_items=items,
        initial_model=model,  # Custom model
        item_template=forced_choice_template,
        unlabeled_pool=pool,
        human_ratings=ratings,
    )

    assert len(results) == 2


def test_active_learning_selects_items_from_pool(test_items, forced_choice_template):
    """Test that active learning actually selects items from unlabeled pool."""
    initial_items = test_items(n=10)
    unlabeled_pool = test_items(n=20)

    ratings = {
        str(item.id): "option_a" if i < 5 else "option_b"
        for i, item in enumerate(initial_items)
    }

    selector_config = UncertaintySamplerConfig(method="entropy")
    loop_config = ActiveLearningLoopConfig(max_iterations=2, budget_per_iteration=5)
    loop = ActiveLearningLoop(
        item_selector=UncertaintySampler(config=selector_config),
        config=loop_config,
    )

    model_config = ForcedChoiceModelConfig(num_epochs=1, batch_size=4, device="cpu")
    results = loop.run(
        initial_items=initial_items,
        initial_model=ForcedChoiceModel(config=model_config),
        item_template=forced_choice_template,
        unlabeled_pool=unlabeled_pool,
        human_ratings=ratings,
    )

    # Check that items were selected
    assert len(loop.iteration_history) == 2
    for iteration_result in loop.iteration_history:
        assert len(iteration_result["selected_items"]) <= 5


def test_active_learning_empty_unlabeled_pool(test_items, forced_choice_template):
    """Test active learning with empty unlabeled pool."""
    initial_items = test_items(n=10)
    unlabeled_pool = []

    ratings = {str(item.id): "option_a" for item in initial_items}

    selector_config = UncertaintySamplerConfig(method="entropy")
    loop_config = ActiveLearningLoopConfig(max_iterations=3, budget_per_iteration=5)
    loop = ActiveLearningLoop(
        item_selector=UncertaintySampler(config=selector_config),
        config=loop_config,
    )

    model_config = ForcedChoiceModelConfig(num_epochs=1, batch_size=4, device="cpu")
    results = loop.run(
        initial_items=initial_items,
        initial_model=ForcedChoiceModel(config=model_config),
        item_template=forced_choice_template,
        unlabeled_pool=unlabeled_pool,
        human_ratings=ratings,
    )

    # Should return empty results since no unlabeled pool
    assert len(results) == 0


def test_active_learning_without_ratings_stops(test_items, forced_choice_template):
    """Test that loop stops if no ratings provided."""
    initial_items = test_items(n=10)
    unlabeled_pool = test_items(n=20)

    selector_config = UncertaintySamplerConfig(method="entropy")
    loop_config = ActiveLearningLoopConfig(max_iterations=3, budget_per_iteration=5)
    loop = ActiveLearningLoop(
        item_selector=UncertaintySampler(config=selector_config),
        config=loop_config,
    )

    model_config = ForcedChoiceModelConfig(num_epochs=1, batch_size=4, device="cpu")
    results = loop.run(
        initial_items=initial_items,
        initial_model=ForcedChoiceModel(config=model_config),
        item_template=forced_choice_template,
        unlabeled_pool=unlabeled_pool,
        human_ratings=None,  # No ratings
    )

    # Should stop early without ratings
    assert len(results) == 0


def test_active_learning_budget_respected(test_items, forced_choice_template):
    """Test that budget per iteration is respected."""
    initial_items = test_items(n=10)
    unlabeled_pool = test_items(n=50)

    ratings = {
        str(item.id): "option_a" if i < 5 else "option_b"
        for i, item in enumerate(initial_items)
    }

    selector_config = UncertaintySamplerConfig(method="entropy")
    loop_config = ActiveLearningLoopConfig(max_iterations=1, budget_per_iteration=7)
    loop = ActiveLearningLoop(
        item_selector=UncertaintySampler(config=selector_config),
        config=loop_config,
    )

    model_config = ForcedChoiceModelConfig(num_epochs=1, batch_size=4, device="cpu")
    loop.run(
        initial_items=initial_items,
        initial_model=ForcedChoiceModel(config=model_config),
        item_template=forced_choice_template,
        unlabeled_pool=unlabeled_pool,
        human_ratings=ratings,
    )

    # Check that selected items respect budget
    assert len(loop.iteration_history[0]["selected_items"]) <= 7


def test_active_learning_small_unlabeled_pool(test_items, forced_choice_template):
    """Test when unlabeled pool is smaller than budget."""
    initial_items = test_items(n=10)
    unlabeled_pool = test_items(n=3)  # Smaller than budget

    ratings = {
        str(item.id): "option_a" if i < 5 else "option_b"
        for i, item in enumerate(initial_items)
    }

    selector_config = UncertaintySamplerConfig(method="entropy")
    loop_config = ActiveLearningLoopConfig(max_iterations=1, budget_per_iteration=10)
    loop = ActiveLearningLoop(
        item_selector=UncertaintySampler(config=selector_config),
        config=loop_config,
    )

    model_config = ForcedChoiceModelConfig(num_epochs=1, batch_size=4, device="cpu")
    loop.run(
        initial_items=initial_items,
        initial_model=ForcedChoiceModel(config=model_config),
        item_template=forced_choice_template,
        unlabeled_pool=unlabeled_pool,
        human_ratings=ratings,
    )

    # Should select all available items
    assert len(loop.iteration_history[0]["selected_items"]) <= 3


def test_active_learning_loop_summary(test_items, forced_choice_template):
    """Test get_summary method."""
    initial_items = test_items(n=10)
    unlabeled_pool = test_items(n=20)

    ratings = {
        str(item.id): "option_a" if i < 5 else "option_b"
        for i, item in enumerate(initial_items)
    }

    selector_config = UncertaintySamplerConfig(method="entropy")
    loop_config = ActiveLearningLoopConfig(max_iterations=2, budget_per_iteration=5)
    loop = ActiveLearningLoop(
        item_selector=UncertaintySampler(config=selector_config),
        config=loop_config,
    )

    model_config = ForcedChoiceModelConfig(num_epochs=1, batch_size=4, device="cpu")
    loop.run(
        initial_items=initial_items,
        initial_model=ForcedChoiceModel(config=model_config),
        item_template=forced_choice_template,
        unlabeled_pool=unlabeled_pool,
        human_ratings=ratings,
    )

    summary = loop.get_summary()

    assert summary["total_iterations"] == 2
    assert summary["total_items_selected"] > 0
    assert "convergence_info" in summary


def test_active_learning_model_trains_each_iteration(
    test_items, forced_choice_template
):
    """Test that model is retrained each iteration."""
    initial_items = test_items(n=15)
    unlabeled_pool = test_items(n=30)

    ratings = {
        str(item.id): "option_a" if i < 8 else "option_b"
        for i, item in enumerate(initial_items)
    }

    selector_config = UncertaintySamplerConfig(method="entropy")
    loop_config = ActiveLearningLoopConfig(max_iterations=3, budget_per_iteration=5)
    loop = ActiveLearningLoop(
        item_selector=UncertaintySampler(config=selector_config),
        config=loop_config,
    )

    model_config = ForcedChoiceModelConfig(num_epochs=1, batch_size=4, device="cpu")
    results = loop.run(
        initial_items=initial_items,
        initial_model=ForcedChoiceModel(config=model_config),
        item_template=forced_choice_template,
        unlabeled_pool=unlabeled_pool,
        human_ratings=ratings,
    )

    # Each iteration should have training metrics
    assert len(results) == 3
    for metadata in results:
        assert "accuracy" in metadata.metrics
        assert "train_accuracy" in metadata.metrics or "train_loss" in metadata.metrics


def test_active_learning_different_selection_methods(
    test_items, forced_choice_template
):
    """Test with different uncertainty sampling methods."""
    initial_items = test_items(n=15)
    unlabeled_pool = test_items(n=20)

    ratings = {
        str(item.id): "option_a" if i < 8 else "option_b"
        for i, item in enumerate(initial_items)
    }

    # Test with entropy
    selector_config_entropy = UncertaintySamplerConfig(method="entropy")
    loop_config_entropy = ActiveLearningLoopConfig(
        max_iterations=1, budget_per_iteration=5
    )
    loop_entropy = ActiveLearningLoop(
        item_selector=UncertaintySampler(config=selector_config_entropy),
        config=loop_config_entropy,
    )

    model_config = ForcedChoiceModelConfig(num_epochs=1, batch_size=4, device="cpu")
    results_entropy = loop_entropy.run(
        initial_items=initial_items,
        initial_model=ForcedChoiceModel(config=model_config),
        item_template=forced_choice_template,
        unlabeled_pool=unlabeled_pool.copy(),
        human_ratings=ratings,
    )

    # Test with margin
    selector_config_margin = UncertaintySamplerConfig(method="margin")
    loop_config_margin = ActiveLearningLoopConfig(
        max_iterations=1, budget_per_iteration=5
    )
    loop_margin = ActiveLearningLoop(
        item_selector=UncertaintySampler(config=selector_config_margin),
        config=loop_config_margin,
    )

    model_config2 = ForcedChoiceModelConfig(num_epochs=1, batch_size=4, device="cpu")
    results_margin = loop_margin.run(
        initial_items=initial_items,
        initial_model=ForcedChoiceModel(config=model_config2),
        item_template=forced_choice_template,
        unlabeled_pool=unlabeled_pool.copy(),
        human_ratings=ratings,
    )

    # Both should complete successfully
    assert len(results_entropy) == 1
    assert len(results_margin) == 1
