"""Active learning loop orchestration.

This module orchestrates the iterative active learning loop (stages 3-6):
construct items → deploy experiment → collect data → train model → select
next items. It manages convergence detection and coordinates all components.
"""

from __future__ import annotations

from datetime import UTC
from typing import TYPE_CHECKING, TypedDict

import numpy as np

from bead.active_learning.selection import ItemSelector
from bead.active_learning.trainers.base import ModelMetadata
from bead.config.models import ActiveLearningLoopConfig
from bead.evaluation.convergence import ConvergenceDetector
from bead.items.models import Item, ItemTemplate

if TYPE_CHECKING:
    from bead.active_learning.models.base import ActiveLearningModel


class IterationResult(TypedDict):
    """Results from a single active learning iteration.

    Attributes
    ----------
    iteration : int
        Iteration number.
    selected_items : list[Item]
        Items selected for annotation in this iteration.
    model : TwoAFCModel
        Updated model after this iteration.
    metadata : ModelMetadata | None
        Training metadata if model was retrained, None otherwise.
    """

    iteration: int
    selected_items: list[Item]
    model: ActiveLearningModel
    metadata: ModelMetadata | None


class ActiveLearningLoop:
    """Orchestrates the active learning loop (stages 3-6).

    Manages the iterative process of selecting informative items,
    training models on collected data, and determining when to stop.

    Note: Phase 22 (data collection) is not yet implemented, so this
    loop uses placeholder interfaces for deployment and data collection.
    The focus is on the selection logic and loop orchestration.

    Parameters
    ----------
    item_selector : ItemSelector
        Algorithm for selecting informative items.
    config : ActiveLearningLoopConfig | None
        Configuration object. If None, uses default configuration.

    Attributes
    ----------
    item_selector : ItemSelector
        Item selection algorithm.
    config : ActiveLearningLoopConfig
        Loop configuration.
    iteration_history : list[IterationResult]
        History of all iterations with structured results.

    Examples
    --------
    >>> from bead.active_learning.selection import UncertaintySampler
    >>> from bead.config.models import ActiveLearningLoopConfig
    >>> import numpy as np
    >>> selector = UncertaintySampler()
    >>> config = ActiveLearningLoopConfig(  # doctest: +SKIP
    ...     max_iterations=5,
    ...     budget_per_iteration=100
    ... )
    >>> loop = ActiveLearningLoop(  # doctest: +SKIP
    ...     item_selector=selector,
    ...     config=config
    ... )
    """

    def __init__(
        self,
        item_selector: ItemSelector,
        config: ActiveLearningLoopConfig | None = None,
    ) -> None:
        """Initialize active learning loop.

        Parameters
        ----------
        item_selector : ItemSelector
            Algorithm for selecting items.
        config : ActiveLearningLoopConfig | None
            Configuration object. If None, uses default configuration.
        """
        self.item_selector = item_selector
        self.config = config or ActiveLearningLoopConfig()
        self.iteration_history: list[IterationResult] = []

    def run(
        self,
        initial_items: list[Item],
        initial_model: ActiveLearningModel,
        item_template: ItemTemplate,
        unlabeled_pool: list[Item],
        human_ratings: dict[str, str] | None = None,
        convergence_detector: ConvergenceDetector | None = None,
    ) -> list[ModelMetadata]:
        """Run the complete active learning loop.

        Parameters
        ----------
        initial_items : list[Item]
            Initial labeled items for training.
        initial_model : ActiveLearningModel
            Model instance to use for active learning.
        item_template : ItemTemplate
            Template used to construct all items. Required for validating
            model compatibility with task type.
        unlabeled_pool : list[Item]
            Pool of unlabeled items to select from.
        human_ratings : dict[str, str] | None
            Human ratings mapping item_id to option names.
        convergence_detector : ConvergenceDetector | None
            Detector for checking convergence to human-level performance.
            If provided, will check convergence after each iteration.

        Returns
        -------
        list[ModelMetadata]
            Metadata for all trained models across iterations.

        Raises
        ------
        ValueError
            If stopping_criterion is invalid or threshold not provided when needed.

        Notes
        -----
        Stopping criteria and performance thresholds are configured via
        the `config` parameter passed to __init__.

        Examples
        --------
        >>> from uuid import uuid4
        >>> from bead.items.models import Item
        >>> from bead.config.models import ActiveLearningLoopConfig
        >>> selector = UncertaintySampler()  # doctest: +SKIP
        >>> config = ActiveLearningLoopConfig(max_iterations=3)  # doctest: +SKIP
        >>> loop = ActiveLearningLoop(  # doctest: +SKIP
        ...     item_selector=selector,
        ...     config=config
        ... )
        >>> # Run would typically be called here with real data
        """
        # Validate inputs based on config
        stopping_criterion = self.config.stopping_criterion
        performance_threshold = self.config.performance_threshold
        metric_name = self.config.metric_name

        if (
            stopping_criterion == "performance_threshold"
            and performance_threshold is None
        ):
            raise ValueError(
                "performance_threshold must be provided in config when using "
                "performance_threshold stopping criterion"
            )

        if stopping_criterion == "convergence" and convergence_detector is None:
            raise ValueError(
                "convergence_detector must be provided when using "
                "convergence stopping criterion"
            )

        current_model: ActiveLearningModel = initial_model

        # Validate model compatibility with item task type
        if item_template.task_type not in current_model.supported_task_types:
            raise ValueError(
                f"Model {type(current_model).__name__} does not support "
                f"task type '{item_template.task_type}'. "
                f"Supported types: {current_model.supported_task_types}"
            )

        # Validate all initial items for structural compatibility
        for item in initial_items:
            current_model.validate_item_compatibility(item, item_template)

        # Validate all unlabeled items for structural compatibility
        for item in unlabeled_pool:
            current_model.validate_item_compatibility(item, item_template)

        model_history: list[ModelMetadata] = []
        current_unlabeled = unlabeled_pool.copy()
        labeled_items = initial_items.copy()

        # Check if we have any unlabeled items to start with
        if not current_unlabeled:
            return model_history

        # Run iterations
        for iteration in range(self.config.max_iterations):
            # Extract labels for current labeled items
            if human_ratings is None:
                # No ratings provided, can't train
                break

            labels = [
                human_ratings.get(str(item.id), "option_a") for item in labeled_items
            ]

            # Train model
            train_metrics = current_model.train(items=labeled_items, labels=labels)

            # Evaluate model on labeled items
            predictions = current_model.predict(labeled_items)
            pred_labels = [p.predicted_class for p in predictions]

            # Compute accuracy
            from bead.evaluation.model_metrics import ModelMetrics

            metrics_calculator = ModelMetrics()
            accuracy = metrics_calculator.accuracy(labels, pred_labels)

            # Create metadata
            from datetime import datetime
            from pathlib import Path

            training_config_dict = {
                "iteration": iteration,
                "max_iterations": self.config.max_iterations,
                "budget_per_iteration": self.config.budget_per_iteration,
            }

            metadata = ModelMetadata(
                model_name="ActiveLearningModel",
                framework="custom",
                training_config=training_config_dict,
                training_data_path=Path("active_learning_data"),
                metrics={"accuracy": accuracy, **train_metrics},
                training_time=0.0,
                training_timestamp=datetime.now(UTC).isoformat(),
            )
            model_history.append(metadata)

            # Run one iteration for item selection
            iteration_result = self.run_iteration(
                iteration=iteration,
                unlabeled_items=current_unlabeled,
                current_model=current_model,
            )

            # Store results
            self.iteration_history.append(iteration_result)

            # Update state
            selected_items = iteration_result["selected_items"]
            current_model = iteration_result["model"]

            # Add selected items to labeled set
            labeled_items.extend(selected_items)

            # Remove selected items from unlabeled pool
            selected_ids = {item.id for item in selected_items}
            current_unlabeled = [
                item for item in current_unlabeled if item.id not in selected_ids
            ]

            # Check stopping criteria
            if stopping_criterion == "max_iterations":
                # Will stop naturally at max_iterations
                pass
            elif stopping_criterion == "performance_threshold":
                if metadata and metric_name in metadata.metrics:
                    if metadata.metrics[metric_name] >= performance_threshold:  # type: ignore
                        break
            elif stopping_criterion == "convergence":
                if convergence_detector is not None and metadata is not None:
                    # Compute human baseline on first iteration
                    if iteration == 0 and human_ratings is not None:
                        convergence_detector.compute_human_baseline(human_ratings)

                    # Check if converged
                    if metric_name in metadata.metrics:
                        converged = convergence_detector.check_convergence(
                            model_accuracy=metadata.metrics[metric_name],
                            iteration=iteration + 1,
                        )

                        if converged:
                            print(f"✓ Converged at iteration {iteration + 1}")
                            break

            # Check if unlabeled pool is exhausted
            if not current_unlabeled:
                break

        return model_history

    def run_iteration(
        self,
        iteration: int,
        unlabeled_items: list[Item],
        current_model: ActiveLearningModel,
    ) -> IterationResult:
        """Run one iteration of the active learning loop.

        Steps:
        1. Select informative items using uncertainty sampling
        2. (Placeholder) Deploy experiment for data collection
        3. (Placeholder) Wait for and collect data
        4. (Placeholder) Train new model on augmented dataset
        5. Return results

        Parameters
        ----------
        iteration : int
            Current iteration number.
        unlabeled_items : list[Item]
            Unlabeled items available for selection.
        current_model : ActiveLearningModel
            Current trained model for making predictions.

        Returns
        -------
        IterationResult
            Structured iteration results containing:
            - iteration: Iteration number
            - selected_items: List of selected items
            - model: Updated model
            - metadata: Training metadata if available

        Examples
        --------
        >>> from uuid import uuid4
        >>> from bead.items.models import Item
        >>> import numpy as np
        >>> selector = UncertaintySampler()
        >>> loop = ActiveLearningLoop(
        ...     item_selector=selector,
        ...     trainer=None,
        ...     predict_fn=lambda m, i: np.array([0.5, 0.5]),
        ...     max_iterations=5,
        ...     budget_per_iteration=2
        ... )
        >>> items = [
        ...     Item(item_template_id=uuid4(), rendered_elements={})
        ...     for _ in range(5)
        ... ]
        >>> result = loop.run_iteration(0, items, None)
        >>> len(result["selected_items"])
        2
        >>> result["iteration"]
        0
        """
        # Step 1: Select items using active learning
        budget = min(self.config.budget_per_iteration, len(unlabeled_items))

        def model_predict_fn(model: ActiveLearningModel, item: Item) -> np.ndarray:
            """Get prediction probabilities for a single item."""
            proba = model.predict_proba([item])
            return proba[0]

        selected_items = self.item_selector.select(
            items=unlabeled_items,
            model=current_model,
            predict_fn=model_predict_fn,
            budget=budget,
        )

        # Step 2: Deploy experiment (PLACEHOLDER - Phase 22 not yet implemented)
        # In the future, this would:
        # - Create experiment lists using ListPartitioner
        # - Generate jsPsych experiment using JsPsychExperimentGenerator
        # - Export to JATOS format
        # - Return deployment info for manual upload

        # Step 3: Collect data (PLACEHOLDER - Phase 22 not yet implemented)
        # In the future, this would:
        # - Wait for participants to complete experiments
        # - Use JATOSDataCollector to download results
        # - Use ProlificDataCollector to get participant metadata
        # - Use DataMerger to merge JATOS and Prolific data

        # Step 4: Train new model (PLACEHOLDER - training data not available)
        # In the future, this would:
        # - Merge old training data with new collected data
        # - Call trainer.train() with augmented dataset
        # - Return updated model and metadata

        # For now, return placeholder results
        return IterationResult(
            iteration=iteration,
            selected_items=selected_items,
            model=current_model,  # Unchanged for now
            metadata=None,  # Would contain training metrics
        )

    def check_convergence(
        self,
        metrics_history: list[dict[str, float]],
        metric_name: str = "accuracy",
        patience: int = 3,
        min_delta: float = 0.01,
    ) -> bool:
        """Check if model performance has converged.

        Uses early stopping logic: if performance hasn't improved by
        at least min_delta for patience iterations, consider converged.

        For metrics where lower is better (like "loss"), the logic checks
        if the best (lowest) value is from more than patience iterations ago.

        Parameters
        ----------
        metrics_history : list[dict[str, float]]
            History of metrics from each iteration.
        metric_name : str
            Name of metric to track.
        patience : int
            Number of iterations without improvement before stopping.
        min_delta : float
            Minimum change to count as improvement.

        Returns
        -------
        bool
            True if converged, False otherwise.

        Examples
        --------
        >>> loop = ActiveLearningLoop(  # doctest: +SKIP
        ...     item_selector=UncertaintySampler(),
        ...     trainer=None,
        ...     predict_fn=lambda m, i: np.array([0.5, 0.5])
        ... )
        >>> # Improving performance - not converged
        >>> history = [
        ...     {"accuracy": 0.7},
        ...     {"accuracy": 0.75},
        ...     {"accuracy": 0.8}
        ... ]
        >>> loop.check_convergence(history, metric_name="accuracy", patience=2)
        False
        >>> # No improvement for 3 iterations - converged
        >>> history = [
        ...     {"accuracy": 0.8},
        ...     {"accuracy": 0.81},
        ...     {"accuracy": 0.805},
        ...     {"accuracy": 0.81}
        ... ]
        >>> loop.check_convergence(
        ...     history, metric_name="accuracy", patience=3, min_delta=0.02
        ... )
        True
        """
        if len(metrics_history) < patience + 1:
            return False

        # Get recent metrics
        recent_metrics = [m[metric_name] for m in metrics_history[-(patience + 1) :]]

        # Determine if lower is better (like loss) or higher is better (like accuracy)
        is_lower_better = metric_name.lower() in ["loss", "error", "mse", "rmse", "mae"]

        if is_lower_better:
            # For loss metrics, best means minimum
            best_metric = min(recent_metrics)
            best_idx = recent_metrics.index(best_metric)

            # If best is from patience or more iterations ago, check convergence
            if best_idx <= len(recent_metrics) - patience - 1:
                # Check that degradation from best to current is < min_delta
                current_metric = recent_metrics[-1]
                degradation = current_metric - best_metric

                if degradation >= min_delta:
                    return True  # Performance degraded, converged

        else:
            # For accuracy metrics, best means maximum
            best_metric = max(recent_metrics)
            best_idx = recent_metrics.index(best_metric)

            # If best is from patience or more iterations ago, converged
            if best_idx <= len(recent_metrics) - patience - 1:
                # Check that improvement from best to current is < min_delta
                current_metric = recent_metrics[-1]
                improvement = best_metric - current_metric

                if improvement >= min_delta:
                    return False  # Still improving

                return True

        return False

    def get_summary(self) -> dict[str, int | dict[str, int]]:
        """Get summary statistics of the active learning loop.

        Returns
        -------
        dict[str, int | dict[str, int]]
            Summary dictionary with the following keys:

            total_iterations : int
                Total number of iterations run.
            total_items_selected : int
                Total items selected across all iterations.
            convergence_info : dict[str, int]
                Configuration parameters (max_iterations, budget_per_iteration).

        Examples
        --------
        >>> selector = UncertaintySampler()
        >>> loop = ActiveLearningLoop(
        ...     item_selector=selector,
        ...     trainer=None,
        ...     predict_fn=lambda m, i: np.array([0.5, 0.5])
        ... )
        >>> summary = loop.get_summary()
        >>> summary["total_iterations"]
        0
        >>> summary["total_items_selected"]
        0
        """
        total_items = sum(
            len(iteration["selected_items"]) for iteration in self.iteration_history
        )

        return {
            "total_iterations": len(self.iteration_history),
            "total_items_selected": total_items,
            "convergence_info": {
                "max_iterations": self.config.max_iterations,
                "budget_per_iteration": self.config.budget_per_iteration,
            },
        }
