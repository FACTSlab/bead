#!/usr/bin/env python3
"""Simulate the complete active learning pipeline with synthetic judgments.

This script demonstrates the sash.simulation framework on the argument structure
project. It:
1. Loads 2AFC pairs from items/2afc_pairs.jsonl
2. Simulates human judgments using the sash.simulation framework
3. Trains model on simulated data
4. Uses active learning to select next batch
5. Repeats until convergence

The simulation uses the LMBasedAnnotator with temperature noise to generate
probabilistic judgments based on language model scores.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from sash.active_learning.loop import ActiveLearningLoop
from sash.active_learning.models.forced_choice import ForcedChoiceModel
from sash.active_learning.selection import UncertaintySampler
from sash.config.models import (
    ActiveLearningLoopConfig,
    ForcedChoiceModelConfig,
    NoiseModelConfig,
    SimulatedAnnotatorConfig,
    UncertaintySamplerConfig,
)
from sash.evaluation.convergence import ConvergenceDetector
from sash.evaluation.interannotator import InterAnnotatorMetrics
from sash.evaluation.model_metrics import ModelMetrics
from sash.items.models import Item, ItemTemplate, PresentationSpec, TaskSpec, TaskType
from sash.simulation.annotators.base import SimulatedAnnotator


def load_2afc_pairs(path: Path, limit: int | None = None, skip: int = 0) -> list[Item]:
    """Load 2AFC pairs from JSONL.

    Parameters
    ----------
    path : Path
        Path to JSONL file
    limit : int | None
        Maximum number of items to load
    skip : int
        Number of items to skip at start

    Returns
    -------
    list[Item]
        List of items
    """
    items = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i < skip:
                continue
            if limit and (i - skip) >= limit:
                break
            data = json.loads(line)
            items.append(Item(**data))
    return items


def get_forced_choice_template() -> ItemTemplate:
    """Create ItemTemplate for 2AFC forced choice task.

    Returns
    -------
    ItemTemplate
        Template configured for forced_choice task using proper TaskSpec
    """
    return ItemTemplate(
        name="2AFC Forced Choice",
        judgment_type="preference",
        task_type="forced_choice",
        task_spec=TaskSpec(
            prompt="Which sentence sounds more natural?",
            options=["option_a", "option_b"],
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )


def run_simulation(
    initial_size: int = 50,
    budget_per_iteration: int = 20,
    max_iterations: int = 10,
    convergence_threshold: float = 0.05,
    temperature: float = 1.0,
    random_state: int | None = None,
    output_dir: Path | None = None,
    max_items: int | None = None,
) -> dict[str, Any]:
    """Run complete simulation of active learning pipeline.

    Parameters
    ----------
    initial_size : int
        Initial training set size
    budget_per_iteration : int
        Items to annotate per iteration
    max_iterations : int
        Maximum AL iterations
    convergence_threshold : float
        Convergence threshold for stopping
    temperature : float
        Temperature for simulated judgments (higher = more noise)
    random_state : int | None
        Random seed
    output_dir : Path | None
        Directory to save simulation results
    max_items : int | None
        Maximum total items to use (for quick testing)

    Returns
    -------
    dict[str, Any]
        Simulation results including convergence metrics
    """
    print("=" * 80)
    print("SIMULATION: Argument Structure Active Learning Pipeline")
    print("=" * 80)
    print("Configuration:")
    print(f"  Initial size: {initial_size}")
    print(f"  Budget/iteration: {budget_per_iteration}")
    print(f"  Max iterations: {max_iterations}")
    print(f"  Temperature: {temperature}")
    print(f"  Random state: {random_state}")
    print()

    # Setup output directory
    if output_dir is None:
        output_dir = Path("simulation_output")
    output_dir.mkdir(exist_ok=True)

    # Set random seed
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    # [1/7] Load data
    print("[1/7] Loading data...")
    pairs_path = Path("items/2afc_pairs.jsonl")

    if not pairs_path.exists():
        raise FileNotFoundError(
            f"2AFC pairs not found: {pairs_path}\nRun: make 2afc-pairs"
        )

    # Load and sample data
    if max_items is None:
        max_items = initial_size + budget_per_iteration * max_iterations

    all_pairs = load_2afc_pairs(pairs_path, limit=max_items)

    if len(all_pairs) < initial_size:
        raise ValueError(
            f"Not enough items: need {initial_size}, found {len(all_pairs)}"
        )

    # Shuffle and split
    random.shuffle(all_pairs)
    initial_items = all_pairs[:initial_size]
    unlabeled_pool = all_pairs[initial_size:]

    print(f"  Loaded {len(all_pairs)} 2AFC pairs")
    print(f"  Initial set: {len(initial_items)}")
    print(f"  Unlabeled pool: {len(unlabeled_pool)}")
    print()

    # [2/7] Setup simulated annotator
    print("[2/7] Setting up simulated annotator...")

    # Create annotator configuration using sash.simulation framework
    annotator_config = SimulatedAnnotatorConfig(
        strategy="lm_score",
        model_output_key="lm_score",
        noise_model=NoiseModelConfig(
            noise_type="temperature",
            temperature=temperature,
        ),
        random_state=random_state,
        fallback_to_random=True,
    )

    # Create annotator from configuration
    annotator = SimulatedAnnotator.from_config(annotator_config)

    print(f"  Strategy: lm_score")
    print(f"  Temperature: {temperature}")
    print(f"  Random state: {random_state}")
    print()

    # [3/7] Generate initial annotations
    print("[3/7] Generating initial annotations...")

    # Create ItemTemplate for the simulation
    item_template = get_forced_choice_template()

    # Generate initial annotations using the simulation framework
    human_ratings = annotator.annotate_batch(initial_items, item_template)
    print(f"  Generated {len(human_ratings)} initial annotations")

    # Compute simulated human agreement (sample twice with different seeds)
    # Create two new annotators with different random states for agreement calculation
    annotator_sample1 = SimulatedAnnotator.from_config(
        annotator_config.model_copy(update={"random_state": (random_state or 0) + 1000})
    )
    annotator_sample2 = SimulatedAnnotator.from_config(
        annotator_config.model_copy(update={"random_state": (random_state or 0) + 2000})
    )

    sample1 = annotator_sample1.annotate_batch(initial_items, item_template)
    sample2 = annotator_sample2.annotate_batch(initial_items, item_template)

    labels1 = [sample1[str(item.id)] for item in initial_items]
    labels2 = [sample2[str(item.id)] for item in initial_items]

    inter_annotator = InterAnnotatorMetrics()
    human_agreement = inter_annotator.cohens_kappa(labels1, labels2)

    print(f"  Simulated human agreement (Cohen's κ): {human_agreement:.3f}")
    print()

    # [4/7] Setup convergence detection
    print("[4/7] Setting up convergence detection...")
    convergence_detector = ConvergenceDetector(
        human_agreement_metric="accuracy",  # Using accuracy as proxy for kappa
        convergence_threshold=convergence_threshold,
        min_iterations=2,
        alpha=0.05,
    )
    print(f"  Convergence threshold: {convergence_threshold}")
    print()

    # [5/7] Setup active learning
    print("[5/7] Setting up active learning...")

    # Create model with configuration
    model_config = ForcedChoiceModelConfig(
        model_name="bert-base-uncased",
        num_epochs=3,
        batch_size=16,
        device="cpu",
    )
    model = ForcedChoiceModel(config=model_config)

    # Create selector with configuration
    selector_config = UncertaintySamplerConfig(method="entropy")
    item_selector = UncertaintySampler(config=selector_config)

    # Create loop with configuration
    loop_config = ActiveLearningLoopConfig(
        max_iterations=max_iterations,
        budget_per_iteration=budget_per_iteration,
    )
    loop = ActiveLearningLoop(
        item_selector=item_selector,
        config=loop_config,
    )

    print("  Strategy: Uncertainty sampling (entropy)")
    print("  Model: ForcedChoiceModel (BERT-based)")
    print()

    # [6/7] Run active learning loop
    print("[6/7] Running active learning loop...")
    print()

    iteration_results = []
    current_labeled = initial_items.copy()
    current_unlabeled = unlabeled_pool.copy()
    converged = False

    for iteration in range(max_iterations):
        print(f"  Iteration {iteration + 1}/{max_iterations}")
        print("  " + "-" * 70)

        # Extract labels
        labels = [human_ratings[str(item.id)] for item in current_labeled]

        # Train model
        print(f"    Training on {len(current_labeled)} items...")
        train_metrics = model.train(current_labeled, labels)
        print(f"    Train accuracy: {train_metrics['train_accuracy']:.3f}")

        # Evaluate on held-out data
        # Sample from unlabeled pool for testing
        test_size = min(50, len(current_unlabeled))
        if test_size > 0:
            test_items = random.sample(current_unlabeled, test_size)
            test_annotations = annotator.annotate_batch(test_items, item_template)
            test_labels = [test_annotations[str(item.id)] for item in test_items]

            predictions = model.predict(test_items)
            pred_labels = [p.predicted_class for p in predictions]

            metrics_calc = ModelMetrics()
            test_accuracy = metrics_calc.accuracy(test_labels, pred_labels)
            print(f"    Test accuracy: {test_accuracy:.3f}")
        else:
            # No unlabeled items left, use training accuracy
            test_accuracy = train_metrics["train_accuracy"]
            print(f"    Test accuracy: {test_accuracy:.3f} (using train)")

        # Store results
        iteration_results.append(
            {
                "iteration": iteration + 1,
                "train_accuracy": train_metrics["train_accuracy"],
                "test_accuracy": test_accuracy,
                "n_labeled": len(current_labeled),
                "n_unlabeled": len(current_unlabeled),
            }
        )

        # Check convergence
        converged = convergence_detector.check_convergence(
            model_accuracy=test_accuracy,
            iteration=iteration + 1,
        )

        gap = abs(test_accuracy - human_agreement)
        print(f"    Agreement gap: {gap:.3f}")

        if converged:
            print("    ✓ Converged!")
            break

        # Select next batch
        if not current_unlabeled:
            print("    No more unlabeled items")
            break

        n_select = min(budget_per_iteration, len(current_unlabeled))
        print(f"    Selecting {n_select} items for annotation...")

        selected_items = item_selector.select(
            model=model,
            unlabeled_items=current_unlabeled,
            n_select=n_select,
        )

        # Simulate annotations for selected items using the simulation framework
        new_annotations = annotator.annotate_batch(selected_items, item_template)
        human_ratings.update(new_annotations)

        # Update sets
        current_labeled.extend(selected_items)
        current_unlabeled = [
            item
            for item in current_unlabeled
            if str(item.id) not in {str(s.id) for s in selected_items}
        ]

        print()

    # [7/7] Summary
    print("[7/7] Simulation complete!")
    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Iterations completed: {len(iteration_results)}")
    print(f"Total annotations: {len(human_ratings)}")
    print(f"Final test accuracy: {iteration_results[-1]['test_accuracy']:.3f}")
    print(f"Simulated human agreement: {human_agreement:.3f}")
    print(
        f"Final gap: {abs(iteration_results[-1]['test_accuracy'] - human_agreement):.3f}"
    )

    if converged:
        print("Status: ✓ CONVERGED")
    else:
        print("Status: ⚠ MAX ITERATIONS REACHED")
    print()

    # Save results
    results = {
        "config": {
            "initial_size": initial_size,
            "budget_per_iteration": budget_per_iteration,
            "max_iterations": max_iterations,
            "convergence_threshold": convergence_threshold,
            "temperature": temperature,
            "random_state": random_state,
        },
        "human_agreement": human_agreement,
        "iterations": iteration_results,
        "converged": converged,
        "total_annotations": len(human_ratings),
    }

    results_path = output_dir / "simulation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to: {results_path}")
    print()

    return results


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Simulate active learning pipeline with synthetic judgments"
    )
    parser.add_argument(
        "--initial-size",
        type=int,
        default=50,
        help="Initial training set size (default: 50)",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=20,
        help="Items to annotate per iteration (default: 20)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=10,
        help="Maximum AL iterations (default: 10)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.05,
        help="Convergence threshold (default: 0.05)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Judgment noise temperature (default: 1.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed (default: None)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("simulation_output"),
        help="Output directory (default: simulation_output)",
    )
    parser.add_argument(
        "--max-items",
        type=int,
        default=None,
        help="Maximum total items to use (default: None = use all needed)",
    )

    args = parser.parse_args()

    run_simulation(
        initial_size=args.initial_size,
        budget_per_iteration=args.budget,
        max_iterations=args.max_iterations,
        convergence_threshold=args.threshold,
        temperature=args.temperature,
        random_state=args.seed,
        output_dir=args.output_dir,
        max_items=args.max_items,
    )


if __name__ == "__main__":
    main()
