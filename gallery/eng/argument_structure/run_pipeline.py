#!/usr/bin/env python3
"""Run the complete argument structure active learning pipeline.

This script orchestrates the full pipeline:
1. Load configuration
2. Load 2AFC pairs
3. Set up convergence detection
4. Set up active learning components
5. Run active learning loop with convergence detection
6. Report results and save outputs

The pipeline continues until the model converges to human-level inter-annotator
agreement, as measured by Krippendorff's alpha or other configured metrics.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from bead.active_learning.loop import ActiveLearningLoop
from bead.active_learning.selection import UncertaintySampler
from bead.active_learning.trainers.base import ModelMetadata
from bead.evaluation.convergence import ConvergenceDetector
from bead.items.item import Item


def load_config(config_path: Path) -> dict[str, Any]:
    """Load configuration from YAML file.

    Parameters
    ----------
    config_path : Path
        Path to configuration YAML file.

    Returns
    -------
    dict[str, Any]
        Configuration dictionary.
    """
    print(f"Loading configuration from {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    print("✓ Configuration loaded")
    return config


def load_2afc_pairs(path: Path, limit: int | None = None, skip: int = 0) -> list[Item]:
    """Load 2AFC pairs from JSONL file.

    Parameters
    ----------
    path : Path
        Path to 2AFC pairs JSONL file.
    limit : int | None
        Maximum number of items to load. If None, load all.
    skip : int
        Number of items to skip from the beginning.

    Returns
    -------
    list[Item]
        List of 2AFC item pairs.
    """
    print(f"Loading 2AFC pairs from {path}")
    items = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i < skip:
                continue
            if limit and len(items) >= limit:
                break
            data = json.loads(line)
            items.append(Item(**data))

    print(f"✓ Loaded {len(items)} 2AFC pairs")
    if skip > 0:
        print(f"  (skipped first {skip} items)")
    return items


def setup_convergence_detector(config: dict[str, Any]) -> ConvergenceDetector:
    """Set up convergence detector from configuration.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary.

    Returns
    -------
    ConvergenceDetector
        Configured convergence detector.
    """
    print("Setting up convergence detection...")
    conv_config = config["training"]["convergence"]

    detector = ConvergenceDetector(
        human_agreement_metric=conv_config["metric"],
        convergence_threshold=conv_config["threshold"],
        min_iterations=conv_config["min_iterations"],
        alpha=conv_config.get("alpha", 0.05),
    )

    print("✓ Convergence detector initialized")
    print(f"  - Metric: {conv_config['metric']}")
    print(f"  - Threshold: {conv_config['threshold']}")
    print(f"  - Min iterations: {conv_config['min_iterations']}")

    return detector


def setup_active_learning(config: dict[str, Any]) -> tuple[UncertaintySampler, Any]:
    """Set up active learning components from configuration.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary.

    Returns
    -------
    tuple[UncertaintySampler, Any]
        Tuple of (item selector, trainer).
        Trainer is None for now (placeholder for future implementation).
    """
    print("Setting up active learning components...")
    al_config = config["active_learning"]

    # Set up item selector
    if al_config["strategy"] == "uncertainty_sampling":
        selector = UncertaintySampler(method=al_config["method"])
        print(f"✓ Item selector: {al_config['strategy']}")
        print(f"  - Method: {al_config['method']}")
    else:
        raise ValueError(f"Unknown AL strategy: {al_config['strategy']}")

    # Trainer placeholder (Phase 22 - not yet implemented)
    trainer = None
    print("  - Trainer: Not implemented (using placeholder)")

    return selector, trainer


def predict_2afc(model: Any, item: Item) -> np.ndarray:
    """Prediction function for 2AFC items.

    This is a placeholder implementation that returns random probabilities.
    In a real implementation, this would use the trained model to predict
    which option is more acceptable.

    Parameters
    ----------
    model : Any
        Trained model (currently unused).
    item : Item
        2AFC item to predict on.

    Returns
    -------
    np.ndarray
        Array of shape (2,) with probabilities for each option.
    """
    # Placeholder: return random probabilities
    # In real implementation, would use model to score both options
    return np.array([0.5, 0.5])


def load_human_ratings(
    human_ratings_path: Path | None,
) -> dict[str, list[Any]] | None:
    """Load human ratings for computing baseline agreement.

    Parameters
    ----------
    human_ratings_path : Path | None
        Path to human ratings file (JSONL format).
        If None, returns None.

    Returns
    -------
    dict[str, list[Any]] | None
        Dictionary mapping rater IDs to their ratings.
        Returns None if no human ratings available.
    """
    if human_ratings_path is None or not human_ratings_path.exists():
        print("  - Human ratings: Not available (will be collected during deployment)")
        return None

    print(f"Loading human ratings from {human_ratings_path}")
    # Placeholder implementation
    # Real implementation would parse JSONL with format:
    # {"rater_id": "r1", "item_id": "...", "response": 0 or 1}
    return None


def print_header(title: str) -> None:
    """Print a formatted header.

    Parameters
    ----------
    title : str
        Title to display.
    """
    print("\n" + "=" * 70)
    print(title.upper())
    print("=" * 70)


def print_section(section: str) -> None:
    """Print a formatted section header.

    Parameters
    ----------
    section : str
        Section title.
    """
    print(f"\n{section}")
    print("-" * len(section))


def print_results(results: list[ModelMetadata]) -> None:
    """Print final results from active learning.

    Parameters
    ----------
    results : list[ModelMetadata]
        List of model metadata from each iteration.
    """
    print_section("Active Learning Results")

    if not results:
        print("No training iterations completed.")
        return

    print(f"\nCompleted {len(results)} iterations")

    # Print per-iteration results
    print("\nIteration Summary:")
    print(f"{'Iter':<6} {'Accuracy':<12} {'Precision':<12} {'Recall':<12} {'F1':<12}")
    print("-" * 60)

    for i, metadata in enumerate(results, 1):
        metrics = metadata.metrics
        acc = metrics.get("accuracy", 0.0)
        prec = metrics.get("precision", 0.0)
        rec = metrics.get("recall", 0.0)
        f1 = metrics.get("f1", 0.0)
        print(f"{i:<6} {acc:<12.4f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f}")

    # Print final metrics
    print_section("Final Model Performance")
    final_metrics = results[-1].metrics
    for metric, value in sorted(final_metrics.items()):
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")


def save_results(
    results: list[ModelMetadata], output_path: Path, config: dict[str, Any]
) -> None:
    """Save results to JSON file.

    Parameters
    ----------
    results : list[ModelMetadata]
        List of model metadata from each iteration.
    output_path : Path
        Path to save results JSON file.
    config : dict[str, Any]
        Configuration dictionary.
    """
    print_section("Saving Results")

    # Prepare results for serialization
    results_data = {
        "config": {
            "project": config["project"],
            "active_learning": config["active_learning"],
            "convergence": config["training"]["convergence"],
        },
        "iterations": [
            {
                "iteration": i + 1,
                "metrics": metadata.metrics,
                "model_info": {
                    "n_train_samples": metadata.n_train_samples,
                    "n_val_samples": metadata.n_val_samples,
                    "training_time": metadata.training_time,
                },
            }
            for i, metadata in enumerate(results)
        ],
        "final_metrics": results[-1].metrics if results else {},
    }

    with open(output_path, "w") as f:
        json.dump(results_data, f, indent=2)

    print(f"✓ Results saved to {output_path}")


def main(args: argparse.Namespace) -> None:
    """Run the complete active learning pipeline.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments.
    """
    print_header("Argument Structure Active Learning Pipeline")

    # Determine base directory
    if args.config:
        config_path = Path(args.config)
        base_dir = config_path.parent
    else:
        base_dir = Path(__file__).parent
        config_path = base_dir / "config.yaml"

    print(f"\nBase directory: {base_dir}")
    print(f"Configuration: {config_path}")

    # Load configuration
    print_section("[1/7] Loading Configuration")
    try:
        config = load_config(config_path)
    except Exception as e:
        print(f"✗ Error loading configuration: {e}")
        sys.exit(1)

    # Set up convergence detection
    print_section("[2/7] Setting Up Convergence Detection")
    try:
        convergence_detector = setup_convergence_detector(config)
    except Exception as e:
        print(f"✗ Error setting up convergence detector: {e}")
        sys.exit(1)

    # Set up active learning components
    print_section("[3/7] Setting Up Active Learning")
    try:
        selector, trainer = setup_active_learning(config)
    except Exception as e:
        print(f"✗ Error setting up active learning: {e}")
        sys.exit(1)

    # Create active learning loop
    al_config = config["active_learning"]
    loop = ActiveLearningLoop(
        item_selector=selector,
        trainer=trainer,
        predict_fn=predict_2afc,
        max_iterations=al_config["max_iterations"],
        budget_per_iteration=al_config["budget_per_iteration"],
    )
    print("✓ Active learning loop initialized")
    print(f"  - Max iterations: {al_config['max_iterations']}")
    print(f"  - Budget per iteration: {al_config['budget_per_iteration']}")

    # Load 2AFC pairs
    print_section("[4/7] Loading 2AFC Pairs")
    pairs_path = base_dir / config["paths"]["2afc_pairs"]

    # Use command-line overrides if provided
    initial_size = args.initial_size or al_config.get("initial_training_size", 100)
    unlabeled_size = args.unlabeled_size or 500

    try:
        initial_items = load_2afc_pairs(pairs_path, limit=initial_size, skip=0)
        unlabeled_pool = load_2afc_pairs(
            pairs_path, limit=unlabeled_size, skip=initial_size
        )
    except Exception as e:
        print(f"✗ Error loading 2AFC pairs: {e}")
        sys.exit(1)

    print("\nData split:")
    print(f"  - Initial training set: {len(initial_items)} items")
    print(f"  - Unlabeled pool: {len(unlabeled_pool)} items")

    # Load human ratings (if available)
    print_section("[5/7] Loading Human Ratings")
    human_ratings_path = base_dir / "data" / "human_ratings.jsonl"
    human_ratings = load_human_ratings(
        human_ratings_path if human_ratings_path.exists() else None
    )

    # Run active learning loop
    print_section("[6/7] Running Active Learning Loop")
    print("\nStarting active learning with convergence detection...")
    print("=" * 70)

    if args.dry_run:
        print("\n⚠️  DRY RUN MODE - No actual training will occur")
        print("=" * 70)
        results = []
    else:
        try:
            results = loop.run(
                initial_items=initial_items,
                initial_model=None,
                unlabeled_pool=unlabeled_pool,
                human_ratings=human_ratings,
                convergence_detector=convergence_detector,
                stopping_criterion=al_config["stopping_criterion"],
                metric_name=config["training"]["convergence"]["metric"],
            )
        except Exception as e:
            print(f"\n✗ Error during active learning: {e}")
            traceback.print_exc()
            sys.exit(1)

    print("\n" + "=" * 70)
    print("✓ Active learning complete!")

    # Print and save results
    print_section("[7/7] Results")
    if not args.dry_run:
        print_results(results)

        # Save results
        output_path = base_dir / "results" / "pipeline_results.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        save_results(results, output_path, config)

    # Print summary
    print_header("Pipeline Complete")
    summary = loop.get_summary()
    print(f"\nTotal iterations: {summary['total_iterations']}")
    print(f"Total items selected: {summary['total_items_selected']}")

    if not args.dry_run and results:
        final_acc = results[-1].metrics.get("accuracy", 0.0)
        print(f"Final accuracy: {final_acc:.4f}")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the argument structure active learning pipeline"
    )

    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration YAML file (default: config.yaml in script dir)",
    )

    parser.add_argument(
        "--initial-size",
        type=int,
        help="Size of initial training set (overrides config)",
    )

    parser.add_argument(
        "--unlabeled-size",
        type=int,
        help="Size of unlabeled pool (overrides config)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run in dry-run mode (load data but don't train)",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
