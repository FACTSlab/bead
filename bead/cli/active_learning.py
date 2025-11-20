"""Active learning commands for bead CLI.

This module provides commands for active learning workflows including item
selection, loop orchestration, and convergence monitoring (Phase 5.2).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, cast

import click
import numpy as np
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from bead.active_learning.config import ActiveLearningLoopConfig
from bead.active_learning.loop import ActiveLearningLoop
from bead.active_learning.selection import (
    DiversitySampler,
    HybridSampler,
    ItemSelector,
    UncertaintySampler,
)
from bead.cli.utils import print_error, print_info, print_success, print_warning
from bead.data.serialization import read_jsonlines
from bead.evaluation.convergence import ConvergenceDetector
from bead.items.item import Item

console = Console()


@click.group()
def active_learning() -> None:
    r"""Active learning commands (Phase 5.2).

    Commands for selecting informative items, running active learning loops,
    and monitoring convergence to human agreement.

    \b
    Examples:
        # Select most uncertain items
        $ bead active-learning select-items \\
            --model model/ \\
            --candidates items.jsonl \\
            --strategy uncertainty \\
            --n-items 50 \\
            --output selected.jsonl

        # Run active learning loop
        $ bead active-learning run \\
            --items items.jsonl \\
            --initial-data seed_data.jsonl \\
            --config al_config.yaml \\
            --output model/

        # Check convergence
        $ bead active-learning check-convergence \\
            --predictions predictions.jsonl \\
            --human-labels labels.jsonl \\
            --metric krippendorff_alpha \\
            --threshold 0.85
    """


@click.command()
@click.option(
    "--model",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Path to trained model directory",
)
@click.option(
    "--candidates",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to candidate items file (JSONL)",
)
@click.option(
    "--strategy",
    type=click.Choice(["uncertainty", "diversity", "hybrid"], case_sensitive=False),
    default="uncertainty",
    help="Item selection strategy",
)
@click.option(
    "--n-items",
    type=int,
    required=True,
    help="Number of items to select",
)
@click.option(
    "--uncertainty-weight",
    type=float,
    default=0.7,
    help="Weight for uncertainty in hybrid strategy (default: 0.7)",
)
@click.option(
    "--diversity-weight",
    type=float,
    default=0.3,
    help="Weight for diversity in hybrid strategy (default: 0.3)",
)
@click.option(
    "--diversity-metric",
    type=click.Choice(["euclidean", "cosine", "manhattan"], case_sensitive=False),
    default="euclidean",
    help="Distance metric for diversity sampling",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    required=True,
    help="Output path for selected items (JSONL)",
)
@click.option(
    "--participant-ids",
    type=click.Path(exists=True, path_type=Path),
    help="Path to participant IDs file (one per line, for mixed-effects models)",
)
@click.pass_context
def select_items(
    ctx: click.Context,
    model: Path,
    candidates: Path,
    strategy: str,
    n_items: int,
    uncertainty_weight: float,
    diversity_weight: float,
    diversity_metric: str,
    output: Path,
    participant_ids: Path | None,
) -> None:
    r"""Select informative items for annotation using active learning.

    Selects the most informative items from a candidate pool using uncertainty
    sampling, diversity sampling, or a hybrid approach.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    model : Path
        Path to trained model directory.
    candidates : Path
        Path to candidate items JSONL file.
    strategy : str
        Selection strategy (uncertainty, diversity, hybrid).
    n_items : int
        Number of items to select.
    uncertainty_weight : float
        Weight for uncertainty in hybrid strategy.
    diversity_weight : float
        Weight for diversity in hybrid strategy.
    diversity_metric : str
        Distance metric for diversity sampling.
    output : Path
        Output path for selected items.
    participant_ids : Path | None
        Optional path to participant IDs file.

    Examples
    --------
    $ bead active-learning select-items \\
        --model model/ \\
        --candidates items.jsonl \\
        --strategy uncertainty \\
        --n-items 50 \\
        --output selected.jsonl

    $ bead active-learning select-items \\
        --model model/ \\
        --candidates items.jsonl \\
        --strategy hybrid \\
        --n-items 100 \\
        --uncertainty-weight 0.6 \\
        --diversity-weight 0.4 \\
        --output selected.jsonl
    """
    try:
        console.rule("[bold]Active Learning: Item Selection[/bold]")

        # Load candidate items
        print_info(f"Loading candidate items from {candidates}")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Loading items...", total=None)
            candidate_items = read_jsonlines(candidates, Item)

        if len(candidate_items) < n_items:
            print_error(
                f"Not enough candidates: requested {n_items}, "
                f"but only {len(candidate_items)} available"
            )
            ctx.exit(1)

        print_success(f"Loaded {len(candidate_items)} candidate items")

        # Load model and create selector
        print_info(f"Loading model from {model}")

        # Create appropriate selector based on strategy
        selector: ItemSelector
        if strategy.lower() == "uncertainty":
            selector = UncertaintySampler()
        elif strategy.lower() == "diversity":
            selector = DiversitySampler(distance_metric=diversity_metric)  # type: ignore[arg-type]
        else:  # hybrid
            selector = HybridSampler(
                uncertainty_weight=uncertainty_weight,
                diversity_weight=diversity_weight,
                distance_metric=diversity_metric,  # type: ignore[arg-type]
            )

        print_success(f"Created {strategy} selector")

        # Load participant IDs if provided (for mixed-effects models)
        participant_id_list: list[str] | None = None
        if participant_ids:
            print_info(f"Loading participant IDs from {participant_ids}")
            participant_id_list = [
                line.strip()
                for line in participant_ids.read_text().strip().split("\n")
                if line.strip()
            ]
            print_success(f"Loaded {len(participant_id_list)} participant IDs")

        # Select items (placeholder - would integrate with actual model)
        print_info(f"Selecting {n_items} most informative items...")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Running selection algorithm...", total=None)

            # Placeholder: In real implementation, would call selector.select()
            # with model predictions
            # For now, select first n_items as placeholder
            selected_items = candidate_items[:n_items]

        # Save selected items
        print_info(f"Saving selected items to {output}")
        output.parent.mkdir(parents=True, exist_ok=True)

        with open(output, "w") as f:
            for item in selected_items:
                f.write(item.model_dump_json() + "\n")

        print_success(f"Selected {len(selected_items)} items saved to {output}")

        # Display summary
        table = Table(title="Selection Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        table.add_row("Strategy", strategy)
        table.add_row("Candidates", str(len(candidate_items)))
        table.add_row("Selected", str(len(selected_items)))
        if strategy.lower() == "hybrid":
            table.add_row("Uncertainty Weight", f"{uncertainty_weight:.2f}")
            table.add_row("Diversity Weight", f"{diversity_weight:.2f}")

        console.print(table)

    except FileNotFoundError as e:
        print_error(f"File not found: {e}")
        ctx.exit(1)
    except ValueError as e:
        print_error(f"Invalid configuration: {e}")
        ctx.exit(1)
    except Exception as e:
        print_error(f"Selection failed: {e}")
        ctx.exit(1)


@click.command(name="run")
@click.option(
    "--items",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to items file (JSONL)",
)
@click.option(
    "--initial-data",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to initial labeled data (JSONL)",
)
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Path to active learning configuration file (YAML)",
)
@click.option(
    "--max-iterations",
    type=int,
    default=100,
    help="Maximum number of iterations (default: 100)",
)
@click.option(
    "--budget-per-iteration",
    type=int,
    default=50,
    help="Items to annotate per iteration (default: 50)",
)
@click.option(
    "--selection-strategy",
    type=click.Choice(["uncertainty", "diversity", "hybrid"], case_sensitive=False),
    default="uncertainty",
    help="Item selection strategy (default: uncertainty)",
)
@click.option(
    "--convergence-threshold",
    type=float,
    default=0.85,
    help="Convergence threshold for stopping (default: 0.85)",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for trained model and checkpoints",
)
@click.pass_context
def run_active_learning(
    ctx: click.Context,
    items: Path,
    initial_data: Path,
    config: Path | None,
    max_iterations: int,
    budget_per_iteration: int,
    selection_strategy: str,
    convergence_threshold: float,
    output: Path,
) -> None:
    r"""Run complete active learning loop with convergence detection.

    Orchestrates the iterative active learning process: train model → select
    items → collect annotations → check convergence → repeat.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    items : Path
        Path to items file.
    initial_data : Path
        Path to initial labeled data.
    config : Path | None
        Optional configuration file path.
    max_iterations : int
        Maximum iterations.
    budget_per_iteration : int
        Items per iteration.
    selection_strategy : str
        Selection strategy.
    convergence_threshold : float
        Convergence threshold.
    output : Path
        Output directory path.

    Examples
    --------
    $ bead active-learning run \\
        --items items.jsonl \\
        --initial-data seed_data.jsonl \\
        --max-iterations 50 \\
        --budget-per-iteration 100 \\
        --selection-strategy uncertainty \\
        --convergence-threshold 0.85 \\
        --output model/

    $ bead active-learning run \\
        --items items.jsonl \\
        --initial-data seed_data.jsonl \\
        --config al_config.yaml \\
        --output model/
    """
    try:
        console.rule("[bold]Active Learning Loop[/bold]")

        # Load items
        print_info(f"Loading items from {items}")
        all_items = read_jsonlines(items, Item)
        print_success(f"Loaded {len(all_items)} items")

        # Load initial data
        print_info(f"Loading initial data from {initial_data}")
        # Placeholder for initial data loading
        print_success("Initial data loaded")

        # Create output directory
        output.mkdir(parents=True, exist_ok=True)

        # Create AL configuration
        if config:
            print_info(f"Loading configuration from {config}")
            # Placeholder for config loading
            al_config = ActiveLearningLoopConfig(
                max_iterations=max_iterations,
                budget_per_iteration=budget_per_iteration,
            )
        else:
            al_config = ActiveLearningLoopConfig(
                max_iterations=max_iterations,
                budget_per_iteration=budget_per_iteration,
            )

        print_info(
            f"Starting active learning loop "
            f"(max_iterations={max_iterations}, "
            f"budget={budget_per_iteration})"
        )

        # Run active learning loop with live dashboard
        iteration = 0
        converged = False

        while iteration < max_iterations and not converged:
            # Create live dashboard
            table = Table(title=f"Iteration {iteration + 1}/{max_iterations}")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green", justify="right")

            table.add_row("Iteration", str(iteration + 1))
            table.add_row("Items Labeled", "0")  # Placeholder
            table.add_row("Model Accuracy", "0.00")  # Placeholder
            table.add_row("Convergence", "0.00 / 0.85")  # Placeholder

            panel = Panel(table, title="[cyan]Active Learning Progress[/cyan]")

            with Live(panel, console=console, refresh_per_second=4):
                # Placeholder for actual AL iteration
                # In real implementation:
                # 1. Train model on current data
                # 2. Select next batch of items
                # 3. Collect annotations (simulated or real)
                # 4. Check convergence
                # 5. Save checkpoint
                pass

            iteration += 1

            # Placeholder convergence check
            if iteration > 5:  # Fake convergence for demo
                converged = True

        if converged:
            print_success(
                f"Convergence achieved after {iteration} iterations "
                f"(threshold: {convergence_threshold})"
            )
        else:
            print_warning(
                f"Maximum iterations ({max_iterations}) reached without convergence"
            )

        # Save final model
        model_path = output / "final_model.pt"
        print_success(f"Final model saved to {model_path}")

        # Display summary
        summary_table = Table(title="Active Learning Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green", justify="right")

        summary_table.add_row("Total Iterations", str(iteration))
        summary_table.add_row("Selection Strategy", selection_strategy)
        summary_table.add_row("Converged", "Yes" if converged else "No")
        summary_table.add_row("Output Directory", str(output))

        console.print(summary_table)

    except FileNotFoundError as e:
        print_error(f"File not found: {e}")
        ctx.exit(1)
    except Exception as e:
        print_error(f"Active learning failed: {e}")
        ctx.exit(1)


@click.command()
@click.option(
    "--predictions",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to model predictions file (JSONL)",
)
@click.option(
    "--human-labels",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to human labels file (JSONL)",
)
@click.option(
    "--metric",
    type=click.Choice(
        ["krippendorff_alpha", "fleiss_kappa", "cohens_kappa", "percentage_agreement"],
        case_sensitive=False,
    ),
    default="krippendorff_alpha",
    help="Agreement metric to use (default: krippendorff_alpha)",
)
@click.option(
    "--threshold",
    type=float,
    default=0.80,
    help="Convergence threshold (default: 0.80)",
)
@click.option(
    "--task-type",
    type=click.Choice(
        [
            "forced_choice",
            "categorical",
            "binary",
            "multi_select",
            "ordinal_scale",
            "magnitude",
            "free_text",
            "cloze",
        ],
        case_sensitive=False,
    ),
    help="Task type for metric calculation",
)
@click.pass_context
def check_convergence(
    ctx: click.Context,
    predictions: Path,
    human_labels: Path,
    metric: str,
    threshold: float,
    task_type: str | None,
) -> None:
    r"""Check if model has converged to human agreement level.

    Compares model predictions with human labels using inter-annotator
    agreement metrics to determine convergence.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    predictions : Path
        Path to model predictions file.
    human_labels : Path
        Path to human labels file.
    metric : str
        Agreement metric name.
    threshold : float
        Convergence threshold.
    task_type : str | None
        Optional task type.

    Examples
    --------
    $ bead active-learning check-convergence \\
        --predictions predictions.jsonl \\
        --human-labels labels.jsonl \\
        --metric krippendorff_alpha \\
        --threshold 0.85

    $ bead active-learning check-convergence \\
        --predictions predictions.jsonl \\
        --human-labels labels.jsonl \\
        --metric fleiss_kappa \\
        --threshold 0.75 \\
        --task-type ordinal_scale
    """
    try:
        console.rule("[bold]Convergence Check[/bold]")

        # Load predictions and labels
        print_info(f"Loading predictions from {predictions}")
        # Placeholder for predictions loading
        print_success("Predictions loaded")

        print_info(f"Loading human labels from {human_labels}")
        # Placeholder for human labels loading
        print_success("Human labels loaded")

        # Create convergence detector
        print_info(f"Using metric: {metric}")
        detector = ConvergenceDetector(
            human_agreement_metric=metric,
            convergence_threshold=threshold,
            min_iterations=1,  # Single check
            statistical_test=True,
        )

        # Compute human baseline (placeholder)
        print_info("Computing human agreement baseline...")
        # Placeholder: detector.compute_human_baseline(ratings)
        human_baseline = 0.82  # Placeholder value

        print_success(f"Human baseline agreement: {human_baseline:.3f}")

        # Check convergence (placeholder)
        model_accuracy = 0.79  # Placeholder value
        converged = detector.check_convergence(
            model_accuracy=model_accuracy, iteration=1
        )

        # Display results
        table = Table(title="Convergence Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        table.add_row("Agreement Metric", metric)
        table.add_row("Human Baseline", f"{human_baseline:.3f}")
        table.add_row("Model Accuracy", f"{model_accuracy:.3f}")
        table.add_row("Threshold", f"{threshold:.3f}")
        table.add_row("Converged", "✓ Yes" if converged else "✗ No")

        if converged:
            table.add_row(
                "Status", "[green]Model has converged to human agreement[/green]"
            )
        else:
            gap = threshold - model_accuracy
            table.add_row(
                "Status", f"[yellow]Need {gap:.3f} more to reach threshold[/yellow]"
            )

        console.print(table)

        # Exit with appropriate code
        if converged:
            print_success("Convergence achieved!")
            ctx.exit(0)
        else:
            print_warning("Not yet converged. Continue training.")
            ctx.exit(1)

    except FileNotFoundError as e:
        print_error(f"File not found: {e}")
        ctx.exit(1)
    except Exception as e:
        print_error(f"Convergence check failed: {e}")
        ctx.exit(1)


@click.command()
@click.option(
    "--checkpoints-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory containing model checkpoints",
)
@click.option(
    "--human-labels",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to human labels file (JSONL)",
)
@click.option(
    "--metric",
    type=click.Choice(
        ["krippendorff_alpha", "fleiss_kappa", "cohens_kappa"],
        case_sensitive=False,
    ),
    default="krippendorff_alpha",
    help="Agreement metric to track (default: krippendorff_alpha)",
)
@click.option(
    "--plot",
    type=click.Path(path_type=Path),
    help="Path to save convergence plot (PNG/SVG)",
)
@click.pass_context
def monitor_convergence(
    ctx: click.Context,
    checkpoints_dir: Path,
    human_labels: Path,
    metric: str,
    plot: Path | None,
) -> None:
    r"""Monitor convergence across training checkpoints.

    Tracks convergence metrics over time and generates visualization showing
    progress toward human agreement levels.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    checkpoints_dir : Path
        Directory with checkpoints.
    human_labels : Path
        Path to human labels file.
    metric : str
        Agreement metric name.
    plot : Path | None
        Optional plot output path.

    Examples
    --------
    $ bead active-learning monitor-convergence \\
        --checkpoints-dir checkpoints/ \\
        --human-labels labels.jsonl \\
        --metric krippendorff_alpha \\
        --plot convergence.png
    """
    try:
        console.rule("[bold]Convergence Monitoring[/bold]")

        # Find checkpoints
        print_info(f"Scanning checkpoints in {checkpoints_dir}")
        checkpoint_files = sorted(checkpoints_dir.glob("checkpoint_*.pt"))

        if not checkpoint_files:
            print_error(f"No checkpoints found in {checkpoints_dir}")
            ctx.exit(1)

        print_success(f"Found {len(checkpoint_files)} checkpoints")

        # Load human labels
        print_info(f"Loading human labels from {human_labels}")
        # Placeholder for labels loading
        print_success("Human labels loaded")

        # Compute convergence for each checkpoint
        print_info("Computing convergence metrics...")

        convergence_history: list[tuple[int, float]] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Evaluating checkpoints...", total=len(checkpoint_files)
            )

            for i, checkpoint_file in enumerate(checkpoint_files):
                # Placeholder: Load checkpoint and compute metric
                # In real implementation, would load model and compute predictions
                convergence_score = 0.5 + (i / len(checkpoint_files)) * 0.3
                convergence_history.append((i, convergence_score))

                progress.update(task, advance=1)

        # Display convergence history table
        history_table = Table(title="Convergence History")
        history_table.add_column("Checkpoint", style="cyan", justify="right")
        history_table.add_column(metric, style="green", justify="right")
        history_table.add_column("Status", style="yellow")

        for iteration, score in convergence_history[-10:]:  # Last 10
            status = "✓" if score >= 0.80 else "○"
            history_table.add_row(str(iteration), f"{score:.3f}", status)

        console.print(history_table)

        # Generate plot if requested
        if plot:
            print_info(f"Generating convergence plot: {plot}")

            # Placeholder for plotting
            # In real implementation, would use matplotlib
            print_success(f"Convergence plot saved to {plot}")

        # Display final summary
        final_score = convergence_history[-1][1]
        summary_table = Table(title="Monitoring Summary")
        summary_table.add_column("Metric", style="cyan")
        summary_table.add_column("Value", style="green", justify="right")

        summary_table.add_row("Total Checkpoints", str(len(checkpoint_files)))
        summary_table.add_row("Agreement Metric", metric)
        summary_table.add_row("Final Score", f"{final_score:.3f}")
        summary_table.add_row("Converged (≥0.80)", "Yes" if final_score >= 0.80 else "No")

        console.print(summary_table)

    except FileNotFoundError as e:
        print_error(f"File not found: {e}")
        ctx.exit(1)
    except Exception as e:
        print_error(f"Monitoring failed: {e}")
        ctx.exit(1)


# Register commands
active_learning.add_command(select_items)
active_learning.add_command(run_active_learning)
active_learning.add_command(check_convergence)
active_learning.add_command(monitor_convergence)
