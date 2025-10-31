"""List partitioning commands for bead CLI.

This module provides commands for partitioning items into experiment lists
(Stage 4 of the bead pipeline).
"""

from __future__ import annotations

import json
from pathlib import Path

import click
from pydantic import ValidationError
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from bead.cli.utils import print_error, print_info, print_success
from bead.items.models import Item
from bead.lists.models import ExperimentList
from bead.lists.partitioner import ListPartitioner

console = Console()


@click.group()
def lists() -> None:
    r"""List construction commands (Stage 4).

    Commands for partitioning items into experiment lists.

    \b
    Examples:
        $ bead lists partition items.jsonl lists/ --n-lists 5 --strategy balanced
        $ bead lists list lists/
        $ bead lists validate lists/list_0.jsonl
        $ bead lists show-stats lists/
    """


@click.command()
@click.argument("items_file", type=click.Path(exists=True, path_type=Path))
@click.argument("output_dir", type=click.Path(path_type=Path))
@click.option(
    "--strategy",
    type=click.Choice(["balanced", "random", "stratified"]),
    default="balanced",
    help="Partitioning strategy",
)
@click.option(
    "--n-lists",
    type=int,
    required=True,
    help="Number of lists to create",
)
@click.option(
    "--random-seed",
    type=int,
    help="Random seed for reproducibility",
)
@click.pass_context
def partition(
    ctx: click.Context,
    items_file: Path,
    output_dir: Path,
    strategy: str,
    n_lists: int,
    random_seed: int | None,
) -> None:
    r"""Partition items into experiment lists.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    items_file : Path
        Path to items file.
    output_dir : Path
        Output directory for list files.
    strategy : str
        Partitioning strategy.
    n_lists : int
        Number of lists to create.
    random_seed : int | None
        Random seed for reproducibility.

    Examples
    --------
    # Balanced partitioning
    $ bead lists partition items.jsonl lists/ --n-lists 5 --strategy balanced

    # Random partitioning with seed
    $ bead lists partition items.jsonl lists/ --n-lists 5 \\
        --strategy random --random-seed 42

    # Stratified partitioning
    $ bead lists partition items.jsonl lists/ --n-lists 5 --strategy stratified
    """
    try:
        if n_lists < 1:
            print_error("--n-lists must be >= 1")
            ctx.exit(1)

        # Load items
        print_info(f"Loading items from {items_file}")
        items: list[Item] = []
        with open(items_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item_data = json.loads(line)
                item = Item(**item_data)
                items.append(item)

        if len(items) == 0:
            print_error("No items found in file")
            ctx.exit(1)

        print_info(f"Loaded {len(items)} items")

        # Extract item UUIDs and create minimal metadata
        item_uuids = [item.id for item in items]
        metadata = {
            item.id: {"template_id": str(item.item_template_id)} for item in items
        }

        # Create partitioner
        partitioner = ListPartitioner(random_seed=random_seed)

        # Partition items
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task(
                f"Partitioning {len(items)} items into {n_lists} lists...", total=None
            )
            experiment_lists = partitioner.partition(
                items=item_uuids,
                n_lists=n_lists,
                strategy=strategy,
                metadata=metadata,
            )

        # Save lists
        output_dir.mkdir(parents=True, exist_ok=True)
        for exp_list in experiment_lists:
            list_file = output_dir / f"list_{exp_list.list_number}.jsonl"
            with open(list_file, "w", encoding="utf-8") as f:
                f.write(exp_list.model_dump_json() + "\n")

        print_success(
            f"Created {len(experiment_lists)} lists "
            f"with {len(items)} items: {output_dir}"
        )

        # Show distribution
        console.print("\n[cyan]Distribution:[/cyan]")
        for exp_list in experiment_lists:
            console.print(
                f"  list_{exp_list.list_number}: {len(exp_list.item_refs)} items"
            )

    except ValidationError as e:
        print_error(f"Validation error: {e}")
        ctx.exit(1)
    except Exception as e:
        print_error(f"Failed to partition items: {e}")
        ctx.exit(1)


@click.command(name="list")
@click.option(
    "--directory",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path.cwd(),
    help="Directory to search for list files",
)
@click.option(
    "--pattern",
    default="*.jsonl",
    help="File pattern to match (default: *.jsonl)",
)
@click.pass_context
def list_lists(
    ctx: click.Context,
    directory: Path,
    pattern: str,
) -> None:
    """List experiment list files in a directory.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    directory : Path
        Directory to search.
    pattern : str
        File pattern to match.

    Examples
    --------
    $ bead lists list
    $ bead lists list --directory experiment_lists/
    $ bead lists list --pattern "list_*.jsonl"
    """
    try:
        files = list(directory.glob(pattern))

        if not files:
            print_info(f"No files found in {directory} matching {pattern}")
            return

        table = Table(title=f"Experiment Lists in {directory}")
        table.add_column("File", style="cyan")
        table.add_column("List #", justify="right", style="yellow")
        table.add_column("Items", justify="right", style="green")
        table.add_column("Name", style="white")

        for file_path in sorted(files):
            try:
                with open(file_path, encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    if not first_line:
                        continue

                list_data = json.loads(first_line)
                exp_list = ExperimentList(**list_data)

                table.add_row(
                    str(file_path.name),
                    str(exp_list.list_number),
                    str(len(exp_list.item_refs)),
                    exp_list.name,
                )
            except Exception:
                continue

        console.print(table)

    except Exception as e:
        print_error(f"Failed to list experiment lists: {e}")
        ctx.exit(1)


@click.command()
@click.argument("list_file", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def validate(ctx: click.Context, list_file: Path) -> None:
    """Validate an experiment list file.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    list_file : Path
        Path to experiment list file.

    Examples
    --------
    $ bead lists validate list_0.jsonl
    """
    try:
        print_info(f"Validating experiment list: {list_file}")

        with open(list_file, encoding="utf-8") as f:
            first_line = f.readline().strip()
            if not first_line:
                print_error("File is empty")
                ctx.exit(1)

        list_data = json.loads(first_line)
        exp_list = ExperimentList(**list_data)

        print_success(
            f"Experiment list is valid: {exp_list.name} "
            f"({len(exp_list.item_refs)} items)"
        )

    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON: {e}")
        ctx.exit(1)
    except ValidationError as e:
        print_error(f"Validation error: {e}")
        ctx.exit(1)
    except Exception as e:
        print_error(f"Failed to validate experiment list: {e}")
        ctx.exit(1)


@click.command()
@click.argument(
    "lists_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.pass_context
def show_stats(ctx: click.Context, lists_dir: Path) -> None:
    """Show statistics about experiment lists in a directory.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    lists_dir : Path
        Directory containing list files.

    Examples
    --------
    $ bead lists show-stats lists/
    """
    try:
        print_info(f"Analyzing experiment lists in: {lists_dir}")

        list_files = list(lists_dir.glob("*.jsonl"))

        if not list_files:
            print_error("No list files found")
            ctx.exit(1)

        lists_data: list[ExperimentList] = []
        for file_path in list_files:
            try:
                with open(file_path, encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    if first_line:
                        list_data = json.loads(first_line)
                        exp_list = ExperimentList(**list_data)
                        lists_data.append(exp_list)
            except Exception:
                continue

        if not lists_data:
            print_error("No valid experiment lists found")
            ctx.exit(1)

        # Calculate statistics
        total_lists = len(lists_data)
        item_counts = [len(exp_list.item_refs) for exp_list in lists_data]
        total_items = sum(item_counts)
        avg_items = total_items / total_lists if total_lists > 0 else 0
        min_items = min(item_counts) if item_counts else 0
        max_items = max(item_counts) if item_counts else 0

        # Display statistics
        table = Table(title="Experiment List Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        table.add_row("Total Lists", str(total_lists))
        table.add_row("Total Items", str(total_items))
        table.add_row("", "")  # Separator
        table.add_row("Avg Items per List", f"{avg_items:.1f}")
        table.add_row("Min Items per List", str(min_items))
        table.add_row("Max Items per List", str(max_items))

        console.print(table)

        # Show per-list breakdown
        console.print("\n[cyan]Per-List Breakdown:[/cyan]")
        for exp_list in sorted(lists_data, key=lambda x: x.list_number):
            console.print(f"  {exp_list.name}: {len(exp_list.item_refs)} items")

    except Exception as e:
        print_error(f"Failed to show statistics: {e}")
        ctx.exit(1)


# Register commands
lists.add_command(partition)
lists.add_command(list_lists)
lists.add_command(validate)
lists.add_command(show_stats)
