"""Training commands for bead CLI.

This module provides commands for collecting data and training judgment prediction
models (Stage 6 of the bead pipeline).
"""

from __future__ import annotations

import json
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from bead.cli.utils import print_error, print_info, print_success
from bead.data_collection.jatos import JATOSDataCollector

console = Console()


@click.group()
def training() -> None:
    r"""Training commands (Stage 6).

    Commands for collecting data and training judgment prediction models.

    \b
    Examples:
        $ bead training collect-data results.jsonl \\
            --jatos-url https://jatos.example.com \\
            --api-token TOKEN --study-id 123
        $ bead training show-data-stats results.jsonl
    """


@click.command()
@click.argument("output_file", type=click.Path(path_type=Path))
@click.option("--jatos-url", required=True, help="JATOS server URL")
@click.option("--api-token", required=True, help="JATOS API token")
@click.option("--study-id", required=True, type=int, help="JATOS study ID")
@click.option("--component-id", type=int, help="Filter by component ID (optional)")
@click.option("--worker-type", help="Filter by worker type (optional)")
@click.pass_context
def collect_data(
    ctx: click.Context,
    output_file: Path,
    jatos_url: str,
    api_token: str,
    study_id: int,
    component_id: int | None,
    worker_type: str | None,
) -> None:
    r"""Collect judgment data from JATOS.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    output_file : Path
        Output path for collected data.
    jatos_url : str
        JATOS server URL.
    api_token : str
        JATOS API token.
    study_id : int
        JATOS study ID.
    component_id : int | None
        Component ID to filter by.
    worker_type : str | None
        Worker type to filter by.

    Examples
    --------
    $ bead training collect-data results.jsonl \\
        --jatos-url https://jatos.example.com \\
        --api-token my-token \\
        --study-id 123

    $ bead training collect-data results.jsonl \\
        --jatos-url https://jatos.example.com \\
        --api-token my-token \\
        --study-id 123 \\
        --component-id 456 \\
        --worker-type Prolific
    """
    try:
        print_info(f"Collecting data from JATOS study {study_id}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Downloading results from JATOS...", total=None)

            collector = JATOSDataCollector(
                base_url=jatos_url,
                api_token=api_token,
                study_id=study_id,
            )

            results = collector.download_results(
                output_path=output_file,
                component_id=component_id,
                worker_type=worker_type,
            )

        print_success(f"Collected {len(results)} results: {output_file}")

    except Exception as e:
        print_error(f"Failed to collect data: {e}")
        ctx.exit(1)


@click.command()
@click.argument("data_file", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def show_data_stats(ctx: click.Context, data_file: Path) -> None:
    """Show statistics about collected data.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    data_file : Path
        Path to data file.

    Examples
    --------
    $ bead training show-data-stats results.jsonl
    """
    try:
        print_info(f"Analyzing data: {data_file}")

        # Load and analyze data
        results: list[dict[str, object]] = []
        with open(data_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                result: dict[str, object] = json.loads(line)
                results.append(result)

        if not results:
            print_error("No data found in file")
            ctx.exit(1)

        # Calculate statistics
        total_results = len(results)

        # Count unique workers if available
        worker_ids: set[object] = set()
        for result in results:
            if "worker_id" in result:
                worker_ids.add(result["worker_id"])

        # Count response types if available
        response_types: dict[str, int] = {}
        for result in results:
            if "data" in result:
                data: object = result["data"]
                if isinstance(data, dict):
                    for key in data.keys():  # type: ignore[var-annotated]
                        key_str = str(key)  # type: ignore[arg-type]
                        response_types[key_str] = response_types.get(key_str, 0) + 1

        # Display statistics
        table = Table(title="Data Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        table.add_row("Total Results", str(total_results))
        if worker_ids:
            table.add_row("Unique Workers", str(len(worker_ids)))

        if response_types:
            table.add_row("", "")  # Separator
            for resp_type, count in sorted(response_types.items()):
                table.add_row(f"Response Type: {resp_type}", str(count))

        console.print(table)

    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON in data file: {e}")
        ctx.exit(1)
    except Exception as e:
        print_error(f"Failed to show statistics: {e}")
        ctx.exit(1)


# Register commands
training.add_command(collect_data)
training.add_command(show_data_stats)
