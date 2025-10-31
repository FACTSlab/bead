"""Deployment commands for bead CLI.

This module provides commands for generating and deploying jsPsych experiments
(Stage 5 of the bead pipeline).
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID

import click
from pydantic import ValidationError
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from bead.cli.utils import print_error, print_info, print_success
from bead.deployment.jatos.api import JATOSClient
from bead.deployment.jatos.exporter import JATOSExporter
from bead.deployment.jspsych.config import ExperimentConfig
from bead.deployment.jspsych.generator import JsPsychExperimentGenerator
from bead.items.models import Item, ItemTemplate
from bead.lists.models import ExperimentList

console = Console()


@click.group()
def deployment() -> None:
    r"""Deployment commands (Stage 5).

    Commands for generating and deploying jsPsych experiments.

    \b
    Examples:
        $ bead deployment generate lists/ items.jsonl experiment/
        $ bead deployment export-jatos experiment/ study.jzip \\
            --title "My Study"
        $ bead deployment upload-jatos study.jzip \\
            --jatos-url https://jatos.example.com --api-token TOKEN
        $ bead deployment validate experiment/
    """


@click.command()
@click.argument(
    "lists_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.argument("items_file", type=click.Path(exists=True, path_type=Path))
@click.argument("output_dir", type=click.Path(path_type=Path))
@click.option(
    "--experiment-type",
    type=click.Choice(["likert_rating", "forced_choice", "magnitude_estimation"]),
    default="likert_rating",
    help="Type of experiment",
)
@click.option("--title", default="Experiment", help="Experiment title")
@click.option("--description", default="", help="Experiment description")
@click.option(
    "--instructions", default="Please complete the task.", help="Instructions text"
)
@click.pass_context
def generate(
    ctx: click.Context,
    lists_dir: Path,
    items_file: Path,
    output_dir: Path,
    experiment_type: str,
    title: str,
    description: str,
    instructions: str,
) -> None:
    r"""Generate jsPsych experiment from lists and items.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    lists_dir : Path
        Directory containing experiment list files.
    items_file : Path
        Path to items file.
    output_dir : Path
        Output directory for generated experiment.
    experiment_type : str
        Type of experiment to generate.
    title : str
        Experiment title.
    description : str
        Experiment description.
    instructions : str
        Instructions text.

    Examples
    --------
    $ bead deployment generate lists/ items.jsonl experiment/ \\
        --experiment-type likert_rating \\
        --title "Acceptability Study" \\
        --instructions "Rate each sentence from 1 to 7"
    """
    try:
        # Load experiment lists
        print_info(f"Loading experiment lists from {lists_dir}")
        list_files = list(lists_dir.glob("*.jsonl"))
        if not list_files:
            print_error(f"No list files found in {lists_dir}")
            ctx.exit(1)

        experiment_lists: list[ExperimentList] = []
        for list_file in list_files:
            with open(list_file, encoding="utf-8") as f:
                first_line = f.readline().strip()
                if first_line:
                    list_data = json.loads(first_line)
                    exp_list = ExperimentList(**list_data)
                    experiment_lists.append(exp_list)

        print_info(f"Loaded {len(experiment_lists)} experiment lists")

        # Load items
        print_info(f"Loading items from {items_file}")
        items_dict: dict[UUID, Item] = {}
        with open(items_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item_data = json.loads(line)
                item = Item(**item_data)
                items_dict[item.id] = item

        print_info(f"Loaded {len(items_dict)} items")

        # Create empty templates dict (simplified for CLI)
        templates_dict: dict[UUID, ItemTemplate] = {}

        # Create experiment config
        config = ExperimentConfig(
            experiment_type=experiment_type,  # type: ignore
            title=title,
            description=description,
            instructions=instructions,
        )

        # Generate experiment
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Generating jsPsych experiment...", total=None)

            generator = JsPsychExperimentGenerator(
                config=config,
                output_dir=output_dir,
            )
            output_path = generator.generate(
                lists=experiment_lists,
                items=items_dict,
                templates=templates_dict,
            )

        print_success(f"Generated jsPsych experiment: {output_path}")

    except ValidationError as e:
        print_error(f"Validation error: {e}")
        ctx.exit(1)
    except Exception as e:
        print_error(f"Failed to generate experiment: {e}")
        ctx.exit(1)


@click.command()
@click.argument(
    "experiment_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.argument("output_file", type=click.Path(path_type=Path))
@click.option("--title", required=True, help="Study title for JATOS")
@click.option("--description", default="", help="Study description")
@click.option("--component-title", default="Main Experiment", help="Component title")
@click.pass_context
def export_jatos(
    ctx: click.Context,
    experiment_dir: Path,
    output_file: Path,
    title: str,
    description: str,
    component_title: str,
) -> None:
    r"""Export experiment to JATOS .jzip file.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    experiment_dir : Path
        Directory containing generated experiment.
    output_file : Path
        Output path for .jzip file.
    title : str
        Study title for JATOS.
    description : str
        Study description.
    component_title : str
        Component title.

    Examples
    --------
    $ bead deployment export-jatos experiment/ study.jzip \\
        --title "Acceptability Study" \\
        --description "Rating task for linguistic acceptability"
    """
    try:
        print_info(f"Exporting experiment from {experiment_dir}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Creating JATOS package...", total=None)

            exporter = JATOSExporter(
                study_title=title,
                study_description=description,
            )
            exporter.export(
                experiment_dir=experiment_dir,
                output_path=output_file,
                component_title=component_title,
            )

        print_success(f"Created JATOS package: {output_file}")

    except FileNotFoundError as e:
        print_error(f"File not found: {e}")
        ctx.exit(1)
    except ValueError as e:
        print_error(f"Invalid experiment: {e}")
        ctx.exit(1)
    except Exception as e:
        print_error(f"Failed to export to JATOS: {e}")
        ctx.exit(1)


@click.command()
@click.argument("jzip_file", type=click.Path(exists=True, path_type=Path))
@click.option("--jatos-url", required=True, help="JATOS server URL")
@click.option("--api-token", required=True, help="JATOS API token")
@click.pass_context
def upload_jatos(
    ctx: click.Context,
    jzip_file: Path,
    jatos_url: str,
    api_token: str,
) -> None:
    r"""Upload .jzip file to JATOS server.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    jzip_file : Path
        Path to .jzip file.
    jatos_url : str
        JATOS server URL.
    api_token : str
        JATOS API token.

    Examples
    --------
    $ bead deployment upload-jatos study.jzip \\
        --jatos-url https://jatos.example.com \\
        --api-token my-api-token
    """
    try:
        print_info(f"Uploading {jzip_file} to {jatos_url}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Uploading to JATOS...", total=None)

            client = JATOSClient(base_url=jatos_url, api_token=api_token)
            study_id: int = client.import_study(jzip_file)  # type: ignore[attr-defined]

        print_success(f"Uploaded study to JATOS (Study ID: {study_id})")

    except Exception as e:
        print_error(f"Failed to upload to JATOS: {e}")
        ctx.exit(1)


@click.command()
@click.argument(
    "experiment_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.pass_context
def validate(ctx: click.Context, experiment_dir: Path) -> None:
    """Validate generated experiment structure.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    experiment_dir : Path
        Directory containing generated experiment.

    Examples
    --------
    $ bead deployment validate experiment/
    """
    try:
        print_info(f"Validating experiment: {experiment_dir}")

        # Check required files
        required_files = [
            "index.html",
            "css/experiment.css",
            "js/experiment.js",
            "data/timeline.json",
        ]

        missing_files: list[str] = []
        for file_path in required_files:
            full_path = experiment_dir / file_path
            if not full_path.exists():
                missing_files.append(file_path)

        if missing_files:
            print_error("Missing required files:")
            for file_path in missing_files:
                console.print(f"  [red]✗[/red] {file_path}")
            ctx.exit(1)

        # Validate timeline.json
        timeline_file = experiment_dir / "data" / "timeline.json"
        with open(timeline_file, encoding="utf-8") as f:
            timeline_data: object = json.load(f)

        if not isinstance(timeline_data, list):
            print_error("timeline.json must be a list")
            ctx.exit(1)

        print_success(
            f"Experiment structure is valid ({len(timeline_data)} trials)"  # type: ignore[arg-type]
        )

    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON in timeline.json: {e}")
        ctx.exit(1)
    except Exception as e:
        print_error(f"Failed to validate experiment: {e}")
        ctx.exit(1)


# Register commands
deployment.add_command(generate)
deployment.add_command(export_jatos)
deployment.add_command(upload_jatos)
deployment.add_command(validate)
