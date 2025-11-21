"""Workflow orchestration commands for the bead CLI.

This module provides commands for managing end-to-end pipeline workflows,
including running complete pipelines, resuming interrupted workflows, and
rolling back to previous stages.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import yaml
from rich.console import Console
from rich.table import Table

from bead.cli.utils import JsonValue, print_error, print_info, print_success

console = Console()


# ============================================================================
# State Management Utilities
# ============================================================================


def get_state_file(project_dir: Path) -> Path:
    """Get path to workflow state file.

    Parameters
    ----------
    project_dir : Path
        Project directory path.

    Returns
    -------
    Path
        Path to .bead/workflow_state.json
    """
    bead_dir = project_dir / ".bead"
    bead_dir.mkdir(exist_ok=True)
    return bead_dir / "workflow_state.json"


def load_state(project_dir: Path) -> dict[str, JsonValue]:
    """Load workflow state from file.

    Parameters
    ----------
    project_dir : Path
        Project directory path.

    Returns
    -------
    dict[str, JsonValue]
        Workflow state dictionary.
    """
    state_file = get_state_file(project_dir)
    if not state_file.exists():
        return {"stages": {}, "last_run": None}

    with open(state_file) as f:
        return json.load(f)


def save_state(project_dir: Path, state: dict[str, JsonValue]) -> None:
    """Save workflow state to file.

    Parameters
    ----------
    project_dir : Path
        Project directory path.
    state : dict[str, JsonValue]
        Workflow state dictionary.
    """
    state_file = get_state_file(project_dir)
    with open(state_file, "w") as f:
        json.dump(state, f, indent=2, default=str)


def update_stage_state(
    project_dir: Path, stage: str, status: str, error: str | None = None
) -> None:
    """Update state for a specific stage.

    Parameters
    ----------
    project_dir : Path
        Project directory path.
    stage : str
        Stage name.
    status : str
        Stage status ('pending', 'running', 'completed', 'failed').
    error : str | None
        Error message if status is 'failed'.
    """
    state = load_state(project_dir)
    if not isinstance(state.get("stages"), dict):
        state["stages"] = {}

    stages_dict = state["stages"]
    assert isinstance(stages_dict, dict)

    stages_dict[stage] = {
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "error": error,
    }

    state["last_run"] = datetime.now().isoformat()
    save_state(project_dir, state)


def detect_stage_completion(project_dir: Path, stage: str) -> bool:
    """Detect if a stage has been completed by checking filesystem.

    Parameters
    ----------
    project_dir : Path
        Project directory path.
    stage : str
        Stage name.

    Returns
    -------
    bool
        True if stage appears completed.
    """
    # Check for expected output files
    if stage == "resources":
        return (project_dir / "lexicons").exists() and any(
            (project_dir / "lexicons").glob("*.jsonl")
        )
    elif stage == "templates":
        return (project_dir / "templates").exists() and any(
            (project_dir / "templates").glob("*.jsonl")
        )
    elif stage == "items":
        return (project_dir / "items").exists() and any(
            (project_dir / "items").glob("*.jsonl")
        )
    elif stage == "lists":
        return (project_dir / "lists").exists() and any(
            (project_dir / "lists").glob("*.jsonl")
        )
    elif stage == "deployment":
        return (project_dir / "experiments").exists() and any(
            (project_dir / "experiments").iterdir()
        )
    elif stage == "training":
        return (project_dir / "models").exists() and any(
            (project_dir / "models").iterdir()
        )

    return False


# ============================================================================
# Workflow Templates
# ============================================================================

WORKFLOW_TEMPLATES = {
    "acceptability-study": {
        "name": "Acceptability Judgment Study",
        "description": "Collect acceptability judgments on linguistic stimuli",
        "config": {
            "project": {
                "name": "acceptability_study",
                "language_code": "eng",
                "description": "Acceptability judgment experiment",
            },
            "paths": {
                "lexicons_dir": "lexicons",
                "templates_dir": "templates",
                "items_dir": "items",
                "lists_dir": "lists",
                "experiments_dir": "experiments",
            },
            "templates": {"filling_strategy": "exhaustive"},
            "items": {"validation_enabled": True},
            "lists": {"n_lists": 10},
            "deployment": {"platform": "jatos", "jspsych_version": "8.0.0"},
        },
    },
    "forced-choice": {
        "name": "Forced Choice Study",
        "description": "2AFC or 3AFC comparison judgments",
        "config": {
            "project": {
                "name": "forced_choice_study",
                "language_code": "eng",
                "description": "Forced choice experiment",
            },
            "paths": {
                "lexicons_dir": "lexicons",
                "templates_dir": "templates",
                "items_dir": "items",
                "lists_dir": "lists",
                "experiments_dir": "experiments",
            },
            "templates": {"filling_strategy": "stratified"},
            "items": {"validation_enabled": True},
            "lists": {"n_lists": 20},
            "deployment": {"platform": "jatos", "jspsych_version": "8.0.0"},
        },
    },
    "ordinal-scale": {
        "name": "Ordinal Scale Study",
        "description": "Likert scale or slider ratings",
        "config": {
            "project": {
                "name": "ordinal_scale_study",
                "language_code": "eng",
                "description": "Ordinal scale experiment",
            },
            "paths": {
                "lexicons_dir": "lexicons",
                "templates_dir": "templates",
                "items_dir": "items",
                "lists_dir": "lists",
                "experiments_dir": "experiments",
            },
            "templates": {"filling_strategy": "random", "max_combinations": 500},
            "items": {"validation_enabled": True},
            "lists": {"n_lists": 15},
            "deployment": {"platform": "jatos", "jspsych_version": "8.0.0"},
        },
    },
    "active-learning": {
        "name": "Active Learning Study",
        "description": "Human-in-the-loop training with convergence detection",
        "config": {
            "project": {
                "name": "active_learning_study",
                "language_code": "eng",
                "description": "Active learning experiment",
            },
            "paths": {
                "lexicons_dir": "lexicons",
                "templates_dir": "templates",
                "items_dir": "items",
                "lists_dir": "lists",
                "experiments_dir": "experiments",
                "models_dir": "models",
            },
            "templates": {"filling_strategy": "stratified"},
            "items": {"validation_enabled": True},
            "lists": {"n_lists": 10},
            "deployment": {"platform": "jatos", "jspsych_version": "8.0.0"},
            "training": {
                "framework": "huggingface",
                "epochs": 10,
                "convergence": {"metric": "krippendorff_alpha", "threshold": 0.80},
            },
        },
    },
}


# ============================================================================
# Stage Execution Utilities
# ============================================================================


def _execute_stage(
    stage: str,
    config: dict[str, Any],
    project_dir: Path,
    verbose: bool,
) -> None:
    """Execute a specific pipeline stage.

    Parameters
    ----------
    stage : str
        Stage name ('resources', 'templates', 'items', 'lists',
        'deployment', 'training').
    config : dict[str, Any]
        Configuration dictionary from YAML.
    project_dir : Path
        Project directory path.
    verbose : bool
        Whether to show detailed command output.

    Raises
    ------
    RuntimeError
        If stage execution fails.
    """
    if stage == "resources":
        _execute_resources_stage(config, project_dir, verbose)
    elif stage == "templates":
        _execute_templates_stage(config, project_dir, verbose)
    elif stage == "items":
        _execute_items_stage(config, project_dir, verbose)
    elif stage == "lists":
        _execute_lists_stage(config, project_dir, verbose)
    elif stage == "deployment":
        _execute_deployment_stage(config, project_dir, verbose)
    elif stage == "training":
        _execute_training_stage(config, project_dir, verbose)
    else:
        raise ValueError(f"Unknown stage: {stage}")


def _execute_resources_stage(
    config: dict[str, Any], project_dir: Path, verbose: bool
) -> None:
    """Execute resources stage (lexicon and template creation).

    This stage typically involves manual creation of lexicons and templates
    or importing from external sources. For now, we validate that the
    required directories exist.

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary.
    project_dir : Path
        Project directory.
    verbose : bool
        Verbose output flag.

    Raises
    ------
    RuntimeError
        If resources directory doesn't exist or is empty.
    """
    paths = config.get("paths", {})
    lexicons_dir = project_dir / paths.get("lexicons_dir", "lexicons")
    templates_dir = project_dir / paths.get("templates_dir", "templates")

    # Check that resources exist
    if not lexicons_dir.exists():
        raise RuntimeError(f"Lexicons directory not found: {lexicons_dir}")

    if not templates_dir.exists():
        raise RuntimeError(f"Templates directory not found: {templates_dir}")

    # Count files
    lexicon_files = list(lexicons_dir.glob("*.jsonl"))
    template_files = list(templates_dir.glob("*.jsonl"))

    if not lexicon_files:
        raise RuntimeError(f"No lexicon files found in {lexicons_dir}")

    if not template_files:
        raise RuntimeError(f"No template files found in {templates_dir}")

    console.print(
        f"[green]✓[/green] Found {len(lexicon_files)} lexicon(s) "
        f"and {len(template_files)} template(s)"
    )


def _execute_templates_stage(
    config: dict[str, Any], project_dir: Path, verbose: bool
) -> None:
    """Execute templates stage (template filling).

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary.
    project_dir : Path
        Project directory.
    verbose : bool
        Verbose output flag.

    Raises
    ------
    RuntimeError
        If template filling fails.
    """
    paths = config.get("paths", {})
    templates_config = config.get("templates", {})

    templates_dir = project_dir / paths.get("templates_dir", "templates")
    lexicons_dir = project_dir / paths.get("lexicons_dir", "lexicons")
    output_dir = project_dir / paths.get("filled_templates_dir", "filled_templates")
    output_dir.mkdir(exist_ok=True)

    # Get template and lexicon files
    template_files = list(templates_dir.glob("*.jsonl"))
    lexicon_files = list(lexicons_dir.glob("*.jsonl"))

    if not template_files:
        raise RuntimeError(f"No template files found in {templates_dir}")
    if not lexicon_files:
        raise RuntimeError(f"No lexicon files found in {lexicons_dir}")

    # Build command for each template file
    strategy = templates_config.get("filling_strategy", "exhaustive")

    for template_file in template_files:
        output_file = output_dir / f"filled_{template_file.name}"

        cmd = [
            "bead",
            "templates",
            "fill",
            str(template_file),
            *[str(f) for f in lexicon_files],
            str(output_file),
            "--strategy",
            strategy,
        ]

        console.print(f"[cyan]Filling template: {template_file.name}[/cyan]")
        _run_command(cmd, verbose)


def _execute_items_stage(
    config: dict[str, Any], project_dir: Path, verbose: bool
) -> None:
    """Execute items stage (item construction).

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary.
    project_dir : Path
        Project directory.
    verbose : bool
        Verbose output flag.

    Raises
    ------
    RuntimeError
        If item construction fails.
    """
    paths = config.get("paths", {})
    items_config = config.get("items", {})

    filled_dir = project_dir / paths.get("filled_templates_dir", "filled_templates")
    output_dir = project_dir / paths.get("items_dir", "items")
    output_dir.mkdir(exist_ok=True)

    # Get filled template files
    filled_files = list(filled_dir.glob("*.jsonl"))

    if not filled_files:
        raise RuntimeError(f"No filled templates found in {filled_dir}")

    # Build item construction command
    task_type = items_config.get("task_type")

    if task_type:
        # Use task-type-specific command if specified
        output_file = output_dir / "items.jsonl"

        cmd = [
            "bead",
            "items",
            "construct",
            *[str(f) for f in filled_files],
            str(output_file),
            "--task-type",
            task_type,
        ]
    else:
        # Use generic construct command
        output_file = output_dir / "items.jsonl"

        cmd = [
            "bead",
            "items",
            "construct",
            *[str(f) for f in filled_files],
            str(output_file),
        ]

    console.print(
        f"[cyan]Constructing items from {len(filled_files)} template(s)[/cyan]"
    )
    _run_command(cmd, verbose)


def _execute_lists_stage(
    config: dict[str, Any], project_dir: Path, verbose: bool
) -> None:
    """Execute lists stage (list partitioning).

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary.
    project_dir : Path
        Project directory.
    verbose : bool
        Verbose output flag.

    Raises
    ------
    RuntimeError
        If list partitioning fails.
    """
    paths = config.get("paths", {})
    lists_config = config.get("lists", {})

    items_dir = project_dir / paths.get("items_dir", "items")
    output_dir = project_dir / paths.get("lists_dir", "lists")
    output_dir.mkdir(exist_ok=True)

    # Get item files
    item_files = list(items_dir.glob("*.jsonl"))

    if not item_files:
        raise RuntimeError(f"No item files found in {items_dir}")

    # Get first item file (typically there's just one)
    item_file = item_files[0]

    # Build partitioning command
    n_lists = lists_config.get("n_lists", 10)

    cmd = [
        "bead",
        "lists",
        "partition",
        str(item_file),
        str(output_dir),
        "--n-lists",
        str(n_lists),
    ]

    # Add constraints if specified
    if "list_constraints" in lists_config:
        constraints_file = project_dir / lists_config["list_constraints"]
        if constraints_file.exists():
            cmd.extend(["--list-constraints", str(constraints_file)])

    if "batch_constraints" in lists_config:
        constraints_file = project_dir / lists_config["batch_constraints"]
        if constraints_file.exists():
            cmd.extend(["--batch-constraints", str(constraints_file)])

    console.print(f"[cyan]Partitioning items into {n_lists} lists[/cyan]")
    _run_command(cmd, verbose)


def _execute_deployment_stage(
    config: dict[str, Any], project_dir: Path, verbose: bool
) -> None:
    """Execute deployment stage (experiment generation).

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary.
    project_dir : Path
        Project directory.
    verbose : bool
        Verbose output flag.

    Raises
    ------
    RuntimeError
        If deployment generation fails.
    """
    paths = config.get("paths", {})
    deployment_config = config.get("deployment", {})

    lists_dir = project_dir / paths.get("lists_dir", "lists")
    items_dir = project_dir / paths.get("items_dir", "items")
    output_dir = project_dir / paths.get("experiments_dir", "experiments")
    output_dir.mkdir(exist_ok=True)

    # Get list and item files
    item_files = list(items_dir.glob("*.jsonl"))

    if not item_files:
        raise RuntimeError(f"No item files found in {items_dir}")

    if not lists_dir.exists():
        raise RuntimeError(f"Lists directory not found: {lists_dir}")

    item_file = item_files[0]

    # Build deployment command
    cmd = [
        "bead",
        "deployment",
        "generate",
        str(lists_dir),
        str(item_file),
        str(output_dir),
    ]

    # Add distribution strategy (required)
    dist_strategy = deployment_config.get("distribution_strategy", "balanced")
    cmd.extend(["--distribution-strategy", dist_strategy])

    # Add experiment type if specified
    if "experiment_type" in deployment_config:
        cmd.extend(["--experiment-type", deployment_config["experiment_type"]])

    console.print(
        f"[cyan]Generating deployment with {dist_strategy} distribution[/cyan]"
    )
    _run_command(cmd, verbose)


def _execute_training_stage(
    config: dict[str, Any], project_dir: Path, verbose: bool
) -> None:
    """Execute training stage (model training).

    Parameters
    ----------
    config : dict[str, Any]
        Configuration dictionary.
    project_dir : Path
        Project directory.
    verbose : bool
        Verbose output flag.

    Raises
    ------
    RuntimeError
        If training fails.
    """
    paths = config.get("paths", {})
    _ = config.get("training", {})  # For future use

    items_dir = project_dir / paths.get("items_dir", "items")
    output_dir = project_dir / paths.get("models_dir", "models")
    output_dir.mkdir(exist_ok=True)

    # Get item files
    item_files = list(items_dir.glob("*.jsonl"))

    if not item_files:
        raise RuntimeError(f"No item files found in {items_dir}")

    item_file = item_files[0]

    # Training typically requires data collection first
    # For workflow orchestration, we just validate the setup
    console.print(
        "[yellow]⚠[/yellow] Training stage requires data collection. "
        "Skipping automated execution."
    )
    console.print(
        "[cyan]ℹ[/cyan] After data collection, run: "
        f"bead training train-model --items {item_file} --data <data.jsonl>"
    )


def _run_command(cmd: list[str], verbose: bool) -> None:
    """Run a subprocess command with error handling.

    Parameters
    ----------
    cmd : list[str]
        Command and arguments to execute.
    verbose : bool
        Whether to show command output in real-time.

    Raises
    ------
    RuntimeError
        If command execution fails.
    """
    if verbose:
        console.print(f"[dim]Running: {' '.join(cmd)}[/dim]")

    try:
        if verbose:
            # Show output in real-time
            result = subprocess.run(cmd, check=True, text=True)
        else:
            # Capture output
            result = subprocess.run(
                cmd, capture_output=True, text=True, check=True
            )

        if result.stdout and verbose:
            console.print(result.stdout)

    except subprocess.CalledProcessError as e:
        error_msg = f"Command failed: {' '.join(cmd)}"
        if e.stderr:
            error_msg += f"\n{e.stderr}"
        raise RuntimeError(error_msg) from e
    except FileNotFoundError as e:
        raise RuntimeError(
            f"Command not found: {cmd[0]}. Is bead installed?"
        ) from e


# ============================================================================
# Workflow Commands
# ============================================================================


@click.group()
def workflow() -> None:
    """Manage end-to-end pipeline workflows.

    Examples
    --------
    Run complete pipeline:
        $ bead workflow run --config bead.yaml

    Initialize new project:
        $ bead workflow init acceptability-study

    Check workflow status:
        $ bead workflow status

    Resume interrupted workflow:
        $ bead workflow resume

    Rollback to previous stage:
        $ bead workflow rollback deployment
    """


@workflow.command()
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="bead.yaml",
    help="Path to configuration file",
)
@click.option(
    "--stages",
    type=str,
    default=None,
    help="Comma-separated list of stages to run (default: all)",
)
@click.option(
    "--from-stage",
    type=click.Choice(
        ["resources", "templates", "items", "lists", "deployment", "training"]
    ),
    default=None,
    help="Start from this stage",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Show what would be executed without running",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Show detailed command output",
)
def run(
    config_path: Path,
    stages: str | None,
    from_stage: str | None,
    dry_run: bool,
    verbose: bool,
) -> None:
    """Run complete pipeline workflow.

    Executes all pipeline stages sequentially:
    1. resources - Create lexicons and templates
    2. templates - Fill templates with lexicon items
    3. items - Construct experimental items
    4. lists - Partition items into experiment lists
    5. deployment - Generate jsPsych experiments
    6. training - Train models with active learning (optional)

    The workflow tracks progress and can be resumed if interrupted.

    Examples
    --------
    Run all stages:
        $ bead workflow run --config bead.yaml

    Run specific stages:
        $ bead workflow run --stages resources,templates,items

    Start from items stage:
        $ bead workflow run --from-stage items

    Dry run to preview:
        $ bead workflow run --dry-run
    """
    project_dir = config_path.parent
    console.rule("[bold]Pipeline Workflow Execution[/bold]")

    # Load configuration
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print_error(f"Failed to load config: {e}")
        sys.exit(1)

    # Determine which stages to run
    all_stages = [
        "resources",
        "templates",
        "items",
        "lists",
        "deployment",
        "training",
    ]

    if stages:
        selected_stages = [s.strip() for s in stages.split(",")]
    elif from_stage:
        start_idx = all_stages.index(from_stage)
        selected_stages = all_stages[start_idx:]
    else:
        selected_stages = all_stages

    # Show plan
    print_info(f"Configuration: {config_path}")
    print_info(f"Stages to run: {', '.join(selected_stages)}")

    if dry_run:
        console.print("\n[yellow]DRY RUN MODE - No commands will be executed[/yellow]")
        for stage in selected_stages:
            console.print(f"  • Would execute: [cyan]{stage}[/cyan] stage")
        return

    # Execute stages
    _ = load_state(project_dir)  # For resume/status compatibility
    failed = False

    for stage in selected_stages:
        console.rule(f"[bold cyan]Stage: {stage}[/bold cyan]")

        try:
            update_stage_state(project_dir, stage, "running")

            # Execute the stage
            _execute_stage(stage, config, project_dir, verbose)

            update_stage_state(project_dir, stage, "completed")
            print_success(f"{stage} stage completed")

        except RuntimeError as e:
            update_stage_state(project_dir, stage, "failed", str(e))
            print_error(f"Stage '{stage}' failed: {e}")
            failed = True
            break
        except Exception as e:
            update_stage_state(project_dir, stage, "failed", str(e))
            print_error(f"Stage '{stage}' failed with unexpected error: {e}")
            failed = True
            break

    if not failed:
        console.rule("[bold green]Pipeline Complete[/bold green]")
        print_success("All stages completed successfully")
    else:
        console.rule("[bold red]Pipeline Failed[/bold red]")
        print_error("Pipeline execution failed. Use 'bead workflow resume' to continue")
        sys.exit(1)


@workflow.command()
@click.argument("template", type=click.Choice(list(WORKFLOW_TEMPLATES.keys())))
@click.option(
    "--output-dir",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output directory (default: current directory)",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Overwrite existing files",
)
def init(template: str, output_dir: Path | None, force: bool) -> None:
    """Initialize new project from template.

    Creates a complete project structure with configuration file,
    directory layout, and example files.

    Available templates:
    - acceptability-study: Acceptability judgment experiments
    - forced-choice: 2AFC or 3AFC comparison judgments
    - ordinal-scale: Likert scale or slider ratings
    - active-learning: Human-in-the-loop training

    Examples
    --------
    Initialize acceptability study:
        $ bead workflow init acceptability-study

    Initialize in specific directory:
        $ bead workflow init forced-choice --output-dir my-project

    Overwrite existing files:
        $ bead workflow init ordinal-scale --force
    """
    if output_dir is None:
        output_dir = Path.cwd()

    template_spec = WORKFLOW_TEMPLATES[template]
    console.rule(f"[bold]Initialize: {template_spec['name']}[/bold]")

    # Create directory structure
    config_data: Any = template_spec["config"]
    paths: dict[str, Any] = config_data.get("paths", {})

    dirs_to_create: list[str] = [
        str(paths.get("lexicons_dir", "lexicons")),
        str(paths.get("templates_dir", "templates")),
        str(paths.get("items_dir", "items")),
        str(paths.get("lists_dir", "lists")),
        str(paths.get("experiments_dir", "experiments")),
    ]

    if "models_dir" in paths:
        dirs_to_create.append(str(paths["models_dir"]))

    for dir_name in dirs_to_create:
        dir_path = output_dir / dir_name
        if dir_path.exists() and not force:
            console.print(f"[yellow]⚠[/yellow] Directory exists: {dir_name}")
        else:
            dir_path.mkdir(parents=True, exist_ok=True)
            console.print(f"[green]✓[/green] Created: {dir_name}/")

    # Create configuration file
    config_file = output_dir / "bead.yaml"
    if config_file.exists() and not force:
        print_error(f"Configuration file exists: {config_file}")
        print_info("Use --force to overwrite")
        sys.exit(1)

    with open(config_file, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
    print_success(f"Created configuration: {config_file}")

    # Create .gitignore
    gitignore_file = output_dir / ".gitignore"
    gitignore_content = """# bead workflow ignores
.bead/
.cache/
*.pyc
__pycache__/
*.jsonl
experiments/
models/
"""
    with open(gitignore_file, "w") as f:
        f.write(gitignore_content)
    print_success("Created .gitignore")

    console.print("\n[bold green]✓ Project initialized[/bold green]")
    console.print("\n[bold]Next steps:[/bold]")
    console.print("  1. Edit bead.yaml to configure your experiment")
    console.print("  2. Create lexicon files in lexicons/")
    console.print("  3. Create template files in templates/")
    console.print("  4. Run: bead workflow run")


@workflow.command()
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="bead.yaml",
    help="Path to configuration file",
)
def status(config_path: Path) -> None:
    """Show current workflow status.

    Displays completion status for each pipeline stage by checking
    both the workflow state file and the filesystem.

    Examples
    --------
    Show workflow status:
        $ bead workflow status

    Use custom config:
        $ bead workflow status --config my-config.yaml
    """
    project_dir = config_path.parent
    console.rule("[bold]Workflow Status[/bold]")

    # Load state
    state = load_state(project_dir)
    stages_state: dict[str, Any] = state.get("stages", {})  # type: ignore

    # Check filesystem
    all_stages = [
        "resources",
        "templates",
        "items",
        "lists",
        "deployment",
        "training",
    ]

    table = Table(title="Pipeline Stage Status")
    table.add_column("Stage", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Last Updated")

    for stage in all_stages:
        stage_info: dict[str, Any] = stages_state.get(stage, {})

        # Check filesystem
        fs_complete = detect_stage_completion(project_dir, stage)

        # Determine status
        if stage_info.get("status") == "completed":
            status = "[green]✓ Completed[/green]"
            timestamp = str(stage_info.get("timestamp", "Unknown"))
        elif stage_info.get("status") == "failed":
            status = "[red]✗ Failed[/red]"
            timestamp = str(stage_info.get("timestamp", "Unknown"))
        elif fs_complete:
            status = "[yellow]⚠ Detected[/yellow]"
            timestamp = "Filesystem check"
        else:
            status = "[dim]○ Pending[/dim]"
            timestamp = "-"

        table.add_row(stage, status, timestamp)

    console.print(table)

    if state.get("last_run"):
        console.print(f"\n[dim]Last run: {state.get('last_run')}[/dim]")


@workflow.command()
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="bead.yaml",
    help="Path to configuration file",
)
def resume(config_path: Path) -> None:
    """Resume interrupted workflow.

    Reads the workflow state file and continues execution from the
    last incomplete stage, skipping stages that have already completed.

    Examples
    --------
    Resume workflow:
        $ bead workflow resume

    Resume with custom config:
        $ bead workflow resume --config my-config.yaml
    """
    project_dir = config_path.parent
    console.rule("[bold]Resume Workflow[/bold]")

    # Load state
    state = load_state(project_dir)
    stages_value = state.get("stages", {})

    # Validate stages is a dict
    if not isinstance(stages_value, dict):
        print_error("Invalid workflow state. Use 'bead workflow run' to start.")
        sys.exit(1)

    stages_state: dict[str, Any] = stages_value

    if not stages_state:
        print_error("No workflow state found. Use 'bead workflow run' to start.")
        sys.exit(1)

    # Find last completed stage
    all_stages = [
        "resources",
        "templates",
        "items",
        "lists",
        "deployment",
        "training",
    ]

    last_completed_idx = -1
    for i, stage in enumerate(all_stages):
        stage_info: dict[str, Any] = stages_state.get(stage, {})  # type: ignore[assignment]
        if stage_info.get("status") == "completed":
            last_completed_idx = i

    if last_completed_idx == len(all_stages) - 1:
        print_success("All stages completed. Nothing to resume.")
        return

    # Resume from next stage
    resume_from = all_stages[last_completed_idx + 1]
    console.print(f"[cyan]Resuming from stage: {resume_from}[/cyan]")
    console.print(f"[dim]Last completed: {all_stages[last_completed_idx]}[/dim]\n")

    # Invoke run command with from-stage
    ctx = click.get_current_context()
    ctx.invoke(
        run,
        config_path=config_path,
        stages=None,
        from_stage=resume_from,
        dry_run=False,
        verbose=False,
    )


@workflow.command()
@click.argument(
    "stage",
    type=click.Choice(
        ["resources", "templates", "items", "lists", "deployment", "training"]
    ),
)
@click.option(
    "--config",
    "-c",
    "config_path",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default="bead.yaml",
    help="Path to configuration file",
)
@click.option(
    "--force",
    is_flag=True,
    default=False,
    help="Skip confirmation prompt",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Show what would be deleted without deleting",
)
def rollback(stage: str, config_path: Path, force: bool, dry_run: bool) -> None:
    """Rollback to previous stage.

    Deletes outputs from the specified stage and all subsequent stages,
    allowing you to re-run from that point. Also updates the workflow
    state file.

    Examples
    --------
    Rollback to items stage:
        $ bead workflow rollback items

    Dry run to preview:
        $ bead workflow rollback deployment --dry-run

    Skip confirmation:
        $ bead workflow rollback templates --force
    """
    project_dir = config_path.parent
    console.rule("[bold yellow]Rollback Workflow[/bold yellow]")

    # Determine stages to delete
    all_stages = [
        "resources",
        "templates",
        "items",
        "lists",
        "deployment",
        "training",
    ]

    stage_idx = all_stages.index(stage)
    stages_to_delete = all_stages[stage_idx:]

    # Map stages to directories
    stage_dirs: dict[str, list[str]] = {
        "resources": ["lexicons", "templates"],
        "templates": ["filled_templates"],
        "items": ["items"],
        "lists": ["lists"],
        "deployment": ["experiments"],
        "training": ["models"],
    }

    dirs_to_delete: list[str] = []
    for s in stages_to_delete:
        dirs_to_delete.extend(stage_dirs.get(s, []))

    # Show what will be deleted
    console.print("[yellow]Will rollback stages:[/yellow]")
    for s in stages_to_delete:
        console.print(f"  • {s}")

    console.print("\n[yellow]Will delete directories:[/yellow]")
    for dir_name in dirs_to_delete:
        dir_path = project_dir / dir_name
        if dir_path.exists():
            file_count = sum(1 for _ in dir_path.rglob("*") if _.is_file())
            console.print(f"  • {dir_name}/ ({file_count} files)")

    if dry_run:
        console.print("\n[cyan]DRY RUN MODE - No files will be deleted[/cyan]")
        return

    # Confirm deletion
    if not force:
        console.print()
        if not click.confirm(
            "Are you sure you want to delete these files?", default=False
        ):
            print_info("Rollback cancelled")
            return

    # Delete directories
    console.print()
    for dir_name in dirs_to_delete:
        dir_path = project_dir / dir_name
        if dir_path.exists():
            try:
                shutil.rmtree(dir_path)
                console.print(f"[green]✓[/green] Deleted: {dir_name}/")
            except Exception as e:
                print_error(f"Failed to delete {dir_name}/: {e}")

    # Update state
    state = load_state(project_dir)
    stages_state: dict[str, Any] = state.get("stages", {})  # type: ignore
    for s in stages_to_delete:
        if s in stages_state:
            del stages_state[s]
    save_state(project_dir, state)

    console.print(f"\n[green]✓[/green] Rolled back to stage: [cyan]{stage}[/cyan]")
    print_info("Run 'bead workflow run --from-stage " + stage + "' to continue")


@workflow.command(name="list-templates")
def list_templates() -> None:
    """List available workflow templates.

    Shows all predefined workflow templates with descriptions and
    configuration requirements.

    Examples
    --------
    List templates:
        $ bead workflow list-templates
    """
    console.rule("[bold]Available Workflow Templates[/bold]")

    table = Table(title="Workflow Templates")
    table.add_column("Template ID", style="cyan")
    table.add_column("Name", style="bold")
    table.add_column("Description")

    for template_id, template_spec in WORKFLOW_TEMPLATES.items():
        table.add_row(
            template_id,
            str(template_spec["name"]),
            str(template_spec["description"]),
        )

    console.print(table)

    console.print("\n[bold]Usage:[/bold]")
    console.print("  bead workflow init <template-id>")
    console.print("\n[bold]Example:[/bold]")
    console.print("  bead workflow init acceptability-study")
