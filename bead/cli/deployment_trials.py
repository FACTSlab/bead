"""Trial configuration commands for jsPsych deployment.

This module provides commands for configuring jsPsych trial parameters,
including timing, response collection, and trial-type-specific settings.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from bead.cli.utils import print_error, print_success

console = Console()


@click.group(name="trials")
def deployment_trials() -> None:
    r"""Trial configuration commands for jsPsych experiments.

    Commands for configuring trial parameters, including timing,
    response collection, and trial-type-specific settings.

    \b
    Examples:
        $ bead deployment trials configure-rating \\
            --min-value 1 --max-value 7 \\
            --min-label "Completely unnatural" \\
            --max-label "Completely natural" \\
            --output rating_config.json

        $ bead deployment trials configure-choice \\
            --button-html '<button class="choice-btn">%choice%</button>' \\
            --output choice_config.json
    """


@click.command()
@click.argument("output_file", type=click.Path(path_type=Path))
@click.option(
    "--min-value",
    type=int,
    default=1,
    help="Minimum value of the rating scale (default: 1)",
)
@click.option(
    "--max-value",
    type=int,
    default=7,
    help="Maximum value of the rating scale (default: 7)",
)
@click.option(
    "--step",
    type=int,
    default=1,
    help="Step size for the scale (default: 1)",
)
@click.option(
    "--min-label",
    default="Strongly Disagree",
    help="Label for minimum value",
)
@click.option(
    "--max-label",
    default="Strongly Agree",
    help="Label for maximum value",
)
@click.option(
    "--show-numeric-labels",
    is_flag=True,
    help="Show numeric labels on buttons (Likert only)",
)
@click.option(
    "--required",
    is_flag=True,
    default=True,
    help="Require response (slider only)",
)
@click.pass_context
def configure_rating(
    ctx: click.Context,
    output_file: Path,
    min_value: int,
    max_value: int,
    step: int,
    min_label: str,
    max_label: str,
    show_numeric_labels: bool,
    required: bool,
) -> None:
    r"""Configure rating scale trial parameters.

    Creates a configuration file for rating scale trials (Likert or slider).
    This configuration can be used when generating jsPsych experiments.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    output_file : Path
        Output path for configuration JSON file.
    min_value : int
        Minimum value of the rating scale.
    max_value : int
        Maximum value of the rating scale.
    step : int
        Step size for the scale.
    min_label : str
        Label for minimum value.
    max_label : str
        Label for maximum value.
    show_numeric_labels : bool
        Show numeric labels on buttons (Likert only).
    required : bool
        Require response (slider only).

    Examples
    --------
    $ bead deployment trials configure-rating rating_config.json

    $ bead deployment trials configure-rating rating_config.json \\
        --min-value 1 --max-value 5 --step 1 \\
        --min-label "Completely unnatural" \\
        --max-label "Completely natural"

    $ bead deployment trials configure-rating likert_config.json \\
        --min-value 1 --max-value 7 --show-numeric-labels
    """
    try:
        # Validate parameters
        if min_value >= max_value:
            print_error("min-value must be less than max-value")
            ctx.exit(1)

        if step <= 0:
            print_error("step must be positive")
            ctx.exit(1)

        if (max_value - min_value) % step != 0:
            print_error(
                f"Range ({max_value - min_value}) must be divisible by step ({step})"
            )
            ctx.exit(1)

        # Create configuration
        config: dict[str, Any] = {
            "type": "rating_scale",
            "min_value": min_value,
            "max_value": max_value,
            "step": step,
            "min_label": min_label,
            "max_label": max_label,
            "show_numeric_labels": show_numeric_labels,
            "required": required,
        }

        # Create output directory if needed
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Write configuration
        output_file.write_text(
            json.dumps(config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print_success(f"Rating configuration saved: {output_file}")

        # Show summary table
        table = Table(title="Rating Scale Configuration", show_header=False)
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Min Value", str(min_value))
        table.add_row("Max Value", str(max_value))
        table.add_row("Step", str(step))
        table.add_row("Min Label", min_label)
        table.add_row("Max Label", max_label)
        table.add_row("Show Numeric Labels", "Yes" if show_numeric_labels else "No")
        table.add_row("Required", "Yes" if required else "No")
        table.add_row("Scale Points", str((max_value - min_value) // step + 1))

        console.print(table)

    except Exception as e:
        print_error(f"Failed to create rating configuration: {e}")
        ctx.exit(1)


@click.command()
@click.argument("output_file", type=click.Path(path_type=Path))
@click.option(
    "--button-html",
    default='<button class="jspsych-btn">%choice%</button>',
    help="HTML template for choice buttons (%choice% is replaced with option text)",
)
@click.option(
    "--enable-keyboard",
    is_flag=True,
    default=True,
    help="Enable keyboard shortcuts for choices",
)
@click.option(
    "--randomize-position",
    is_flag=True,
    default=False,
    help="Randomize position of choices (for forced choice)",
)
@click.pass_context
def configure_choice(
    ctx: click.Context,
    output_file: Path,
    button_html: str,
    enable_keyboard: bool,
    randomize_position: bool,
) -> None:
    r"""Configure choice trial parameters.

    Creates a configuration file for choice trials (binary or forced choice).
    This configuration can be used when generating jsPsych experiments.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    output_file : Path
        Output path for configuration JSON file.
    button_html : str
        HTML template for choice buttons.
    enable_keyboard : bool
        Enable keyboard shortcuts for choices.
    randomize_position : bool
        Randomize position of choices.

    Examples
    --------
    $ bead deployment trials configure-choice choice_config.json

    $ bead deployment trials configure-choice choice_config.json \\
        --button-html '<button class="my-btn">%choice%</button>' \\
        --randomize-position

    $ bead deployment trials configure-choice binary_config.json \\
        --button-html '<button class="yes-no-btn">%choice%</button>' \\
        --enable-keyboard
    """
    try:
        # Validate button HTML
        if "%choice%" not in button_html:
            print_error("button-html must contain %choice% placeholder")
            ctx.exit(1)

        # Create configuration
        config: dict[str, Any] = {
            "type": "choice",
            "button_html": button_html,
            "enable_keyboard": enable_keyboard,
            "randomize_position": randomize_position,
        }

        # Create output directory if needed
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Write configuration
        output_file.write_text(
            json.dumps(config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print_success(f"Choice configuration saved: {output_file}")

        # Show summary
        kb_status = 'Enabled' if enable_keyboard else 'Disabled'
        rand_status = 'Yes' if randomize_position else 'No'
        summary_panel = Panel(
            f"[cyan]Button HTML:[/cyan]\n{button_html}\n\n"
            f"[cyan]Keyboard Shortcuts:[/cyan] {kb_status}\n"
            f"[cyan]Randomize Position:[/cyan] {rand_status}",
            title="[bold]Choice Configuration Summary[/bold]",
            border_style="green",
        )
        console.print(summary_panel)

    except Exception as e:
        print_error(f"Failed to create choice configuration: {e}")
        ctx.exit(1)


@click.command()
@click.argument("output_file", type=click.Path(path_type=Path))
@click.option(
    "--duration-ms",
    type=int,
    help="Duration each chunk is displayed (milliseconds)",
)
@click.option(
    "--isi-ms",
    type=int,
    default=0,
    help="Inter-stimulus interval between chunks (milliseconds, default: 0)",
)
@click.option(
    "--timeout-ms",
    type=int,
    help="Maximum time to wait for response (milliseconds, optional)",
)
@click.option(
    "--mask-char",
    default="",
    help="Character to use for masking hidden text (e.g., '#')",
)
@click.option(
    "--cumulative",
    is_flag=True,
    default=False,
    help="Display chunks cumulatively (each chunk remains visible)",
)
@click.pass_context
def configure_timing(
    ctx: click.Context,
    output_file: Path,
    duration_ms: int | None,
    isi_ms: int,
    timeout_ms: int | None,
    mask_char: str,
    cumulative: bool,
) -> None:
    r"""Configure timing parameters for trials.

    Creates a configuration file for presentation timing (e.g., self-paced
    reading, RSVP). This configuration can be used when generating jsPsych
    experiments with timed presentation.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    output_file : Path
        Output path for configuration JSON file.
    duration_ms : int | None
        Duration each chunk is displayed (milliseconds).
    isi_ms : int
        Inter-stimulus interval between chunks (milliseconds).
    timeout_ms : int | None
        Maximum time to wait for response (milliseconds).
    mask_char : str
        Character to use for masking hidden text.
    cumulative : bool
        Display chunks cumulatively.

    Examples
    --------
    $ bead deployment trials configure-timing timing_config.json \\
        --duration-ms 500 --isi-ms 100

    $ bead deployment trials configure-timing rsvp_config.json \\
        --duration-ms 300 --isi-ms 50 --mask-char "#"

    $ bead deployment trials configure-timing spr_config.json \\
        --isi-ms 0 --cumulative --mask-char "#"
    """
    try:
        # Validate parameters
        if isi_ms < 0:
            print_error("isi-ms must be non-negative")
            ctx.exit(1)

        if duration_ms is not None and duration_ms <= 0:
            print_error("duration-ms must be positive")
            ctx.exit(1)

        if timeout_ms is not None and timeout_ms <= 0:
            print_error("timeout-ms must be positive")
            ctx.exit(1)

        # Create configuration
        config: dict[str, Any] = {
            "type": "timing",
            "isi_ms": isi_ms,
            "cumulative": cumulative,
        }

        if duration_ms is not None:
            config["duration_ms"] = duration_ms

        if timeout_ms is not None:
            config["timeout_ms"] = timeout_ms

        if mask_char:
            config["mask_char"] = mask_char

        # Create output directory if needed
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Write configuration
        output_file.write_text(
            json.dumps(config, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        print_success(f"Timing configuration saved: {output_file}")

        # Show summary table
        table = Table(title="Timing Configuration", show_header=False)
        table.add_column("Parameter", style="cyan")
        table.add_column("Value", style="green")

        if duration_ms is not None:
            table.add_row("Duration", f"{duration_ms} ms")
        else:
            table.add_row("Duration", "Self-paced")

        table.add_row("ISI", f"{isi_ms} ms")

        if timeout_ms is not None:
            table.add_row("Timeout", f"{timeout_ms} ms")

        if mask_char:
            table.add_row("Mask Character", f"'{mask_char}'")

        table.add_row("Cumulative", "Yes" if cumulative else "No")

        console.print(table)

    except Exception as e:
        print_error(f"Failed to create timing configuration: {e}")
        ctx.exit(1)


@click.command()
@click.argument("config_files", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.pass_context
def show_config(ctx: click.Context, config_files: tuple[Path, ...]) -> None:
    """Display trial configuration files.

    Shows the contents of one or more trial configuration JSON files
    in a formatted table.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    config_files : tuple[Path, ...]
        Paths to configuration files to display.

    Examples
    --------
    $ bead deployment trials show-config rating_config.json

    $ bead deployment trials show-config rating_config.json choice_config.json
    """
    try:
        if not config_files:
            print_error("No configuration files specified")
            ctx.exit(1)

        for config_file in config_files:
            # Load configuration
            config = json.loads(config_file.read_text(encoding="utf-8"))

            # Create table
            table = Table(
                title=f"Configuration: {config_file.name}",
                show_header=False,
            )
            table.add_column("Parameter", style="cyan")
            table.add_column("Value", style="green")

            # Display all key-value pairs
            for key, value in sorted(config.items()):
                table.add_row(key, str(value))

            console.print(table)
            console.print()  # Blank line between tables

    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON in configuration file: {e}")
        ctx.exit(1)
    except Exception as e:
        print_error(f"Failed to show configuration: {e}")
        ctx.exit(1)


# Register commands
deployment_trials.add_command(configure_rating)
deployment_trials.add_command(configure_choice)
deployment_trials.add_command(configure_timing)
deployment_trials.add_command(show_config)
