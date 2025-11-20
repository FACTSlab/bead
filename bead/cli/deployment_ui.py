"""UI customization commands for jsPsych deployment.

This module provides commands for customizing the appearance of jsPsych
experiments using Material Design themes and custom CSS.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import click
from rich.console import Console
from rich.panel import Panel

from bead.cli.utils import print_error, print_info, print_success
from bead.deployment.jspsych.ui.styles import MaterialDesignStylesheet

console = Console()


@click.group(name="ui")
def deployment_ui() -> None:
    r"""UI customization commands for jsPsych experiments.

    Commands for applying themes, generating CSS, and customizing the
    appearance of jsPsych experiments.

    \b
    Examples:
        $ bead deployment ui generate-css experiment/css/custom.css \\
            --theme dark --primary-color "#1976D2"
        $ bead deployment ui customize experiment/ \\
            --theme dark --primary-color "#1976D2"
    """


@click.command()
@click.argument("output_file", type=click.Path(path_type=Path))
@click.option(
    "--theme",
    type=click.Choice(["light", "dark", "auto"], case_sensitive=False),
    default="light",
    help="Color theme (light, dark, or auto for system preference)",
)
@click.option(
    "--primary-color",
    default="#6200EE",
    help="Primary color as hex code (default: Material Purple)",
)
@click.option(
    "--secondary-color",
    default="#03DAC6",
    help="Secondary color as hex code (default: Material Teal)",
)
@click.pass_context
def generate_css(
    ctx: click.Context,
    output_file: Path,
    theme: Literal["light", "dark", "auto"],
    primary_color: str,
    secondary_color: str,
) -> None:
    r"""Generate Material Design CSS for jsPsych experiment.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    output_file : Path
        Output path for generated CSS file.
    theme : Literal["light", "dark", "auto"]
        Color theme.
    primary_color : str
        Primary color as hex code.
    secondary_color : str
        Secondary color as hex code.

    Examples
    --------
    $ bead deployment ui generate-css experiment/css/custom.css

    $ bead deployment ui generate-css experiment/css/dark.css \\
        --theme dark --primary-color "#1976D2"

    $ bead deployment ui generate-css experiment/css/material.css \\
        --theme auto --primary-color "#6200EE" --secondary-color "#03DAC6"
    """
    try:
        print_info(f"Generating Material Design CSS (theme: {theme})")

        # Validate color codes
        if not _is_valid_hex_color(primary_color):
            print_error(f"Invalid primary color: {primary_color}")
            print_info("Color must be a valid hex code (e.g., #1976D2)")
            ctx.exit(1)

        if not _is_valid_hex_color(secondary_color):
            print_error(f"Invalid secondary color: {secondary_color}")
            print_info("Color must be a valid hex code (e.g., #03DAC6)")
            ctx.exit(1)

        # Create output directory if needed
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Generate CSS
        stylesheet = MaterialDesignStylesheet()
        css = stylesheet.generate_css(
            theme=theme,
            primary_color=primary_color,
            secondary_color=secondary_color,
        )

        # Write to file
        output_file.write_text(css, encoding="utf-8")

        print_success(f"Generated CSS: {output_file}")

        # Show preview
        preview_panel = Panel(
            f"[cyan]Theme:[/cyan] {theme}\n"
            f"[cyan]Primary:[/cyan] {primary_color}\n"
            f"[cyan]Secondary:[/cyan] {secondary_color}\n"
            f"[cyan]Lines:[/cyan] {len(css.splitlines())}",
            title="[bold]CSS Preview[/bold]",
            border_style="green",
        )
        console.print(preview_panel)

    except Exception as e:
        print_error(f"Failed to generate CSS: {e}")
        ctx.exit(1)


@click.command()
@click.argument("experiment_dir", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--theme",
    type=click.Choice(["light", "dark", "auto"], case_sensitive=False),
    default="light",
    help="Color theme (light, dark, or auto for system preference)",
)
@click.option(
    "--primary-color",
    default="#6200EE",
    help="Primary color as hex code (default: Material Purple)",
)
@click.option(
    "--secondary-color",
    default="#03DAC6",
    help="Secondary color as hex code (default: Material Teal)",
)
@click.option(
    "--css-file",
    type=click.Path(exists=True, path_type=Path),
    help="Path to custom CSS file to include (optional)",
)
@click.option(
    "--output-name",
    default="experiment.css",
    help="Output CSS filename (default: experiment.css)",
)
@click.pass_context
def customize(
    ctx: click.Context,
    experiment_dir: Path,
    theme: Literal["light", "dark", "auto"],
    primary_color: str,
    secondary_color: str,
    css_file: Path | None,
    output_name: str,
) -> None:
    r"""Apply UI customization to jsPsych experiment directory.

    Generates Material Design CSS and optionally merges with custom CSS.
    Writes the final CSS to the experiment's css/ directory.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    experiment_dir : Path
        Path to experiment directory.
    theme : Literal["light", "dark", "auto"]
        Color theme.
    primary_color : str
        Primary color as hex code.
    secondary_color : str
        Secondary color as hex code.
    css_file : Path | None
        Optional path to custom CSS file.
    output_name : str
        Output CSS filename.

    Examples
    --------
    $ bead deployment ui customize experiment/

    $ bead deployment ui customize experiment/ \\
        --theme dark --primary-color "#1976D2"

    $ bead deployment ui customize experiment/ \\
        --theme auto --css-file custom.css --output-name styles.css
    """
    try:
        print_info(f"Customizing UI for experiment: {experiment_dir}")

        # Validate color codes
        if not _is_valid_hex_color(primary_color):
            print_error(f"Invalid primary color: {primary_color}")
            print_info("Color must be a valid hex code (e.g., #1976D2)")
            ctx.exit(1)

        if not _is_valid_hex_color(secondary_color):
            print_error(f"Invalid secondary color: {secondary_color}")
            print_info("Color must be a valid hex code (e.g., #03DAC6)")
            ctx.exit(1)

        # Create css directory if needed
        css_dir = experiment_dir / "css"
        css_dir.mkdir(parents=True, exist_ok=True)

        # Generate Material Design CSS
        stylesheet = MaterialDesignStylesheet()
        material_css = stylesheet.generate_css(
            theme=theme,
            primary_color=primary_color,
            secondary_color=secondary_color,
        )

        # Merge with custom CSS if provided
        final_css = material_css
        if css_file:
            print_info(f"Merging with custom CSS: {css_file}")
            custom_css = css_file.read_text(encoding="utf-8")
            final_css = material_css + "\n\n/* Custom CSS */\n\n" + custom_css

        # Write to output file
        output_path = css_dir / output_name
        output_path.write_text(final_css, encoding="utf-8")

        print_success(f"CSS written to: {output_path}")

        # Show summary
        summary_panel = Panel(
            f"[cyan]Theme:[/cyan] {theme}\n"
            f"[cyan]Primary:[/cyan] {primary_color}\n"
            f"[cyan]Secondary:[/cyan] {secondary_color}\n"
            f"[cyan]Custom CSS:[/cyan] {'Yes' if css_file else 'No'}\n"
            f"[cyan]Output:[/cyan] {output_path.relative_to(experiment_dir)}\n"
            f"[cyan]Lines:[/cyan] {len(final_css.splitlines())}",
            title="[bold]UI Customization Summary[/bold]",
            border_style="green",
        )
        console.print(summary_panel)

        # Update index.html to reference the CSS file
        index_html = experiment_dir / "index.html"
        if index_html.exists():
            _update_index_html_css_reference(index_html, output_name)
            print_success("Updated index.html CSS reference")
        else:
            print_info("index.html not found - skipping CSS reference update")

    except Exception as e:
        print_error(f"Failed to customize UI: {e}")
        ctx.exit(1)


def _is_valid_hex_color(color: str) -> bool:
    """Validate hex color code.

    Parameters
    ----------
    color : str
        Color string to validate.

    Returns
    -------
    bool
        True if valid hex color, False otherwise.

    Examples
    --------
    >>> _is_valid_hex_color("#1976D2")
    True
    >>> _is_valid_hex_color("1976D2")
    False
    >>> _is_valid_hex_color("#GGG")
    False
    """
    if not color.startswith("#"):
        return False
    hex_part = color[1:]
    if len(hex_part) not in (3, 6):
        return False
    try:
        int(hex_part, 16)
        return True
    except ValueError:
        return False


def _update_index_html_css_reference(index_html: Path, css_filename: str) -> None:
    """Update index.html to reference the CSS file.

    Parameters
    ----------
    index_html : Path
        Path to index.html file.
    css_filename : str
        CSS filename to reference.
    """
    html_content = index_html.read_text(encoding="utf-8")

    # Check if CSS link already exists
    css_link = f'<link rel="stylesheet" href="css/{css_filename}">'
    if css_link in html_content:
        return  # Already has the correct reference

    # Find </head> tag and insert CSS link before it
    if "</head>" in html_content:
        html_content = html_content.replace(
            "</head>",
            f"  {css_link}\n</head>",
        )
        index_html.write_text(html_content, encoding="utf-8")


# Register commands
deployment_ui.add_command(generate_css)
deployment_ui.add_command(customize)
