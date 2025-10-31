"""Template filling commands for bead CLI.

This module provides commands for filling templates with lexical items
(Stage 2 of the bead pipeline).
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
from bead.resources.lexicon import Lexicon
from bead.resources.template_collection import TemplateCollection
from bead.templates.filler import FilledTemplate
from bead.templates.strategies import (
    ExhaustiveStrategy,
    RandomStrategy,
    StrategyFiller,
    StratifiedStrategy,
)

console = Console()


@click.group()
def templates() -> None:
    r"""Template filling commands (Stage 2).

    Commands for filling templates with lexical items using various strategies.

    \b
    Examples:
        $ bead templates fill template.jsonl lexicon.jsonl filled.jsonl \\
            --strategy exhaustive
        $ bead templates fill template.jsonl lexicon.jsonl filled.jsonl \\
            --strategy random --max-combinations 100
        $ bead templates list-filled filled.jsonl
        $ bead templates validate-filled filled.jsonl
        $ bead templates show-stats filled.jsonl
    """


@click.command()
@click.argument("template_file", type=click.Path(exists=True, path_type=Path))
@click.argument("lexicon_file", type=click.Path(exists=True, path_type=Path))
@click.argument("output_file", type=click.Path(path_type=Path))
@click.option(
    "--strategy",
    type=click.Choice(["exhaustive", "random", "stratified"]),
    default="exhaustive",
    help="Filling strategy to use",
)
@click.option(
    "--max-combinations",
    type=int,
    help="Maximum combinations for random/stratified strategies",
)
@click.option(
    "--random-seed",
    type=int,
    help="Random seed for reproducibility",
)
@click.option(
    "--grouping-property",
    help="Property for stratified strategy (e.g., 'pos', 'features.tense')",
)
@click.option(
    "--language-code",
    help="ISO 639 language code to filter items",
)
@click.pass_context
def fill(
    ctx: click.Context,
    template_file: Path,
    lexicon_file: Path,
    output_file: Path,
    strategy: str,
    max_combinations: int | None,
    random_seed: int | None,
    grouping_property: str | None,
    language_code: str | None,
) -> None:
    r"""Fill templates with lexical items.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    template_file : Path
        Path to template file.
    lexicon_file : Path
        Path to lexicon file.
    output_file : Path
        Path to output filled templates file.
    strategy : str
        Filling strategy name.
    max_combinations : int | None
        Maximum number of combinations.
    random_seed : int | None
        Random seed for reproducibility.
    grouping_property : str | None
        Property for stratified sampling.
    language_code : str | None
        ISO 639 language code filter.

    Examples
    --------
    # Exhaustive filling
    $ bead templates fill template.jsonl lexicon.jsonl filled.jsonl \\
        --strategy exhaustive

    # Random sampling
    $ bead templates fill template.jsonl lexicon.jsonl filled.jsonl \\
        --strategy random --max-combinations 100 --random-seed 42

    # Stratified sampling
    $ bead templates fill template.jsonl lexicon.jsonl filled.jsonl \\
        --strategy stratified --max-combinations 100 --grouping-property pos
    """
    try:
        # Validate strategy-specific options
        if strategy in ("random", "stratified") and max_combinations is None:
            print_error(f"--max-combinations required for {strategy} strategy")
            ctx.exit(1)

        if strategy == "stratified" and grouping_property is None:
            print_error("--grouping-property required for stratified strategy")
            ctx.exit(1)

        # Load lexicon
        print_info(f"Loading lexicon from {lexicon_file}")
        lexicon = Lexicon.from_jsonl(str(lexicon_file), "lexicon")
        print_info(f"Loaded {len(lexicon)} lexical items")

        # Load templates
        print_info(f"Loading templates from {template_file}")
        template_collection = TemplateCollection.from_jsonl(
            str(template_file), "templates"
        )
        print_info(f"Loaded {len(template_collection)} templates")

        # Create strategy
        filling_strategy: ExhaustiveStrategy | RandomStrategy | StratifiedStrategy
        if strategy == "exhaustive":
            filling_strategy = ExhaustiveStrategy()
        elif strategy == "random":
            assert max_combinations is not None
            filling_strategy = RandomStrategy(
                n_samples=max_combinations,
                seed=random_seed,
            )
        elif strategy == "stratified":
            assert max_combinations is not None
            assert grouping_property is not None
            filling_strategy = StratifiedStrategy(
                n_samples=max_combinations,
                grouping_property=grouping_property,
                seed=random_seed,
            )
        else:
            print_error(f"Unknown strategy: {strategy}")
            ctx.exit(1)

        # Create filler
        filler = StrategyFiller(lexicon=lexicon, strategy=filling_strategy)

        # Fill templates with progress
        all_filled: list[FilledTemplate] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Filling {len(template_collection)} templates...",
                total=len(template_collection),
            )

            for template in template_collection:
                try:
                    filled_templates = filler.fill(template, language_code)
                    all_filled.extend(filled_templates)
                    progress.advance(task)
                except ValueError as e:
                    print_error(f"Failed to fill template '{template.name}': {e}")
                    continue

        # Save filled templates
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for filled in all_filled:
                f.write(filled.model_dump_json() + "\n")

        print_success(
            f"Created {len(all_filled)} filled templates from "
            f"{len(template_collection)} templates: {output_file}"
        )

    except ValidationError as e:
        print_error(f"Validation error: {e}")
        ctx.exit(1)
    except Exception as e:
        print_error(f"Failed to fill templates: {e}")
        ctx.exit(1)


@click.command()
@click.option(
    "--directory",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path.cwd(),
    help="Directory to search for filled template files",
)
@click.option(
    "--pattern",
    default="*.jsonl",
    help="File pattern to match (default: *.jsonl)",
)
@click.pass_context
def list_filled(
    ctx: click.Context,
    directory: Path,
    pattern: str,
) -> None:
    """List filled template files in a directory.

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
    $ bead templates list-filled
    $ bead templates list-filled --directory filled_templates/
    $ bead templates list-filled --pattern "filled_*.jsonl"
    """
    try:
        files = list(directory.glob(pattern))

        if not files:
            print_info(f"No files found in {directory} matching {pattern}")
            return

        table = Table(title=f"Filled Templates in {directory}")
        table.add_column("File", style="cyan")
        table.add_column("Count", justify="right", style="yellow")
        table.add_column("Strategy", style="green")
        table.add_column("Sample", style="white")

        for file_path in sorted(files):
            try:
                # Count filled templates and get metadata
                with open(file_path, encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]

                if not lines:
                    continue

                count = len(lines)

                # Parse first filled template for metadata
                first_data = json.loads(lines[0])
                strategy_name = first_data.get("strategy_name", "N/A")
                rendered = first_data.get("rendered_text", "N/A")

                # Truncate long rendered text
                if len(rendered) > 40:
                    rendered = rendered[:37] + "..."

                table.add_row(
                    str(file_path.name),
                    str(count),
                    strategy_name,
                    rendered,
                )
            except Exception:
                # Skip files that can't be parsed
                continue

        console.print(table)

    except Exception as e:
        print_error(f"Failed to list filled templates: {e}")
        ctx.exit(1)


@click.command()
@click.argument("filled_file", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def validate_filled(ctx: click.Context, filled_file: Path) -> None:
    """Validate a filled templates file.

    Checks that all filled templates are properly formatted.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    filled_file : Path
        Path to filled templates file.

    Examples
    --------
    $ bead templates validate-filled filled.jsonl
    """
    try:
        print_info(f"Validating filled templates: {filled_file}")

        count = 0
        errors: list[str] = []

        with open(filled_file, encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    filled_data = json.loads(line)
                    FilledTemplate(**filled_data)
                    count += 1
                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_num}: Invalid JSON - {e}")
                except ValidationError as e:
                    errors.append(f"Line {line_num}: Validation error - {e}")

        if errors:
            print_error(f"Validation failed with {len(errors)} errors:")
            for error in errors[:10]:
                console.print(f"  [red]✗[/red] {error}")
            if len(errors) > 10:
                console.print(f"  ... and {len(errors) - 10} more errors")
            ctx.exit(1)
        else:
            print_success(f"Filled templates file is valid: {count} filled templates")

    except Exception as e:
        print_error(f"Failed to validate filled templates: {e}")
        ctx.exit(1)


@click.command()
@click.argument("filled_file", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def show_stats(ctx: click.Context, filled_file: Path) -> None:
    """Show statistics about filled templates.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    filled_file : Path
        Path to filled templates file.

    Examples
    --------
    $ bead templates show-stats filled.jsonl
    """
    try:
        print_info(f"Analyzing filled templates: {filled_file}")

        # Collect statistics
        total_count = 0
        templates_seen: set[str] = set()
        strategies_used: dict[str, int] = {}
        text_lengths: list[int] = []

        with open(filled_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    filled_data = json.loads(line)
                    filled = FilledTemplate(**filled_data)

                    total_count += 1
                    templates_seen.add(filled.template_name)
                    strategies_used[filled.strategy_name] = (
                        strategies_used.get(filled.strategy_name, 0) + 1
                    )
                    text_lengths.append(len(filled.rendered_text))

                except Exception:
                    continue

        if total_count == 0:
            print_error("No valid filled templates found")
            ctx.exit(1)

        # Calculate statistics
        avg_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0
        min_length = min(text_lengths) if text_lengths else 0
        max_length = max(text_lengths) if text_lengths else 0

        # Display statistics table
        table = Table(title="Filled Template Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        table.add_row("Total Filled Templates", str(total_count))
        table.add_row("Unique Template Names", str(len(templates_seen)))
        table.add_row("", "")  # Separator

        for strategy, count in sorted(strategies_used.items()):
            table.add_row(f"Strategy: {strategy}", str(count))

        table.add_row("", "")  # Separator
        table.add_row("Avg Text Length", f"{avg_length:.1f}")
        table.add_row("Min Text Length", str(min_length))
        table.add_row("Max Text Length", str(max_length))

        console.print(table)

        # Show sample templates
        if templates_seen:
            console.print("\n[cyan]Sample Template Names:[/cyan]")
            for name in sorted(templates_seen)[:5]:
                console.print(f"  • {name}")
            if len(templates_seen) > 5:
                console.print(f"  ... and {len(templates_seen) - 5} more")

    except Exception as e:
        print_error(f"Failed to show statistics: {e}")
        ctx.exit(1)


# Register commands
templates.add_command(fill)
templates.add_command(list_filled)
templates.add_command(validate_filled)
templates.add_command(show_stats)
