"""Resource management commands for bead CLI.

This module provides commands for creating, listing, and validating
lexicons and templates (Stage 1 of the bead pipeline).
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

import click
from pydantic import ValidationError
from rich.console import Console
from rich.table import Table

from bead.cli.utils import print_error, print_info, print_success
from bead.resources.lexicon import LexicalItem, Lexicon
from bead.resources.structures import Slot, Template
from bead.resources.template_collection import TemplateCollection

console = Console()


@click.group()
def resources() -> None:
    r"""Resource management commands (Stage 1).

    Commands for creating, validating, and managing lexicons and templates.

    \b
    Examples:
        $ bead resources create-lexicon lexicon.jsonl --name verbs \\
            --from-csv verbs.csv
        $ bead resources create-template template.jsonl --name transitive \\
            --template-string "{subject} {verb} {object}"
        $ bead resources list-lexicons --directory lexicons/
        $ bead resources validate-lexicon lexicon.jsonl
    """


@resources.command()
@click.argument("output_file", type=click.Path(path_type=Path))
@click.option("--name", required=True, help="Lexicon name")
@click.option(
    "--from-csv",
    "csv_file",
    type=click.Path(exists=True, path_type=Path),
    help="Create from CSV file (requires 'lemma' column, optional 'pos', 'form', etc.)",
)
@click.option(
    "--from-json",
    "json_file",
    type=click.Path(exists=True, path_type=Path),
    help="Create from JSON file (array of lexical item objects)",
)
@click.option("--language-code", help="ISO 639 language code (e.g., 'eng', 'en')")
@click.option("--description", help="Description of the lexicon")
@click.pass_context
def create_lexicon(
    ctx: click.Context,
    output_file: Path,
    name: str,
    csv_file: Path | None,
    json_file: Path | None,
    language_code: str | None,
    description: str | None,
) -> None:
    r"""Create a lexicon from various sources.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    output_file : Path
        Path to output lexicon file.
    name : str
        Name for the lexicon.
    csv_file : Path | None
        Path to CSV source file.
    json_file : Path | None
        Path to JSON source file.
    language_code : str | None
        ISO 639 language code.
    description : str | None
        Description of the lexicon.

    Examples
    --------
    # Create from CSV file
    $ bead resources create-lexicon lexicon.jsonl --name verbs --from-csv verbs.csv

    # Create from JSON file
    $ bead resources create-lexicon lexicon.jsonl --name verbs --from-json verbs.json

    # With language code
    $ bead resources create-lexicon lexicon.jsonl --name verbs \\
        --from-csv verbs.csv --language-code eng
    """
    try:
        # Validate that exactly one source is provided
        sources = [csv_file, json_file]
        provided_sources = [s for s in sources if s is not None]

        if len(provided_sources) == 0:
            print_error("Must provide one source: --from-csv or --from-json")
            ctx.exit(1)
        elif len(provided_sources) > 1:
            print_error("Only one source allowed: --from-csv or --from-json")
            ctx.exit(1)

        # Create lexicon
        lexicon = Lexicon(
            name=name,
            language_code=language_code,
            description=description,
        )

        # Load from source
        if csv_file:
            print_info(f"Loading lexical items from CSV: {csv_file}")
            with open(csv_file, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "lemma" not in row:
                        print_error("CSV must have 'lemma' column")
                        ctx.exit(1)

                    item_data: dict[str, Any] = {"lemma": row["lemma"]}
                    if "pos" in row and row["pos"]:
                        item_data["pos"] = row["pos"]
                    if "form" in row and row["form"]:
                        item_data["form"] = row["form"]
                    if "source" in row and row["source"]:
                        item_data["source"] = row["source"]

                    # Extract features (columns with feature_ prefix)
                    features: dict[str, Any] = {}
                    for key, value in row.items():
                        if key.startswith("feature_") and value:
                            features[key[8:]] = value
                    if features:
                        item_data["features"] = features

                    # Extract attributes (columns with attr_ prefix)
                    attributes: dict[str, Any] = {}
                    for key, value in row.items():
                        if key.startswith("attr_") and value:
                            attributes[key[5:]] = value
                    if attributes:
                        item_data["attributes"] = attributes

                    item = LexicalItem(**item_data)
                    lexicon.add(item)

        elif json_file:
            print_info(f"Loading lexical items from JSON: {json_file}")
            with open(json_file, encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                print_error("JSON file must contain an array of lexical items")
                ctx.exit(1)

            for item_data in data:  # type: ignore[var-annotated]
                if not isinstance(item_data, dict):
                    continue
                item = LexicalItem(**item_data)  # type: ignore[arg-type]
                lexicon.add(item)

        # Save lexicon
        output_file.parent.mkdir(parents=True, exist_ok=True)
        lexicon.to_jsonl(str(output_file))

        print_success(
            f"Created lexicon '{name}' with {len(lexicon)} items: {output_file}"
        )

    except ValidationError as e:
        print_error(f"Validation error: {e}")
        ctx.exit(1)
    except Exception as e:
        print_error(f"Failed to create lexicon: {e}")
        ctx.exit(1)


@resources.command()
@click.argument("output_file", type=click.Path(path_type=Path))
@click.option("--name", required=True, help="Template name")
@click.option(
    "--template-string",
    required=True,
    help="Template string with {slot_name} placeholders",
)
@click.option("--language-code", help="ISO 639 language code")
@click.option("--description", help="Template description")
@click.option(
    "--slot",
    "slots",
    multiple=True,
    help=(
        "Slot definition in format: name:required "
        "(e.g., 'subject:true', 'object:false')"
    ),
)
@click.pass_context
def create_template(
    ctx: click.Context,
    output_file: Path,
    name: str,
    template_string: str,
    language_code: str | None,
    description: str | None,
    slots: tuple[str, ...],
) -> None:
    r"""Create a template with slots and constraints.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    output_file : Path
        Path to output template file.
    name : str
        Name for the template.
    template_string : str
        Template string with {slot_name} placeholders.
    language_code : str | None
        ISO 639 language code.
    description : str | None
        Description of the template.
    slots : tuple[str, ...]
        Slot definitions in format "name:required".

    Examples
    --------
    # Create simple template
    $ bead resources create-template template.jsonl \\
        --name transitive \\
        --template-string "{subject} {verb} {object}"

    # With slot specifications
    $ bead resources create-template template.jsonl \\
        --name transitive \\
        --template-string "{subject} {verb} {object}" \\
        --slot subject:true \\
        --slot verb:true \\
        --slot object:false
    """
    try:
        # Parse slot definitions
        slot_dict: dict[str, Slot] = {}

        # Extract slot names from template string
        slot_names = re.findall(r"\{(\w+)\}", template_string)

        if not slot_names:
            print_error(
                "Template string must contain at least one {slot_name} placeholder"
            )
            ctx.exit(1)

        # Parse explicit slot definitions
        explicit_slots: dict[str, bool] = {}
        for slot_def in slots:
            if ":" not in slot_def:
                print_error(
                    f"Invalid slot definition: {slot_def}. Use format 'name:required'"
                )
                ctx.exit(1)

            slot_name, required_str = slot_def.split(":", 1)
            required = required_str.lower() in ("true", "yes", "1")
            explicit_slots[slot_name] = required

        # Create slot objects for all slot names in template
        for slot_name in slot_names:
            required = explicit_slots.get(slot_name, True)
            slot_dict[slot_name] = Slot(name=slot_name, required=required)

        # Create template
        template = Template(
            name=name,
            template_string=template_string,
            slots=slot_dict,
            language_code=language_code,
            description=description,
        )

        # Create collection and add template
        collection = TemplateCollection(
            name=f"{name}_collection",
            language_code=language_code,
        )
        collection.add(template)

        # Save collection
        output_file.parent.mkdir(parents=True, exist_ok=True)
        collection.to_jsonl(str(output_file))

        print_success(
            f"Created template '{name}' with {len(slot_dict)} slots: {output_file}"
        )

    except ValidationError as e:
        print_error(f"Validation error: {e}")
        ctx.exit(1)
    except Exception as e:
        print_error(f"Failed to create template: {e}")
        ctx.exit(1)


@resources.command()
@click.option(
    "--directory",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path.cwd(),
    help="Directory to search for lexicon files",
)
@click.option(
    "--pattern",
    default="*.jsonl",
    help="File pattern to match (default: *.jsonl)",
)
@click.pass_context
def list_lexicons(
    ctx: click.Context,
    directory: Path,
    pattern: str,
) -> None:
    """List available lexicons in a directory.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    directory : Path
        Directory to search for lexicon files.
    pattern : str
        File pattern to match.

    Examples
    --------
    $ bead resources list-lexicons
    $ bead resources list-lexicons --directory lexicons/
    $ bead resources list-lexicons --pattern "verb*.jsonl"
    """
    try:
        lexicon_files = list(directory.glob(pattern))

        if not lexicon_files:
            print_info(f"No lexicon files found in {directory} matching {pattern}")
            return

        table = Table(title=f"Lexicons in {directory}")
        table.add_column("File", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Items", justify="right", style="yellow")
        table.add_column("Language", style="magenta")

        for file_path in sorted(lexicon_files):
            try:
                # Try to load first item to get lexicon metadata
                with open(file_path, encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    if not first_line:
                        continue

                # Count total lines
                with open(file_path, encoding="utf-8") as f:
                    item_count = sum(1 for line in f if line.strip())

                # Parse first item to get metadata
                item_data = json.loads(first_line)
                lexicon_name = file_path.stem
                language = item_data.get("language_code", "N/A")

                table.add_row(
                    str(file_path.name),
                    lexicon_name,
                    str(item_count),
                    language,
                )
            except Exception:
                # Skip files that can't be parsed
                continue

        console.print(table)

    except Exception as e:
        print_error(f"Failed to list lexicons: {e}")
        ctx.exit(1)


@resources.command()
@click.option(
    "--directory",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path.cwd(),
    help="Directory to search for template files",
)
@click.option(
    "--pattern",
    default="*.jsonl",
    help="File pattern to match (default: *.jsonl)",
)
@click.pass_context
def list_templates(
    ctx: click.Context,
    directory: Path,
    pattern: str,
) -> None:
    """List available templates in a directory.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    directory : Path
        Directory to search for template files.
    pattern : str
        File pattern to match.

    Examples
    --------
    $ bead resources list-templates
    $ bead resources list-templates --directory templates/
    $ bead resources list-templates --pattern "trans*.jsonl"
    """
    try:
        template_files = list(directory.glob(pattern))

        if not template_files:
            print_info(f"No template files found in {directory} matching {pattern}")
            return

        table = Table(title=f"Templates in {directory}")
        table.add_column("File", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Slots", justify="right", style="yellow")
        table.add_column("Template String", style="white")

        for file_path in sorted(template_files):
            try:
                # Load first template
                with open(file_path, encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    if not first_line:
                        continue

                # Parse template
                template_data = json.loads(first_line)
                template_name = template_data.get("name", file_path.stem)
                slot_count = len(template_data.get("slots", {}))
                template_str = template_data.get("template_string", "N/A")

                # Truncate long template strings
                if len(template_str) > 50:
                    template_str = template_str[:47] + "..."

                table.add_row(
                    str(file_path.name),
                    template_name,
                    str(slot_count),
                    template_str,
                )
            except Exception:
                # Skip files that can't be parsed
                continue

        console.print(table)

    except Exception as e:
        print_error(f"Failed to list templates: {e}")
        ctx.exit(1)


@resources.command()
@click.argument("lexicon_file", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def validate_lexicon(ctx: click.Context, lexicon_file: Path) -> None:
    """Validate a lexicon file.

    Checks that the lexicon file is properly formatted and all items are valid.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    lexicon_file : Path
        Path to lexicon file to validate.

    Examples
    --------
    $ bead resources validate-lexicon lexicon.jsonl
    """
    try:
        print_info(f"Validating lexicon: {lexicon_file}")

        item_count = 0
        errors: list[str] = []

        with open(lexicon_file, encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    item_data = json.loads(line)
                    LexicalItem(**item_data)
                    item_count += 1
                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_num}: Invalid JSON - {e}")
                except ValidationError as e:
                    errors.append(f"Line {line_num}: Validation error - {e}")

        if errors:
            print_error(f"Validation failed with {len(errors)} errors:")
            for error in errors[:10]:  # Show first 10 errors
                console.print(f"  [red]✗[/red] {error}")
            if len(errors) > 10:
                console.print(f"  ... and {len(errors) - 10} more errors")
            ctx.exit(1)
        else:
            print_success(f"Lexicon is valid: {item_count} items")

    except Exception as e:
        print_error(f"Failed to validate lexicon: {e}")
        ctx.exit(1)


@resources.command()
@click.argument("template_file", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def validate_template(ctx: click.Context, template_file: Path) -> None:
    """Validate a template file.

    Checks that the template file is properly formatted and all templates are valid.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    template_file : Path
        Path to template file to validate.

    Examples
    --------
    $ bead resources validate-template templates.jsonl
    """
    try:
        print_info(f"Validating template: {template_file}")

        template_count = 0
        errors: list[str] = []

        with open(template_file, encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    template_data = json.loads(line)
                    Template(**template_data)
                    template_count += 1
                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_num}: Invalid JSON - {e}")
                except ValidationError as e:
                    errors.append(f"Line {line_num}: Validation error - {e}")

        if errors:
            print_error(f"Validation failed with {len(errors)} errors:")
            for error in errors[:10]:  # Show first 10 errors
                console.print(f"  [red]✗[/red] {error}")
            if len(errors) > 10:
                console.print(f"  ... and {len(errors) - 10} more errors")
            ctx.exit(1)
        else:
            print_success(f"Template file is valid: {template_count} templates")

    except Exception as e:
        print_error(f"Failed to validate template: {e}")
        ctx.exit(1)
