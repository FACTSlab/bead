#!/usr/bin/env python3
"""Generate jsPsych/JATOS deployment from experiment lists.

This script loads experiment lists and 2AFC pairs, randomly selects a subset of lists,
and generates a jsPsych experiment that can be exported to JATOS.
"""

from __future__ import annotations

import argparse
import random
import sys
import traceback
from pathlib import Path
from uuid import UUID

import yaml
from protocol import acceptability_family, build_protocol
from rich.console import Console
from rich.progress import track
from rich.table import Table

from bead.deployment.distribution import (
    DistributionStrategyType,
    ListDistributionStrategy,
)
from bead.deployment.jatos.exporter import JATOSExporter
from bead.deployment.jspsych.config import (
    ChoiceConfig,
    ExperimentConfig,
    InstructionPage,
    InstructionsConfig,
)
from bead.deployment.jspsych.generator import JsPsychExperimentGenerator
from bead.items.item import Item
from bead.items.item_template import ItemTemplate
from bead.lists import ExperimentList
from bead.protocol.items import family_to_item_template

console = Console()

_INSTRUCTIONS_PLUGIN_TAG = (
    '<script src="https://unpkg.com/@jspsych/plugin-instructions@2.0.0"></script>'
)


def _inject_instructions_plugin(html_path: Path) -> None:
    """Insert the jsPsych instructions plugin script tag if missing."""
    html = html_path.read_text(encoding="utf-8")
    if "plugin-instructions" not in html:
        html = html.replace(
            '<script src="https://unpkg.com/@jspsych/plugin-preload',
            f'{_INSTRUCTIONS_PLUGIN_TAG}\n    <script src="https://unpkg.com/@jspsych/plugin-preload',
        )
        html_path.write_text(html, encoding="utf-8")


# (file, English text bead hard-codes, ui_text key or "{lang}", required).
# Optional entries only exist in some builds (the redirect line needs an
# on_finish_url).
_LOCALIZE: tuple[tuple[str, str, str, bool], ...] = (
    ("index.html", '<html lang="en">', "{lang}", True),
    # Progress-bar label: bead never sets message_progress_bar, so jsPsych
    # shows its default "Completion Progress"; insert it next to the bar flag.
    ("js/experiment.js", "show_progress_bar: true,", "progress_bar", False),
    ("js/experiment.js", 'button_label_previous: "Previous"', "previous", True),
    ("js/experiment.js", "<h2>✓ Complete</h2>", "complete_heading", True),
    ("js/experiment.js", "<p>Thank you for participating!</p>", "thanks", True),
    ("js/experiment.js", "<p>Your responses have been recorded.</p>", "recorded", True),
    ("js/experiment.js", "<p><em>You may close this window.</em></p>", "close_window",
     False),
    ("js/experiment.js", "<p><em>You will be redirected shortly...</em></p>",
     "redirecting", False),
    ("js/experiment.js", "<h2>Experiment Error</h2>", "error_heading", True),
    ("js/experiment.js", "<p><strong>Error:</strong>", "error_label", True),
    ("js/experiment.js", "<p>Please contact the experiment administrator.</p>",
     "contact_admin", True),
    ("js/gallery-bundle.js", 'bead-alternative-label">Option 1</div>', "option_1",
     True),
    ("js/gallery-bundle.js", 'bead-alternative-label">Option 2</div>', "option_2",
     True),
    ("js/gallery-bundle.js", "\n          Select\n        </button>", "select", True),
)


def _localized(english: str, key: str, ui_text: dict[str, str], lang: str) -> str:
    """Swap the English text inside ``english`` for its ui_text translation."""
    if key == "{lang}":
        return f'<html lang="{lang}">'
    if key == "progress_bar":
        return f'{english}\n            message_progress_bar: "{ui_text[key]}",'
    inner = {
        "previous": "Previous", "complete_heading": "✓ Complete",
        "thanks": "Thank you for participating!",
        "recorded": "Your responses have been recorded.",
        "close_window": "You may close this window.",
        "redirecting": "You will be redirected shortly...",
        "error_heading": "Experiment Error", "error_label": "Error:",
        "contact_admin": "Please contact the experiment administrator.",
        "option_1": "Option 1", "option_2": "Option 2", "select": "Select",
    }[key]
    return english.replace(inner, ui_text[key])


def localize_experiment(
    list_dir: Path, ui_text: dict[str, str], lang: str
) -> list[str]:
    """Replace the English UI text bead hard-codes in a generated experiment.

    bead's jsPsych templates and forced-choice plugin fix these strings (the
    trial ``button_label`` is ignored by the plugin), and bead is read-only
    here, so they are rewritten in the generated files -- same approach as
    _inject_instructions_plugin. Returns the required strings that were not
    found, so a bead template change surfaces as a warning, not silent
    English.
    """
    missing: list[str] = []
    contents: dict[str, str] = {}
    for rel, english, key, required in _LOCALIZE:
        path = list_dir / rel
        text = contents.setdefault(rel, path.read_text(encoding="utf-8"))
        if english not in text:
            if required:
                missing.append(f"{rel}: {english.strip()!r}")
            continue
        contents[rel] = text.replace(english, _localized(english, key, ui_text, lang))
    for rel, text in contents.items():
        (list_dir / rel).write_text(text, encoding="utf-8")
    return missing


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_experiment_lists(lists_path: Path) -> list[ExperimentList]:
    """Load experiment lists from JSONL file."""
    with console.status(f"[bold]Loading experiment lists from {lists_path}...[/bold]"):
        lists = []
        with open(lists_path) as f:
            for line in f:
                lists.append(ExperimentList.model_validate_json(line))

    console.print(f"[green]✓[/green] Loaded {len(lists)} experiment lists")
    return lists


def load_items_by_uuid(pairs_path: Path) -> dict[UUID, Item]:
    """Load all 2AFC pairs and index by UUID."""
    with console.status(f"[bold]Loading 2AFC pairs from {pairs_path}...[/bold]"):
        items_dict = {}
        with open(pairs_path) as f:
            for line in f:
                item = Item.model_validate_json(line)
                items_dict[item.id] = item

    console.print(f"[green]✓[/green] Loaded {len(items_dict)} 2AFC pairs")
    return items_dict


def build_instructions(experiment: dict) -> InstructionsConfig:
    """Instruction pages and button labels from config (Korean by default)."""
    pages = experiment.get("instructions") or [
        "You will see pairs of sentences. Please select the sentence that "
        "sounds more natural to you."
    ]
    if isinstance(pages, str):
        pages = [pages]
    return InstructionsConfig(
        pages=tuple(InstructionPage(content=page) for page in pages),
        button_label_next=experiment.get("button_label_next", "Next"),
        button_label_finish=experiment.get("button_label_finish", "Begin Experiment"),
    )


def build_item_template(config_path: Path) -> ItemTemplate:
    """Build the 2AFC ItemTemplate from the protocol declared in config.yaml.

    The prompt shown above each pair (어느 문장이 더 자연스러운가요?) comes
    from ``protocol.families[].anchor.canonical_prompt`` via bead's
    :func:`family_to_item_template` bridge, as in the English gallery, so it
    has one source of truth shared with the active-learning scripts.
    """
    family = acceptability_family(build_protocol(config_path))
    return family_to_item_template(family, judgment_type="acceptability")


def main() -> None:
    """Generate jsPsych/JATOS deployment."""
    parser = argparse.ArgumentParser(
        description="Generate jsPsych/JATOS deployment from experiment lists"
    )
    parser.add_argument(
        "--n-lists",
        type=int,
        default=20,
        help="Number of lists to randomly select (default: 20)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="deployment",
        help="Output directory for deployment files (default: deployment)",
    )
    parser.add_argument(
        "--no-jatos",
        action="store_true",
        help="Skip JATOS .jzip export",
    )
    args = parser.parse_args()

    # Determine base directory
    base_dir = Path(__file__).parent
    config_path = base_dir / "config.yaml"

    console.rule("[bold]jsPsych/JATOS Deployment Generation[/bold]")
    console.print(f"Base directory: [cyan]{base_dir}[/cyan]")
    console.print(f"Configuration: [cyan]{config_path}[/cyan]")
    console.print(f"Lists to select: [cyan]{args.n_lists}[/cyan]")
    console.print(f"Output directory: [cyan]{args.output_dir}[/cyan]\n")

    # Load configuration
    console.rule("[1/6] Loading Configuration")
    config = load_config(config_path)
    deployment_config = config.get("deployment", {})
    jspsych_config = deployment_config.get("jspsych", {})
    experiment_config_dict = deployment_config.get("experiment", {})

    console.print("[green]✓[/green] Configuration loaded")
    console.print(
        f"  • Platform: [cyan]{deployment_config.get('platform', 'jatos')}[/cyan]"
    )
    console.print("  • Experiment type: [cyan]forced_choice (2AFC)[/cyan]\n")

    # Load experiment lists
    console.rule("[2/6] Loading Experiment Lists")
    lists_path = base_dir / config["paths"]["experiment_lists"]
    all_lists = load_experiment_lists(lists_path)

    # Randomly select subset of lists
    random.seed(config.get("deployment", {}).get("random_seed", 42))
    n_lists = min(args.n_lists, len(all_lists))
    selected_lists = random.sample(all_lists, n_lists)
    console.print(
        f"[green]✓[/green] Randomly selected {n_lists} lists from "
        f"{len(all_lists)} available\n"
    )

    # Load all 2AFC pairs
    console.rule("[3/6] Loading 2AFC Pairs")
    pairs_path = base_dir / config["paths"]["2afc_pairs"]
    items_dict = load_items_by_uuid(pairs_path)

    # Item template (and prompt) from the protocol
    console.print()
    console.rule("[4/6] Creating Item Template")
    template = build_item_template(config_path)
    templates_dict = {template.id: template}
    console.print(
        "[green]✓[/green] Built 2AFC ItemTemplate from the protocol "
        f"(prompt: {template.task_spec.prompt})\n"
    )

    # Update all items to reference this template (immutable update)
    items_dict = {
        uid: item.with_(item_template_id=template.id)
        for uid, item in items_dict.items()
    }

    # Create ExperimentConfig for jsPsych (base configuration)
    console.rule("[5/6] Generating jsPsych Experiments")

    # Extract distribution strategy from config
    dist_config_dict = deployment_config.get("distribution_strategy", {})
    strategy_type = dist_config_dict.get("strategy_type", "balanced")
    distribution_strategy = ListDistributionStrategy(
        strategy_type=DistributionStrategyType(strategy_type),
        max_participants=dist_config_dict.get("max_participants"),
        error_on_exhaustion=dist_config_dict.get("error_on_exhaustion", True),
        debug_mode=dist_config_dict.get("debug_mode", False),
        debug_list_index=dist_config_dict.get("debug_list_index", 0),
    )

    base_config_dict = {
        "experiment_type": "forced_choice",
        "title": experiment_config_dict.get(
            "title", "Sentence Acceptability Judgments"
        ),
        "description": experiment_config_dict.get(
            "description", "Rate which sentence sounds more natural"
        ),
        "instructions": build_instructions(experiment_config_dict),
        "randomize_trial_order": jspsych_config.get("randomize_order", True),
        "show_progress_bar": True,
        # Prolific redirect on completion (bead builds the completion URL);
        # null in config = no redirect.
        "prolific_completion_code": jspsych_config.get("prolific_completion_code"),
        "distribution_strategy": distribution_strategy,
    }

    choice_config = ChoiceConfig(
        randomize_choice_order=jspsych_config.get("randomize_choices", True),
        required=True,
    )

    # Generate TWO versions: local (standalone) and jatos (deployment)
    versions = [
        ("local", False),   # Standalone version for local testing
        ("jatos", True),    # JATOS version for deployment
    ]

    for version_name, use_jatos in versions:
        console.print(
            f"\n[bold]Generating {version_name} version "
            f"(use_jatos={use_jatos})[/bold]"
        )

        # Create version-specific config
        experiment_config = ExperimentConfig(
            **base_config_dict,
            use_jatos=use_jatos,
        )

        # Create version-specific output directory
        output_dir = base_dir / args.output_dir / version_name
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, exp_list in track(
            enumerate(selected_lists),
            description=f"Generating {version_name} experiments",
            total=len(selected_lists),
        ):
            list_output_dir = output_dir / f"list_{i + 1:02d}"

            generator = JsPsychExperimentGenerator(
                config=experiment_config,
                output_dir=list_output_dir,
                choice_config=choice_config,
            )

            try:
                generator.generate(
                    lists=[exp_list],
                    items=items_dict,
                    templates=templates_dict,
                )
                _inject_instructions_plugin(list_output_dir / "index.html")
                missing = localize_experiment(
                    list_output_dir,
                    experiment_config_dict.get("ui_text", {}),
                    experiment_config_dict.get("language", "ko"),
                )
                if missing and i == 0:
                    console.print(
                        "[yellow]⚠[/yellow] English UI text not found (bead "
                        f"template changed?): {missing}"
                    )
            except Exception as e:
                console.print(
                    f"[red]✗[/red] Error generating {version_name} list {i + 1}: {e}"
                )
                traceback.print_exc()
                sys.exit(1)

        console.print(
            f"[green]✓[/green] Generated {n_lists} {version_name} experiments"
        )

    console.print()

    # Export to JATOS if requested
    if not args.no_jatos:
        console.rule("[6/6] Exporting to JATOS")

        exporter = JATOSExporter(
            study_title=config["project"]["name"],
            study_description=config["project"].get(
                "description", "Argument structure acceptability study"
            ),
        )

        jatos_dir = base_dir / args.output_dir / "jatos"

        for i, _ in track(
            enumerate(selected_lists),
            description="Exporting to JATOS",
            total=len(selected_lists),
        ):
            list_dir = jatos_dir / f"list_{i + 1:02d}"
            jzip_path = jatos_dir / f"list_{i + 1:02d}.jzip"

            try:
                exporter.export(
                    experiment_dir=list_dir,
                    output_path=jzip_path,
                    component_title=f"List {i + 1}",
                )
            except Exception as e:
                console.print(f"[red]✗[/red] Error exporting list {i + 1}: {e}")
                traceback.print_exc()

        console.print(f"[green]✓[/green] Exported {n_lists} JATOS packages\n")
    else:
        console.rule("[6/6] Skipping JATOS Export")
        console.print("[dim]  Use --no-jatos flag to disable JATOS export[/dim]\n")

    # Print summary table
    console.rule("[bold]Deployment Summary[/bold]")
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("Lists generated:", f"[cyan]{n_lists}[/cyan]")
    table.add_row("Versions:", "[cyan]local (standalone) + jatos (deployment)[/cyan]")
    table.add_row("Output directory:", f"[cyan]{base_dir / args.output_dir}[/cyan]")
    table.add_row(
        "Total items deployed:",
        f"[cyan]{sum(len(lst.item_refs) for lst in selected_lists)}[/cyan]",
    )
    table.add_row(
        "Items per list:",
        f"[cyan]{[len(lst.item_refs) for lst in selected_lists]}[/cyan]",
    )

    if not args.no_jatos:
        table.add_row("", "")
        jatos_glob = base_dir / args.output_dir / "jatos"
        table.add_row("JATOS packages:", f"[cyan]{jatos_glob}/*.jzip[/cyan]")

    console.print(table)

    console.print(
        "\n[dim]Local version: Open deployment/local/list_XX/index.html "
        "in browser[/dim]"
    )
    if not args.no_jatos:
        console.print(
            "[dim]JATOS version: Upload .jzip files to your JATOS server[/dim]"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]✗ Unexpected error: {e}[/red]")
        import traceback

        traceback.print_exc()
        sys.exit(1)
