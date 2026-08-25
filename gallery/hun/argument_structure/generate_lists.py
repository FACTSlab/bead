#!/usr/bin/env python3
"""Generate experiment lists from 2AFC pairs.

This script loads 2AFC pairs and partitions them into balanced experimental lists
according to the configuration in config.yaml.
"""

from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from uuid import UUID

import yaml
from rich.console import Console
from rich.table import Table

from bead.items.item import Item
from bead.lists import (
    BalanceConstraint,
    DiversityConstraint,
    ListCollection,
    UniquenessConstraint,
)
from bead.lists.constraints import (
    BatchBalanceConstraint,
    BatchCoverageConstraint,
    BatchDiversityConstraint,
    BatchMinOccurrenceConstraint,
    GroupedQuantileConstraint,
)
from bead.lists.partitioner import ListPartitioner

console = Console()


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_2afc_pairs(pairs_path: Path) -> tuple[list[Item], dict[UUID, dict]]:
    """Load 2AFC pairs and extract metadata for constraint checking.

    Returns
    -------
    tuple[list[Item], dict[UUID, dict]]
        Tuple of (items list, metadata dict keyed by item UUID).
    """
    with console.status(f"[bold]Loading 2AFC pairs from {pairs_path}...[/bold]"):
        items = []
        metadata_dict = {}

        with open(pairs_path) as f:
            for line in f:
                data = json.loads(line)
                item = Item(**data)
                items.append(item)

                metadata_dict[item.id] = {
                    "metadata": dict(item.item_metadata)
                }

    console.print(f"[green]✓[/green] Loaded {len(items)} 2AFC pairs")
    return items, metadata_dict


def select_balanced_candidate_pool(
    items: list[Item],
    total_items_needed: int,
    random_seed: int = 42,
) -> list[UUID]:
    """Select a balanced candidate pool before list partitioning.

    The older ENG/KOR script takes the first N pairs from the JSONL file. Since
    pair files are usually written with all same-verb pairs first, that can make
    a nominally balanced list impossible before the partitioner even starts.
    This implementation selects equal same-verb and different-verb candidate
    pools first. The same-verb half is round-robin sampled across contrast_type
    so one contrast does not dominate the pool.
    """
    rng = random.Random(random_seed)
    by_pair_type: dict[str, list[Item]] = defaultdict(list)
    for item in items:
        by_pair_type[item.item_metadata.get("pair_type", "unknown")].append(item)

    same_target = total_items_needed // 2
    different_target = total_items_needed - same_target

    same_items = by_pair_type.get("same_verb", [])
    different_items = by_pair_type.get("different_verb", [])

    if len(same_items) < same_target or len(different_items) < different_target:
        raise ValueError(
            "Not enough pairs for a 50/50 candidate pool: "
            f"need {same_target} same-verb and {different_target} different-verb, "
            f"but found {len(same_items)} and {len(different_items)}."
        )

    same_by_contrast: dict[str, list[Item]] = defaultdict(list)
    for item in same_items:
        same_by_contrast[item.item_metadata.get("contrast_type", "unknown")].append(item)

    for group in same_by_contrast.values():
        rng.shuffle(group)

    selected_same: list[Item] = []
    contrast_names = sorted(same_by_contrast)
    while len(selected_same) < same_target and contrast_names:
        next_names = []
        for contrast_name in contrast_names:
            group = same_by_contrast[contrast_name]
            if group and len(selected_same) < same_target:
                selected_same.append(group.pop())
            if group:
                next_names.append(contrast_name)
        contrast_names = next_names

    rng.shuffle(different_items)
    selected_different = different_items[:different_target]

    selected = selected_same + selected_different
    rng.shuffle(selected)
    return [item.id for item in selected]


def build_constraints(config: dict) -> tuple[list, list]:
    """Build list and batch constraints from config."""
    list_constraints = []
    batch_constraints = []

    for constraint_spec in config.get("lists", {}).get("constraints", []):
        constraint_type = constraint_spec["type"]

        if constraint_type == "balance":
            list_constraints.append(
                BalanceConstraint(
                    property_expression=constraint_spec["property_expression"],
                    target_counts=constraint_spec.get("target_counts", {}),
                )
            )

        elif constraint_type == "uniqueness":
            list_constraints.append(
                UniquenessConstraint(
                    property_expression=constraint_spec["property_expression"]
                )
            )

        elif constraint_type == "grouped_quantile":
            list_constraints.append(
                GroupedQuantileConstraint(
                    property_expression=constraint_spec["property_expression"],
                    group_by_expression=constraint_spec[
                        "group_by_expression"
                    ],
                    n_quantiles=constraint_spec["n_quantiles"],
                    items_per_quantile=constraint_spec[
                        "items_per_quantile"
                    ],
                )
            )

        elif constraint_type == "diversity":
            list_constraints.append(
                DiversityConstraint(
                    property_expression=constraint_spec[
                        "property_expression"
                    ],
                    min_unique_values=constraint_spec[
                        "min_unique_values"
                    ],
                )
            )

    for constraint_spec in config.get(
        "lists", {}
    ).get("batch_constraints", []):
        constraint_type = constraint_spec["type"]

        if constraint_type == "coverage":
            batch_constraints.append(
                BatchCoverageConstraint(
                    property_expression=constraint_spec[
                        "property_expression"
                    ],
                    target_values=constraint_spec["target_values"],
                    min_coverage=constraint_spec.get(
                        "min_coverage", 1.0
                    ),
                )
            )

        elif constraint_type == "balance":
            batch_constraints.append(
                BatchBalanceConstraint(
                    property_expression=constraint_spec[
                        "property_expression"
                    ],
                    target_distribution=constraint_spec.get(
                        "target_distribution", {}
                    ),
                    tolerance=constraint_spec.get("tolerance", 0.05),
                )
            )

        elif constraint_type == "min_occurrence":
            batch_constraints.append(
                BatchMinOccurrenceConstraint(
                    property_expression=constraint_spec[
                        "property_expression"
                    ],
                    min_occurrences=constraint_spec[
                        "min_occurrences"
                    ],
                )
            )

        elif constraint_type == "diversity":
            batch_constraints.append(
                BatchDiversityConstraint(
                    property_expression=constraint_spec[
                        "property_expression"
                    ],
                    max_lists_per_value=constraint_spec.get(
                        "max_lists_per_value"
                    ),
                )
            )

    return list_constraints, batch_constraints


def main() -> None:
    """Generate experiment lists from 2AFC pairs."""
    base_dir = Path(__file__).parent
    config_path = base_dir / "config.yaml"

    console.rule("[bold]Experiment List Generation[/bold]")
    console.print(f"Base directory: [cyan]{base_dir}[/cyan]")
    console.print(
        f"Configuration: [cyan]{config_path}[/cyan]\n"
    )

    console.rule("[1/5] Loading Configuration")
    config = load_config(config_path)

    list_config = config["lists"]
    n_lists = list_config["n_lists"]
    items_per_list = list_config["items_per_list"]
    strategy = list_config.get("strategy", "balanced")

    console.print("[green]✓[/green] Configuration loaded")
    console.print(
        f"  • Lists to generate: [cyan]{n_lists}[/cyan]"
    )
    console.print(
        f"  • Items per list: [cyan]{items_per_list}[/cyan]"
    )
    console.print(f"  • Strategy: [cyan]{strategy}[/cyan]\n")

    console.rule("[2/5] Loading 2AFC Pairs")

    pairs_path = base_dir / config["paths"]["2afc_pairs"]
    items, metadata_dict = load_2afc_pairs(pairs_path)

    console.print()
    console.rule("[3/5] Building Constraints")

    list_constraints, batch_constraints = build_constraints(config)

    console.print(
        f"[green]✓[/green] Built "
        f"{len(list_constraints)} list constraints"
    )
    console.print(
        f"[green]✓[/green] Built "
        f"{len(batch_constraints)} batch constraints\n"
    )

    console.rule("[4/5] Partitioning Items")

    partitioner = ListPartitioner(random_seed=42)
    item_uuids = [item.id for item in items]

    try:
        total_items_needed = n_lists * items_per_list
        selected_uuids = select_balanced_candidate_pool(
            items,
            total_items_needed=total_items_needed,
            random_seed=42,
        )

        if batch_constraints:
            with console.status(
                "[bold]Using batch-constrained partitioning...[/bold]"
            ):
                experiment_lists = (
                    partitioner.partition_with_batch_constraints(
                        items=selected_uuids,
                        n_lists=n_lists,
                        list_constraints=list_constraints,
                        batch_constraints=batch_constraints,
                        metadata=metadata_dict,
                    )
                )

        else:
            with console.status(
                "[bold]Using standard partitioning...[/bold]"
            ):
                experiment_lists = partitioner.partition(
                    items=selected_uuids,
                    n_lists=n_lists,
                    constraints=list_constraints,
                    strategy=strategy,
                    metadata=metadata_dict,
                )

        console.print(
            f"[green]✓[/green] Created "
            f"{len(experiment_lists)} lists"
        )

        for i, exp_list in enumerate(experiment_lists):
            console.print(
                f"  • List {i + 1}: "
                f"[cyan]{len(exp_list.item_refs)}[/cyan] items"
            )

    except Exception as e:
        console.print(
            f"[red]✗[/red] Error during partitioning: {e}"
        )

        import traceback

        traceback.print_exc()
        sys.exit(1)

    console.print()
    console.rule("[5/5] Saving Lists")

    output_path = (
        base_dir / config["paths"]["experiment_lists"]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    from uuid import uuid4

    list_collection = ListCollection(
        name="argument_structure_lists",
        source_items_id=uuid4(),
        lists=experiment_lists,
        partitioning_strategy=strategy,
        partitioning_config={
            "n_lists": len(experiment_lists),
            "items_per_list": items_per_list,
            "n_list_constraints": len(list_constraints),
            "n_batch_constraints": len(batch_constraints),
        },
        partitioning_stats={
            "total_items": sum(
                len(exp_list.item_refs)
                for exp_list in experiment_lists
            ),
        },
    )

    with open(output_path, "w") as f:
        for exp_list in list_collection.lists:
            f.write(exp_list.model_dump_json() + "\n")

    console.print(
        f"[green]✓[/green] Saved "
        f"{len(experiment_lists)} lists to "
        f"[cyan]{output_path}[/cyan]\n"
    )

    console.rule("[bold]Summary[/bold]")

    table = Table(
        show_header=False,
        box=None,
        padding=(0, 2),
    )

    table.add_row(
        "Total lists:",
        f"[cyan]{len(experiment_lists)}[/cyan]",
    )

    table.add_row(
        "Total items distributed:",
        (
            f"[cyan]"
            f"{sum(len(exp_list.item_refs) for exp_list in experiment_lists)}"
            f"[/cyan]"
        ),
    )

    table.add_row(
        "Items per list:",
        (
            f"[cyan]"
            f"{[len(exp_list.item_refs) for exp_list in experiment_lists]}"
            f"[/cyan]"
        ),
    )

    console.print(table)


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        console.print(
            "\n[yellow]⚠️ Interrupted by user[/yellow]"
        )
        sys.exit(130)

    except Exception as e:
        console.print(
            f"\n[red]✗ Unexpected error: {e}[/red]"
        )

        import traceback

        traceback.print_exc()
        sys.exit(1)