#!/usr/bin/env python3
"""Generate experiment lists from 2AFC pairs.

This script loads 2AFC pairs and partitions them into balanced experimental lists
according to the configuration in config.yaml.
"""

from __future__ import annotations

import argparse
import random
import sys
import traceback
from pathlib import Path
from uuid import UUID, uuid4

import yaml
from rich.console import Console
from rich.table import Table

from bead.dsl.evaluator import DSLEvaluator
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
from bead.lists.experiment_list import ExperimentList
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
                item = Item.model_validate_json(line)
                items.append(item)

                # Build metadata dict for constraint checking
                # The DSL expects: item.metadata.pair_type format
                # So we need: {"metadata": {item_metadata values}}
                metadata_dict[item.id] = {"metadata": dict(item.item_metadata)}

    console.print(f"[green]✓[/green] Loaded {len(items)} 2AFC pairs")
    return items, metadata_dict


def pool_values(expression: str, metadata_dict: dict[UUID, dict]) -> list:
    """Every distinct value of ``expression`` over the item pool, sorted.

    Used for coverage ``target_values: "from_pool"``: the targets are whatever
    the pair pool actually contains (e.g. every template id), so the
    constraint is satisfiable by construction and survives regeneration.
    """
    evaluator = DSLEvaluator()
    values = {
        evaluator.evaluate(expression, {"item": item_metadata})
        for item_metadata in metadata_dict.values()
    }
    return sorted(values, key=str)


def build_constraints(
    config: dict, metadata_dict: dict[UUID, dict]
) -> tuple[list, list]:
    """Build list and batch constraints from config.

    Returns
    -------
    tuple[list, list]
        Tuple of (list_constraints, batch_constraints).
    """
    list_constraints = []
    batch_constraints = []

    # Parse list constraints from config
    for constraint_spec in config.get("lists", {}).get("constraints", []):
        constraint_type = constraint_spec["type"]

        if constraint_type == "balance":
            list_constraints.append(
                BalanceConstraint(
                    constraint_type="balance",
                    property_expression=constraint_spec["property_expression"],
                    target_counts=constraint_spec.get("target_counts", {}),
                )
            )
        elif constraint_type == "uniqueness":
            list_constraints.append(
                UniquenessConstraint(
                    constraint_type="uniqueness",
                    property_expression=constraint_spec["property_expression"],
                )
            )
        elif constraint_type == "grouped_quantile":
            list_constraints.append(
                GroupedQuantileConstraint(
                    constraint_type="grouped_quantile",
                    property_expression=constraint_spec["property_expression"],
                    group_by_expression=constraint_spec["group_by_expression"],
                    n_quantiles=constraint_spec["n_quantiles"],
                    items_per_quantile=constraint_spec["items_per_quantile"],
                )
            )
        elif constraint_type == "diversity":
            list_constraints.append(
                DiversityConstraint(
                    constraint_type="diversity",
                    property_expression=constraint_spec["property_expression"],
                    min_unique_values=constraint_spec["min_unique_values"],
                )
            )

    # Parse batch constraints from config
    for constraint_spec in config.get("lists", {}).get("batch_constraints", []):
        constraint_type = constraint_spec["type"]

        if constraint_type == "coverage":
            target_values = constraint_spec["target_values"]
            if target_values == "from_pool":
                target_values = pool_values(
                    constraint_spec["property_expression"], metadata_dict
                )
                console.print(
                    f"  coverage targets from pool: {len(target_values)} values of "
                    f"{constraint_spec['property_expression']}"
                )
            batch_constraints.append(
                BatchCoverageConstraint(
                    constraint_type="coverage",
                    property_expression=constraint_spec["property_expression"],
                    target_values=tuple(target_values),
                    min_coverage=constraint_spec.get("min_coverage", 1.0),
                )
            )
        elif constraint_type == "balance":
            batch_constraints.append(
                BatchBalanceConstraint(
                    constraint_type="balance",
                    property_expression=constraint_spec["property_expression"],
                    target_distribution=constraint_spec.get("target_distribution", {}),
                    tolerance=constraint_spec.get("tolerance", 0.05),
                )
            )
        elif constraint_type == "min_occurrence":
            batch_constraints.append(
                BatchMinOccurrenceConstraint(
                    constraint_type="min_occurrence",
                    property_expression=constraint_spec["property_expression"],
                    min_occurrences=constraint_spec["min_occurrences"],
                )
            )
        elif constraint_type == "diversity":
            batch_constraints.append(
                BatchDiversityConstraint(
                    constraint_type="diversity",
                    property_expression=constraint_spec["property_expression"],
                    max_lists_per_value=constraint_spec.get("max_lists_per_value"),
                )
            )

    return list_constraints, batch_constraints


def _distinct(
    items: list[UUID], expression: str, metadata_dict: dict[UUID, dict],
    evaluator: DSLEvaluator,
) -> set[str]:
    """Distinct values of ``expression`` over ``items``."""
    return {
        str(evaluator.evaluate(expression, {"item": metadata_dict[i]}))
        for i in items
    }


def _cover_first(
    shuffled: list[UUID], expressions: tuple[str, ...],
    metadata_dict: dict[UUID, dict], evaluator: DSLEvaluator,
) -> list[UUID]:
    """Reorder ``shuffled`` so one item per value of each expression comes first.

    The first occurrence (in the already-random order) of every value is moved
    to the front; the rest keep their random order. Taking the first N of the
    result then covers every value whenever N is at least the number of
    values.
    """
    seen: set[tuple[str, str]] = set()
    front: list[UUID] = []
    rest: list[UUID] = []
    for item_id in shuffled:
        new = False
        for expression in expressions:
            key = (
                expression,
                str(evaluator.evaluate(expression, {"item": metadata_dict[item_id]})),
            )
            if key not in seen:
                seen.add(key)
                new = True
        (front if new else rest).append(item_id)
    return front + rest


def split_partition(
    partitioner: ListPartitioner,
    item_uuids: list[UUID],
    metadata_dict: dict[UUID, dict],
    split_by: dict,
    n_lists: int,
    list_constraints: list,
    seed: int,
    cover_expressions: tuple[str, ...] = (),
) -> list[ExperimentList]:
    """Partition each stratum separately, then merge list i of every stratum.

    ``split_by`` gives a property expression and the exact number of items
    per list for each of its values (e.g. 25 same_verb + 25 different_verb).
    bead's greedy partitioner treats a list-level balance constraint as
    pass/fail, so enforcing 25/25 that way made list sizes drift (47-54)
    without balancing much (same_verb 18-30 per list). Partitioning each
    stratum into n_lists x per_list and concatenating gives every list
    exactly the configured counts and size. List constraints still apply
    within each stratum (a balance constraint on the split property itself
    is dropped -- it is satisfied by construction).

    Each stratum's pool is a seeded random sample, except that it first takes
    one random item for every value of each ``cover_expressions`` property
    (the batch coverage constraints, e.g. frame_id), then fills up at random.
    Batch constraints are only reported on this path, not enforced by swaps,
    and many frames have just 1-3 pairs, so a purely random pool could miss
    one (it did: 137/138 frames). Seeding the pool guarantees coverage of
    every value present in the stratum.
    """
    evaluator = DSLEvaluator()
    expression = split_by["property_expression"]
    per_list: dict[str, int] = split_by["per_list"]
    by_value: dict[str, list[UUID]] = {value: [] for value in per_list}
    for item_id in item_uuids:
        value = str(evaluator.evaluate(expression, {"item": metadata_dict[item_id]}))
        if value in by_value:
            by_value[value].append(item_id)

    constraints = [
        c for c in list_constraints
        if not (
            isinstance(c, BalanceConstraint)
            and c.property_expression == expression
        )
    ]
    rng = random.Random(seed)
    per_stratum: list[list[ExperimentList]] = []
    for value, count in per_list.items():
        pool = by_value[value]
        needed = n_lists * count
        if len(pool) < needed:
            raise ValueError(
                f"split_by {expression}={value!r}: need {needed} items "
                f"({n_lists} lists x {count}), pool has {len(pool)}"
            )
        rng.shuffle(pool)
        pool = _cover_first(pool, cover_expressions, metadata_dict, evaluator)
        if len(pool) > needed and any(
            len(_distinct(pool, e, metadata_dict, evaluator)) > needed
            for e in cover_expressions
        ):
            console.print(
                f"[yellow]⚠[/yellow] {value}: more coverage values than the "
                f"{needed} items drawn; coverage can't be complete"
            )
        # "balanced" is bead's greedy strategy; the configured
        # lists.strategy name isn't a bead strategy and was never applied.
        lists = partitioner.partition(
            items=pool[:needed], n_lists=n_lists, constraints=constraints,
            strategy="balanced", metadata=metadata_dict,
        )
        per_stratum.append(lists)
        console.print(
            f"  • {value}: {needed} items -> {n_lists} lists x {count}"
        )

    merged = []
    for i in range(n_lists):
        refs = tuple(ref for lists in per_stratum for ref in lists[i].item_refs)
        merged_list = per_stratum[0][i].with_(
            item_refs=refs,
            list_constraints=tuple(list_constraints),
            list_metadata={
                **per_stratum[0][i].list_metadata,
                "split_by": expression,
                "split_counts": dict(per_list),
            },
        )
        # No public API for these; the same scorer bead's own partitioner uses.
        merged.append(merged_list.with_(
            balance_metrics=partitioner._compute_balance_metrics(  # noqa: SLF001
                merged_list, list_constraints, metadata_dict
            )
        ))
    return merged


def report_batch_constraints(
    partitioner: ListPartitioner,
    lists: list[ExperimentList],
    batch_constraints: list,
    metadata_dict: dict[UUID, dict],
    tolerance: float = 0.05,
) -> None:
    """Score each batch constraint on the final lists and flag any unmet.

    The split path builds lists outside partition_with_batch_constraints, so
    batch constraints are checked here instead of driving swaps (on this
    pool they were already met without any refinement).
    """
    for constraint in batch_constraints:
        score = partitioner._compute_batch_constraint_score(  # noqa: SLF001
            lists, constraint, metadata_dict
        )
        ok = score >= 1.0 - tolerance
        mark = "[green]✓[/green]" if ok else "[yellow]⚠[/yellow]"
        console.print(
            f"  {mark} batch {constraint.constraint_type} "
            f"({constraint.property_expression}): score {score:.3f}"
        )


def main() -> None:
    """Generate experiment lists from 2AFC pairs."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n-lists", type=int, default=None, help="Override n_lists from config"
    )
    args = parser.parse_args()

    # Determine base directory
    base_dir = Path(__file__).parent
    config_path = base_dir / "config.yaml"

    console.rule("[bold]Experiment List Generation[/bold]")
    console.print(f"Base directory: [cyan]{base_dir}[/cyan]")
    console.print(f"Configuration: [cyan]{config_path}[/cyan]\n")

    # Load configuration
    console.rule("[1/5] Loading Configuration")
    config = load_config(config_path)
    list_config = config["lists"]
    n_lists = args.n_lists if args.n_lists is not None else list_config["n_lists"]
    items_per_list = list_config["items_per_list"]
    strategy = list_config.get("strategy", "balanced")

    console.print("[green]✓[/green] Configuration loaded")
    console.print(f"  • Lists to generate: [cyan]{n_lists}[/cyan]")
    console.print(f"  • Items per list: [cyan]{items_per_list}[/cyan]")
    console.print(f"  • Strategy: [cyan]{strategy}[/cyan]\n")

    # Load 2AFC pairs
    console.rule("[2/5] Loading 2AFC Pairs")
    pairs_path = base_dir / config["paths"]["2afc_pairs"]
    items, metadata_dict = load_2afc_pairs(pairs_path)

    # Build constraints
    console.print()
    console.rule("[3/5] Building Constraints")
    list_constraints, batch_constraints = build_constraints(config, metadata_dict)
    console.print(f"[green]✓[/green] Built {len(list_constraints)} list constraints")
    console.print(
        f"[green]✓[/green] Built {len(batch_constraints)} batch constraints\n"
    )

    # Partition items into lists
    console.rule("[4/5] Partitioning Items")
    seed = list_config.get("random_seed", 42)
    partitioner = ListPartitioner(random_seed=seed)
    # Pool = a seeded random sample of the needed size, not the first N in
    # file order: the pairs file is ordered by cell, so a prefix cut kept
    # only 21% of NP_V_S and 42% of NP_V_PP_S pairs.
    item_uuids = [item.id for item in items]
    random.Random(seed).shuffle(item_uuids)
    total_items_needed = n_lists * items_per_list
    selected_uuids = item_uuids[:total_items_needed]

    split_by = list_config.get("split_by")
    try:
        if split_by:
            per_list_total = sum(split_by["per_list"].values())
            if per_list_total != items_per_list:
                raise ValueError(
                    f"split_by.per_list sums to {per_list_total}, "
                    f"items_per_list is {items_per_list}"
                )
            with console.status("[bold]Partitioning each stratum...[/bold]"):
                experiment_lists = split_partition(
                    partitioner, [item.id for item in items], metadata_dict,
                    split_by, n_lists, list_constraints, seed,
                    cover_expressions=tuple(
                        c.property_expression for c in batch_constraints
                        if isinstance(c, BatchCoverageConstraint)
                    ),
                )
            report_batch_constraints(
                partitioner, experiment_lists, batch_constraints, metadata_dict
            )
        # Use the appropriate partitioning method based on batch constraints
        elif batch_constraints:
            with console.status("[bold]Using batch-constrained partitioning...[/bold]"):
                experiment_lists = partitioner.partition_with_batch_constraints(
                    items=selected_uuids,
                    n_lists=n_lists,
                    list_constraints=list_constraints,
                    batch_constraints=batch_constraints,
                    metadata=metadata_dict,
                )
        else:
            with console.status("[bold]Using standard partitioning...[/bold]"):

                experiment_lists = partitioner.partition(
                    items=selected_uuids,
                    n_lists=n_lists,
                    constraints=list_constraints,
                    strategy=strategy,
                    metadata=metadata_dict,
                )

        console.print(f"[green]✓[/green] Created {len(experiment_lists)} lists")
        for i, exp_list in enumerate(experiment_lists):
            console.print(
                f"  • List {i + 1}: [cyan]{len(exp_list.item_refs)}[/cyan] items"
            )

    except Exception as e:
        console.print(f"[red]✗[/red] Error during partitioning: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Save lists
    console.print()
    console.rule("[5/5] Saving Lists")
    output_path = base_dir / config["paths"]["experiment_lists"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create list collection

    list_collection = ListCollection(
        name="argument_structure_lists",
        source_items_id=uuid4(),  # Generate a UUID for the source items
        lists=experiment_lists,
        partitioning_strategy=strategy,
        partitioning_config={
            "n_lists": len(experiment_lists),
            "items_per_list": items_per_list,
            "n_list_constraints": len(list_constraints),
            "n_batch_constraints": len(batch_constraints),
        },
        partitioning_stats={
            "total_items": sum(len(lst.item_refs) for lst in experiment_lists),
        },
    )

    # Save as JSONL
    with open(output_path, "w") as f:
        for exp_list in list_collection.lists:
            f.write(exp_list.model_dump_json() + "\n")

    console.print(
        f"[green]✓[/green] Saved {len(experiment_lists)} lists to "
        f"[cyan]{output_path}[/cyan]\n"
    )

    # Print summary table
    console.rule("[bold]Summary[/bold]")
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("Total lists:", f"[cyan]{len(experiment_lists)}[/cyan]")
    table.add_row(
        "Total items distributed:",
        f"[cyan]{sum(len(lst.item_refs) for lst in experiment_lists)}[/cyan]",
    )
    table.add_row(
        "Items per list:",
        f"[cyan]{[len(lst.item_refs) for lst in experiment_lists]}[/cyan]",
    )
    console.print(table)


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
