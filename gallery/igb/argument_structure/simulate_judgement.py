#!/usr/bin/env python3
"""Simulate human judgments for 2AFC pairs and output as CSV/JSONL.

This is a MODIFIED version of simulate_pipeline.py that:
- KEEPS: Human judgment simulation using LM scores
- REMOVES: Active learning loop, model training, convergence detection
- ADDS: CSV and JSONL output of simulated choices

Based on the original simulate_pipeline.py from the bead framework.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Any

import numpy as np

from bead.config.simulation import NoiseModelConfig, SimulatedAnnotatorConfig
from bead.evaluation.interannotator import InterAnnotatorMetrics
from bead.items.item import Item
from bead.items.item_template import ItemTemplate, PresentationSpec, TaskSpec
from bead.simulation.annotators.base import SimulatedAnnotator


def load_2afc_pairs(path: Path, limit: int | None = None, skip: int = 0) -> list[Item]:
    """Load 2AFC pairs from JSONL.

    Parameters
    ----------
    path : Path
        Path to JSONL file
    limit : int | None
        Maximum number of items to load
    skip : int
        Number of items to skip at start

    Returns
    -------
    list[Item]
        List of items
    """
    items = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i < skip:
                continue
            if limit and (i - skip) >= limit:
                break
            data = json.loads(line)
            items.append(Item(**data))
    return items


def get_forced_choice_template() -> ItemTemplate:
    """Create ItemTemplate for 2AFC forced choice task.

    Returns
    -------
    ItemTemplate
        Template configured for forced_choice task using proper TaskSpec
    """
    return ItemTemplate(
        name="2AFC Forced Choice",
        judgment_type="preference",
        task_type="forced_choice",
        task_spec=TaskSpec(
            prompt="Which sentence sounds more natural?",
            options=["option_a", "option_b"],
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )


def run_simulation(
    n_items: int | None = None,
    temperature: float = 1.0,
    random_state: int | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Simulate human judgments and output as CSV/JSONL.

    Parameters
    ----------
    pairs_path : Path
        Path to 2AFC pairs JSONL file
    n_items : int | None
        Number of items to simulate (None = all)
    temperature : float
        Temperature for simulated judgments (higher = more noise)
    random_state : int | None
        Random seed for reproducibility
    output_dir : Path | None
        Directory to save outputs

    Returns
    -------
    dict[str, Any]
        Simulation results
    """
    print("=" * 80)
    print("SIMULATION: Igbo Acceptability Judgments")
    print("=" * 80)
    print("Configuration:")
    print(f"  N items: {n_items or 'all'}")
    print(f"  Temperature: {temperature}")
    print(f"  Random state: {random_state}")
    print()

    # Setup output directory
    if output_dir is None:
        output_dir = Path("simulation_output")
    output_dir.mkdir(exist_ok=True)

    # Set random seed
    if random_state is not None:
        random.seed(random_state)
        np.random.seed(random_state)

    # [1/4] Load data
    print("[1/4] Loading data...")
    pairs_path = Path("items/2afc_pairs.jsonl")
    
    if not pairs_path.exists():
        raise FileNotFoundError(f"2AFC pairs not found: {pairs_path}")

    all_pairs = load_2afc_pairs(pairs_path, limit=n_items)
    print(f"  Loaded {len(all_pairs)} 2AFC pairs")
    print()

    # [2/4] Setup simulated annotator
    # This creates the "fake human" that makes choices based on LM scores
    print("[2/4] Setting up simulated annotator...")

    # Create annotator configuration using bead.simulation framework
    annotator_config = SimulatedAnnotatorConfig(
        strategy="lm_score",
        model_output_key="lm_score",
        noise_model=NoiseModelConfig(
            noise_type="temperature",
            temperature=temperature,
        ),
        random_state=random_state,
        fallback_to_random=True,
    )

    annotator = SimulatedAnnotator.from_config(annotator_config)
    item_template = get_forced_choice_template()

    print(f"  Strategy: lm_score")
    print(f"  Temperature: {temperature}")
    print(f"  Random state: {random_state}")
    print()

     # [3/4] Generate simulated annotations
    print("[3/4] Generating simulated annotations...")

    annotations = annotator.annotate_batch(all_pairs, item_template)
    print(f"  Generated {len(annotations)} annotations")

    # Create second annotator for agreement calculation
    annotator2 = SimulatedAnnotator.from_config(
        annotator_config.model_copy(update={"random_state": (random_state or 0) + 1000})
    )
    annotations2 = annotator2.annotate_batch(all_pairs, item_template)

    # Compute agreement
    labels1 = [annotations[str(item.id)] for item in all_pairs]
    labels2 = [annotations2[str(item.id)] for item in all_pairs]

    inter_annotator = InterAnnotatorMetrics()
    human_agreement = inter_annotator.cohens_kappa(labels1, labels2)

    print(f"  Simulated human agreement (Cohen's κ): {human_agreement:.3f}")
    print()

    # [4/4] Create output data and save
    print("[4/4] Creating output files...")

    # Build output data with all relevant fields
    output_data = []
    for item in all_pairs:
        item_id = str(item.id)
        choice = annotations[item_id]  # "option_a" or "option_b"
        
        # Get metadata
        meta = item.item_metadata
        rendered = item.rendered_elements
        
        output_data.append({
            "pair_id": item_id,
            "sentence_a": rendered.get("option_a", ""),
            "sentence_b": rendered.get("option_b", ""),
            "lm_score_a": meta.get("lm_score_a", 0),
            "lm_score_b": meta.get("lm_score_b", 0),
            "lm_score_diff": meta.get("lm_score_diff", 0),
            "choice": choice,
            "choice_numeric": 0 if choice == "option_a" else 1,
            "verb": meta.get("verb", ""),
            "pair_type": meta.get("pair_type", ""),
            "template1": meta.get("template1", ""),
            "template2": meta.get("template2", ""),
            "quantile": meta.get("quantile", 0),
        })

    # Count choices
    n_chose_a = sum(1 for d in output_data if d["choice_numeric"] == 0)
    n_chose_b = len(output_data) - n_chose_a
    print(f"  Chose A: {n_chose_a} ({100*n_chose_a/len(output_data):.1f}%)")
    print(f"  Chose B: {n_chose_b} ({100*n_chose_b/len(output_data):.1f}%)")

    # Save as CSV
    csv_path = output_dir / "simulated_judgments.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_data[0].keys())
        writer.writeheader()
        writer.writerows(output_data)
    print(f" Saved: {csv_path}")

    # Save as JSONL
    jsonl_path = output_dir / "simulated_judgments.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for item in output_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f" Saved: {jsonl_path}")

    # Save summary results
    results = {
        "config": {
            "n_items": len(all_pairs),
            "temperature": temperature,
            "random_state": random_state,
        },
        "summary": {
            "n_chose_a": n_chose_a,
            "n_chose_b": n_chose_b,
            "pct_chose_a": n_chose_a / len(output_data),
            "pct_chose_b": n_chose_b / len(output_data),
        },
        # "agreement": agreement,
    }

    results_path = output_dir / "simulation_summary.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f" Saved: {results_path}")

    print()
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total items simulated: {len(output_data)}")
    print()
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Simulate human judgments for 2AFC pairs"
    )
    parser.add_argument(
        "--n-items",
        type=int,
        default=None,
        help="Number of items to simulate (default: all)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Judgment noise temperature (default: 1.0)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("simulation_output"),
        help="Output directory (default: simulation_output)",
    )

    args = parser.parse_args()

    run_simulation(
        n_items=args.n_items,
        temperature=args.temperature,
        random_state=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()