#!/usr/bin/env python3
"""Bin scored concrete templates within frame_class x aspect cells and select per bin.

Input: score_concrete_templates.py output (default: the --sample-per-cell sample), one
row per concrete template with ``raw_score`` and the template's stratification
metadata (``frame_class``, ``aspect``, ``adjunct_tier``, ``obliques``).

Why per cell: scores are verb-predictability, so they differ systematically
by frame shape (complement frames' matrix verb is predictable, bare
intransitives' isn't) and by aspect. Global quantiles would sort by frame
shape instead of naturalness -- in the pilot the top global quintile was 33%
clausal against an 11% base rate. Within a cell, bins are equal-count
quantiles of ``score_field``; cells too small for ``n_bins`` bins of at least
``min_per_bin`` concrete templates get fewer bins (NP_V: 19 per aspect -> 2).

Selection draws ``per_bin`` concrete templates from every bin, so low-scoring
(anomalous) concrete templates are kept as deliberately as high-scoring (natural) ones
-- the experiment, not the model, decides what's acceptable. Each pick goes
to the template used least so far in the cell (random concrete template within it),
so selections spread across the cell's case frames instead of clustering.

Rows without a valid verb (``beam_had_valid_candidate`` false) are excluded
from binning -- their score is a fallback, not a measurement.

Outputs:
  output   selected concrete templates, each input row plus
           concrete_template_id, cell, bin (1 = lowest), n_bins,
           bin_score_min, bin_score_max
  summary  per cell: n, n_bins, and per bin its score range, size, selected

Parameters live in config.yaml ``concrete_template_binning``; CLI flags override.
"""

import argparse
import hashlib
import json
import logging
import random
from collections import defaultdict
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


def concrete_template_id(row: dict) -> str:
    """Return a stable id for a concrete template (template + filled text)."""
    key = f"{row['frame_name']}\t{row['prefix']}\t{row['suffix']}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def cell_bins(n: int, n_bins: int, min_per_bin: int) -> int:
    """Return how many bins a cell of ``n`` concrete templates gets."""
    return max(1, min(n_bins, n // min_per_bin))


def assign_bins(rows: list[dict], score_field: str, k: int) -> list[list[dict]]:
    """Split ``rows`` into ``k`` equal-count bins by score, lowest first.

    Rank-based, so ties and skew can't leave a bin empty; ties are broken by
    concrete_template_id to keep the split deterministic.
    """
    ordered = sorted(rows, key=lambda r: (r[score_field], r["concrete_template_id"]))
    bins: list[list[dict]] = [[] for _ in range(k)]
    for rank, row in enumerate(ordered):
        bins[rank * k // len(ordered)].append(row)
    return bins


def select_from_bin(
    rows: list[dict], n: int, rng: random.Random, usage: dict[str, int]
) -> list[dict]:
    """Pick ``n`` rows from one bin, spreading picks across templates.

    Each pick goes to the template used least so far in this *cell*
    (``usage``, shared across the cell's bins and updated here), ties broken
    at random, so a cell with more templates than picks per bin still covers
    as many of its templates as its total picks allow.
    """
    by_frame: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_frame[row["frame_name"]].append(row)
    for queue in by_frame.values():
        rng.shuffle(queue)
    tiebreak = {name: rng.random() for name in by_frame}
    picked: list[dict] = []
    while len(picked) < n:
        available = [name for name, queue in by_frame.items() if queue]
        if not available:
            break
        name = min(available, key=lambda t: (usage.get(t, 0), tiebreak[t]))
        picked.append(by_frame[name].pop())
        usage[name] = usage.get(name, 0) + 1
    return picked


def main(
    input_path: Path,
    output_path: Path,
    summary_path: Path,
    score_field: str,
    n_bins: int,
    min_per_bin: int,
    per_bin: int,
    seed: int,
) -> None:
    """Bin every cell, select per bin, and write selections and a summary."""
    rng = random.Random(seed)
    with open(input_path) as f:
        rows = [json.loads(line) for line in f]
    valid = [r for r in rows if r.get("beam_had_valid_candidate", True)]
    logger.info(
        f"Loaded {len(rows)} concrete templates from {input_path}; "
        f"{len(rows) - len(valid)} without a valid verb excluded"
    )

    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in valid:
        row["concrete_template_id"] = concrete_template_id(row)
        cells[(row["frame_class"], row["aspect"])].append(row)

    selected: list[dict] = []
    summary: list[dict] = []
    for (frame_class, aspect), cell_rows in sorted(cells.items()):
        k = cell_bins(len(cell_rows), n_bins, min_per_bin)
        cell_summary = {
            "frame_class": frame_class, "aspect": aspect,
            "n": len(cell_rows), "n_bins": k, "bins": [],
        }
        usage: dict[str, int] = {}
        for b, bin_rows in enumerate(assign_bins(cell_rows, score_field, k)):
            lo = min(r[score_field] for r in bin_rows)
            hi = max(r[score_field] for r in bin_rows)
            picks = select_from_bin(bin_rows, per_bin, rng, usage)
            for row in picks:
                selected.append({
                    **row,
                    "cell": f"{frame_class}|{aspect}",
                    "bin": b + 1,
                    "n_bins": k,
                    "bin_score_min": lo,
                    "bin_score_max": hi,
                })
            cell_summary["bins"].append({
                "bin": b + 1, "n": len(bin_rows),
                "score_min": round(lo, 4), "score_max": round(hi, 4),
                "selected": len(picks),
                "frames_selected": len({r["frame_name"] for r in picks}),
            })
        summary.append(cell_summary)
        logger.info(
            f"{frame_class:15s} {aspect:9s} n={len(cell_rows):4d} bins={k} "
            f"selected={sum(b['selected'] for b in cell_summary['bins'])}"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for row in selected:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    config_used = {
        "input": str(input_path), "score_field": score_field, "n_bins": n_bins,
        "min_per_bin": min_per_bin, "per_bin": per_bin, "seed": seed,
    }
    with open(summary_path, "w") as f:
        json.dump(
            {"config": config_used, "n_selected": len(selected), "cells": summary},
            f, ensure_ascii=False, indent=1,
        )
    logger.info(
        f"Selected {len(selected)} concrete templates from {len(cells)} cells -> "
        f"{output_path} (summary: {summary_path})"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Bin scored concrete templates per frame_class x aspect cell and "
        "select an equal number per bin"
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--score-field")
    parser.add_argument("--n-bins", type=int)
    parser.add_argument("--min-per-bin", type=int)
    parser.add_argument("--per-bin", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)["concrete_template_binning"]

    def pick(cli_value: object, key: str) -> object:
        """Return the CLI value if given, else the config value."""
        return cfg[key] if cli_value is None else cli_value

    main(
        input_path=Path(pick(args.input, "input")),
        output_path=Path(pick(args.output, "output")),
        summary_path=Path(pick(args.summary, "summary")),
        score_field=str(pick(args.score_field, "score_field")),
        n_bins=int(pick(args.n_bins, "n_bins")),
        min_per_bin=int(pick(args.min_per_bin, "min_per_bin")),
        per_bin=int(pick(args.per_bin, "per_bin")),
        seed=int(pick(args.seed, "seed")),
    )
