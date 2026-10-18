#!/usr/bin/env python3
"""Build candidate 2AFC pairs, score them, bin by score gap and select.

Past-tense test path (config ``past_test``), after sample_concrete_templates.py.
Every verb is in its past form (먹었다), and every sentence is used once.

1. Split. Each frame's concrete templates are shuffled; a third go to
   different_verb and the rest to same_verb (which uses two per pair), so
   both pair types draw on every frame and no template is used twice.
2. Build ``pair_candidate_factor`` x the needed pairs per type:
   - different_verb: one template, two verbs from the deck.
   - same_verb: one deck verb in two templates of different frame classes.
     The partner is the remaining template with the closest
     ``carrier_percentile``; a different subject noun costs
     ``subject_mismatch_cost`` (in percentile units), so partners differ
     mainly in argument structure and neither carrier is much odder.
   The deck (create_concrete_template_pairs.VerbDeck) deals every verb once
   before any repeats.
3. Score each sentence with its real verb (score_given_verbs) and remove the
   verb's rarity and length (rarity_residuals); ``lm_score_diff`` is the
   absolute difference between the two sides.
4. Bin. ``lm_score_diff`` is cut into ``n_bins`` equal-count bins per pair
   type (``quantile``, 1 = smallest gap = hardest).
5. Select n_lists x per_list pairs per type (lists.split_by), the same number
   from every bin, round robin over (pair type, bin). Each pick is the
   candidate whose verbs have been used least so far, then whose frames have,
   ties at random, so verb and frame coverage are as even as the budget
   allows. Every selected pair goes into a list.

Outputs: pairs_output (Items, read by generate_lists.py) and pairs_summary
(candidates, bin edges, coverage, rarity fit).

Parameters live in config.yaml ``past_test``; CLI flags override.
"""

import argparse
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

import yaml
from create_concrete_template_pairs import (
    VERBS_PATH,
    VerbDeck,
    load_verb_forms,
    score_specs,
    to_item,
)
from score_concrete_templates import MODEL_NAME, load_templates
from transformers import AutoModelForMaskedLM, AutoTokenizer

from bead.resources.lexical_item import LexicalItem
from bead.resources.template import Template

logger = logging.getLogger(__name__)

FORM = "PST"


def subject(ct: dict) -> str | None:
    """Return the subject noun lemma of a concrete template, if any."""
    filler = ct["slot_fillers"].get("noun_subj")
    return filler["lemma"] if filler else None


def split_by_frame(
    rows: list[dict], rng: random.Random
) -> tuple[list[dict], list[dict]]:
    """Split each frame's templates: a third to different_verb, the rest same_verb."""
    by_frame: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_frame[row["frame_name"]].append(row)
    different, same = [], []
    for name in sorted(by_frame):
        frame_rows = by_frame[name]
        rng.shuffle(frame_rows)
        cut = round(len(frame_rows) / 3)
        different += frame_rows[:cut]
        same += frame_rows[cut:]
    rng.shuffle(different)
    rng.shuffle(same)
    return different, same


def match_partners(
    pool: list[dict], n: int, mismatch_cost: float
) -> tuple[list[tuple[dict, dict]], int]:
    """Pair up to ``n`` anchors with partners of another frame class.

    Anchors are taken in pool order; each takes the remaining template of a
    different frame class with the smallest |carrier_percentile difference|
    + ``mismatch_cost`` if the subject nouns differ. Returns the pairs and the
    number of anchors left without a partner.
    """
    alive = list(range(len(pool)))
    used: set[int] = set()
    pairs: list[tuple[dict, dict]] = []
    unmatched = 0
    for i in range(len(pool)):
        if len(pairs) >= n:
            break
        if i in used:
            continue
        used.add(i)
        anchor = pool[i]
        alive = [j for j in alive if j not in used]
        options = [j for j in alive if pool[j]["frame_class"] != anchor["frame_class"]]
        if not options:
            unmatched += 1
            continue
        best = min(
            options,
            key=lambda j: abs(
                pool[j]["carrier_percentile"] - anchor["carrier_percentile"]
            ) + (mismatch_cost if subject(pool[j]) != subject(anchor) else 0.0),
        )
        used.add(best)
        pairs.append((anchor, pool[best]))
    return pairs, unmatched


def build_candidates(
    rows: list[dict],
    templates: dict[str, Template],
    verb_forms: dict[str, dict[str, LexicalItem]],
    n_per_type: int,
    mismatch_cost: float,
    rng: random.Random,
) -> tuple[list[dict], dict[str, int]]:
    """Plan the candidate pairs (not yet scored); see module docstring."""
    different_pool, same_pool = split_by_frame(rows, rng)
    deck = VerbDeck(list(verb_forms), rng)
    specs: list[dict] = []
    for ct in different_pool[:n_per_type]:
        v1 = deck.draw()
        v2 = deck.draw(avoid=frozenset({v1}))
        template = templates[ct["frame_name"]]
        specs.append({
            "pair_type": "different_verb", "form": FORM,
            "sides": [(template, ct, verb_forms[v1][FORM]),
                      (template, ct, verb_forms[v2][FORM])],
        })
    same_pairs, unmatched = match_partners(same_pool, n_per_type, mismatch_cost)
    for anchor, partner in same_pairs:
        verb = verb_forms[deck.draw()][FORM]
        specs.append({
            "pair_type": "same_verb", "form": FORM,
            "sides": [(templates[anchor["frame_name"]], anchor, verb),
                      (templates[partner["frame_name"]], partner, verb)],
        })
    counts = {
        "different_verb": min(n_per_type, len(different_pool)),
        "same_verb": len(same_pairs),
        "same_verb_unmatched_anchors": unmatched,
    }
    return specs, counts


def assign_bins(diffs: list[float], n_bins: int) -> list[int]:
    """Equal-count bins of ``diffs`` by rank, 1 = smallest."""
    order = sorted(range(len(diffs)), key=lambda i: diffs[i])
    bins = [0] * len(diffs)
    for rank, i in enumerate(order):
        bins[i] = 1 + rank * n_bins // len(diffs)
    return bins


def spec_verbs(spec: dict) -> list[str]:
    """Return the distinct verb lemmas of a pair spec."""
    return sorted({verb.lemma for _, _, verb in spec["sides"]})


def spec_frames(spec: dict) -> list[str]:
    """Return the frame of each template slot in a pair spec (one per template)."""
    if spec["pair_type"] == "different_verb":
        return [spec["sides"][0][1]["frame_name"]]
    return [ct["frame_name"] for _, ct, _ in spec["sides"]]


def spec_fillers(spec: dict) -> set[tuple[str, str, str]]:
    """Return the (frame, slot, filler lemma) positions a pair spec shows."""
    return {
        (ct["frame_name"], slot, filler["lemma"])
        for _, ct, _ in spec["sides"]
        for slot, filler in ct["slot_fillers"].items()
    }


def select(
    specs: list[dict],
    bins: list[int],
    targets: dict[str, int],
    n_bins: int,
    rng: random.Random,
) -> list[int]:
    """Pick spec indices: equal per (type, bin), least-used verbs, frames, fillers.

    A greedy pass fills the bins round robin, then repair_verb_coverage swaps
    in pairs carrying verbs the greedy pass missed.
    """
    cells: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, spec in enumerate(specs):
        cells[(spec["pair_type"], bins[i])].append(i)
    need: dict[tuple[str, int], int] = {}
    for pair_type, total in targets.items():
        base, extra = divmod(total, n_bins)
        for b in range(1, n_bins + 1):
            need[(pair_type, b)] = base + (1 if b <= extra else 0)
    tiebreak = [rng.random() for _ in specs]
    verb_use: Counter[str] = Counter()
    frame_use: Counter[str] = Counter()
    filler_use: Counter[tuple[str, str, str]] = Counter()
    chosen: list[int] = []
    while any(need[c] > 0 and cells[c] for c in need):
        for cell in sorted(need):
            if need[cell] <= 0 or not cells[cell]:
                continue
            best = min(
                cells[cell],
                key=lambda i: (
                    sum(verb_use[v] for v in spec_verbs(specs[i])),
                    sum(frame_use[f] for f in spec_frames(specs[i])),
                    sum(filler_use[f] for f in spec_fillers(specs[i])),
                    tiebreak[i],
                ),
            )
            cells[cell].remove(best)
            need[cell] -= 1
            chosen.append(best)
            verb_use.update(spec_verbs(specs[best]))
            frame_use.update(spec_frames(specs[best]))
            filler_use.update(spec_fillers(specs[best]))
    short = {f"{t}|{b}": n for (t, b), n in need.items() if n > 0}
    if short:
        logger.warning(f"Bins short of candidates: {short}")
    return repair_verb_coverage(specs, bins, chosen)


def repair_verb_coverage(
    specs: list[dict], bins: list[int], chosen: list[int]
) -> list[int]:
    """Swap in unselected pairs that carry verbs no selected pair has.

    For each missing verb, a candidate carrying it replaces a selected pair
    of the same (type, bin) whose verbs all appear in other selected pairs,
    so no verb is lost and the bin counts are unchanged.
    """
    chosen = chosen[:]
    in_chosen = set(chosen)
    verb_use = Counter(v for i in chosen for v in spec_verbs(specs[i]))
    carriers: dict[str, list[int]] = defaultdict(list)
    for i, spec in enumerate(specs):
        if i not in in_chosen:
            for v in spec_verbs(spec):
                carriers[v].append(i)
    all_verbs = {v for spec in specs for v in spec_verbs(spec)}
    swaps = 0
    for verb in sorted(all_verbs - set(verb_use)):
        for cand in carriers[verb]:
            if verb_use[verb] > 0:
                break
            if cand in in_chosen:
                continue
            cell = (specs[cand]["pair_type"], bins[cand])
            cand_verbs = set(spec_verbs(specs[cand]))
            out = next((
                p for p in chosen
                if (specs[p]["pair_type"], bins[p]) == cell
                and not cand_verbs & set(spec_verbs(specs[p]))
                and all(verb_use[u] > 1 for u in spec_verbs(specs[p]))
            ), None)
            if out is None:
                continue
            chosen[chosen.index(out)] = cand
            in_chosen.discard(out)
            in_chosen.add(cand)
            verb_use.subtract(spec_verbs(specs[out]))
            verb_use.update(cand_verbs)
            swaps += 1
    logger.info(
        f"Verb-coverage repair: {swaps} swaps; "
        f"{len(all_verbs - {v for v, n in verb_use.items() if n > 0})} verbs "
        "still missing"
    )
    return chosen


def main(
    input_path: Path,
    output_path: Path,
    summary_path: Path,
    templates_summary_path: Path,
    targets: dict[str, int],
    pair_candidate_factor: int,
    n_bins: int,
    mismatch_cost: float,
    drop_adjectives: bool,
    batch_size: int,
    seed: int,
) -> None:
    """Build, score, bin, select and write the 2AFC pairs."""
    rng = random.Random(seed)
    with open(input_path) as f:
        rows = [json.loads(line) for line in f]
    templates = {t.name: t for t in load_templates()}
    verb_forms = load_verb_forms(VERBS_PATH, drop_adjectives)
    n_candidates = pair_candidate_factor * max(targets.values())
    logger.info(
        f"{len(rows)} concrete templates, {len(verb_forms)} verbs; building "
        f"{n_candidates} candidate pairs per type"
    )
    specs, counts = build_candidates(
        rows, templates, verb_forms, n_candidates, mismatch_cost, rng
    )
    logger.info(f"Candidates: {counts}; scoring {2 * len(specs)} sentences")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    model.eval()
    side_scores, fit = score_specs(tok, model, specs, batch_size)
    diffs = [
        abs(side_scores[2 * i]["score"] - side_scores[2 * i + 1]["score"])
        for i in range(len(specs))
    ]
    bins = [0] * len(specs)
    bin_edges: dict[str, list[list[float]]] = {}
    for pair_type in targets:
        idx = [i for i, s in enumerate(specs) if s["pair_type"] == pair_type]
        type_bins = assign_bins([diffs[i] for i in idx], n_bins)
        for i, b in zip(idx, type_bins, strict=True):
            bins[i] = b
        bin_edges[pair_type] = [
            [round(min(diffs[i] for i, tb in zip(idx, type_bins, strict=True)
                       if tb == b), 3),
             round(max(diffs[i] for i, tb in zip(idx, type_bins, strict=True)
                       if tb == b), 3)]
            for b in range(1, n_bins + 1)
        ]

    chosen = select(specs, bins, targets, n_bins, rng)
    items = []
    for i in chosen:
        item = to_item(specs[i], (side_scores[2 * i], side_scores[2 * i + 1]), rng)
        items.append(item.with_(item_metadata={
            **item.item_metadata, "quantile": bins[i], "n_quantiles": n_bins,
        }))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for item in items:
            f.write(item.model_dump_json() + "\n")

    verb_use = Counter(v for i in chosen for v in spec_verbs(specs[i]))
    frame_use = Counter(f for i in chosen for f in spec_frames(specs[i]))
    seen_fillers: set[tuple[str, str, str]] = set()
    for i in chosen:
        for _, ct, _ in specs[i]["sides"]:
            for slot, filler in ct["slot_fillers"].items():
                seen_fillers.add((ct["frame_name"], slot, filler["lemma"]))
    with open(templates_summary_path) as f:
        frames_summary = json.load(f)["frames"]
    inventory = {
        (fr["frame_name"], slot)
        for fr in frames_summary for slot in fr["filler_coverage"]
    }
    filler_available = sum(
        c[1] for fr in frames_summary for c in fr["filler_coverage"].values()
    )
    filler_covered = sum(1 for f, s, _ in seen_fillers if (f, s) in inventory)
    same = [i for i in chosen if specs[i]["pair_type"] == "same_verb"]
    pct_gap = [
        abs(specs[i]["sides"][0][1]["carrier_percentile"]
            - specs[i]["sides"][1][1]["carrier_percentile"])
        for i in same
    ]
    summary = {
        "n_pairs": len(items),
        "per_type": dict(Counter(specs[i]["pair_type"] for i in chosen)),
        "per_bin": {
            pair_type: dict(sorted(Counter(
                bins[i] for i in chosen if specs[i]["pair_type"] == pair_type
            ).items()))
            for pair_type in targets
        },
        "candidates": counts,
        "bin_edges_lm_score_diff": bin_edges,
        "verbs_total": len(verb_forms),
        "verbs_covered": len(verb_use),
        "verbs_used_twice_or_more": sum(1 for n in verb_use.values() if n > 1),
        "frames_covered": len(frame_use),
        "frame_template_slots_min_max": [min(frame_use.values()),
                                         max(frame_use.values())],
        "filler_positions_covered": filler_covered,
        "filler_positions_available": filler_available,
        "same_verb_mean_carrier_percentile_gap": (
            round(sum(pct_gap) / len(pct_gap), 4) if pct_gap else None
        ),
        "same_verb_same_subject": sum(
            1 for i in same
            if subject(specs[i]["sides"][0][1]) == subject(specs[i]["sides"][1][1])
        ),
        "form": FORM, "score_model": MODEL_NAME, "seed": seed, "rarity_fit": fit,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    logger.info(
        f"Wrote {len(items)} pairs to {output_path}: {summary['per_type']}; "
        f"verbs {len(verb_use)}/{len(verb_forms)}, frames {len(frame_use)}, "
        f"filler positions {filler_covered}/{filler_available} "
        f"(summary: {summary_path})"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Build, score, bin and select past-tense 2AFC pairs"
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    cfg = config["past_test"]
    per_list = config["lists"]["split_by"]["per_list"]

    main(
        input_path=args.input or Path(cfg["templates_output"]),
        output_path=args.output or Path(config["paths"]["2afc_pairs"]),
        summary_path=args.summary or Path(cfg["pairs_summary"]),
        templates_summary_path=Path(cfg["templates_summary"]),
        targets={t: cfg["n_lists"] * n for t, n in per_list.items()},
        pair_candidate_factor=int(cfg["pair_candidate_factor"]),
        n_bins=int(cfg["n_bins"]),
        mismatch_cost=float(cfg["subject_mismatch_cost"]),
        drop_adjectives=bool(cfg["drop_adjectives"]),
        batch_size=int(cfg["score_batch_size"]),
        seed=int(cfg["seed"] if args.seed is None else args.seed),
    )
