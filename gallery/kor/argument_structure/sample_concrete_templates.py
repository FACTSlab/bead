#!/usr/bin/env python3
"""Sample concrete templates per frame and give each a two-mask carrier score.

Past-tense test path (config ``past_test``): simple-aspect frames only. It
replaces beam decoding and template binning (score_concrete_templates.py ->
bin_concrete_templates.py) with one sampling step; select_pairs.py follows.

Sampling. The template budget is what select_pairs.py needs for its candidate
pairs: ``pair_candidate_factor`` x ``n_lists`` x (different_verb per list +
2 x same_verb per list), since a different_verb pair uses one template and a
same_verb pair two.
It is split evenly across the 46 frames; a frame with fewer valid fillings
than its share (NP_V: 19, one per subject noun) gets all of them and its
leftover goes to the others. Within a frame, ``template_candidate_factor`` x
the share is drawn uniformly
(score_concrete_templates.sample_concrete_template_prefix_suffix) and the
share is picked from it greedily: each pick is the candidate whose fillers
have been used least so far in that frame, so every noun, spatial noun,
complex postposition and complement verb appears in every slot it can fill
before any repeats. Particle slots (이/가, 을/를, 로/으로) follow their noun,
so they are not balanced.

Carrier score. The verb is replaced by ``[MASK][MASK]`` + the word-internal
piece ``##다`` in the sentence as participants see it (다른 for a repeated
noun): the first mask is the stem, the second the past-tense piece.
klue/bert-base splits 70% of the lexicon's past forms exactly this way
(먹/##었/##다, 사랑/##했/##다). The masks are filled in order, so the two
pieces belong to one verb (predicting both at once gave 있+했 -> 있했다):
  1. With both masked, take the ``stem_top_k`` most probable Hangul
     word-initial pieces at the first mask.
  2. For each, fill it in and take the most probable past-tense piece at the
     second: a word-internal Hangul syllable ending in ㅆ (었/았/했/됐/였...),
     the past-tense marker. Without this restriction it picked present
     ##된 (전달된다) and non-verb pieces (##성이).
  3. carrier_score = the best log p(stem) + log p(tense | stem).
The score says how strongly the carrier predicts *some*
past-tense verb, not any lexicon verb, and a low score can mean an open
carrier (사람이 ___) as much as an odd one. It is kept as metadata and used
to match same_verb partners, not to bin or filter. ``carrier_percentile`` is
its rank within the frame class (0 = lowest), since raw scores differ
systematically between classes.

Outputs:
  templates_output   one row per concrete template: frame metadata, rendered
                     prefix/suffix, slot_fillers, concrete_template_id,
                     carrier_score, carrier_best (e.g. 먹었다),
                     carrier_percentile
  templates_summary  per frame: share, candidates drawn, and filler coverage
                     (distinct fillers seen per slot vs available)

Parameters live in config.yaml ``past_test``; CLI flags override.
"""

import argparse
import hashlib
import json
import logging
import random
from collections import Counter, defaultdict
from pathlib import Path

import torch
import yaml
from create_concrete_template_pairs import split_at_verb
from score_concrete_templates import (
    MODEL_NAME,
    ConcreteTemplate,
    load_lexicon_items,
    load_templates,
    sample_concrete_template_prefix_suffix,
)
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from bead.resources.template import Template
from bead.templates.resolver import ConstraintResolver

logger = logging.getLogger(__name__)

# Slots with at most this many distinct lemmas are particle allomorphs
# (이/가, 을/를, 로/으로), determined by the preceding noun.
ALLOMORPH_MAX_LEMMAS = 2


def is_hangul(char: str) -> bool:
    """Return whether ``char`` is a precomposed Hangul syllable."""
    return "가" <= char <= "힣"


def content_slots(candidates: list[ConcreteTemplate]) -> list[str]:
    """Return the slots whose fillers vary beyond particle allomorphy."""
    if not candidates:
        return []
    return [
        name for name in candidates[0].fillers
        if len({ct.fillers[name].lemma for ct in candidates}) > ALLOMORPH_MAX_LEMMAS
    ]


def allocate(n_total: int, available: dict[str, int]) -> dict[str, int]:
    """Split ``n_total`` evenly over frames, capping each at what it has.

    A capped frame's leftover is shared among the rest, repeatedly, until no
    frame is short; the remainder of an uneven split goes to the first frames
    in name order.
    """
    shares: dict[str, int] = {}
    open_frames = sorted(available)
    remaining = n_total
    while open_frames:
        even = remaining // len(open_frames)
        capped = [f for f in open_frames if available[f] <= even]
        if not capped:
            extra = remaining - even * len(open_frames)
            for i, name in enumerate(open_frames):
                shares[name] = even + (1 if i < extra else 0)
            break
        for name in capped:
            shares[name] = available[name]
            remaining -= available[name]
        open_frames = [f for f in open_frames if f not in capped]
    return shares


def pick_balanced(
    candidates: list[ConcreteTemplate], n: int, rng: random.Random
) -> list[ConcreteTemplate]:
    """Pick ``n`` candidates, each time the one with the least-used fillers.

    Usage is counted per (slot, lemma) over the content slots, so fillers
    spread evenly over every slot; ties go to the earlier candidate in a
    random order.
    """
    slots = content_slots(candidates)
    pool = candidates[:]
    rng.shuffle(pool)
    usage: Counter[tuple[str, str]] = Counter()
    picked: list[ConcreteTemplate] = []
    for _ in range(min(n, len(pool))):
        best = min(
            range(len(pool)),
            key=lambda j: sum(usage[(s, pool[j].fillers[s].lemma)] for s in slots),
        )
        ct = pool.pop(best)
        picked.append(ct)
        for s in slots:
            usage[(s, ct.fillers[s].lemma)] += 1
    return picked


def concrete_template_row(template: Template, ct: ConcreteTemplate) -> dict:
    """Build the output row for one concrete template (before scoring)."""
    row: dict = {
        "frame_name": template.name,
        "frame_id": str(template.id),
        "slot_fillers": {
            name: {"id": str(item.id), "lemma": item.lemma, "form": item.form}
            for name, item in ct.fillers.items()
        },
    }
    for key in ("adjunct_tier", "frame_class", "obliques", "aspect", "n_args",
                "comp_type"):
        if key in template.metadata:
            row[key] = template.metadata[key]
    row["cell"] = f"{row['frame_class']}|{row['aspect']}"
    # The text participants see (다른 for a repeated noun), not the raw fill.
    row["prefix"], row["suffix"] = split_at_verb(template, row)
    key = f"{template.name}\t{row['prefix']}\t{row['suffix']}"
    row["concrete_template_id"] = hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
    return row


def has_ssang_siot_final(char: str) -> bool:
    """Return whether a Hangul syllable ends in ㅆ (the past-tense marker)."""
    return is_hangul(char) and (ord(char) - ord("가")) % 28 == 20


def _log_probs_at(
    model: PreTrainedModel,
    tok: PreTrainedTokenizerBase,
    seqs: list[list[int]],
    positions: list[int],
) -> torch.Tensor:
    """Run one padded batch; return log-probabilities at one position per row."""
    width = max(len(s) for s in seqs)
    input_ids = torch.full((len(seqs), width), tok.pad_token_id)
    attention = torch.zeros((len(seqs), width), dtype=torch.long)
    for i, seq in enumerate(seqs):
        input_ids[i, :len(seq)] = torch.tensor(seq)
        attention[i, :len(seq)] = 1
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention).logits
    rows = torch.arange(len(seqs))
    return torch.log_softmax(logits[rows, torch.tensor(positions)], dim=-1)


def carrier_scores(
    tok: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    prefix_suffix: list[tuple[str, str]],
    batch_size: int,
    stem_top_k: int,
) -> list[tuple[float, str]]:
    """Score carriers as ``prefix [MASK][MASK]##다 suffix``; see module docstring.

    Returns, per carrier, the best log p(stem) + log p(tense | stem) and the
    surface the two pieces spell (stem + tense + 다).
    """
    vocab = tok.convert_ids_to_tokens(list(range(model.config.vocab_size)))
    stem_ok = torch.tensor([bool(t) and is_hangul(t[0]) for t in vocab])
    tense_ok = torch.tensor([
        t.startswith("##") and len(t) == 3 and has_ssang_siot_final(t[2])
        for t in vocab
    ])
    da_id = tok.convert_tokens_to_ids("##다")
    mask_id = tok.mask_token_id
    results: list[tuple[float, str]] = []
    for start in range(0, len(prefix_suffix), batch_size):
        batch = prefix_suffix[start:start + batch_size]
        heads, tails = [], []
        for prefix, suffix in batch:
            heads.append(tok(prefix, add_special_tokens=False)["input_ids"])
            tails.append(tok(suffix, add_special_tokens=False)["input_ids"])
        cls, sep = tok.cls_token_id, tok.sep_token_id
        stem_lp = _log_probs_at(
            model, tok,
            [[cls, *h, mask_id, mask_id, da_id, *t, sep]
             for h, t in zip(heads, tails, strict=True)],
            [1 + len(h) for h in heads],
        ).masked_fill(~stem_ok, float("-inf"))
        stem_scores, stem_ids = stem_lp.topk(stem_top_k, dim=-1)
        filled = [(i, int(s)) for i in range(len(batch)) for s in stem_ids[i]]
        tense_lp = _log_probs_at(
            model, tok,
            [[cls, *heads[i], s, mask_id, da_id, *tails[i], sep] for i, s in filled],
            [2 + len(heads[i]) for i, _ in filled],
        ).masked_fill(~tense_ok, float("-inf"))
        tense_scores, tense_ids = tense_lp.max(dim=-1)
        for i in range(len(batch)):
            options = [
                (float(stem_scores[i, k] + tense_scores[i * stem_top_k + k]),
                 int(stem_ids[i, k]), int(tense_ids[i * stem_top_k + k]))
                for k in range(stem_top_k)
            ]
            score, stem_id, tense_id = max(options)
            results.append((score, vocab[stem_id] + vocab[tense_id][2:] + "다"))
    return results


def main(
    output_path: Path,
    summary_path: Path,
    n_templates: int,
    template_candidate_factor: int,
    stem_top_k: int,
    batch_size: int,
    seed: int,
) -> None:
    """Sample, score and write the concrete templates of every simple frame."""
    rng = random.Random(seed)
    items = load_lexicon_items()
    templates = [t for t in load_templates() if t.metadata.get("aspect") == "simple"]
    resolver = ConstraintResolver()
    cache: dict = {}

    base_share = -(-n_templates // len(templates))
    candidates: dict[str, list[ConcreteTemplate]] = {}
    for template in templates:
        candidates[template.name] = sample_concrete_template_prefix_suffix(
            template, items, resolver, cache,
            template_candidate_factor * base_share, rng,
        )
    shares = allocate(n_templates, {n: len(c) for n, c in candidates.items()})
    by_name = {t.name: t for t in templates}

    rows: list[dict] = []
    summary_frames: list[dict] = []
    for name in sorted(shares):
        picked = pick_balanced(candidates[name], shares[name], rng)
        slots = content_slots(candidates[name])
        coverage = {
            s: [len({ct.fillers[s].lemma for ct in picked}),
                len({ct.fillers[s].lemma for ct in candidates[name]})]
            for s in slots
        }
        summary_frames.append({
            "frame_name": name,
            "frame_class": by_name[name].metadata["frame_class"],
            "share": shares[name], "candidates": len(candidates[name]),
            "filler_coverage": coverage,
        })
        rows.extend(concrete_template_row(by_name[name], ct) for ct in picked)
    logger.info(
        f"Sampled {len(rows)} concrete templates from {len(templates)} frames "
        f"(shares {min(shares.values())}-{max(shares.values())})"
    )

    logger.info(f"Loading {MODEL_NAME}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    model.eval()
    scores = carrier_scores(
        tok, model, [(r["prefix"], r["suffix"]) for r in rows], batch_size,
        stem_top_k,
    )
    by_class: dict[str, list[dict]] = defaultdict(list)
    for row, (score, surface) in zip(rows, scores, strict=True):
        row["carrier_score"] = score
        row["carrier_best"] = surface
        by_class[row["frame_class"]].append(row)
    for class_rows in by_class.values():
        class_rows.sort(key=lambda r: (r["carrier_score"], r["concrete_template_id"]))
        for rank, row in enumerate(class_rows):
            row["carrier_percentile"] = rank / max(1, len(class_rows) - 1)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    covered = sum(c[0] for fr in summary_frames for c in fr["filler_coverage"].values())
    available = sum(
        c[1] for fr in summary_frames for c in fr["filler_coverage"].values()
    )
    with open(summary_path, "w") as f:
        json.dump(
            {"n_templates": len(rows), "n_frames": len(templates),
             "score_model": MODEL_NAME, "seed": seed,
             "filler_positions_covered": covered,
             "filler_positions_available": available,
             "frames": summary_frames},
            f, ensure_ascii=False, indent=1,
        )
    logger.info(
        f"Wrote {len(rows)} concrete templates to {output_path}; filler "
        f"positions covered {covered}/{available} (summary: {summary_path})"
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Sample concrete templates per simple-aspect frame and "
        "score each with the two-mask carrier score"
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    cfg = config["past_test"]
    per_list = config["lists"]["split_by"]["per_list"]
    templates_per_list = per_list["different_verb"] + 2 * per_list["same_verb"]

    main(
        output_path=args.output or Path(cfg["templates_output"]),
        summary_path=args.summary or Path(cfg["templates_summary"]),
        n_templates=cfg["pair_candidate_factor"] * cfg["n_lists"] * templates_per_list,
        template_candidate_factor=int(cfg["template_candidate_factor"]),
        stem_top_k=int(cfg["stem_top_k"]),
        batch_size=int(cfg["score_batch_size"]),
        seed=int(cfg["seed"] if args.seed is None else args.seed),
    )
