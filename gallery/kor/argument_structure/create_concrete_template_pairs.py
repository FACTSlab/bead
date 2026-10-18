#!/usr/bin/env python3
"""Build 2AFC items from binned concrete templates.

Terminology: a *frame* is the abstract variant (e.g. ``NP-NOM NP-ACC ___``, a
row of templates/generic_frames.jsonl); a *concrete template* is a frame with
everything filled except the verb and its form (e.g. ``사람이 물건을 ___``).

Input: bin_concrete_templates.py output -- concrete templates selected evenly
across within-cell score bins, each carrying its slot fillers. Replaces the
exhaustive-fill path (fill_templates.py -> create_2afc_pairs.py) and keeps its
pair_type / verb / verb1 / lm_score_diff / quantile fields, so
generate_lists.py's constraints apply; frame references are ``frame_id`` /
``frame_name`` / ``frame1`` / ``frame2``. Two pair types:

- ``different_verb``: one concrete template, two different verbs in the same
  form (same tense for finite frames; the -고 participle for progressives).
  Holds the frame and nouns fixed and varies the verb.
- ``same_verb``: one verb in two concrete templates that share aspect but
  differ in frame_class. Holds the verb fixed and varies the argument
  structure.

Verbs come from the lexicon's verb-slot pool (every lemma with present,
past and participle forms: UniMorph verbs and, unless ``drop_adjectives``,
predicate adjectives), dealt from a shuffled deck so each lemma is used once
before any repeats. Nothing
restricts which verbs or arguments combine -- ungrammatical combinations are
wanted; the experiment decides acceptability.

Each sentence is rendered for participants with
fill_templates.render_template_kor (다른 for repeated nouns) and scored with
score_concrete_templates.score_given_verbs: the verb's log-probability in that
sentence, the same measure the decoder's raw_score uses. That raw score is
mostly the verb's own rarity and length, so it is corrected (rarity_residuals)
before use: ``lm_score_diff`` = |corrected_a - corrected_b|, and pairs get
quantiles of it per pair_type (lists.quantile_bins), as the list constraints
expect. Raw scores and the fit are kept. Option order is randomized. Each
item carries ``rendered_elements`` ``option_a`` / ``option_b`` (read by
bead's forced-choice model in active learning) and the protocol anchor name.

Parameters live in config.yaml ``concrete_template_pairs``; CLI flags override.
"""

import argparse
import json
import logging
import random
from collections import defaultdict
from pathlib import Path
from uuid import UUID

import numpy as np
import yaml
from fill_templates import render_template_kor
from protocol import ACCEPTABILITY_ANCHOR_NAME
from score_concrete_templates import MODEL_NAME, load_templates, score_given_verbs
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from bead.items.forced_choice import create_forced_choice_item
from bead.items.item import Item, MetadataValue
from bead.lists.stratification import assign_quantiles_by_uuid
from bead.resources.lexical_item import LexicalItem
from bead.resources.template import Template

logger = logging.getLogger(__name__)

VERBS_PATH = Path("lexicons/verbs.jsonl")
_SENTINEL = "\x00VERB\x00"


class VerbDeck:
    """Deal verb lemmas evenly: shuffle all, deal in order, reshuffle when empty."""

    def __init__(self, lemmas: list[str], rng: random.Random) -> None:
        self._lemmas = sorted(lemmas)
        self._rng = rng
        self._deck: list[str] = []

    def draw(self, avoid: frozenset[str] = frozenset()) -> str:
        """Return the next lemma not in ``avoid``."""
        for _ in range(2):
            for i in range(len(self._deck) - 1, -1, -1):
                if self._deck[i] not in avoid:
                    return self._deck.pop(i)
            self._deck = self._lemmas[:]
            self._rng.shuffle(self._deck)
        raise ValueError("no lemma available outside `avoid`")


def load_verb_forms(
    path: Path, drop_adjectives: bool = False
) -> dict[str, dict[str, LexicalItem]]:
    """Map lemma -> {PRS, PST, PTCP: item} for lemmas with all three forms.

    One item per form key (the first listed); lemmas lacking a form are
    dropped so every pair can use any form. With ``drop_adjectives``, lemmas
    with any UniMorph ADJ-tagged form (predicate adjectives: 행복하다,
    찬란하다) are left out -- they share the verb slot's pos but are
    ungrammatical in every progressive and accusative-object frame by
    construction.
    """
    forms: dict[str, dict[str, LexicalItem]] = defaultdict(dict)
    adjectives: set[str] = set()
    with open(path) as f:
        for line in f:
            item = LexicalItem.model_validate_json(line)
            features = item.features
            if str(features.get("unimorph_features", "")).startswith("ADJ"):
                adjectives.add(item.lemma)
            if features.get("verb_form") == "V.PTCP":
                key = "PTCP"
            elif features.get("verb_form") is None and features.get("tense"):
                key = str(features["tense"])
            else:
                continue
            forms[item.lemma].setdefault(key, item)
    return {
        lemma: f for lemma, f in forms.items()
        if {"PRS", "PST", "PTCP"} <= set(f)
        and not (drop_adjectives and lemma in adjectives)
    }


def form_key(concrete_template: dict, rng: random.Random) -> str:
    """Pick the verb form key for a pair: PTCP for progressives, else a tense."""
    if concrete_template["aspect"] != "simple":
        return "PTCP"
    return rng.choice(["PRS", "PST"])


def render(
    template: Template, concrete_template: dict, verb: LexicalItem
) -> tuple[str, str]:
    """Render a concrete template with a verb; return (sentence, verb surface)."""
    fillers = {
        name: LexicalItem(
            lemma=f["lemma"], form=f["form"], language_code="kor"
        )
        for name, f in concrete_template["slot_fillers"].items()
    }
    fillers["verb"] = verb
    sentence = render_template_kor(template.template_string, fillers, template.slots)
    return sentence, verb.form or verb.lemma


def split_at_verb(
    template: Template, concrete_template: dict
) -> tuple[str, str]:
    """Return the rendered (prefix, suffix) around the verb, for scoring."""
    sentinel = LexicalItem(lemma=_SENTINEL, form=_SENTINEL, language_code="kor")
    sentence, _ = render(template, concrete_template, sentinel)
    prefix, suffix = sentence.split(_SENTINEL)
    return prefix, suffix


def concrete_template_meta(
    concrete_template: dict, tag: str
) -> dict[str, MetadataValue]:
    """Per-side stratification metadata, keys suffixed with ``tag``.

    Score fields are copied when the concrete template has them: ``bin`` /
    ``n_bins`` / ``raw_score`` from bin_concrete_templates.py, ``carrier_score``
    / ``carrier_percentile`` from sample_concrete_templates.py.
    """
    meta: dict[str, MetadataValue] = {
        f"concrete_template_id{tag}": concrete_template["concrete_template_id"],
        f"frame{tag}": concrete_template["frame_name"],
        f"frame_id{tag}": concrete_template["frame_id"],
        f"cell{tag}": concrete_template["cell"],
        f"frame_class{tag}": concrete_template["frame_class"],
        f"adjunct_tier{tag}": concrete_template["adjunct_tier"],
        f"slot_filler_ids{tag}": {
            name: f["id"] for name, f in concrete_template["slot_fillers"].items()
        },
    }
    for key, field in (
        ("bin", "concrete_template_bin"),
        ("n_bins", "concrete_template_n_bins"),
        ("raw_score", "concrete_template_raw_score"),
        ("carrier_score", "carrier_score"),
        ("carrier_percentile", "carrier_percentile"),
    ):
        if key in concrete_template:
            meta[f"{field}{tag}"] = concrete_template[key]
    return meta


def build_pairs(
    concrete_templates: list[dict],
    templates: dict[str, Template],
    verb_forms: dict[str, dict[str, LexicalItem]],
    n_different: int,
    n_same: int,
    rng: random.Random,
) -> list[dict]:
    """Plan every pair: which concrete templates, which verbs, which form.

    Returns pair specs (not yet scored or turned into Items). A same_verb
    partner is dealt from a per-aspect rotation of concrete templates, skipping ones of
    the same frame_class and preferring the same subject noun, so partners
    spread across frames and differ mainly in argument structure.
    """
    deck = VerbDeck(list(verb_forms), rng)
    by_aspect: dict[str, list[dict]] = defaultdict(list)
    for ct in concrete_templates:
        by_aspect[ct["aspect"]].append(ct)
    partner_queues: dict[str, list[dict]] = {a: [] for a in by_aspect}

    def subject(ct: dict) -> str | None:
        filler = ct["slot_fillers"].get("noun_subj")
        return filler["lemma"] if filler else None

    def next_partner(ct: dict) -> dict:
        """Next concrete template in the aspect's rotation with another frame_class.

        Prefers one with the same subject noun, so the pair differs in
        argument structure rather than in every noun; falls back to any.
        """
        queue = partner_queues[ct["aspect"]]
        for _ in range(2):
            candidates = [
                i for i in range(len(queue) - 1, -1, -1)
                if queue[i]["frame_class"] != ct["frame_class"]
            ]
            same_subject = [i for i in candidates if subject(queue[i]) == subject(ct)]
            if same_subject or candidates:
                return queue.pop((same_subject or candidates)[0])
            queue[:] = by_aspect[ct["aspect"]][:]
            rng.shuffle(queue)
        raise ValueError(f"no partner with another frame_class for {ct['cell']}")

    specs: list[dict] = []
    for ct in concrete_templates:
        template = templates[ct["frame_name"]]
        for _ in range(n_different):
            key = form_key(ct, rng)
            v1 = deck.draw()
            v2 = deck.draw(avoid=frozenset({v1}))
            specs.append({
                "pair_type": "different_verb", "form": key,
                "sides": [(template, ct, verb_forms[v1][key]),
                          (template, ct, verb_forms[v2][key])],
            })
        for _ in range(n_same):
            partner = next_partner(ct)
            key = form_key(ct, rng)
            verb = verb_forms[deck.draw()][key]
            specs.append({
                "pair_type": "same_verb", "form": key,
                "sides": [(template, ct, verb),
                          (templates[partner["frame_name"]], partner, verb)],
            })
    return specs


def rarity_residuals(
    raw: list[float], baseline: list[float], n_tokens: list[int]
) -> tuple[list[float], dict[str, float]]:
    """Remove the verb's own rarity and length from its in-sentence score.

    A verb's log-probability in a sentence is mostly about the verb: in the
    first run, its no-argument ``baseline`` log-probability and token count
    explained 81% of the variance, so different_verb score differences
    sorted pairs by verb rarity. Plain PMI (raw - baseline) over-corrects,
    favoring rare, long verbs (a verb alone is so improbable that any context
    raises it, more so per extra token), and so does per-token PMI (SLOR).
    Instead fit ``raw ~ b0 + b1*baseline + b2*tokens`` by least squares over
    this run's sentences and keep the residual: the part of the score not
    explained by which verb it is, i.e. how much this frame favors this verb
    relative to verbs of similar rarity. Linearly uncorrelated with both by
    construction. same_verb pairs share verb and ending on both sides, so
    their differences are unchanged. Returns the residuals and the fit
    (coefficients, R^2, correlations before/after) for provenance.
    """
    x = np.column_stack([np.ones(len(raw)), baseline, n_tokens])
    y = np.asarray(raw)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta

    def corr(a: np.ndarray, b: list[float] | list[int]) -> float:
        return float(np.corrcoef(a, b)[0, 1])

    fit = {
        "intercept": float(beta[0]),
        "beta_baseline": float(beta[1]),
        "beta_tokens": float(beta[2]),
        "r2": float(1 - resid.var() / y.var()),
        "corr_tokens_raw": corr(y, n_tokens),
        "corr_tokens_corrected": corr(resid, n_tokens),
        "corr_baseline_raw": corr(y, baseline),
        "corr_baseline_corrected": corr(resid, baseline),
    }
    return [float(v) for v in resid], fit


def to_item(spec: dict, scores: tuple[dict, dict], rng: random.Random) -> Item:
    """Turn a scored pair spec into a forced-choice Item (random option order).

    ``scores`` holds, per side, the ``raw``, ``baseline``, ``n_tokens`` and
    rarity-corrected ``score`` (see rarity_residuals).
    """
    sides = list(zip(spec["sides"], scores, strict=True))
    if rng.random() < 0.5:
        sides.reverse()
    (t_a, ct_a, v_a), s_a = sides[0]
    (t_b, ct_b, v_b), s_b = sides[1]
    text_a, surface_a = render(t_a, ct_a, v_a)
    text_b, surface_b = render(t_b, ct_b, v_b)
    metadata: dict[str, MetadataValue] = {
        "anchor": ACCEPTABILITY_ANCHOR_NAME,
        "pair_type": spec["pair_type"],
        "verb_form": spec["form"],
        "aspect": ct_a["aspect"],
        "verb_a": v_a.lemma, "verb_b": v_b.lemma,
        "verb_surface_a": surface_a, "verb_surface_b": surface_b,
        "verb_id_a": str(v_a.id), "verb_id_b": str(v_b.id),
        # lm_score_* is the rarity-corrected score (what lists stratify on);
        # raw/baseline/token count are kept for provenance.
        "lm_score_a": s_a["score"], "lm_score_b": s_b["score"],
        "lm_score_diff": abs(s_a["score"] - s_b["score"]),
        "lm_raw_a": s_a["raw"], "lm_raw_b": s_b["raw"],
        "lm_raw_diff": abs(s_a["raw"] - s_b["raw"]),
        "lm_baseline_a": s_a["baseline"], "lm_baseline_b": s_b["baseline"],
        "lm_n_tokens_a": s_a["n_tokens"], "lm_n_tokens_b": s_b["n_tokens"],
        "lm_score_kind": "rarity_residual",
        "score_model": MODEL_NAME,
        **concrete_template_meta(ct_a, "_a"),
        **concrete_template_meta(ct_b, "_b"),
    }
    if spec["pair_type"] == "different_verb":
        metadata |= {
            "frame_id": ct_a["frame_id"],
            "frame_name": ct_a["frame_name"],
            "verb1": v_a.lemma, "verb2": v_b.lemma,
        }
        item_template_id = UUID(ct_a["frame_id"])
    else:
        metadata |= {
            "frame_id": ct_a["frame_id"],
            "verb": v_a.lemma,
            "frame1": ct_a["frame_name"],
            "frame2": ct_b["frame_name"],
        }
        item_template_id = None
    item = create_forced_choice_item(
        text_a, text_b, item_template_id=item_template_id, metadata=metadata
    )
    # bead's ForcedChoiceModel (active learning) reads each option's text from
    # rendered_elements under the item template's option names, which the
    # active-learning scripts set to option_a / option_b (as in English).
    return item.with_(rendered_elements={"option_a": text_a, "option_b": text_b})


def main(
    input_path: Path,
    output_path: Path,
    n_different: int,
    n_same: int,
    quantile_bins: int,
    batch_size: int,
    seed: int,
    summary_path: Path,
    drop_adjectives: bool = False,
) -> None:
    """Plan, score and write 2AFC pairs from selected concrete templates."""
    rng = random.Random(seed)
    with open(input_path) as f:
        concrete_templates = [json.loads(line) for line in f]
    missing = [s for s in concrete_templates if "slot_fillers" not in s]
    if missing:
        raise SystemExit(
            f"{len(missing)} concrete templates lack slot_fillers; re-run "
            "score_concrete_templates.py and bin_concrete_templates.py"
        )
    templates = {t.name: t for t in load_templates()}
    verb_forms = load_verb_forms(VERBS_PATH, drop_adjectives)
    logger.info(
        f"{len(concrete_templates)} concrete templates, {len(verb_forms)} verb "
        "lemmas with PRS/PST/PTCP forms"
    )

    specs = build_pairs(
        concrete_templates, templates, verb_forms, n_different, n_same, rng
    )
    logger.info(f"Planned {len(specs)} pairs; scoring {2 * len(specs)} sentences")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    model.eval()
    side_scores, fit = score_specs(tok, model, specs, batch_size)

    items = [
        to_item(spec, (side_scores[2 * i], side_scores[2 * i + 1]), rng)
        for i, spec in enumerate(specs)
    ]
    quantiles = assign_quantiles_by_uuid(
        item_ids=[it.id for it in items],
        item_metadata={it.id: it.item_metadata for it in items},
        property_key="lm_score_diff",
        n_quantiles=quantile_bins,
        stratify_by_key="pair_type",
    )
    items = [
        it.with_(item_metadata={**it.item_metadata, "quantile": quantiles[it.id]})
        for it in items
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for item in items:
            f.write(item.model_dump_json() + "\n")
    with open(summary_path, "w") as f:
        json.dump(
            {"n_pairs": len(items), "n_sentences": 2 * len(specs),
             "drop_adjectives": drop_adjectives, "n_verb_lemmas": len(verb_forms),
             "score_model": MODEL_NAME, "rarity_fit": fit},
            f, indent=1,
        )
    counts = defaultdict(int)
    for item in items:
        counts[str(item.item_metadata["pair_type"])] += 1
    verbs_used = {
        str(item.item_metadata[k])
        for item in items for k in ("verb_a", "verb_b")
    }
    logger.info(
        f"Wrote {len(items)} pairs to {output_path}: {dict(counts)}; "
        f"{len(verbs_used)} distinct verb lemmas"
    )


def score_specs(
    tok: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    specs: list[dict],
    batch_size: int,
) -> tuple[list[dict], dict[str, float]]:
    """Score both sides of every pair spec with its real verb.

    Returns one dict per side, in spec order (side a, side b, next spec...),
    holding the ``raw`` in-sentence log-probability, the verb-alone
    ``baseline``, ``n_tokens`` and the rarity-corrected ``score`` (see
    rarity_residuals), plus the correction fit.
    """
    triples: list[tuple[str, str, str]] = []
    for spec in specs:
        for template, ct, verb in spec["sides"]:
            prefix, suffix = split_at_verb(template, ct)
            triples.append((prefix, verb.form or verb.lemma, suffix))
    raw = score_given_verbs(tok, model, triples, batch_size=batch_size)
    # Baseline: the verb with only the sentence's own ending ('.', ' 있다.',
    # ' 있었다.') -- its log-probability with no arguments, i.e. its rarity.
    baseline_keys = sorted({(verb, suffix) for _, verb, suffix in triples})
    baseline_scores = score_given_verbs(
        tok, model, [("", v, sfx) for v, sfx in baseline_keys], batch_size=batch_size
    )
    baseline_of = dict(zip(baseline_keys, baseline_scores, strict=True))
    baseline = [baseline_of[(verb, suffix)] for _, verb, suffix in triples]
    n_tokens = [
        len(tok(verb, add_special_tokens=False)["input_ids"])
        for _, verb, _ in triples
    ]
    corrected, fit = rarity_residuals(raw, baseline, n_tokens)
    logger.info(
        f"Rarity correction: raw ~ {fit['intercept']:.2f} + "
        f"{fit['beta_baseline']:.2f}*baseline + {fit['beta_tokens']:.2f}*tokens "
        f"(R^2 {fit['r2']:.2f}); corr(score, tokens) {fit['corr_tokens_raw']:+.2f} "
        f"-> {fit['corr_tokens_corrected']:+.2f}, corr(score, baseline) "
        f"{fit['corr_baseline_raw']:+.2f} -> {fit['corr_baseline_corrected']:+.2f}"
    )
    side_scores = [
        {"raw": r, "baseline": b, "n_tokens": n, "score": c}
        for r, b, n, c in zip(raw, baseline, n_tokens, corrected, strict=True)
    ]
    return side_scores, fit


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description="Build 2AFC pairs from binned concrete templates"
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--different-verb-per-concrete-template", type=int)
    parser.add_argument("--same-verb-per-concrete-template", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--drop-adjectives", action=argparse.BooleanOptionalAction, default=None,
        help="Leave predicate adjectives out of the verb deck (config: "
        "drop_adjectives).",
    )
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    cfg = config["concrete_template_pairs"]

    def pick(cli_value: object, key: str) -> object:
        """Return the CLI value if given, else the config value."""
        return cfg[key] if cli_value is None else cli_value

    main(
        input_path=Path(str(pick(args.input, "input"))),
        output_path=Path(str(pick(args.output, "output"))),
        n_different=int(str(pick(args.different_verb_per_concrete_template,
                                  "different_verb_per_concrete_template"))),
        n_same=int(str(pick(args.same_verb_per_concrete_template,
                            "same_verb_per_concrete_template"))),
        quantile_bins=int(config["lists"].get("quantile_bins", 10)),
        batch_size=int(cfg["score_batch_size"]),
        seed=int(str(pick(args.seed, "seed"))),
        summary_path=Path(str(pick(args.summary, "scoring_summary"))),
        drop_adjectives=bool(pick(args.drop_adjectives, "drop_adjectives")),
    )
