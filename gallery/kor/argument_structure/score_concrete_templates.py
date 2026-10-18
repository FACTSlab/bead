#!/usr/bin/env python3
"""Score concrete templates via verb masking.

Terminology: a *frame* is the abstract variant (a row of
templates/generic_frames.jsonl, e.g. ``NP-NOM NP-ACC ___``); a *concrete
template* is a frame with every slot filled (bleached nouns, particles)
except the verb and its form (e.g. ``사람이 물건을 ___``). This script scores
concrete templates -- all of them, or a per-cell / per-frame sample.

Why iterative decoding, not a single [MASK]: 94.4% of real finite Korean verb
forms are multi-token under klue/bert-base's WordPiece tokenizer (measured
directly), so a single mask can only ever surface the 5.6% that happen to be
one subword. A single simultaneous multi-mask (e.g. 3x [MASK] read in one
forward pass) was tried and produces ungrammatical output, because each mask
position is predicted with the OTHER mask positions still masked, not
conditioned on what's actually chosen there -- Korean verb conjugation has
real sequential stem-ending dependencies (ㄹ-drop, vowel harmony) that
breaks. Causal-LM beam generation (polyglot-ko) was also tried and degenerated
into non-Korean noise on these short synthetic prefixes. Iterative masked
decoding -- fill one position, substitute the REAL predicted token, mask the
next position (now with genuine bidirectional context: the real stem on one
side, the real sentence-final suffix on the other) -- produces grammatical
output and naturally unifies short and long verb completions in one beam
search, without needing to pick a single fixed mask count in advance.

Generic across every template family (base/adjunct/progressive/spatial/complex/
complement) -- iterates each template's own slots and constraints via
ConstraintResolver, with no per-family branching. Same principle as bead's
MixedFillingStrategy (dispatches by slot name, never by template name) and
generate_templates.py's derive_progressive() (transforms built Templates,
not per-family spec tuples).

Output: concrete_template_scores/concrete_template_scores.jsonl (default; the
per-cell sample goes to cell_sample_scores.jsonl), one row per concrete template:
  {frame_name, prefix, suffix, raw_surface, raw_score, raw_n_tokens,
   beam_mass, beam_n_valid, raw_is_light_verb, beam_had_valid_candidate,
   frame_id, slot_fillers {slot: {id, lemma, form}},
   + the template's adjunct_tier/frame_class/obliques/aspect/n_args}

raw_score (chosen beam path's cumulative log-prob) is the bin score.
raw_surface/raw_score are the first beam candidate (searched in rank order,
no extra forward passes) that's non-empty and not structurally degenerate --
see _is_degenerate -- falling back to the literal top-1 (flagged via
beam_had_valid_candidate=False) if nothing in the beam qualifies.

An earlier version of this script also searched the beam for a
light-verb-excluded "clean" alternative, on the theory that 하다/되다-family
completions (a generic Korean light verb/copula) could inflate a concrete template's
apparent coherence. Re-checked against this mechanism's own output and
dropped: the flag fired on only 4/40 concrete templates in an early quick check,
and only 2/40 overall produced a genuinely different verb family after
exclusion -- the other 2 just swapped to another 하다-inflection the
blacklist didn't happen to list. Not worth the complexity it would take to
make that robust (lemma-based matching instead of a fixed string list;
conditional handling for complement/quotative templates, where 하다 is the
*correct* matrix verb, not a hedge -- exclusion was making results worse
there). raw_is_light_verb is kept as a cheap diagnostic flag for later
inspection, not used to alter the pick. beam_had_valid_candidate uses a
structural check rather than validating against verbs.jsonl (that lexicon has real
coverage gaps -- 아니다/되다 missing entirely, formal -ㅂ니다/습니다 register
not extracted -- that would incorrectly reject good completions).
"""

import argparse
import itertools
import json
import logging
import math
import random
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import NamedTuple, TextIO

import torch
import yaml
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from bead.resources.lexical_item import LexicalItem
from bead.resources.template import Slot, Template
from bead.templates.resolver import ConstraintResolver

logger = logging.getLogger(__name__)

LEXICON_DIR = Path("lexicons")
TEMPLATES_PATH = Path("templates/generic_frames.jsonl")
OUTPUT_PATH = Path("concrete_template_scores/concrete_template_scores.jsonl")
MODEL_NAME = "klue/bert-base"

# Grammaticalized light-verb surface forms, used only for the raw_is_light_verb
# diagnostic flag -- not to exclude or search for an alternative (see module
# docstring for why that was tried and dropped).
LIGHT_VERB_FORMS = {"한다", "합니다", "했다", "하고", "되다", "된다", "됩니다", "됐다"}


def load_lexicon_items() -> list[LexicalItem]:
    """Load every lexical item across all lexicon files into one flat pool."""
    items: list[LexicalItem] = []
    for path in sorted(LEXICON_DIR.glob("*.jsonl")):
        with open(path) as f:
            for line in f:
                items.append(LexicalItem.model_validate_json(line))
    return items


def load_templates() -> list[Template]:
    """Load every generated Template from templates/generic_frames.jsonl."""
    with open(TEMPLATES_PATH) as f:
        return [Template.model_validate_json(line) for line in f]


# Verb forms whose surface must end in a fixed ending, keyed by the verb
# slot's verb_form. The progressive's V.PTCP form is always stem + 고
# (먹고/가고/변하고 있다); without this the decoder keeps filling mask positions
# after the participle is complete (e.g. '[알고가고] 있다', '[변해고 하고] 있었다').
REQUIRED_ENDINGS: dict[str, str] = {"V.PTCP": "고"}
# Finite verb slots (every non-progressive template): all finite forms in the
# lexicon are declaratives ending in 다 (한다/했다/합니다/아닙니다). Without a
# stopping rule the decoder relies on the model padding with '.', and when it
# dislikes the frame it keeps filling positions instead (한다다니다, 한다이다,
# 예 예 예: 10/7,007 pilot concrete templates had no valid verb, 8 in inst+goal frames).
FINITE_ENDING = "다"


# Syllables that only arise from the -아/어 infinitive contraction, never as a
# verb stem's last syllable, so '<syllable>고' is not a participle (변해고 ->
# 변하고, 살아고 -> 살고). Checked against all 2,674 V.PTCP forms in
# verbs.jsonl with zero false rejections. 러 is deliberately absent (이러고/
# 그러고 are real), as are 켜/펴 (켜다/펴다 stems).
_INFINITIVE_SYLLABLES = frozenset(
    {"해", "아", "어", "여", "써", "떠", "꺼", "커", "퍼"}
)
_STEM_FINAL_YEO = frozenset({"켜", "펴"})


def _is_contracted_participle(surface: str) -> bool:
    """Check whether ``surface`` ends in an infinitive-contraction syllable + 고.

    Structural check, not a lexicon lookup, for the same reason as
    _is_degenerate: verbs.jsonl lacks real verbs the model proposes (되다,
    벗어나다). Rejects any open syllable in _INFINITIVE_SYLLABLES or whose
    vowel is ㅘ/ㅙ/ㅝ (봐/돼/줘) or ㅕ (져/쳐/려, except 켜/펴) before 고.
    """
    word = surface.split()[-1] if surface.split() else ""
    if len(word) < 2 or not word.endswith("고"):
        return False
    syllable = word[-2]
    code = ord(syllable) - 0xAC00
    if not 0 <= code < 11172 or code % 28:  # not Hangul, or closed syllable
        return False
    medial = (code // 28) % 21  # ㅘ=9, ㅙ=10, ㅕ=6, ㅝ=14
    return (
        syllable in _INFINITIVE_SYLLABLES
        or medial in (9, 10, 14)
        or (medial == 6 and syllable not in _STEM_FINAL_YEO)
    )


def required_verb_ending(template: Template) -> str | None:
    """Return the ending the verb completion must end with.

    Read off the verb slot's constraint: ``verb_form=='V.PTCP'`` (every
    progressive template) requires 고, any other verb slot is finite and
    requires 다 -- no per-family logic. None only if there's no verb slot.
    """
    verb = template.slots.get("verb")
    if verb is None:
        return None
    for constraint in verb.constraints:
        for verb_form, ending in REQUIRED_ENDINGS.items():
            check = f"self.features.get('verb_form')=='{verb_form}'"
            if check in constraint.expression:
                return ending
    return FINITE_ENDING


def slot_candidates(
    slot: Slot,
    items: list[LexicalItem],
    resolver: ConstraintResolver,
    cache: dict[tuple[str, ...], list[LexicalItem]],
) -> list[LexicalItem]:
    """Filter items by a slot's constraints, cached by expression tuple.

    Many slots across different templates share the identical constraint
    expression (e.g. every noun-type slot uses the same NOUN check, every
    verb slot uses the same finiteness check) -- all 1,111 constraints across
    the 121 templates have empty `context` (verified directly), so a slot's
    candidate list is a pure function of its expression tuple, independent of
    which template/slot it came from. `ConstraintResolver.evaluate_slot_
    constraints` doesn't cache internally and is expensive per call, so
    re-evaluating it against the full ~10,012-item lexicon pool once per
    unique expression tuple (instead of once per slot instance) was the fix
    for a scoring slowdown -- measured >120s wall-clock
    for just 4 templates before this, since the ~20 distinct constraint
    expressions across all 121 templates were each being re-evaluated from
    scratch dozens of times.
    """
    key = tuple(c.expression for c in slot.constraints)
    if key not in cache:
        cache[key] = [
            it for it in items
            if resolver.evaluate_slot_constraints(it, slot.constraints)
        ]
    return cache[key]


class ConcreteTemplate(NamedTuple):
    """A template filled except for its verb slot.

    ``prefix``/``suffix`` are the raw text around ``{verb}`` (surface forms,
    no 다른 rendering); ``fillers`` maps each non-verb slot to its lexical
    item, so the concrete template can be re-rendered or referenced by item id.
    """

    prefix: str
    suffix: str
    fillers: dict[str, LexicalItem]


def iter_concrete_templates(
    template: Template,
    items: list[LexicalItem],
    resolver: ConstraintResolver,
    slot_candidate_cache: dict[tuple[str, ...], list[LexicalItem]],
) -> Iterator[ConcreteTemplate]:
    """Yield every valid concrete template of one template.

    Covers every valid non-verb slot filling, respecting per-slot constraints
    and template-level fc_agree constraints -- generic over whatever slots
    the template happens to have (aux, spatial_noun, complex_postp_*,
    comp_subj, comp_verb_adn/decl, all case markers, ...).
    """
    if "{verb}" not in template.template_string:
        return
    head_tmpl, tail_tmpl = template.template_string.split("{verb}", 1)

    slot_names = [n for n in template.slots if n != "verb"]
    if not slot_names:
        if "{" not in head_tmpl and "{" not in tail_tmpl:
            yield ConcreteTemplate(head_tmpl, tail_tmpl, {})
        return

    per_slot_candidates: dict[str, list[LexicalItem]] = {}
    for name in slot_names:
        slot = template.slots[name]
        cands = slot_candidates(slot, items, resolver, slot_candidate_cache)
        if not cands:
            return
        per_slot_candidates[name] = cands

    for combo in itertools.product(*(per_slot_candidates[n] for n in slot_names)):
        filled = dict(zip(slot_names, combo, strict=True))
        if template.constraints and not resolver.evaluate_template_constraints(
            filled, template.constraints
        ):
            continue
        rendered = _render_concrete_template(head_tmpl, tail_tmpl, filled)
        if rendered is not None:
            yield ConcreteTemplate(*rendered, filled)


def iter_concrete_template_prefix_suffix(
    template: Template,
    items: list[LexicalItem],
    resolver: ConstraintResolver,
    slot_candidate_cache: dict[tuple[str, ...], list[LexicalItem]],
) -> Iterator[tuple[str, str]]:
    """Yield (prefix, suffix) text around the verb slot for one template."""
    for concrete_template in iter_concrete_templates(
        template, items, resolver, slot_candidate_cache
    ):
        yield concrete_template.prefix, concrete_template.suffix


def _render_concrete_template(
    head_tmpl: str, tail_tmpl: str, filled: dict[str, LexicalItem]
) -> tuple[str, str] | None:
    """Substitute non-verb fillers into the text before/after ``{verb}``."""
    head, tail = head_tmpl, tail_tmpl
    for name, item in filled.items():
        # Surface form, not lemma: comp verbs (하는/한다 vs lemma 하다) and
        # the past aux (있었다 vs lemma 있다) differ -- same rule as
        # fill_templates.render_template_kor.
        surface = item.form or item.lemma
        head = head.replace("{" + name + "}", surface)
        tail = tail.replace("{" + name + "}", surface)
    if "{" in head or "{" in tail:
        return None  # unresolved slot -- shouldn't happen, skip defensively
    return head, tail


def sample_concrete_template_prefix_suffix(
    template: Template,
    items: list[LexicalItem],
    resolver: ConstraintResolver,
    slot_candidate_cache: dict[tuple[str, ...], list[LexicalItem]],
    n: int,
    rng: random.Random,
    exclude: set[tuple[str, str]] | None = None,
) -> list[ConcreteTemplate]:
    """Draw up to ``n`` distinct concrete templates uniformly at random from a template.

    Uniform over the same valid fillings iter_concrete_templates enumerates: each
    non-verb slot is drawn independently and the draw is rejected unless the
    template constraints (fc_agree) hold. Small templates (at most 4n raw
    combinations) are enumerated and sampled instead, which also returns all
    of them when fewer than n exist. Concrete templates whose (prefix, suffix) is in
    ``exclude`` (already drawn) are never returned, so repeated calls top up
    a template's sample without duplicates.
    """
    exclude = exclude or set()
    if "{verb}" not in template.template_string:
        return []
    slot_names = [name for name in template.slots if name != "verb"]
    per_slot = {
        name: slot_candidates(template.slots[name], items, resolver,
                              slot_candidate_cache)
        for name in slot_names
    }
    if math.prod(len(c) for c in per_slot.values()) <= 4 * n:
        every = [
            ct for ct in iter_concrete_templates(
                template, items, resolver, slot_candidate_cache
            )
            if (ct.prefix, ct.suffix) not in exclude
        ]
        return rng.sample(every, min(n, len(every)))

    head_tmpl, tail_tmpl = template.template_string.split("{verb}", 1)
    seen: set[tuple[str, str]] = set(exclude)
    out: list[ConcreteTemplate] = []
    for _ in range(200 * n):
        if len(out) >= n:
            break
        filled = {name: rng.choice(per_slot[name]) for name in slot_names}
        if template.constraints and not resolver.evaluate_template_constraints(
            filled, template.constraints
        ):
            continue
        rendered = _render_concrete_template(head_tmpl, tail_tmpl, filled)
        if rendered is not None and rendered not in seen:
            seen.add(rendered)
            out.append(ConcreteTemplate(*rendered, filled))
    return out


def plan_cell_samples(
    templates: list[Template],
    items: list[LexicalItem],
    resolver: ConstraintResolver,
    slot_candidate_cache: dict[tuple[str, ...], list[LexicalItem]],
    n_per_cell: int,
    rng: random.Random,
) -> dict[str, list[ConcreteTemplate]]:
    """Draw ``n_per_cell`` concrete templates per frame_class x aspect cell.

    Binning happens within a cell, so each cell gets the same sample size
    regardless of how many templates share it (per-template sampling gave
    single-template cells 50 concrete templates and unstable bin edges, and 16-template
    cells 800). Within a cell the target is split evenly across its templates,
    so a template with many fillings doesn't crowd out the others; a template
    that runs out of fillings hands its unused share to the rest of the cell
    over further rounds. Cells with fewer valid fillings than ``n_per_cell``
    in total (e.g. NP_V, 19 per aspect) get all of them. Returns template
    name -> its drawn concrete templates.
    """
    cells: dict[tuple[str, str], list[Template]] = {}
    for template in templates:
        key = (template.metadata["frame_class"], template.metadata["aspect"])
        cells.setdefault(key, []).append(template)

    plan: dict[str, list[ConcreteTemplate]] = {}
    for (frame_class, aspect), cell_templates in cells.items():
        drawn: dict[str, list[ConcreteTemplate]] = {t.name: [] for t in cell_templates}
        still_open = list(cell_templates)
        remaining = n_per_cell
        while remaining > 0 and still_open:
            # Even split; the remainder goes to a random subset of templates.
            order = still_open[:]
            rng.shuffle(order)
            base, extra = divmod(remaining, len(order))
            next_open: list[Template] = []
            progressed = False
            for k, template in enumerate(order):
                share = base + (1 if k < extra else 0)
                if share == 0:
                    next_open.append(template)
                    continue
                got = sample_concrete_template_prefix_suffix(
                    template, items, resolver, slot_candidate_cache, share, rng,
                    exclude={(ct.prefix, ct.suffix) for ct in drawn[template.name]},
                )
                drawn[template.name] += got
                progressed = progressed or bool(got)
                if len(got) == share:
                    next_open.append(template)  # not exhausted
            remaining = n_per_cell - sum(len(v) for v in drawn.values())
            still_open = next_open
            if not progressed:
                break
        total = sum(len(v) for v in drawn.values())
        logger.info(
            f"cell {frame_class} x {aspect}: {total} concrete templates from "
            f"{len(cell_templates)} template(s)"
            + ("" if total >= n_per_cell else " (all available fillings)")
        )
        plan.update(drawn)
    return plan


# Standalone copula forms. 이다 always attaches to a noun (사람이다, 문제입니다),
# so as the first word of a verb completion after a space it is a decoding
# artifact (e.g. '사건에 반해서 입니다'), not a predicate.
_STANDALONE_COPULA = frozenset(
    {"입니다", "이다", "이었다", "였다", "이에요", "예요", "이야", "입니까", "이며"}
)


_FIRST_STEP_WIDEN = 4

# A finished declarative (-다/-니다) followed by the copula: 한다이다, 합니다이다.
# Seen in 13/7,007 pilot completions.
_FINITE_PLUS_COPULA = re.compile(r"다(이다|입니다|이었다|였다)$")


def _is_bad_first_token(token: str, tok: PreTrainedTokenizerBase) -> bool:
    """Check whether ``token`` can't begin a verb completion.

    Structural, verb-side only (no restriction on which arguments fill the
    concrete template): a word-internal '##' piece, punctuation (the path that pads
    '...' and decodes to an empty verb), a special token, or a standalone
    copula (_STANDALONE_COPULA).
    """
    return (
        token.startswith("##")
        or token in tok.all_special_tokens
        or not any(ch.isalnum() for ch in token)
        or token in _STANDALONE_COPULA
    )


def _ending_token_ids(tok: PreTrainedTokenizerBase, ending: str) -> torch.Tensor:
    """Vocabulary ids whose token text (minus any '##') ends in ``ending``."""
    return torch.tensor([
        idx for t, idx in tok.get_vocab().items()
        if t.removeprefix("##").endswith(ending) and t not in tok.all_special_tokens
    ])


def iterative_decode_batch(
    tok: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    prefix_suffix_pairs: list[tuple[str, str]],
    k: int,
    beam_width: int,
    required_ending: str | None = None,
) -> list[list[tuple[list[int], float]]]:
    """Run iterative masked beam decoding of the verb span for concrete templates.

    At each of k steps: build one [MASK] immediately after whatever's been
    decoded so far (real substituted tokens, not further masks) for every
    live beam path across every concrete template, run ONE batched forward
    pass, and expand+prune each concrete template's beam independently.
    Returns, per concrete template, its final beam: a list of
    (token_ids, cumulative_log_prob) sorted best-first.

    With ``required_ending`` (e.g. '고' for progressive participles), decoding
    keeps finished hypotheses separately, as in standard beam search: a path
    whose surface ends in the ending is finished and stops expanding (the
    participle's stopping rule, the counterpart of finite verbs padding with
    '.', which can't work in front of 있다). Besides the top-``beam_width``
    expansions, each live path also proposes its best few *completing*
    tokens, so every concrete template gets a finished candidate even when the model
    prefers another construction -- e.g. resultative -아/어 있다 ('달려 있다'),
    which shares the ``… [V] 있다`` surface: without this, ~15% of progressive
    concrete templates ended with no -고 candidate at all. A dispreferred progressive
    then gets a low score, which is the intended signal. Finished paths that
    fail _is_contracted_participle (변해고) are dropped. Returns finished
    paths first (best-first), then any unfinished live paths.
    """
    n_items = len(prefix_suffix_pairs)
    beams: list[list[tuple[list[int], float]]] = [[([], 0.0)] for _ in range(n_items)]
    finished: list[list[tuple[list[int], float]]] = [[] for _ in range(n_items)]
    ending_ids = (
        _ending_token_ids(tok, required_ending) if required_ending is not None
        else None
    )

    def surface_of(tok_ids: list[int]) -> str:
        return tok.convert_tokens_to_string(tok.convert_ids_to_tokens(tok_ids))

    def add(i: int, tok_ids: list[int], score: float,
            live: list[list[tuple[list[int], float]]]) -> None:
        """Route an expanded path to finished (if complete and valid) or live."""
        if len(tok_ids) == 1 and _is_bad_first_token(
            tok.convert_ids_to_tokens(tok_ids[0]), tok
        ):
            return
        if required_ending is None:
            live[i].append((tok_ids, score))
            return
        surface = surface_of(tok_ids)
        # The bare ending (adverb 다 "all", or 고) isn't a verb yet; it stays
        # live so it can continue (다 + ##릅 + ##니다 = 다릅니다).
        if surface.endswith(required_ending) and surface != required_ending:
            if not _is_contracted_participle(surface):
                finished[i].append((tok_ids, score))
        else:
            live[i].append((tok_ids, score))

    for _ in range(k):
        batch_texts: list[str] = []
        batch_item_idx: list[int] = []
        batch_path_idx: list[int] = []
        for i, (prefix, suffix) in enumerate(prefix_suffix_pairs):
            for path_idx, (tok_ids, _score) in enumerate(beams[i]):
                decoded = surface_of(tok_ids) if tok_ids else ""
                batch_texts.append(prefix + decoded + tok.mask_token + suffix)
                batch_item_idx.append(i)
                batch_path_idx.append(path_idx)

        if not batch_texts:
            break  # no live paths left anywhere
        enc = tok(batch_texts, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**enc).logits
        mask_positions = enc["input_ids"] == tok.mask_token_id

        new_candidates: list[list[tuple[list[int], float]]] = [
            [] for _ in range(n_items)
        ]
        for row in range(len(batch_texts)):
            pos = mask_positions[row].nonzero(as_tuple=True)[0]
            if len(pos) == 0:
                continue
            item_i = batch_item_idx[row]
            tok_ids, score = beams[item_i][batch_path_idx[row]]
            probs = torch.softmax(logits[row, pos[0]], dim=-1)
            # First step looks deeper: _is_bad_first_token can reject several
            # of the top picks ('.', '##…', 입니다), and an all-rejected first
            # step would leave the beam empty. Later steps are unchanged.
            n_top = beam_width if tok_ids else beam_width * _FIRST_STEP_WIDEN
            top = torch.topk(probs, n_top)
            proposals = list(
                zip(top.indices.tolist(), top.values.tolist(), strict=True)
            )
            if ending_ids is not None:
                ending_probs = probs[ending_ids]
                best = torch.topk(ending_probs, beam_width)
                proposals += [
                    (int(ending_ids[j]), p)
                    for j, p in zip(best.indices.tolist(), best.values.tolist(),
                                    strict=True)
                ]
            seen: set[int] = set()
            for idx, p in proposals:
                if p <= 0 or idx in seen:
                    continue
                seen.add(idx)
                add(item_i, tok_ids + [idx], score + math.log(p), new_candidates)

        for i in range(n_items):
            new_candidates[i].sort(key=lambda x: -x[1])
            if required_ending is None:
                beams[i] = new_candidates[i][:beam_width] or beams[i]
            else:
                beams[i] = new_candidates[i][:beam_width]
                finished[i].sort(key=lambda x: -x[1])
                finished[i] = finished[i][:beam_width]

    if required_ending is None:
        return beams
    # Never return an empty beam (summarize_beam reads beam[0]); an empty
    # path falls through to its unflagged fallback.
    return [finished[i] + beams[i] or [([], 0.0)] for i in range(n_items)]


def score_given_verbs(
    tok: PreTrainedTokenizerBase,
    model: PreTrainedModel,
    triples: list[tuple[str, str, str]],
    batch_size: int = 64,
) -> list[float]:
    """Score a given verb in its sentence: sum of log p(token | context).

    ``triples`` are (prefix, verb_surface, suffix). Each verb token is scored
    with a [MASK] in its place, the real preceding verb tokens substituted and
    the sentence continuing after -- the same conditioning
    iterative_decode_batch uses to score a path, so a given verb's score is
    comparable to the decoder's best-verb ``raw_score`` for the concrete template.
    """
    token_ids = [
        tok(verb, add_special_tokens=False)["input_ids"] for _, verb, _ in triples
    ]
    scores = [0.0] * len(triples)
    # Step-major over token positions, flattened into batches of rows.
    jobs = [(i, t) for i, ids in enumerate(token_ids) for t in range(len(ids))]
    for start in range(0, len(jobs), batch_size):
        batch = jobs[start:start + batch_size]
        texts = []
        for i, t in batch:
            prefix, _, suffix = triples[i]
            done = token_ids[i][:t]
            decoded = (
                tok.convert_tokens_to_string(tok.convert_ids_to_tokens(done))
                if done else ""
            )
            texts.append(prefix + decoded + tok.mask_token + suffix)
        enc = tok(texts, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**enc).logits
        mask_positions = enc["input_ids"] == tok.mask_token_id
        for row, (i, t) in enumerate(batch):
            pos = mask_positions[row].nonzero(as_tuple=True)[0][0]
            log_probs = torch.log_softmax(logits[row, pos], dim=-1)
            scores[i] += float(log_probs[token_ids[i][t]])
    return scores


def _decoded_tokens(tok: PreTrainedTokenizerBase, token_ids: list[int]) -> list[str]:
    """Decode a beam path's token ids to a token list.

    Strips any trailing '.' tokens the model chose to pad unused mask
    positions with.
    """
    toks = tok.convert_ids_to_tokens(token_ids)
    while toks and toks[-1] == ".":
        toks = toks[:-1]
    return toks


def _is_degenerate(toks: list[str], tok: PreTrainedTokenizerBase) -> bool:
    """Flag decode artifacts that aren't a real Korean word, structurally.

    Three failure modes found during the session-9 quick check and its
    verification: a literal [UNK] token; an immediately-repeated character in
    the detokenized surface (e.g. '정해져'+'##져' -> '정해져져' -- the second
    token duplicates the *tail* of the first, so checking whole-token
    equality misses it, but checking the surface string's adjacent
    characters catches it directly); and a whole decoded token/morpheme
    repeated anywhere in the span (e.g. '따로 따로 따로', found when
    verifying the fix above -- excluding the first artifact surfaced this
    related one in the same beam). Both checks were verified against every
    known-good completion from this session's testing with zero false
    positives. Deliberately NOT a lexicon-membership check against
    verbs.jsonl: that lexicon has real coverage gaps (아니다/되다 aren't in
    it at all, and only one declarative register was extracted for 하다 --
    합니다 is missing) that would incorrectly reject genuinely good
    completions validated earlier this session (아닙니다, 합니다). A
    structural check has no such false-positive risk.
    """
    if tok.unk_token in toks:
        return True
    stripped = [t[2:] if t.startswith("##") else t for t in toks]
    if len(stripped) != len(set(stripped)):
        return True
    surface = tok.convert_tokens_to_string(toks)
    if _FINITE_PLUS_COPULA.search(surface):
        return True  # e.g. 한다이다: a finished -다 verb can't take the copula
    return any(
        a == b for a, b in zip(surface, surface[1:], strict=False) if a != " "
    )


def _is_valid_completion(
    tok: PreTrainedTokenizerBase, toks: list[str], required_ending: str | None
) -> bool:
    """Check a decoded completion: non-empty, not degenerate, correct ending."""
    if not toks or _is_degenerate(toks, tok):
        return False
    if required_ending is None:
        return True
    surface = tok.convert_tokens_to_string(toks)
    return (
        surface.endswith(required_ending)
        and surface != required_ending
        and not _is_contracted_participle(surface)
    )


def summarize_beam(
    tok: PreTrainedTokenizerBase,
    beam: list[tuple[list[int], float]],
    required_ending: str | None = None,
) -> dict:
    """Summarize a beam's top valid path, plus the beam's total valid mass.

    Searches the beam (already computed, no extra forward passes) in rank
    order for the first candidate that's non-empty and not structurally
    degenerate (see _is_degenerate); falls back to the literal top-1 if none
    qualify, flagged via beam_had_valid_candidate. Also reports a cheap
    light-verb diagnostic flag (not used to alter the pick -- see module
    docstring). With ``required_ending``, a candidate is only valid if its
    surface ends in it and isn't a contracted pseudo-participle (변해고).

    ``beam_mass`` is log(sum of probabilities) over every valid path in the
    beam: how much probability the model puts on *some* acceptable verb
    rather than on its single best one, so a concrete template that many verbs fit
    isn't penalized for spreading probability across them. Bounded by the
    beam (at most ``beam_width`` finished paths), so it is a lower bound on
    the true mass. ``raw_n_tokens`` is the chosen path's token count, for
    length-normalized comparisons. Both are None when no path is valid.
    """
    chosen_ids, chosen_score, had_valid = beam[0][0], beam[0][1], False
    valid_scores: list[float] = []
    for ids, score in beam:
        if _is_valid_completion(tok, _decoded_tokens(tok, ids), required_ending):
            if not had_valid:
                chosen_ids, chosen_score, had_valid = ids, score, True
            valid_scores.append(score)

    raw_surface = tok.convert_tokens_to_string(_decoded_tokens(tok, chosen_ids))
    beam_mass = None
    if valid_scores:
        top = max(valid_scores)
        beam_mass = top + math.log(sum(math.exp(v - top) for v in valid_scores))
    return {
        "raw_surface": raw_surface,
        "raw_score": chosen_score,
        "raw_n_tokens": len(_decoded_tokens(tok, chosen_ids)) if had_valid else None,
        "beam_mass": beam_mass,
        "beam_n_valid": len(valid_scores),
        "raw_is_light_verb": raw_surface in LIGHT_VERB_FORMS,
        "beam_had_valid_candidate": had_valid,
    }


def main(
    limit: int | None,
    chunk_size: int,
    beam_width: int,
    max_verb_tokens: int,
    output_path: Path,
    sample_per_template: int | None = None,
    seed: int = 0,
    sample_per_cell: int | None = None,
) -> None:
    """Score concrete templates across all templates and write results as JSONL.

    Every concrete template by default; with ``sample_per_template``, a seeded uniform
    random sample of that many per template (see
    sample_concrete_template_prefix_suffix); with ``sample_per_cell``, that many per
    frame_class x aspect cell, split evenly across the cell's templates (see
    plan_cell_samples). Each row carries the template's
    stratification metadata (adjunct_tier, frame_class, obliques,
    aspect, n_args) for binning.
    """
    rng = random.Random(seed)
    items = load_lexicon_items()
    templates = load_templates()
    resolver = ConstraintResolver()

    logger.info(f"Loading {MODEL_NAME}...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
    model.eval()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    templates_touched: set[str] = set()
    slot_candidate_cache: dict[tuple[str, ...], list[LexicalItem]] = {}
    if sample_per_template and sample_per_cell:
        raise ValueError("use either sample_per_template or sample_per_cell")
    cell_plan = (
        plan_cell_samples(
            templates, items, resolver, slot_candidate_cache, sample_per_cell, rng
        )
        if sample_per_cell
        else None
    )

    def flush(
        chunk: list[ConcreteTemplate],
        out: TextIO,
        required_ending: str | None,
        template: Template,
    ) -> bool:
        """Decode and write one chunk (one template). True if the limit was hit."""
        nonlocal n_written
        beams = iterative_decode_batch(
            tok, model, [(ct.prefix, ct.suffix) for ct in chunk],
            max_verb_tokens, beam_width, required_ending,
        )
        for concrete_template, beam in zip(chunk, beams, strict=True):
            row = summarize_beam(tok, beam, required_ending)
            row["frame_name"] = template.name
            row["frame_id"] = str(template.id)
            for key in ("adjunct_tier", "frame_class", "obliques", "aspect",
                        "n_args", "comp_type"):
                if key in template.metadata:
                    row[key] = template.metadata[key]
            row["prefix"] = concrete_template.prefix
            row["suffix"] = concrete_template.suffix
            # Stand-off refs to the lexical items filling each slot, plus
            # their surface so the row stays readable after lexicon regen.
            row["slot_fillers"] = {
                name: {"id": str(item.id), "lemma": item.lemma, "form": item.form}
                for name, item in concrete_template.fillers.items()
            }
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_written += 1
            templates_touched.add(template.name)
            if limit and n_written >= limit:
                return True
        return False

    with open(output_path, "w") as out:
        for template in templates:
            ending = required_verb_ending(template)
            concrete_templates: Iterable[ConcreteTemplate]
            if cell_plan is not None:
                concrete_templates = cell_plan.get(template.name, [])
            elif sample_per_template:
                concrete_templates = sample_concrete_template_prefix_suffix(
                    template, items, resolver, slot_candidate_cache,
                    sample_per_template, rng,
                )
            else:
                concrete_templates = iter_concrete_templates(
                    template, items, resolver, slot_candidate_cache
                )
            chunk: list[ConcreteTemplate] = []
            for concrete_template in concrete_templates:
                chunk.append(concrete_template)
                if len(chunk) >= chunk_size:
                    if flush(chunk, out, ending, template):
                        logger.info(f"Reached limit={limit}, stopping.")
                        _report(n_written, templates_touched, templates, output_path)
                        return
                    chunk = []
            if chunk and flush(chunk, out, ending, template):
                logger.info(f"Reached limit={limit}, stopping.")
                _report(n_written, templates_touched, templates, output_path)
                return

    _report(n_written, templates_touched, templates, output_path)


def _report(
    n_written: int,
    templates_touched: set[str],
    templates: list[Template],
    output_path: Path,
) -> None:
    logger.info(f"Wrote {n_written} scored concrete templates to {output_path}")
    logger.info(f"Templates touched: {len(templates_touched)}/{len(templates)}")


_DEFAULTS: dict[str, object] = {
    "output": str(OUTPUT_PATH), "sample_per_cell": None, "sample_per_template": None,
    "seed": 0, "chunk_size": 16, "beam_width": 5, "max_verb_tokens": 3,
}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(
        description=(
            "Score concrete templates (frames filled except for the verb) via "
            "iterative masked beam decoding of the verb span. Defaults come from "
            "config.yaml concrete_template_scoring; flags override."
        )
    )
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Stop after N scored concrete templates."
    )
    parser.add_argument(
        "--chunk-size", type=int, default=None,
        help="Concrete templates decoded together per beam-search chunk."
    )
    parser.add_argument("--beam-width", type=int, default=None)
    parser.add_argument("--max-verb-tokens", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--sample-per-template", type=int, default=None,
        help="Score a seeded uniform random sample of N concrete templates per "
        "frame instead of all of them.",
    )
    parser.add_argument(
        "--sample-per-cell", type=int, default=None,
        help="Score N concrete templates per frame_class x aspect cell, split "
        "evenly across the cell's frames (the binning unit).",
    )
    parser.add_argument(
        "--exhaustive", action="store_true",
        help="Ignore configured sampling and score every concrete template "
        "(~5.5M; not needed).",
    )
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    section: dict[str, object] = {}
    if args.config.exists():
        with open(args.config) as f:
            section = (yaml.safe_load(f) or {}).get("concrete_template_scoring") or {}

    def pick(cli_value: object, key: str) -> object:
        """Return the CLI value if given, else config, else the built-in default."""
        if cli_value is not None:
            return cli_value
        return section.get(key, _DEFAULTS[key])

    per_cell = pick(args.sample_per_cell, "sample_per_cell")
    per_template = pick(args.sample_per_template, "sample_per_template")
    if args.exhaustive:
        per_cell = per_template = None
    elif args.sample_per_template is not None:
        per_cell = None  # an explicit per-frame sample replaces the cell default
    main(
        limit=args.limit,
        chunk_size=int(str(pick(args.chunk_size, "chunk_size"))),
        beam_width=int(str(pick(args.beam_width, "beam_width"))),
        max_verb_tokens=int(str(pick(args.max_verb_tokens, "max_verb_tokens"))),
        output_path=Path(str(pick(args.output, "output"))),
        sample_per_template=int(str(per_template)) if per_template else None,
        seed=int(str(pick(args.seed, "seed"))),
        sample_per_cell=int(str(per_cell)) if per_cell else None,
    )
