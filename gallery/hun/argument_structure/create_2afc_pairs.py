#!/usr/bin/env python3
"""Create controlled Hungarian two-alternative forced-choice pairs.

The pipeline keeps the two broad ENG/KOR pair families:

1. same verb, controlled frame contrast;
2. same frame and fillers, different matrix verb.

Same-verb pairs use explicit contrast specifications so shared fillers remain
constant and the intended structural change is auditable. Different-verb pairs
hold the frame, non-verb fillers, and relevant inflectional features constant.

Verb-frame incompatibility is allowed because it is part of the acceptability
signal. Surface, morphology, or uncontrolled-filler errors are not.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import yaml
from rich.console import Console
from rich.table import Table

from bead.items.item import Item
from bead.items.scoring import LanguageModelScorer
from bead.lists.stratification import assign_quantiles_by_uuid
from bead.templates.filler import FilledTemplate


console = Console()


@dataclass(frozen=True)
class ContrastSpec:
    """One controlled same-verb comparison between two template types."""

    left_template: str
    right_template: str
    contrast_type: str
    description: str
    allow_verb_form_change: bool = False
    allow_tense_change: bool = False


# These are the Stage-1 comparisons that have a clear interpretation. The
# shared subject/object/etc. must match before a pair is created.
SAME_VERB_CONTRASTS = [
    ContrastSpec(
        'subj_nom-verb.',
        'subj_nom-noun_dat-verb.',
        'add_dative',
        'Add a dative-marked dependent',
    ),
    ContrastSpec(
        'subj_nom-verb.',
        'subj_nom-noun_loc-verb.',
        'add_locative',
        'Add a neutral superessive locative dependent',
    ),
    ContrastSpec(
        'subj_nom-verb.',
        'subj_nom-noun_inst-verb.',
        'add_instrumental',
        'Add an instrumental/comitative dependent',
    ),
    ContrastSpec(
        'subj_nom-noun_inst-verb.',
        'subj_nom-noun_inst-noun_loc-verb.',
        'add_locative',
        'Add a locative while keeping the instrumental',
    ),
    ContrastSpec(
        'subj_nom-noun_inst-verb.',
        'subj_nom-noun_inst-noun_dat-verb.',
        'add_dative',
        'Add a dative while keeping the instrumental',
    ),
    ContrastSpec(
        'subj_nom-noun_loc-verb.',
        'subj_nom-noun_loc-noun_dat-verb.',
        'add_dative',
        'Add a dative while keeping the locative',
    ),
    ContrastSpec(
        'tr-indef.',
        'tr_dat-indef.',
        'add_dative',
        'Add a dative to an indefinite-object transitive frame',
    ),
    ContrastSpec(
        'tr-indef.',
        'tr_loc-indef.',
        'add_locative',
        'Add a locative to an indefinite-object transitive frame',
    ),
    ContrastSpec(
        'tr-indef.',
        'tr_ins-indef.',
        'add_instrumental',
        'Add an instrumental to an indefinite-object transitive frame',
    ),
    ContrastSpec(
        'tr_ins-indef.',
        'tr_ins_loc-indef.',
        'add_locative',
        'Add a locative while keeping the instrumental and object',
    ),
    ContrastSpec(
        'tr_ins-indef.',
        'tr_ins_dat-indef.',
        'add_dative',
        'Add a dative while keeping the instrumental and object',
    ),
    ContrastSpec(
        'tr_loc-indef.',
        'tr_loc_dat-indef.',
        'add_dative',
        'Add a dative while keeping the locative and object',
    ),
    ContrastSpec(
        'tr-def.',
        'tr_dat-def.',
        'add_dative',
        'Add a dative to a definite-object transitive frame',
    ),
    ContrastSpec(
        'tr-def.',
        'tr_loc-def.',
        'add_locative',
        'Add a locative to a definite-object transitive frame',
    ),
    ContrastSpec(
        'tr-def.',
        'tr_ins-def.',
        'add_instrumental',
        'Add an instrumental to a definite-object transitive frame',
    ),
    ContrastSpec(
        'tr_ins-def.',
        'tr_ins_loc-def.',
        'add_locative',
        'Add a locative while keeping the instrumental and definite object',
    ),
    ContrastSpec(
        'tr_ins-def.',
        'tr_ins_dat-def.',
        'add_dative',
        'Add a dative while keeping the instrumental and definite object',
    ),
    ContrastSpec(
        'tr_loc-def.',
        'tr_loc_dat-def.',
        'add_dative',
        'Add a dative while keeping the locative and definite object',
    ),
    ContrastSpec(
        'tr-indef.',
        'tr-def.',
        'object_definiteness',
        'Indefinite versus definite direct object',
        allow_verb_form_change=True,
    ),
    ContrastSpec(
        'tr_dat-indef.',
        'tr_dat-def.',
        'object_definiteness',
        'Indefinite versus definite object with a dative',
        allow_verb_form_change=True,
    ),
    ContrastSpec(
        'tr_loc-indef.',
        'tr_loc-def.',
        'object_definiteness',
        'Indefinite versus definite object with a locative',
        allow_verb_form_change=True,
    ),
    ContrastSpec(
        'tr_ins-indef.',
        'tr_ins-def.',
        'object_definiteness',
        'Indefinite versus definite object with an instrumental',
        allow_verb_form_change=True,
    ),
    ContrastSpec(
        'tr_ins_loc-indef.',
        'tr_ins_loc-def.',
        'object_definiteness',
        'Indefinite versus definite object with instrumental + locative',
        allow_verb_form_change=True,
    ),
    ContrastSpec(
        'tr_ins_dat-indef.',
        'tr_ins_dat-def.',
        'object_definiteness',
        'Indefinite versus definite object with instrumental + dative',
        allow_verb_form_change=True,
    ),
    ContrastSpec(
        'tr_loc_dat-indef.',
        'tr_loc_dat-def.',
        'object_definiteness',
        'Indefinite versus definite object with locative + dative',
        allow_verb_form_change=True,
    ),
    ContrastSpec(
        'subj_nom-verb.',
        'subj_nom-verb-hogy_ind.',
        'clausal_complement',
        "Add a finite direct-object hogy ('that') complement",
        allow_verb_form_change=True,
    ),
    ContrastSpec(
        'subj_nom-verb.',
        'subj_nom-eppen-verb_prs.',
        'ongoing_context',
        "Plain present versus éppen ('right now') present",
    ),
    ContrastSpec(
        'tr-indef.',
        'subj_nom-eppen-obj_acc-indf-verb_prs.',
        'ongoing_context',
        'Plain versus éppen with an indefinite object',
    ),
    ContrastSpec(
        'tr-def.',
        'subj_nom-eppen-obj_acc-def-verb_prs.',
        'ongoing_context',
        'Plain versus éppen with a definite object',
    ),
    ContrastSpec(
        'subj_nom-eppen-verb_prs.',
        'subj_nom-eppen-verb_pst.',
        'tense',
        'Present versus past in an éppen context',
        allow_verb_form_change=True,
        allow_tense_change=True,
    ),
    ContrastSpec(
        'subj_nom-eppen-obj_acc-indf-verb_prs.',
        'subj_nom-eppen-obj_acc-indf-verb_pst.',
        'tense',
        'Present versus past with éppen and an indefinite object',
        allow_verb_form_change=True,
        allow_tense_change=True,
    ),
    ContrastSpec(
        'subj_nom-eppen-obj_acc-def-verb_prs.',
        'subj_nom-eppen-obj_acc-def-verb_pst.',
        'tense',
        'Present versus past with éppen and a definite object',
        allow_verb_form_change=True,
        allow_tense_change=True,
    ),

    # ------------------------------------------------------------------
    # Korean-parallel adjuncts.
    #
    # These follow exactly the same shape as the add_dative/add_locative
    # contrasts above: the right-hand frame adds one oblique dependent and
    # nothing else changes. add_comitative is worth reading carefully — the
    # comitative and instrumental frames share Hungarian INS, so the two
    # differ only in the semantic class of the noun, not in case.
    # ------------------------------------------------------------------
    ContrastSpec(
        'subj_nom-verb.',
        'subj_nom-noun_goal-verb.',
        'add_goal',
        'Add a sublative goal dependent',
    ),
    ContrastSpec(
        'subj_nom-verb.',
        'subj_nom-noun_src-verb.',
        'add_source',
        'Add a delative source dependent',
    ),
    ContrastSpec(
        'subj_nom-verb.',
        'subj_nom-noun_com-verb.',
        'add_comitative',
        'Add a comitative dependent (INS on a human noun)',
    ),
    ContrastSpec(
        'subj_nom-verb.',
        'subj_nom-noun_ter-verb.',
        'add_terminative',
        'Add a terminative dependent',
    ),
    ContrastSpec(
        'subj_nom-verb.',
        'subj_nom-noun_init-verb.',
        'add_initiative',
        'Add an ablative starting-point dependent',
    ),
    ContrastSpec(
        'tr-indef.',
        'tr_goal-indef.',
        'add_goal',
        'Add a goal to an indefinite-object transitive frame',
    ),
    ContrastSpec(
        'tr-indef.',
        'tr_src-indef.',
        'add_source',
        'Add a source to an indefinite-object transitive frame',
    ),
    ContrastSpec(
        'tr-indef.',
        'tr_com-indef.',
        'add_comitative',
        'Add a comitative to an indefinite-object transitive frame',
    ),
    ContrastSpec(
        'tr-indef.',
        'tr_ter-indef.',
        'add_terminative',
        'Add a terminative to an indefinite-object transitive frame',
    ),
    ContrastSpec(
        'tr-indef.',
        'tr_init-indef.',
        'add_initiative',
        'Add a starting point to an indefinite-object transitive frame',
    ),
    ContrastSpec(
        'tr-def.',
        'tr_goal-def.',
        'add_goal',
        'Add a goal to a definite-object transitive frame',
    ),
    ContrastSpec(
        'tr-def.',
        'tr_src-def.',
        'add_source',
        'Add a source to a definite-object transitive frame',
    ),
    ContrastSpec(
        'tr-def.',
        'tr_com-def.',
        'add_comitative',
        'Add a comitative to a definite-object transitive frame',
    ),
    ContrastSpec(
        'tr-def.',
        'tr_ter-def.',
        'add_terminative',
        'Add a terminative to a definite-object transitive frame',
    ),
    ContrastSpec(
        'tr-def.',
        'tr_init-def.',
        'add_initiative',
        'Add a starting point to a definite-object transitive frame',
    ),

    # ------------------------------------------------------------------
    # Postposition frames.
    #
    # Only "does the verb tolerate a postpositional dependent at all" is
    # tested here. Contrasting the essive/lative/ablative series against each
    # other (mögött / mögé / mögül) would hold the postposition slot as the
    # manipulation, and shared_slots() currently treats `postposition` as
    # shared material that must match, so every such pair would be rejected.
    # Adding that contrast means excluding `postposition` there first.
    # ------------------------------------------------------------------
    ContrastSpec(
        'subj_nom-verb.',
        'subj_nom-spostp_ess-verb.',
        'add_spatial_postposition',
        'Add a bare spatial postpositional phrase',
    ),
    ContrastSpec(
        'tr-indef.',
        'tr_spostp_ess-indef.',
        'add_spatial_postposition',
        'Add a spatial postpositional phrase to an indefinite-object frame',
    ),
    ContrastSpec(
        'tr-def.',
        'tr_spostp_ess-def.',
        'add_spatial_postposition',
        'Add a spatial postpositional phrase to a definite-object frame',
    ),
    ContrastSpec(
        'subj_nom-verb.',
        'subj_nom-cpostp-verb.',
        'add_complex_postposition',
        'Add a case-governing postpositional phrase',
    ),
    ContrastSpec(
        'tr-indef.',
        'tr_cpostp-indef.',
        'add_complex_postposition',
        'Add a case-governing postpositional phrase to an indefinite-object frame',
    ),
    ContrastSpec(
        'tr-def.',
        'tr_cpostp-def.',
        'add_complex_postposition',
        'Add a case-governing postpositional phrase to a definite-object frame',
    ),

    # ------------------------------------------------------------------
    # Object definiteness in the frames added above.
    # ------------------------------------------------------------------
    ContrastSpec(
        'tr_goal-indef.',
        'tr_goal-def.',
        'object_definiteness',
        'Indefinite versus definite object with a goal',
        allow_verb_form_change=True,
    ),
    ContrastSpec(
        'tr_src-indef.',
        'tr_src-def.',
        'object_definiteness',
        'Indefinite versus definite object with a source',
        allow_verb_form_change=True,
    ),
    ContrastSpec(
        'tr_com-indef.',
        'tr_com-def.',
        'object_definiteness',
        'Indefinite versus definite object with a comitative',
        allow_verb_form_change=True,
    ),
    ContrastSpec(
        'tr_ter-indef.',
        'tr_ter-def.',
        'object_definiteness',
        'Indefinite versus definite object with a terminative',
        allow_verb_form_change=True,
    ),
    ContrastSpec(
        'tr_init-indef.',
        'tr_init-def.',
        'object_definiteness',
        'Indefinite versus definite object with a starting point',
        allow_verb_form_change=True,
    ),
    ContrastSpec(
        'tr_spostp_ess-indef.',
        'tr_spostp_ess-def.',
        'object_definiteness',
        'Indefinite versus definite object with a spatial postposition',
        allow_verb_form_change=True,
    ),
    ContrastSpec(
        'tr_cpostp-indef.',
        'tr_cpostp-def.',
        'object_definiteness',
        'Indefinite versus definite object with a case-governing postposition',
        allow_verb_form_change=True,
    ),
]


def load_config(config_path: Path) -> dict:
    """Load the Hungarian pipeline configuration."""
    with config_path.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def load_filled_templates(path: Path, limit: int | None = None) -> list[FilledTemplate]:
    """Load filled Hungarian templates from JSONL."""
    filled_templates: list[FilledTemplate] = []

    with path.open(encoding="utf-8") as input_file:
        for index, line in enumerate(input_file):
            if limit is not None and index >= limit:
                break
            if line.strip():
                filled_templates.append(FilledTemplate(**json.loads(line)))

    return filled_templates


def load_template_strings(path: Path) -> dict[str, str]:
    """Load canonical template strings keyed by template name.

    Pair generation re-renders every source sentence from these canonical
    templates plus the recorded slot fillers. This means a stale or malformed
    ``rendered_text`` value can never silently propagate into the 2AFC file.
    """
    templates: dict[str, str] = {}
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            data = json.loads(line)
            templates[data["name"]] = data["template_string"]
    return templates


def normalize_surface(text: str) -> str:
    """Normalize whitespace while preserving lexical word boundaries."""
    text = " ".join(text.split())
    for punctuation in (".", ",", "?", "!"):
        text = text.replace(f" {punctuation}", punctuation)
    return text


def render_from_records(template_string: str, slot_records: dict[str, dict]) -> str:
    """Canonically render a sentence from a template and serialized fillers."""
    surfaces = {
        name: (record.get("form") or record.get("lemma"))
        for name, record in slot_records.items()
    }
    rendered = normalize_surface(template_string.format_map(surfaces))
    if "{" in rendered or "}" in rendered:
        raise ValueError(f"Unresolved placeholder after canonical rendering: {rendered!r}")
    if re.search(r",\S", rendered):
        raise ValueError(f"Missing space after comma in canonical rendering: {rendered!r}")
    return rendered


def filler_record(filler) -> dict:
    """Keep the lexical information needed to verify minimal pairs."""
    return {
        "lemma": filler.lemma,
        "form": filler.form or filler.lemma,
        "features": dict(filler.features or {}),
    }


def _surface_is_standalone(text: str, surface: str) -> bool:
    import re
    return re.search(rf"(?<!\w){re.escape(surface)}(?!\w)", text, flags=re.UNICODE) is not None


def validate_filled_surface(filled_template: FilledTemplate) -> None:
    """Reject malformed source sentences before any pair is created."""
    text = filled_template.rendered_text.strip()
    if "{" in text or "}" in text:
        raise ValueError(f"Unresolved placeholder: {text!r}")
    if "," in text and ", " not in text:
        raise ValueError(f"Missing space after comma: {text!r}")
    for slot_name, filler in filled_template.slot_fillers.items():
        surface = filler.form or filler.lemma
        if not _surface_is_standalone(text, surface):
            raise ValueError(
                f"Malformed source item: slot {slot_name!r} surface {surface!r} is not a standalone token in {text!r}. "
                "Regenerate filled templates with the current fill_templates.py."
            )


def convert_filled_templates_to_items(filled_templates: list[FilledTemplate], template_strings: dict[str, str]) -> list[Item]:
    """Convert filled sentences to Items and retain their slot-level metadata."""
    items: list[Item] = []

    for filled_template in filled_templates:
        validate_filled_surface(filled_template)
        verb = filled_template.slot_fillers.get("verb")
        if verb is None:
            continue

        slot_fillers = {
            slot_name: filler_record(filler)
            for slot_name, filler in filled_template.slot_fillers.items()
        }

        template_string = template_strings.get(filled_template.template_name)
        if template_string is None:
            raise ValueError(f"No canonical template string for {filled_template.template_name!r}")

        # Re-render from the template and fillers instead of trusting a stored
        # surface string. This is the authoritative 2AFC sentence.
        text = render_from_records(template_string, slot_fillers)
        stored_text = normalize_surface(filled_template.rendered_text.strip())
        if stored_text != text:
            raise ValueError(
                "Filled-template surface does not round-trip from its canonical template: "
                f"template={filled_template.template_name!r}, stored={stored_text!r}, canonical={text!r}"
            )
        if text:
            text = text[0].upper() + text[1:]

        item = Item(
            item_template_id=filled_template.template_id,
            rendered_elements={"text": text},
            item_metadata={
                "filled_template_id": str(filled_template.id),
                "template_id": str(filled_template.template_id),
                "template_name": filled_template.template_name,
                "template_structure": filled_template.template_name,
                "template_string": template_string,
                "verb_lemma": verb.lemma,
                "verb_form": verb.form or verb.lemma,
                "verb_features": dict(verb.features or {}),
                "slot_fillers": slot_fillers,
                "language": "hun",
                "strategy": filled_template.strategy_name,
            },
        )
        items.append(item)

    return items


def score_items_with_language_model(items: list[Item], cache_dir: Path, model_name: str) -> dict[str, float]:
    """Score the generated Hungarian sentences with the configured LM."""
    scorer = LanguageModelScorer(model_name=model_name, cache_dir=cache_dir, device="cpu", text_key="text")

    temporary_items: list[Item] = []
    original_id_by_temporary_id: dict[object, str] = {}

    for item in items:
        temporary_item = Item(item_template_id=uuid4(), rendered_elements={"text": item.rendered_elements.get("text", "")})
        temporary_items.append(temporary_item)
        original_id_by_temporary_id[temporary_item.id] = str(item.id)

    scores_list = scorer.score_batch(temporary_items)
    return {
        original_id_by_temporary_id[temp_item.id]: float(score)
        for temp_item, score in zip(temporary_items, scores_list, strict=True)
    }


def slot_signature(item: Item, slot_names: set[str], *, use_lemma_for: set[str] | None = None) -> tuple:
    """Return a hashable signature for selected lexical fillers."""
    use_lemma_for = use_lemma_for or set()
    fillers = item.item_metadata.get("slot_fillers", {})
    signature = []

    for slot_name in sorted(slot_names):
        record = fillers.get(slot_name)
        if record is None:
            signature.append((slot_name, None))
            continue
        value = record.get("lemma") if slot_name in use_lemma_for else record.get("form")
        signature.append((slot_name, value))

    return tuple(signature)


def verb_feature_signature(item: Item, *, include_tense: bool = True, include_object_agreement: bool = True) -> tuple:
    """Keep inflectional features fixed when the verb lemma is the manipulation."""
    features = item.item_metadata.get("verb_features", {})
    keys = ["person", "number", "mood", "finiteness"]
    if include_tense:
        keys.append("tense")
    if include_object_agreement:
        keys.append("object_agreement")
    return tuple((key, features.get(key)) for key in keys)


def make_pair(first_item: Item, second_item: Item, lm_scores: dict[str, float], metadata: dict) -> Item:
    """Create one two-sentence forced-choice item."""
    first_score = lm_scores.get(str(first_item.id), 0.0)
    second_score = lm_scores.get(str(second_item.id), 0.0)

    pair_metadata = {
        **metadata,
        "language": "hun",
        "source_item_0_id": str(first_item.id),
        "source_item_1_id": str(second_item.id),
        "option_a_template": first_item.item_metadata.get("template_name"),
        "option_b_template": second_item.item_metadata.get("template_name"),
        "option_a_verb": first_item.item_metadata.get("verb_lemma"),
        "option_b_verb": second_item.item_metadata.get("verb_lemma"),
        "verb_a": first_item.item_metadata.get("verb_lemma"),
        "verb_b": second_item.item_metadata.get("verb_lemma"),
        "template_a": first_item.item_metadata.get("template_name"),
        "template_b": second_item.item_metadata.get("template_name"),
        "option_a_template_string": first_item.item_metadata.get("template_string"),
        "option_b_template_string": second_item.item_metadata.get("template_string"),
        "option_a_verb_form": first_item.item_metadata.get("verb_form"),
        "option_b_verb_form": second_item.item_metadata.get("verb_form"),
        "option_a_verb_features": dict(first_item.item_metadata.get("verb_features", {})),
        "option_b_verb_features": dict(second_item.item_metadata.get("verb_features", {})),
        "option_a_slots": dict(first_item.item_metadata.get("slot_fillers", {})),
        "option_b_slots": dict(second_item.item_metadata.get("slot_fillers", {})),
        "lm_score_a": first_score,
        "lm_score_b": second_score,
        "lm_score_diff": abs(first_score - second_score),
    }

    return Item(
        item_template_id=uuid4(),
        rendered_elements={
            "option_a": first_item.rendered_elements.get("text", ""),
            "option_b": second_item.rendered_elements.get("text", ""),
        },
        item_metadata=pair_metadata,
    )


def shared_slots(first_item: Item, second_item: Item) -> set[str]:
    """Return slot names that occur in both templates, excluding the matrix verb."""
    first = set(first_item.item_metadata.get("slot_fillers", {}))
    second = set(second_item.item_metadata.get("slot_fillers", {}))
    return (first & second) - {"verb", "ongoing", "comp_subject", "comp_verb"}


def same_verb_items_match(first_item: Item, second_item: Item, spec: ContrastSpec) -> bool:
    """Check that a same-verb pair differs only in the intended manipulation."""
    if first_item.item_metadata.get("verb_lemma") != second_item.item_metadata.get("verb_lemma"):
        return False

    first_features = first_item.item_metadata.get("verb_features", {})
    second_features = second_item.item_metadata.get("verb_features", {})

    if not spec.allow_tense_change and first_features.get("tense") != second_features.get("tense"):
        return False

    # For most frame additions the matrix verb itself should be identical.
    if not spec.allow_verb_form_change and first_item.item_metadata.get("verb_form") != second_item.item_metadata.get("verb_form"):
        return False

    common_slots = shared_slots(first_item, second_item)

    # In a definiteness contrast, the object noun should stay the same lemma,
    # while its article and the verb's conjugation are allowed to change.
    if spec.contrast_type == "object_definiteness":
        common_slots.discard("determiner")
        return slot_signature(first_item, common_slots, use_lemma_for={"object"}) == slot_signature(second_item, common_slots, use_lemma_for={"object"})

    # The clausal contrast adds the complement and necessarily changes matrix
    # object agreement. Shared lexical material (especially the subject) stays fixed.
    if spec.contrast_type == "clausal_complement":
        common_slots.discard("determiner")

    return slot_signature(first_item, common_slots) == slot_signature(second_item, common_slots)


def load_preverb_forms(path: Path) -> set[str]:
    """Load the controlled Hungarian preverb inventory."""
    if not path.exists():
        return set()
    forms = set()
    with path.open(encoding="utf-8") as input_file:
        for line in input_file:
            if not line.strip():
                continue
            data = json.loads(line)
            forms.add(data.get("form") or data.get("lemma"))
    return {form for form in forms if form}


def infer_transparently_prefixed_lemmas(items: list[Item], preverbs: set[str]) -> set[str]:
    """Conservatively detect prefixed lemmas when the stripped base is also present.

    This avoids treating an arbitrary word that merely begins with e.g. ``el`` as
    prefixed. It catches transparent Stage-1 cases such as lead = le + ad and
    elintéz = el + intéz when both base and prefixed lemmas occur in the lexicon.
    """
    lemmas = {item.item_metadata.get("verb_lemma") for item in items}
    lemmas.discard(None)
    prefixed = set()
    for lemma in lemmas:
        for preverb in sorted(preverbs, key=len, reverse=True):
            if lemma.startswith(preverb) and len(lemma) > len(preverb):
                base = lemma[len(preverb):]
                if base in lemmas:
                    prefixed.add(lemma)
                    break
    return prefixed


def create_same_verb_pairs(items: list[Item], lm_scores: dict[str, float], prefixed_lemmas: set[str] | None = None) -> list[Item]:
    """Create explicitly defined same-verb/different-frame minimal pairs.

    Prefixed verbs are conservatively excluded from the éppen/tense contrasts in
    Stage 1 because preverb placement/aspect is reserved for the Hungarian-specific
    Stage-2 analysis. They remain available in the nominal and clausal frames.
    """
    prefixed_lemmas = prefixed_lemmas or set()
    by_template_and_verb: dict[tuple[str, str], list[Item]] = defaultdict(list)
    for item in items:
        key = (item.item_metadata.get("template_name"), item.item_metadata.get("verb_lemma"))
        by_template_and_verb[key].append(item)

    verbs = sorted({item.item_metadata.get("verb_lemma") for item in items if item.item_metadata.get("verb_lemma")})
    pairs: list[Item] = []
    seen = set()

    for spec in SAME_VERB_CONTRASTS:
        for verb in verbs:
            if spec.contrast_type in {"ongoing_context", "tense"} and verb in prefixed_lemmas:
                continue
            left_items = by_template_and_verb.get((spec.left_template, verb), [])
            right_items = by_template_and_verb.get((spec.right_template, verb), [])

            for left_item in left_items:
                for right_item in right_items:
                    if not same_verb_items_match(left_item, right_item, spec):
                        continue

                    pair_key = (str(left_item.id), str(right_item.id), spec.contrast_type)
                    if pair_key in seen:
                        continue
                    seen.add(pair_key)

                    pairs.append(make_pair(
                        left_item,
                        right_item,
                        lm_scores,
                        {
                            "pair_type": "same_verb",
                            "contrast_type": spec.contrast_type,
                            "contrast_description": spec.description,
                            "verb": verb,
                            "template1": spec.left_template,
                            "template2": spec.right_template,
                        },
                    ))

    return pairs


def nonverb_signature(item: Item) -> tuple:
    """Identify a sentence frame and all lexical material except the matrix verb."""
    fillers = item.item_metadata.get("slot_fillers", {})
    slot_names = set(fillers) - {"verb"}
    return slot_signature(item, slot_names)


def create_different_verb_pairs(items: list[Item], lm_scores: dict[str, float], prefixed_lemmas: set[str] | None = None) -> list[Item]:
    """Create same-frame pairs where the matrix verb lemma is the only lexical change.

    We pair neighboring lemmas deterministically within each matched group instead
    of taking every possible verb pair. This keeps the full run tractable and avoids
    millions of redundant combinations.
    """
    prefixed_lemmas = prefixed_lemmas or set()
    groups: dict[tuple, dict[str, Item]] = defaultdict(dict)

    for item in items:
        template_name = item.item_metadata.get("template_name", "")
        verb = item.item_metadata.get("verb_lemma")
        if "eppen" in template_name and verb in prefixed_lemmas:
            continue
        key = (
            item.item_metadata.get("template_name"),
            nonverb_signature(item),
            verb_feature_signature(item),
        )
        if verb and verb not in groups[key]:
            groups[key][verb] = item

    pairs: list[Item] = []

    for (template_name, _, _), by_verb in sorted(groups.items(), key=lambda pair: str(pair[0])):
        ordered = [by_verb[verb] for verb in sorted(by_verb)]

        for index in range(0, len(ordered) - 1, 2):
            first_item = ordered[index]
            second_item = ordered[index + 1]
            pairs.append(make_pair(
                first_item,
                second_item,
                lm_scores,
                {
                    "pair_type": "different_verb",
                    "contrast_type": "lexical_verb",
                    "contrast_description": "Same frame and fillers; change only the matrix verb lemma",
                    "template_structure": template_name,
                    "template_id": first_item.item_metadata.get("template_id"),
                    "template1": template_name,
                    "template2": template_name,
                    "verb1": first_item.item_metadata.get("verb_lemma"),
                    "verb2": second_item.item_metadata.get("verb_lemma"),
                    "verb_a": first_item.item_metadata.get("verb_lemma"),
                    "verb_b": second_item.item_metadata.get("verb_lemma"),
                },
            ))

    return pairs


def assign_quantiles_to_pairs(pair_items: list[Item], n_quantiles: int = 10) -> list[Item]:
    """Assign LM score-difference quantiles separately for the two pair families."""
    if not pair_items:
        return pair_items

    item_metadata = {item.id: item.item_metadata for item in pair_items}
    item_ids = [item.id for item in pair_items]
    quantile_assignments = assign_quantiles_by_uuid(
        item_ids=item_ids,
        item_metadata=item_metadata,
        property_key="lm_score_diff",
        n_quantiles=n_quantiles,
        stratify_by_key="pair_type",
    )

    for item in pair_items:
        item.item_metadata["quantile"] = quantile_assignments[item.id]

    return pair_items


def save_items_jsonl(items: list[Item], output_path: Path) -> None:
    """Write forced-choice items to JSONL."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output_file:
        for item in items:
            output_file.write(item.model_dump_json() + "\n")


def print_summary(filled_count: int, item_count: int, pair_items: list[Item], output_path: Path) -> None:
    """Print a short summary and contrast counts."""
    pair_type_counts: dict[str, int] = defaultdict(int)
    contrast_counts: dict[str, int] = defaultdict(int)

    for item in pair_items:
        pair_type_counts[item.item_metadata.get("pair_type", "unknown")] += 1
        contrast_counts[item.item_metadata.get("contrast_type", "unknown")] += 1

    table = Table(title="Hungarian 2AFC generation")
    table.add_column("Stage")
    table.add_column("Count", justify="right")
    table.add_row("Filled templates loaded", f"{filled_count:,}")
    table.add_row("Sentence items created", f"{item_count:,}")
    table.add_row("Same verb / different frame", f"{pair_type_counts['same_verb']:,}")
    table.add_row("Same frame / different verb", f"{pair_type_counts['different_verb']:,}")
    table.add_row("Total controlled pairs", f"{len(pair_items):,}")
    console.print(table)

    if contrast_counts:
        console.print("\n[bold]Contrast types[/bold]")
        for contrast_type, count in sorted(contrast_counts.items()):
            console.print(f"  {contrast_type}: {count:,}")

    console.print(f"\n[green]Saved:[/green] {output_path}")


def get_2afc_config(config: dict) -> dict:
    """Return the 2AFC section with defaults for older config files."""
    return config.get("create_2afc_pairs", config.get("2afc", {}))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create controlled Hungarian 2AFC pairs")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"), help="Path to config.yaml")
    parser.add_argument("--input", type=Path, default=None, help="Override the filled-template JSONL path")
    parser.add_argument("--templates", type=Path, default=None, help="Override canonical template JSONL path")
    parser.add_argument("--output", type=Path, default=None, help="Override the output JSONL path")
    parser.add_argument("--limit", type=int, default=None, help="Only load the first N filled templates")
    parser.add_argument("--model", type=str, default=None, help="Override the Hungarian language-model name")
    parser.add_argument("--quantiles", type=int, default=None, help="Override the number of quantile bins")
    parser.add_argument("--skip-scoring", action="store_true", help="Skip LM scoring for a quick structural check")
    args = parser.parse_args()

    config = load_config(args.config)
    pair_config = get_2afc_config(config)

    input_path = args.input or Path(pair_config.get("input_path", "filled_templates/generic_frames_filled.jsonl"))
    templates_path = args.templates or Path(pair_config.get("templates_path", "templates/generic_frames.jsonl"))
    output_path = args.output or Path(pair_config.get("output_path", "items/2afc_pairs.jsonl"))
    cache_dir = Path(pair_config.get("cache_dir", ".cache/language_models"))
    model_name = args.model or pair_config.get("model_name", "NYTK/PULI-GPT-2")
    n_quantiles = args.quantiles if args.quantiles is not None else int(pair_config.get("n_quantiles", 10))
    preverbs_path = Path(pair_config.get("preverbs_path", "lexicons/preverbs.jsonl"))

    console.rule("[bold]Hungarian 2AFC Pair Generation[/bold]")
    console.print(f"Filled templates: [cyan]{input_path}[/cyan]")
    console.print(f"Canonical templates: [cyan]{templates_path}[/cyan]")
    console.print(f"Output: [cyan]{output_path}[/cyan]")

    filled_templates = load_filled_templates(input_path, limit=args.limit)
    if not filled_templates:
        raise ValueError(f"No filled templates were found in {input_path}")

    template_strings = load_template_strings(templates_path)
    items = convert_filled_templates_to_items(filled_templates, template_strings)
    console.print(f"Loaded [cyan]{len(items):,}[/cyan] sentence items")
    preverbs = load_preverb_forms(preverbs_path)
    prefixed_lemmas = infer_transparently_prefixed_lemmas(items, preverbs)
    if prefixed_lemmas:
        console.print(
            "Stage-1 temporal controls: excluding transparently prefixed lemmas from éppen/tense contrasts: "
            + ", ".join(sorted(prefixed_lemmas))
        )

    if args.skip_scoring:
        console.print("[yellow]Structural check:[/yellow] skipping LM scoring")
        lm_scores = {str(item.id): 0.0 for item in items}
    else:
        console.print(f"Scoring with [cyan]{model_name}[/cyan]")
        lm_scores = score_items_with_language_model(items, cache_dir=cache_dir, model_name=model_name)

    with console.status("[bold]Creating controlled same-verb pairs...[/bold]"):
        same_verb_pairs = create_same_verb_pairs(items, lm_scores, prefixed_lemmas=prefixed_lemmas)

    with console.status("[bold]Creating controlled different-verb pairs...[/bold]"):
        different_verb_pairs = create_different_verb_pairs(items, lm_scores, prefixed_lemmas=prefixed_lemmas)

    pair_items = same_verb_pairs + different_verb_pairs
    if not pair_items:
        raise ValueError("No controlled 2AFC pairs were created. Check that the filled-template file contains repeated verbs and frames.")

    if not args.skip_scoring:
        pair_items = assign_quantiles_to_pairs(pair_items, n_quantiles=n_quantiles)
    else:
        for item in pair_items:
            item.item_metadata["quantile"] = 0

    save_items_jsonl(pair_items, output_path)
    print_summary(len(filled_templates), len(items), pair_items, output_path)


if __name__ == "__main__":
    main()
