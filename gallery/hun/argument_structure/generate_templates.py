#!/usr/bin/env python3
"""
Generate Stage-1 Hungarian argument-structure templates.

This is the Hungarian Stage-1 version of the ENG/KOR argument-structure task.
The basic experimental question is unchanged: which verbs are acceptable in
which frames? The frames are realized with Hungarian morphology.

Korean-aligned structural baseline. Korean marks seven adjunct types with
particles; Hungarian marks the same distinctions with case suffixes, except
for the comitative, which is syncretic with the instrumental:

    Korean adjunct   Korean particle   Hungarian case   Stage-1 realization
    dat              에게              DAT              egy embernek
    loc              에서              SUP              egy helyen
    inst             (으)로            INS              egy eszkozzel
    goal             에                SUB              egy helyre
    source           에서              DEL              egy helyrol
    com              와/과             INS              egy emberrel
    term             까지              TER              egy helyig
    init             부터              ABL              egy helytol

The locative/goal/source triple deliberately uses Hungarian's *surface* series
(SUP -on, SUB -ra, DEL -rol) rather than mixing series, so that the three
frames differ only in direction.

Korean's spatial relational nouns (위, 앞, 뒤 + 에/에서/로) correspond to
Hungarian bare postpositions, which come in the same three-way essive/lative/
ablative series (alatt/ala/alol). Korean's complex postpositions
(에 대해서, 을 통해서) correspond to Hungarian case-governing postpositions
(kepest + ALL, egyutt + INS, keresztul + SUP), where the governed case is tied
to the postposition by an explicit cross-slot constraint. That constraint is
the Hungarian counterpart of Korean's `fc_agree` allomorphy handling.

Hungarian-specific choices:
    - nouns are already case-inflected in the lexicon;
    - Korean INST corresponds to Hungarian INS;
    - every ACC frame has INDF and DEF versions;
    - INDF objects use egy;
    - DEF objects use a/az and DEF verbal conjugation;
    - finite baseline verbs are IND, PRS/PST, 3SG;
    - ongoing templates use eppen rather than a fake Hungarian
      morphological progressive;
    - one finite direct-object hogy + indicative clausal-complement frame is
      included.

Preverbs, pro-drop, focus, argument omission, and potential mood (POT) are
reserved for Stage 2.

Word order
----------
Hungarian's immediately preverbal position is the focus position, so stacking
several arguments there is a marked order. `--word-order` makes that an
explicit parameter rather than a buried assumption:

    preverbal (default)  Egy ember egy targyat egy helyen ad.
    neutral              Egy ember ad egy targyat egy helyen.

`preverbal` reproduces the original Stage-1 output exactly. `neutral` places
the finite verb directly after the subject, which is the unmarked order for a
sentence with no focused constituent. Which one to collect on is a design
decision; both are generated from the same frame inventory so they can be
compared.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from bead.resources import Constraint, Slot, Template


VOWELS = "aáeéiíoóöőuúüű"

PREVERBAL = "preverbal"
NEUTRAL = "neutral"


# ---------------------------------------------------------------------------
# CONTROLLED FILLERS
# ---------------------------------------------------------------------------
#
# Slots listed here are pinned to a specific lemma. They are the frame's fixed
# background material, held constant so that the verb and the frame shape are
# the only things that vary.
#
# Pinning by lemma matters. These slots used to be selected by semantic class
# alone, which only behaved as "controlled" because each class happened to have
# exactly one member. Adding any noun to `location` or `instrument` would have
# silently un-controlled the baseline described in LINGUISTIC_DESIGN_NOTES.md
# without changing a line of template code.

CONTROLLED_LEMMAS = {
    "location": "hely",
    "goal": "hely",
    "source": "hely",
    "terminus": "hely",
    "origin": "hely",
    "instrument": "eszköz",
    "comitative": "ember",
    "postp_object": "hely",
    "complex_postp_object": "ember",
}

# Slots that vary freely within a semantic class, for lexical variety.
OPEN_CLASSES = {
    "subject": ["human", "group"],
    "dative": ["human", "group"],
    # Stage 1 restricts direct objects to inanimates to reduce semantic
    # confounds in verb-frame judgments.
    "object": ["inanimate_object"],
}


# ---------------------------------------------------------------------------
# ADJUNCT REGISTRY
# ---------------------------------------------------------------------------
#
# (slot name, case, description, semantic class, Korean counterpart)
#
# `comitative` is the one entry whose case duplicates another's: Hungarian does
# not distinguish instrumental from comitative morphologically. The two frames
# differ in the semantic class of the noun, not in case, and that is a genuine
# Hungarian fact rather than a gap in the inventory.

ADJUNCTS = {
    "dat": ("dative", "DAT", "dative argument", None, "dat"),
    "loc": ("location", "SUP", "neutral superessive locative argument", "location", "loc"),
    "ins": ("instrument", "INS", "instrumental argument", "instrument", "inst"),
    "goal": ("goal", "SUB", "sublative goal argument", "location", "goal"),
    "src": ("source", "DEL", "delative source argument", "location", "loc.src"),
    "com": ("comitative", "INS", "comitative argument", "human", "com"),
    "ter": ("terminus", "TER", "terminative argument", "location", "term"),
    "init": ("origin", "ABL", "ablative starting-point argument", "location", "init"),
}


# ---------------------------------------------------------------------------
# SLOT HELPERS
# ---------------------------------------------------------------------------

def noun_slot(
    name: str,
    case_name: str | None,
    description: str,
    semantic_classes: list[str] | None = None,
    lemma: str | None = None,
) -> Slot:
    """Create a singular noun slot, optionally constrained for case and lemma.

    `case_name` may be None for a slot whose case is fixed by a cross-slot
    constraint instead, as with a postposition's complement.
    """
    conditions = [
        "self.features.get('pos') == 'NOUN'",
        "self.features.get('number') == 'SG'",
    ]

    if case_name is not None:
        conditions.append(f"self.features.get('case') == '{case_name}'")

    if semantic_classes:
        conditions.append(f"self.features.get('semantic_class') in {semantic_classes!r}")

    if lemma is not None:
        conditions.append(f"self.lemma == '{lemma}'")

    return Slot(
        name=name,
        description=description,
        constraints=[Constraint(expression=" and ".join(conditions))],
    )


def argument_slot(slot_name: str, case_name: str, description: str) -> Slot:
    """Build an argument slot, pinning it if it is a controlled filler."""
    if slot_name in CONTROLLED_LEMMAS:
        return noun_slot(
            slot_name,
            case_name,
            description,
            lemma=CONTROLLED_LEMMAS[slot_name],
        )

    return noun_slot(
        slot_name,
        case_name,
        description,
        semantic_classes=OPEN_CLASSES.get(slot_name),
    )


def determiner_slot(definiteness: str, name: str = "determiner") -> Slot:
    """Create a Hungarian determiner slot for the requested definiteness."""
    return Slot(
        name=name,
        description=f"{definiteness.lower()} Hungarian determiner",
        constraints=[
            Constraint(
                expression=(
                    "self.features.get('pos') == 'DET' "
                    f"and self.features.get('definiteness') == '{definiteness}'"
                )
            )
        ],
    )


def verb_slot(
    object_agreement: str,
    name: str = "verb",
    tense: str | None = None,
    lemma: str | None = None,
) -> Slot:
    """Create a 3SG finite indicative verb slot with object-agreement control."""
    conditions = [
        "self.features.get('pos') == 'V'",
        "self.features.get('finiteness') == 'FIN'",
        "self.features.get('mood') == 'IND'",
        "self.features.get('person') == '3'",
        "self.features.get('number') == 'SG'",
        f"self.features.get('object_agreement') == '{object_agreement}'",
    ]

    if tense is None:
        conditions.append("self.features.get('tense') in ['PRS', 'PST']")
    else:
        conditions.append(f"self.features.get('tense') == '{tense}'")

    if lemma is not None:
        conditions.append(f"self.lemma == '{lemma}'")

    return Slot(
        name=name,
        description=f"3SG finite indicative Hungarian verb ({object_agreement})",
        constraints=[Constraint(expression=" and ".join(conditions))],
    )


def ongoing_particle_slot() -> Slot:
    """Create the controlled `éppen` ongoing-event slot."""
    return Slot(
        name="ongoing",
        description="Hungarian ongoing-event adverb éppen",
        constraints=[
            Constraint(
                expression=(
                    "self.features.get('pos') == 'PART' "
                    "and self.form == 'éppen'"
                )
            )
        ],
    )


def spatial_postposition_slot(series: str) -> Slot:
    """Create a bare (NOM-governing) spatial postposition slot.

    The essive/lative/ablative series is the Hungarian counterpart of the
    Korean spatial-noun contrast between the goal, source and plain locative
    particles.
    """
    return Slot(
        name="postposition",
        description=f"Hungarian bare spatial postposition ({series})",
        constraints=[
            Constraint(
                expression=(
                    "self.features.get('pos') == 'POSTP' "
                    "and self.features.get('postp_type') == 'spatial' "
                    f"and self.features.get('series') == '{series}' "
                    "and self.features.get('stage1') == true"
                )
            )
        ],
    )


def complex_postposition_slot() -> Slot:
    """Create a case-governing postposition slot.

    The governed case is not fixed here; it is tied to the complement noun by
    `postposition_government_constraint`.
    """
    return Slot(
        name="postposition",
        description="Hungarian case-governing postposition",
        constraints=[
            Constraint(
                expression=(
                    "self.features.get('pos') == 'POSTP' "
                    "and self.features.get('postp_type') == 'complex' "
                    "and self.features.get('stage1') == true"
                )
            )
        ],
    )


# ---------------------------------------------------------------------------
# CROSS-SLOT CONSTRAINTS
# ---------------------------------------------------------------------------

def definite_article_constraint(determiner_name: str, noun_name: str) -> Constraint:
    """
    Hungarian:
        a  + consonant-initial noun
        az + vowel-initial noun

    Uses the noun's actual surface form, so no extra noun feature is needed.
    """

    return Constraint(
        expression=(
            f"(({determiner_name}.form == 'az' and "
            f"{noun_name}.form[0].lower() in '{VOWELS}') or "
            f"({determiner_name}.form == 'a' and "
            f"{noun_name}.form[0].lower() not in '{VOWELS}'))"
        )
    )


def postposition_government_constraint(postposition_name: str, noun_name: str) -> Constraint:
    """Require a postposition's complement to bear the case it governs.

    `képest` governs ALL (egy emberhez képest), `együtt` governs INS (egy
    emberrel együtt), `keresztül` governs SUP (egy emberen keresztül), and the
    non-spatial group such as `szerint` and `miatt` governs a caseless NOM
    complement (egy ember szerint).

    This plays the role Korean's `fc_agree` plays for particle allomorphy: the
    lexicon states the requirement and the constraint enforces it, rather than
    the template hard-coding one case per postposition.
    """

    return Constraint(
        expression=(
            f"{postposition_name}.features.get('gov_case') == "
            f"{noun_name}.features.get('case')"
        )
    )


# ---------------------------------------------------------------------------
# WORD ORDER
# ---------------------------------------------------------------------------

def assemble(
    subject_part: str,
    preverbal_parts: list[str],
    verb_part: str,
    trailing_part: str,
    word_order: str,
) -> str:
    """Join a frame's constituents under the requested word order.

    `preverbal_parts` are the object and adjunct phrases. Under `preverbal`
    they sit between the subject and the verb, which is the original Stage-1
    order. Under `neutral` the finite verb follows the subject directly and the
    same phrases follow the verb, which is the unmarked order for a Hungarian
    sentence with no focused constituent.

    `trailing_part` is material that is always final regardless of order, such
    as a hogy-clause.
    """
    if word_order == NEUTRAL:
        ordered = [subject_part, verb_part, *preverbal_parts]
    else:
        ordered = [subject_part, *preverbal_parts, verb_part]

    if trailing_part:
        ordered.append(trailing_part)

    return " ".join(part for part in ordered if part)


def order_tags(word_order: str) -> list[str]:
    return [f"order_{word_order}"]


# ---------------------------------------------------------------------------
# TEMPLATE CONSTRUCTION
# ---------------------------------------------------------------------------

def build_adjunct_slots(
    adjunct_keys: list[str],
) -> tuple[dict[str, Slot], list[str], list[str]]:
    """Return slots, surface phrases, and metadata labels for a set of adjuncts."""
    slots: dict[str, Slot] = {}
    phrases: list[str] = []
    labels: list[str] = []

    for key in adjunct_keys:
        slot_name, case_name, description, _semantic_class, _korean = ADJUNCTS[key]
        slots[slot_name] = argument_slot(slot_name, case_name, description)
        phrases.append(f"egy {{{slot_name}}}")
        labels.append(key)

    return slots, phrases, labels


def make_intransitive_template(
    name: str,
    frame_family: str,
    adjunct_keys: list[str],
    word_order: str,
) -> Template:
    """
    Build:
        NOM (OBL...) V

    Example:
        Egy ember egy helyen alszik.
        'A person sleeps in a place.'

    The sentence is only an example of the frame shape. The experiment still
    tests whether each matrix verb is acceptable in that frame.
    """

    slots = {
        "subject": argument_slot("subject", "NOM", "nominative subject"),
        "verb": verb_slot("INDF"),
    }

    adjunct_slots, phrases, labels = build_adjunct_slots(adjunct_keys)
    slots.update(adjunct_slots)

    template_string = assemble("Egy {subject}", phrases, "{verb}", "", word_order) + "."

    return Template(
        name=name,
        template_string=template_string,
        slots=slots,
        constraints=[],
        description=f"Hungarian Stage-1 {frame_family} frame",
        language_code="hun",
        tags=["stage1", "nominal", "intransitive", *order_tags(word_order)],
        metadata={
            "frame_family": frame_family,
            "transitivity": "intransitive",
            "object_definiteness": None,
            "adjuncts": labels,
            "word_order": word_order,
        },
    )


def make_transitive_template(
    name: str,
    frame_family: str,
    object_definiteness: str,
    adjunct_keys: list[str],
    word_order: str,
) -> Template:
    """
    Build:
        NOM DET ACC (OBL...) V

    Indefinite object:
        Egy ember egy tárgyat ad.
        'A person gives an object.'

    Definite object:
        Egy ember a tárgyat adja.
        'A person gives the object.'

    Hungarian object definiteness is reflected both in the determiner and in
    the verb's definite/indefinite conjugation.
    """

    slots = {
        "subject": argument_slot("subject", "NOM", "nominative subject"),
        "determiner": determiner_slot(object_definiteness),
        "object": argument_slot("object", "ACC", "accusative direct object"),
        "verb": verb_slot(object_definiteness),
    }

    adjunct_slots, adjunct_phrases, labels = build_adjunct_slots(adjunct_keys)
    slots.update(adjunct_slots)

    phrases = ["{determiner} {object}", *adjunct_phrases]
    template_string = assemble("Egy {subject}", phrases, "{verb}", "", word_order) + "."

    constraints = []

    if object_definiteness == "DEF":
        constraints.append(definite_article_constraint("determiner", "object"))

    return Template(
        name=name,
        template_string=template_string,
        slots=slots,
        constraints=constraints,
        description=f"Hungarian Stage-1 {frame_family} frame ({object_definiteness})",
        language_code="hun",
        tags=[
            "stage1",
            "nominal",
            "transitive",
            object_definiteness.lower(),
            *order_tags(word_order),
        ],
        metadata={
            "frame_family": frame_family,
            "transitivity": "transitive",
            "object_definiteness": object_definiteness,
            "adjuncts": labels,
            "word_order": word_order,
        },
    )


# ---------------------------------------------------------------------------
# POSTPOSITION FRAMES
# ---------------------------------------------------------------------------

def make_spatial_postposition_template(
    series: str,
    object_definiteness: str | None,
    word_order: str,
) -> Template:
    """
    Hungarian counterpart of the Korean spatial relational-noun frames.

    Bare postpositions govern a caseless complement:

        Egy ember egy hely mögött alszik.
        'A person sleeps behind a place.'

    The ESS/LAT/ABL series reproduces the Korean locative/goal/source contrast
    with a single postposition paradigm rather than three particles.
    """

    series_name = {"ESS": "essive", "LAT": "lative", "ABL": "ablative"}[series]

    slots: dict[str, Slot] = {
        "subject": argument_slot("subject", "NOM", "nominative subject"),
        "postp_object": argument_slot(
            "postp_object", "NOM", "caseless complement of a bare postposition"
        ),
        "postposition": spatial_postposition_slot(series),
    }

    constraints = []
    phrases = []
    transitivity = "intransitive"

    if object_definiteness is not None:
        transitivity = "transitive"
        slots["determiner"] = determiner_slot(object_definiteness)
        slots["object"] = argument_slot("object", "ACC", "accusative direct object")
        slots["verb"] = verb_slot(object_definiteness)
        phrases.append("{determiner} {object}")

        if object_definiteness == "DEF":
            constraints.append(definite_article_constraint("determiner", "object"))
    else:
        slots["verb"] = verb_slot("INDF")

    phrases.append("egy {postp_object} {postposition}")

    template_string = assemble("Egy {subject}", phrases, "{verb}", "", word_order) + "."

    if object_definiteness is None:
        name = f"subj_nom-spostp_{series.lower()}-verb."
    else:
        suffix = "indef" if object_definiteness == "INDF" else "def"
        name = f"tr_spostp_{series.lower()}-{suffix}."

    return Template(
        name=name,
        template_string=template_string,
        slots=slots,
        constraints=constraints,
        description=(
            f"Hungarian {series_name} bare-postposition frame ({transitivity})"
        ),
        language_code="hun",
        tags=[
            "stage1",
            "postposition",
            "spatial",
            series.lower(),
            transitivity,
            *order_tags(word_order),
        ],
        metadata={
            "frame_family": f"SPOSTP-{series}",
            "transitivity": transitivity,
            "object_definiteness": object_definiteness,
            "postposition_type": "spatial",
            "postposition_series": series,
            "word_order": word_order,
        },
    )


def make_complex_postposition_template(
    object_definiteness: str | None,
    word_order: str,
) -> Template:
    """
    Hungarian counterpart of the Korean complex-postposition frames.

    The postposition and its complement's case covary:

        Egy ember egy emberhez képest dolgozik.   (képest governs ALL)
        Egy ember egy emberrel együtt dolgozik.   (együtt governs INS)
        Egy ember egy ember szerint dolgozik.     (szerint governs NOM)

    The complement slot leaves case open and
    `postposition_government_constraint` ties it to the postposition, so a new
    postposition can be added to the lexicon without editing this template.
    """

    slots: dict[str, Slot] = {
        "subject": argument_slot("subject", "NOM", "nominative subject"),
        "complex_postp_object": noun_slot(
            "complex_postp_object",
            None,
            "case-governed complement of a complex postposition",
            lemma=CONTROLLED_LEMMAS["complex_postp_object"],
        ),
        "postposition": complex_postposition_slot(),
    }

    constraints = [
        postposition_government_constraint("postposition", "complex_postp_object")
    ]

    phrases = []
    transitivity = "intransitive"

    if object_definiteness is not None:
        transitivity = "transitive"
        slots["determiner"] = determiner_slot(object_definiteness)
        slots["object"] = argument_slot("object", "ACC", "accusative direct object")
        slots["verb"] = verb_slot(object_definiteness)
        phrases.append("{determiner} {object}")

        if object_definiteness == "DEF":
            constraints.append(definite_article_constraint("determiner", "object"))
    else:
        slots["verb"] = verb_slot("INDF")

    phrases.append("egy {complex_postp_object} {postposition}")

    template_string = assemble("Egy {subject}", phrases, "{verb}", "", word_order) + "."

    if object_definiteness is None:
        name = "subj_nom-cpostp-verb."
    else:
        suffix = "indef" if object_definiteness == "INDF" else "def"
        name = f"tr_cpostp-{suffix}."

    return Template(
        name=name,
        template_string=template_string,
        slots=slots,
        constraints=constraints,
        description=f"Hungarian case-governing postposition frame ({transitivity})",
        language_code="hun",
        tags=["stage1", "postposition", "complex", transitivity, *order_tags(word_order)],
        metadata={
            "frame_family": "CPOSTP",
            "transitivity": transitivity,
            "object_definiteness": object_definiteness,
            "postposition_type": "complex",
            "word_order": word_order,
        },
    )


# ---------------------------------------------------------------------------
# CLAUSAL COMPLEMENT
# ---------------------------------------------------------------------------

def make_hogy_complement_template(word_order: str) -> Template:
    """
    Baseline finite direct-object hogy-clause.

    The embedded proposition is intentionally controlled:

        hogy egy esemény történik/történt
        'that an event is happening/happened'

    This keeps the embedded clause well formed while the MATRIX verb varies,
    allowing the experiment to test whether the matrix verb licenses a
    finite DIRECT-OBJECT clausal complement.

    Hungarian finite direct-object clauses headed by hogy trigger the objective/
    definite conjugation on a transitive matrix verb (for example, mondja, hogy
    ... "says that ..."). This template therefore requires DEF object agreement.
    A verb that only takes an oblique hogy-clause is a different argument-structure
    type and is not represented by this Stage-1 direct-object-clause frame.

    The matrix verb is already adjacent to the subject, so the word-order
    parameter does not change this frame; it is recorded in the metadata for
    consistency.
    """

    return Template(
        name="subj_nom-verb-hogy_ind.",
        template_string="Egy {subject} {verb}, hogy egy {comp_subject} {comp_verb}.",
        slots={
            "subject": argument_slot("subject", "NOM", "matrix-clause subject"),
            "verb": verb_slot("DEF"),
            "comp_subject": noun_slot(
                "comp_subject",
                "NOM",
                "embedded event subject",
                lemma="esemény",
            ),
            "comp_verb": verb_slot(
                "INDF",
                name="comp_verb",
                lemma="történik",
            ),
        },
        constraints=[],
        description="Hungarian finite indicative direct-object hogy-clause complement",
        language_code="hun",
        tags=["stage1", "clausal", "hogy", "finite", "indicative", *order_tags(word_order)],
        metadata={
            "frame_family": "HOGY-IND",
            "complementizer": "hogy",
            "complement_type": "finite_indicative_direct_object",
            "matrix_object_agreement": "DEF",
            "word_order": word_order,
        },
    )


# ---------------------------------------------------------------------------
# ONGOING / KOREAN-PROGRESSIVE ANALOGUE
# ---------------------------------------------------------------------------

def make_ongoing_intransitive_template(tense: str, word_order: str) -> Template:
    """
    Hungarian does not have the Korean V-go iss progressive paradigm.

    éppen ('right now / just at that moment') creates an ongoing-event
    context. This is an ongoing construction, not a claim that Hungarian has
    a morphological progressive category.

    éppen is placed directly after the subject. In the intransitive frames
    that also makes it immediately preverbal; in the transitive ones the
    object may intervene, depending on word order.
    """

    tense_name = "present" if tense == "PRS" else "past"

    return Template(
        name=f"subj_nom-eppen-verb_{tense.lower()}.",
        template_string="Egy {subject} {ongoing} {verb}.",
        slots={
            "subject": argument_slot("subject", "NOM", "nominative subject"),
            "ongoing": ongoing_particle_slot(),
            "verb": verb_slot("INDF", tense=tense),
        },
        constraints=[],
        description=f"Hungarian {tense_name} ongoing intransitive sentence",
        language_code="hun",
        tags=["stage1", "ongoing", "intransitive", tense.lower(), *order_tags(word_order)],
        metadata={
            "frame_family": "INTR-ONGOING",
            "tense": tense,
            "progressive_morphology": False,
            "word_order": word_order,
        },
    )


def make_ongoing_transitive_template(
    tense: str,
    object_definiteness: str,
    word_order: str,
) -> Template:
    tense_name = "present" if tense == "PRS" else "past"

    constraints = []

    if object_definiteness == "DEF":
        constraints.append(definite_article_constraint("determiner", "object"))

    # éppen sits directly after the subject, so under `preverbal` the object
    # still intervenes between it and the verb (Egy ember éppen a tárgyat
    # adja), and under `neutral` it lands immediately preverbal (Egy ember
    # éppen adja a tárgyat). Attaching it to the subject rather than the verb
    # is what keeps the preverbal output identical to the original Stage-1 set.
    template_string = assemble(
        "Egy {subject} {ongoing}",
        ["{determiner} {object}"],
        "{verb}",
        "",
        word_order,
    ) + "."

    return Template(
        name=f"subj_nom-eppen-obj_acc-{object_definiteness.lower()}-verb_{tense.lower()}.",
        template_string=template_string,
        slots={
            "subject": argument_slot("subject", "NOM", "nominative subject"),
            "ongoing": ongoing_particle_slot(),
            "determiner": determiner_slot(object_definiteness),
            "object": argument_slot("object", "ACC", "accusative direct object"),
            "verb": verb_slot(object_definiteness, tense=tense),
        },
        constraints=constraints,
        description=(
            f"Hungarian {tense_name} ongoing transitive sentence "
            f"({object_definiteness})"
        ),
        language_code="hun",
        tags=[
            "stage1",
            "ongoing",
            "transitive",
            tense.lower(),
            object_definiteness.lower(),
            *order_tags(word_order),
        ],
        metadata={
            "frame_family": "TR-ONGOING",
            "tense": tense,
            "object_definiteness": object_definiteness,
            "progressive_morphology": False,
            "word_order": word_order,
        },
    )


# ---------------------------------------------------------------------------
# FRAME INVENTORY
# ---------------------------------------------------------------------------
#
# Names are load-bearing: create_2afc_pairs.py refers to Stage-1 templates by
# name, so the original names are kept exactly and new families only add names.

INTRANSITIVE_SPECS: list[tuple[str, str, list[str]]] = [
    ("subj_nom-verb.", "INTR", []),
    ("subj_nom-noun_dat-verb.", "INTR-DAT", ["dat"]),
    ("subj_nom-noun_loc-verb.", "INTR-LOC", ["loc"]),
    ("subj_nom-noun_inst-verb.", "INTR-INS", ["ins"]),
    ("subj_nom-noun_inst-noun_loc-verb.", "INTR-INS-LOC", ["ins", "loc"]),
    ("subj_nom-noun_inst-noun_dat-verb.", "INTR-INS-DAT", ["ins", "dat"]),
    ("subj_nom-noun_loc-noun_dat-verb.", "INTR-LOC-DAT", ["loc", "dat"]),
    # Korean-parallel adjuncts added in this revision.
    ("subj_nom-noun_goal-verb.", "INTR-GOAL", ["goal"]),
    ("subj_nom-noun_src-verb.", "INTR-SRC", ["src"]),
    ("subj_nom-noun_com-verb.", "INTR-COM", ["com"]),
    ("subj_nom-noun_ter-verb.", "INTR-TER", ["ter"]),
    ("subj_nom-noun_init-verb.", "INTR-INIT", ["init"]),
]

TRANSITIVE_SPECS: list[tuple[str, list[str]]] = [
    ("TR", []),
    ("TR-DAT", ["dat"]),
    ("TR-LOC", ["loc"]),
    ("TR-INS", ["ins"]),
    ("TR-INS-LOC", ["ins", "loc"]),
    ("TR-INS-DAT", ["ins", "dat"]),
    ("TR-LOC-DAT", ["loc", "dat"]),
    # Korean-parallel adjuncts added in this revision.
    ("TR-GOAL", ["goal"]),
    ("TR-SRC", ["src"]),
    ("TR-COM", ["com"]),
    ("TR-TER", ["ter"]),
    ("TR-INIT", ["init"]),
]

SPATIAL_SERIES = ["ESS", "LAT", "ABL"]

FAMILIES = ["nominal", "clausal", "ongoing", "spatial", "complex"]


def build_templates(
    word_order: str = PREVERBAL,
    families: list[str] | None = None,
    adjunct_filter: set[str] | None = None,
) -> List[Template]:
    """Build the Hungarian Stage-1 template inventory."""
    selected = set(families or FAMILIES)
    templates: List[Template] = []

    def keep(adjunct_keys: list[str]) -> bool:
        if adjunct_filter is None:
            return True
        return set(adjunct_keys) <= adjunct_filter

    if "nominal" in selected:
        for name, frame_family, adjunct_keys in INTRANSITIVE_SPECS:
            if keep(adjunct_keys):
                templates.append(
                    make_intransitive_template(
                        name, frame_family, adjunct_keys, word_order
                    )
                )

        # Each abstract ACC frame receives two Hungarian realizations:
        #     INDF object -> INDF verb
        #     DEF object  -> DEF verb
        for frame_family, adjunct_keys in TRANSITIVE_SPECS:
            if not keep(adjunct_keys):
                continue

            name_base = frame_family.lower().replace("-", "_")

            for definiteness, suffix in (("INDF", "indef"), ("DEF", "def")):
                templates.append(
                    make_transitive_template(
                        name=f"{name_base}-{suffix}.",
                        frame_family=frame_family,
                        object_definiteness=definiteness,
                        adjunct_keys=adjunct_keys,
                        word_order=word_order,
                    )
                )

    if "clausal" in selected:
        templates.append(make_hogy_complement_template(word_order))

    if "ongoing" in selected:
        for tense in ("PRS", "PST"):
            templates.append(make_ongoing_intransitive_template(tense, word_order))

            for definiteness in ("INDF", "DEF"):
                templates.append(
                    make_ongoing_transitive_template(
                        tense=tense,
                        object_definiteness=definiteness,
                        word_order=word_order,
                    )
                )

    if "spatial" in selected:
        for series in SPATIAL_SERIES:
            for definiteness in (None, "INDF", "DEF"):
                templates.append(
                    make_spatial_postposition_template(series, definiteness, word_order)
                )

    if "complex" in selected:
        for definiteness in (None, "INDF", "DEF"):
            templates.append(
                make_complex_postposition_template(definiteness, word_order)
            )

    return templates


# ---------------------------------------------------------------------------
# SAVE
# ---------------------------------------------------------------------------

def main(
    template_limit: int | None = None,
    word_order: str = PREVERBAL,
    families: list[str] | None = None,
    adjuncts: list[str] | None = None,
    output: Path | None = None,
) -> None:
    """Generate and save the Hungarian Stage-1 template inventory."""
    templates = build_templates(
        word_order=word_order,
        families=families,
        adjunct_filter=set(adjuncts) if adjuncts else None,
    )

    if template_limit is not None:
        templates = templates[:template_limit]

    base_dir = Path(__file__).parent
    templates_dir = base_dir / "templates"
    templates_dir.mkdir(exist_ok=True)

    output_path = output or (templates_dir / "generic_frames.jsonl")

    with output_path.open("w", encoding="utf-8") as output_file:
        for template in templates:
            output_file.write(template.model_dump_json() + "\n")

    print(f"Generated {len(templates)} Hungarian Stage-1 templates ({word_order} order).")
    print(f"Saved templates to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Hungarian Stage-1 argument-structure templates"
    )
    parser.add_argument(
        "--limit", type=int, default=None, help="Limit templates generated for testing"
    )
    parser.add_argument(
        "--word-order",
        choices=[PREVERBAL, NEUTRAL],
        default=PREVERBAL,
        help=(
            "Constituent order. 'preverbal' keeps arguments between the subject "
            "and the verb (the original Stage-1 order); 'neutral' places the "
            "finite verb directly after the subject."
        ),
    )
    parser.add_argument(
        "--include",
        nargs="+",
        choices=FAMILIES,
        default=None,
        help="Frame families to generate (default: all)",
    )
    parser.add_argument(
        "--adjuncts",
        nargs="+",
        choices=sorted(ADJUNCTS),
        default=None,
        help="Restrict nominal frames to these adjunct types",
    )
    parser.add_argument(
        "--output", type=Path, default=None, help="Override the output path"
    )
    args = parser.parse_args()

    main(
        template_limit=args.limit,
        word_order=args.word_order,
        families=args.include,
        adjuncts=args.adjuncts,
        output=args.output,
    )
