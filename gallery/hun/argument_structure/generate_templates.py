#!/usr/bin/env python3
"""
Generate Stage-1 Hungarian argument-structure templates.

This is the Hungarian Stage-1 version of the ENG/KOR argument-structure task.
The basic experimental question is unchanged: which verbs are acceptable in
which frames? The frames are realized with Hungarian morphology.

Korean-aligned structural baseline:
    INTR
    INTR + DAT
    INTR + LOC
    INTR + INS
    INTR + INS + LOC
    INTR + INS + DAT
    INTR + LOC + DAT

    TR
    TR + DAT
    TR + LOC
    TR + INS
    TR + INS + LOC
    TR + INS + DAT
    TR + LOC + DAT

Hungarian-specific choices:
    - nouns are already case-inflected in the lexicon;
    - LOC is realized as SUP with hely → helyen for a neutral “at/in a place” baseline;
    - Korean INST corresponds to Hungarian INS;
    - every ACC frame has INDF and DEF versions;
    - INDF objects use egy;
    - DEF objects use a/az and DEF verbal conjugation;
    - finite baseline verbs are IND, PRS/PST, 3SG;
    - ongoing templates use éppen rather than a fake Hungarian
      morphological progressive;
    - one finite direct-object hogy + indicative clausal-complement frame is included.

The richer Hungarian local-case system, preverbs, pro-drop, focus, argument
omission, and potential mood (POT) are reserved for Stage 2.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from bead.resources import Constraint, Slot, Template


VOWELS = "aáeéiíoóöőuúüű"
# SLOT HELPERS
def noun_slot(name: str, case_name: str, description: str, semantic_classes: list[str] | None = None, lemma: str | None = None) -> Slot:
    """Create a case-constrained singular noun slot."""
    conditions = [
        "self.features.get('pos') == 'NOUN'",
        f"self.features.get('case') == '{case_name}'",
        "self.features.get('number') == 'SG'",
    ]

    if semantic_classes:
        conditions.append(f"self.features.get('semantic_class') in {semantic_classes!r}")

    if lemma is not None:
        conditions.append(f"self.lemma == '{lemma}'")

    return Slot(
        name=name,
        description=description,
        constraints=[Constraint(expression=" and ".join(conditions))],
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
# DETERMINER PHONOLOGY
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
# TEMPLATE CONSTRUCTION
def make_intransitive_template(
    name: str,
    frame_family: str,
    obliques: list[tuple[str, str, str]],
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
        "subject": noun_slot("subject", "NOM", "nominative subject", semantic_classes=["human", "group"]),
        "verb": verb_slot("INDF"),
    }

    parts = ["Egy {subject}"]

    for slot_name, case_name, description in obliques:
        if slot_name == "dative":
            slots[slot_name] = noun_slot(slot_name, case_name, description, semantic_classes=["human", "group"])
        elif slot_name == "location":
            slots[slot_name] = noun_slot(slot_name, case_name, description, semantic_classes=["location"])
        elif slot_name == "instrument":
            slots[slot_name] = noun_slot(slot_name, case_name, description, semantic_classes=["instrument"])
        else:
            slots[slot_name] = noun_slot(slot_name, case_name, description)
        parts.append(f"egy {{{slot_name}}}")

    parts.append("{verb}.")

    return Template(
        name=name,
        template_string=" ".join(parts),
        slots=slots,
        constraints=[],
        description=f"Hungarian Stage-1 {frame_family} frame",
        language_code="hun",
        tags=["stage1", "nominal", "intransitive"],
        metadata={
            "frame_family": frame_family,
            "transitivity": "intransitive",
            "object_definiteness": None,
        },
    )


def make_transitive_template(
    name: str,
    frame_family: str,
    object_definiteness: str,
    obliques: list[tuple[str, str, str]],
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
        "subject": noun_slot("subject", "NOM", "nominative subject", semantic_classes=["human", "group"]),
        "determiner": determiner_slot(object_definiteness),
        "object": noun_slot(
            "object",
            "ACC",
            "accusative direct object",
            # Stage 1 restricts direct objects to inanimates to reduce semantic
            # confounds in verb-frame judgments.
            semantic_classes=["inanimate_object"],
        ),
        "verb": verb_slot(object_definiteness),
    }

    parts = ["Egy {subject}", "{determiner} {object}"]

    for slot_name, case_name, description in obliques:
        if slot_name == "dative":
            slots[slot_name] = noun_slot(slot_name, case_name, description, semantic_classes=["human", "group"])
        elif slot_name == "location":
            slots[slot_name] = noun_slot(slot_name, case_name, description, semantic_classes=["location"])
        elif slot_name == "instrument":
            slots[slot_name] = noun_slot(slot_name, case_name, description, semantic_classes=["instrument"])
        else:
            slots[slot_name] = noun_slot(slot_name, case_name, description)
        parts.append(f"egy {{{slot_name}}}")

    parts.append("{verb}.")

    constraints = []

    if object_definiteness == "DEF":
        constraints.append(definite_article_constraint("determiner", "object"))

    return Template(
        name=name,
        template_string=" ".join(parts),
        slots=slots,
        constraints=constraints,
        description=f"Hungarian Stage-1 {frame_family} frame ({object_definiteness})",
        language_code="hun",
        tags=["stage1", "nominal", "transitive", object_definiteness.lower()],
        metadata={
            "frame_family": frame_family,
            "transitivity": "transitive",
            "object_definiteness": object_definiteness,
        },
    )
# CLAUSAL COMPLEMENT
def make_hogy_complement_template() -> Template:
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
    ... “says that ...”). This template therefore requires DEF object agreement.
    A verb that only takes an oblique hogy-clause is a different argument-structure
    type and is not represented by this Stage-1 direct-object-clause frame.
    """

    return Template(
        name="subj_nom-verb-hogy_ind.",
        template_string="Egy {subject} {verb}, hogy egy {comp_subject} {comp_verb}.",
        slots={
            "subject": noun_slot("subject", "NOM", "matrix-clause subject", semantic_classes=["human", "group"]),
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
        tags=["stage1", "clausal", "hogy", "finite", "indicative"],
        metadata={
            "frame_family": "HOGY-IND",
            "complementizer": "hogy",
            "complement_type": "finite_indicative_direct_object",
            "matrix_object_agreement": "DEF",
        },
    )
# ONGOING / KOREAN-PROGRESSIVE ANALOGUE
def make_ongoing_intransitive_template(tense: str) -> Template:
    """
    Hungarian does not have the Korean V-go iss progressive paradigm.

    éppen ('right now / just at that moment') creates an ongoing-event
    context. This is an ongoing construction, not a claim that Hungarian has
    a morphological progressive category.
    """

    tense_name = "present" if tense == "PRS" else "past"

    return Template(
        name=f"subj_nom-eppen-verb_{tense.lower()}.",
        template_string="Egy {subject} {ongoing} {verb}.",
        slots={
            "subject": noun_slot("subject", "NOM", "nominative subject", semantic_classes=["human", "group"]),
            "ongoing": ongoing_particle_slot(),
            "verb": verb_slot("INDF", tense=tense),
        },
        constraints=[],
        description=f"Hungarian {tense_name} ongoing intransitive sentence",
        language_code="hun",
        tags=["stage1", "ongoing", "intransitive", tense.lower()],
        metadata={
            "frame_family": "INTR-ONGOING",
            "tense": tense,
            "progressive_morphology": False,
        },
    )


def make_ongoing_transitive_template(tense: str, object_definiteness: str) -> Template:
    tense_name = "present" if tense == "PRS" else "past"

    constraints = []

    if object_definiteness == "DEF":
        constraints.append(definite_article_constraint("determiner", "object"))

    return Template(
        name=f"subj_nom-eppen-obj_acc-{object_definiteness.lower()}-verb_{tense.lower()}.",
        template_string="Egy {subject} {ongoing} {determiner} {object} {verb}.",
        slots={
            "subject": noun_slot("subject", "NOM", "nominative subject", semantic_classes=["human", "group"]),
            "ongoing": ongoing_particle_slot(),
            "determiner": determiner_slot(object_definiteness),
            "object": noun_slot("object", "ACC", "accusative direct object", semantic_classes=["inanimate_object"]),
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
        ],
        metadata={
            "frame_family": "TR-ONGOING",
            "tense": tense,
            "object_definiteness": object_definiteness,
            "progressive_morphology": False,
        },
    )
# FULL STAGE-1 INVENTORY
def build_templates() -> List[Template]:
    """Build the complete 28-template Hungarian Stage-1 inventory."""
    templates: List[Template] = []
    # KOREAN-ALIGNED INTRANSITIVE STRUCTURAL INVENTORY
    #
    # LOC is realized as Hungarian SUP with hely → helyen for Stage 1.
    # INST is realized as Hungarian INS.
    intransitive_specs = [
        ("subj_nom-verb.", "INTR", []),

        (
            "subj_nom-noun_dat-verb.",
            "INTR-DAT",
            [("dative", "DAT", "dative argument")],
        ),

        (
            "subj_nom-noun_loc-verb.",
            "INTR-LOC",
            [("location", "SUP", "neutral locative argument")],
        ),

        (
            "subj_nom-noun_inst-verb.",
            "INTR-INS",
            [("instrument", "INS", "instrumental-comitative argument")],
        ),

        (
            "subj_nom-noun_inst-noun_loc-verb.",
            "INTR-INS-LOC",
            [
                ("instrument", "INS", "instrumental-comitative argument"),
                ("location", "SUP", "neutral locative argument"),
            ],
        ),

        (
            "subj_nom-noun_inst-noun_dat-verb.",
            "INTR-INS-DAT",
            [
                ("instrument", "INS", "instrumental-comitative argument"),
                ("dative", "DAT", "dative argument"),
            ],
        ),

        (
            "subj_nom-noun_loc-noun_dat-verb.",
            "INTR-LOC-DAT",
            [
                ("location", "SUP", "neutral locative argument"),
                ("dative", "DAT", "dative argument"),
            ],
        ),
    ]

    for name, frame_family, obliques in intransitive_specs:
        templates.append(make_intransitive_template(name, frame_family, obliques))
    # KOREAN-ALIGNED TRANSITIVE STRUCTURAL INVENTORY
    #
    # Each abstract ACC frame receives two Hungarian realizations:
    #
    #     INDF object -> INDF verb
    #     DEF object  -> DEF verb
    transitive_specs = [
        ("TR", []),

        (
            "TR-DAT",
            [("dative", "DAT", "dative argument")],
        ),

        (
            "TR-LOC",
            [("location", "SUP", "neutral locative argument")],
        ),

        (
            "TR-INS",
            [("instrument", "INS", "instrumental-comitative argument")],
        ),

        (
            "TR-INS-LOC",
            [
                ("instrument", "INS", "instrumental-comitative argument"),
                ("location", "SUP", "neutral locative argument"),
            ],
        ),

        (
            "TR-INS-DAT",
            [
                ("instrument", "INS", "instrumental-comitative argument"),
                ("dative", "DAT", "dative argument"),
            ],
        ),

        (
            "TR-LOC-DAT",
            [
                ("location", "SUP", "neutral locative argument"),
                ("dative", "DAT", "dative argument"),
            ],
        ),
    ]

    for frame_family, obliques in transitive_specs:
        name_base = frame_family.lower().replace("-", "_")

        templates.append(
            make_transitive_template(
                name=f"{name_base}-indef.",
                frame_family=frame_family,
                object_definiteness="INDF",
                obliques=obliques,
            )
        )

        templates.append(
            make_transitive_template(
                name=f"{name_base}-def.",
                frame_family=frame_family,
                object_definiteness="DEF",
                obliques=obliques,
            )
        )
    # CLAUSAL COMPLEMENT
    templates.append(make_hogy_complement_template())
    # ONGOING ANALOGUES OF KOREAN PROGRESSIVE TEMPLATES
    for tense in ("PRS", "PST"):
        templates.append(make_ongoing_intransitive_template(tense))

        for definiteness in ("INDF", "DEF"):
            templates.append(
                make_ongoing_transitive_template(
                    tense=tense,
                    object_definiteness=definiteness,
                )
            )

    return templates
# SAVE
def main(template_limit: int | None = None) -> None:
    """Generate and save the Hungarian Stage-1 template inventory."""
    templates = build_templates()

    if template_limit is not None:
        templates = templates[:template_limit]

    base_dir = Path(__file__).parent
    templates_dir = base_dir / "templates"
    templates_dir.mkdir(exist_ok=True)

    output_path = templates_dir / "generic_frames.jsonl"

    with output_path.open("w", encoding="utf-8") as output_file:
        for template in templates:
            output_file.write(template.model_dump_json() + "\n")

    print(f"Generated {len(templates)} Hungarian Stage-1 templates.")
    print(f"Saved templates to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate Hungarian Stage-1 argument-structure templates")
    parser.add_argument("--limit", type=int, default=None, help="Limit templates generated for testing")
    args = parser.parse_args()

    main(template_limit=args.limit)