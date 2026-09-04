#!/usr/bin/env python3
"""Regenerate FRAME_REFERENCE.csv from the template inventory.

The frame reference is documentation, so it drifts the moment a frame is added
or renamed. Generating it from `build_templates()` keeps the two in step, and
`--check` lets the test suite catch a stale file.

The Hungarian examples use `dolgozik` ('works') as a neutral matrix verb, the
same verb the hand-written original used. As in the rest of Stage 1, an example
illustrates the *shape* of a frame; it does not claim that `dolgozik` — or any
other verb — is acceptable in it.

Usage, from `hun/argument_structure`:

    python build_frame_reference.py
    python build_frame_reference.py --check
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

GALLERY_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(GALLERY_DIR))

from generate_templates import PREVERBAL, build_templates  # noqa: E402
from tests.render_samples import build_lexicon  # noqa: E402

TARGET = GALLERY_DIR / "FRAME_REFERENCE.csv"

FIELDNAMES = [
    "frame_number",
    "template_name",
    "frame_family",
    "transitivity",
    "object_definiteness",
    "frame_description",
    "hungarian_example",
    "english_translation",
]

# Matrix verbs for the examples. The point of the reference is to show the
# *shape* of each frame with a grammatical sentence, so the verb is chosen to
# suit the frame rather than held constant: a transitive frame needs a verb
# that takes an object, and a goal or source frame needs a motion verb.
#
# This is presentation only. The generator itself makes no such choice: it
# crosses every verb with every frame, and a verb-frame mismatch there is the
# data the experiment is meant to collect.
MOTION_ADJUNCTS = {"goal", "src", "ter", "init"}

MATRIX_VERBS = {
    ("intransitive", "PRS"): "dolgozik",
    ("intransitive", "PST"): "dolgozott",
    ("motion", "PRS"): "megy",
    ("motion", "PST"): "ment",
    ("transitive_indf", "PRS"): "tesz",
    ("transitive_indf", "PST"): "tett",
    ("transitive_def", "PRS"): "teszi",
    ("transitive_def", "PST"): "tette",
    ("clausal", "PRS"): "mondja",
}


def matrix_verb(template) -> str:
    """Pick an example verb that fits the frame."""
    metadata = template.metadata
    tense = metadata.get("tense", "PRS")

    if metadata.get("frame_family") == "HOGY-IND":
        return MATRIX_VERBS[("clausal", "PRS")]

    if metadata.get("transitivity") == "transitive":
        key = (
            "transitive_def"
            if metadata.get("object_definiteness") == "DEF"
            else "transitive_indf"
        )
        return MATRIX_VERBS[(key, tense)]

    adjuncts = set(metadata.get("adjuncts", []))
    directional = metadata.get("postposition_series") in {"LAT", "ABL"}

    if adjuncts & MOTION_ADJUNCTS or directional:
        return MATRIX_VERBS[("motion", tense)]

    return MATRIX_VERBS[("intransitive", tense)]

SLOT_EXAMPLES = {
    "subject": ("ember", "a person"),
    "object": ("tárgyat", "the object"),
    "determiner": ("egy", "a"),
    "dative": ("csoportnak", "for a group"),
    "location": ("helyen", "at a place"),
    "instrument": ("eszközzel", "with a tool"),
    "goal": ("helyre", "to a place"),
    "source": ("helyről", "from a place"),
    "comitative": ("emberrel", "with a person"),
    "terminus": ("helyig", "as far as a place"),
    "origin": ("helytől", "from a place"),
    "postp_object": ("hely", "a place"),
    "complex_postp_object": ("ember", "a person"),
}

ADJUNCT_GLOSS = {
    "dat": "dative",
    "loc": "locative",
    "ins": "instrumental",
    "goal": "goal",
    "src": "source",
    "com": "comitative",
    "ter": "terminative",
    "init": "starting-point",
}

SERIES_GLOSS = {"ESS": "essive", "LAT": "lative", "ABL": "ablative"}


def describe(template) -> str:
    """Build a short prose description of a frame from its metadata."""
    metadata = template.metadata
    family = metadata.get("frame_family", "")
    parts = ["Subject"]

    if metadata.get("transitivity") == "transitive":
        definiteness = metadata.get("object_definiteness")
        article = "indefinite" if definiteness == "INDF" else "definite"
        parts.append(f"{article} object")

    for adjunct in metadata.get("adjuncts", []):
        parts.append(ADJUNCT_GLOSS[adjunct])

    if metadata.get("postposition_type") == "spatial":
        series = SERIES_GLOSS[metadata["postposition_series"]]
        parts.append(f"{series} bare postposition")
    elif metadata.get("postposition_type") == "complex":
        parts.append("case-governing postposition")

    if family == "HOGY-IND":
        parts.append("finite direct-object hogy-clause")

    if metadata.get("frame_family") in {"INTR-ONGOING", "TR-ONGOING"}:
        tense = "present" if metadata.get("tense") == "PRS" else "past"
        parts.append(f"éppen ({tense})")

    parts.append("verb")

    if metadata.get("object_definiteness") == "DEF" and family != "HOGY-IND":
        return " + ".join(parts) + " (objective conjugation)"

    return " + ".join(parts)


def example_sentence(template) -> tuple[str, str]:
    """Render a Hungarian example and a rough English gloss."""
    surfaces: dict[str, str] = {}

    for slot_name in template.slots:
        if slot_name in SLOT_EXAMPLES:
            surfaces[slot_name] = SLOT_EXAMPLES[slot_name][0]

    definiteness = template.metadata.get("object_definiteness")

    if "determiner" in template.slots:
        surfaces["determiner"] = "a" if definiteness == "DEF" else "egy"

    if "object" in template.slots:
        surfaces["object"] = "tárgyat"

    if "postposition" in template.slots:
        if template.metadata.get("postposition_type") == "spatial":
            surfaces["postposition"] = {
                "ESS": "mögött",
                "LAT": "mögé",
                "ABL": "mögül",
            }[template.metadata["postposition_series"]]
        else:
            surfaces["postposition"] = "szerint"

    if "ongoing" in template.slots:
        surfaces["ongoing"] = "éppen"

    if "comp_subject" in template.slots:
        surfaces["comp_subject"] = "esemény"
        surfaces["comp_verb"] = "történik"

    surfaces["verb"] = matrix_verb(template)

    hungarian = template.template_string.format_map(surfaces)
    hungarian = " ".join(hungarian.split())

    for punctuation in (".", ","):
        hungarian = hungarian.replace(f" {punctuation}", punctuation)

    return hungarian, gloss(template)


def gloss(template) -> str:
    """A rough English rendering, enough to read the table by."""
    metadata = template.metadata
    pieces = ["A person"]

    ongoing = metadata.get("frame_family") in {"INTR-ONGOING", "TR-ONGOING"}
    past = metadata.get("tense") == "PST"

    if metadata.get("transitivity") == "transitive":
        verb = "put" if past else "puts"
        pieces.append(f"is right now putting" if ongoing and not past else verb)
        article = "an" if metadata.get("object_definiteness") == "INDF" else "the"
        pieces.append(f"{article} object")
    elif set(metadata.get("adjuncts", [])) & MOTION_ADJUNCTS or metadata.get(
        "postposition_series"
    ) in {"LAT", "ABL"}:
        pieces.append("went" if past else "goes")
    else:
        verb = "worked" if past else "works"
        pieces.append("is right now working" if ongoing and not past else verb)

    for adjunct in metadata.get("adjuncts", []):
        slot_name = {
            "dat": "dative",
            "loc": "location",
            "ins": "instrument",
            "goal": "goal",
            "src": "source",
            "com": "comitative",
            "ter": "terminus",
            "init": "origin",
        }[adjunct]
        pieces.append(SLOT_EXAMPLES[slot_name][1])

    if metadata.get("postposition_type") == "spatial":
        pieces.append(
            {"ESS": "behind a place", "LAT": "to behind a place", "ABL": "from behind a place"}[
                metadata["postposition_series"]
            ]
        )
    elif metadata.get("postposition_type") == "complex":
        pieces.append("according to a person")

    if metadata.get("frame_family") == "HOGY-IND":
        pieces.append("that an event is happening")

    return " ".join(pieces) + "."


def build_csv_text() -> str:
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=FIELDNAMES, lineterminator="\n")
    writer.writeheader()

    for number, template in enumerate(build_templates(word_order=PREVERBAL), start=1):
        hungarian, english = example_sentence(template)

        writer.writerow(
            {
                "frame_number": number,
                "template_name": template.name,
                "frame_family": template.metadata.get("frame_family", ""),
                "transitivity": template.metadata.get("transitivity", ""),
                "object_definiteness": template.metadata.get("object_definiteness") or "",
                "frame_description": describe(template),
                "hungarian_example": hungarian,
                "english_translation": english,
            }
        )

    return buffer.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--stdout", action="store_true")
    arguments = parser.parse_args()

    text = build_csv_text()

    if arguments.stdout:
        sys.stdout.write(text)
        return 0

    if arguments.check:
        if not TARGET.exists() or TARGET.read_text(encoding="utf-8") != text:
            print("FRAME_REFERENCE.csv is out of date; run build_frame_reference.py")
            return 1

        print("FRAME_REFERENCE.csv is up to date")
        return 0

    TARGET.write_text(text, encoding="utf-8")
    print(f"Wrote {text.count(chr(10)) - 1} frames to {TARGET}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
