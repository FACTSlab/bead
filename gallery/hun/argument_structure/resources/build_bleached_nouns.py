#!/usr/bin/env python3
"""Build the controlled Hungarian bleached-noun paradigm.

`bleached_nouns.csv` used to be maintained by hand. That does not scale past a
handful of lemmas: Hungarian marks 17 productive cases on every noun, so each
new bleached lemma means 17 hand-typed forms, and a single typo becomes a
silent confound in the acceptability data.

This script generates the paradigm instead. Everything that follows from vowel
harmony or from `-v-` assimilation is computed; everything that does not
(accusative linking vowels, stem alternations) is listed explicitly in
`STEMS` and asserted against the generator.

Run from `hun/argument_structure/resources`:

    python build_bleached_nouns.py            # rewrite bleached_nouns.csv
    python build_bleached_nouns.py --check    # verify the CSV is up to date
    python build_bleached_nouns.py --stdout   # print without writing

The `--check` mode is what `tests/test_resources.py` calls, so a hand edit to
the CSV that drifts from the generator fails the test suite.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Vowel harmony
# ---------------------------------------------------------------------------

BACK_VOWELS = set("aáoóuú")
FRONT_ROUNDED_VOWELS = set("öőüű")
FRONT_UNROUNDED_VOWELS = set("eéií")
VOWELS = BACK_VOWELS | FRONT_ROUNDED_VOWELS | FRONT_UNROUNDED_VOWELS

BACK = "back"
FRONT_UNROUNDED = "front_unrounded"
FRONT_ROUNDED = "front_rounded"


def harmony_class(stem: str) -> str:
    """Return the harmony class governing suffix selection for `stem`.

    The standard generalization for native, non-antiharmonic stems:

    - any back vowel in the stem selects back suffixes;
    - otherwise the *last* vowel decides rounding, which only matters for the
      three-way suffixes (ALL `-hoz/-hez/-höz`, SUP `-on/-en/-ön`).

    Front unrounded `i í é` are transparent in the sense that they never force
    a front class on a stem that also contains a back vowel; that falls out of
    the back-vowel check running first.
    """
    letters = [character for character in stem.lower() if character in VOWELS]

    if not letters:
        raise ValueError(f"stem has no vowel: {stem!r}")

    if any(vowel in BACK_VOWELS for vowel in letters):
        return BACK

    if letters[-1] in FRONT_ROUNDED_VOWELS:
        return FRONT_ROUNDED

    return FRONT_UNROUNDED


# ---------------------------------------------------------------------------
# Case suffixes
# ---------------------------------------------------------------------------
#
# Each entry maps a normalized case label to its allomorphs, keyed by harmony
# class. Two-way suffixes list `front_rounded` as `None`, meaning "fall back to
# front_unrounded"; three-way suffixes give all three.
#
# NOM, ACC, INS and TRA are handled separately: NOM is null, ACC needs a
# linking vowel that is not fully predictable, and INS/TRA assimilate.

SUFFIXES: dict[str, dict[str, str | None]] = {
    "DAT": {BACK: "nak", FRONT_UNROUNDED: "nek", FRONT_ROUNDED: None},
    "INE": {BACK: "ban", FRONT_UNROUNDED: "ben", FRONT_ROUNDED: None},
    "ILL": {BACK: "ba", FRONT_UNROUNDED: "be", FRONT_ROUNDED: None},
    "ELA": {BACK: "ból", FRONT_UNROUNDED: "ből", FRONT_ROUNDED: None},
    "ADE": {BACK: "nál", FRONT_UNROUNDED: "nél", FRONT_ROUNDED: None},
    "ALL": {BACK: "hoz", FRONT_UNROUNDED: "hez", FRONT_ROUNDED: "höz"},
    "ABL": {BACK: "tól", FRONT_UNROUNDED: "től", FRONT_ROUNDED: None},
    "SUP": {BACK: "on", FRONT_UNROUNDED: "en", FRONT_ROUNDED: "ön"},
    "SUB": {BACK: "ra", FRONT_UNROUNDED: "re", FRONT_ROUNDED: None},
    "DEL": {BACK: "ról", FRONT_UNROUNDED: "ről", FRONT_ROUNDED: None},
    "TER": {BACK: "ig", FRONT_UNROUNDED: "ig", FRONT_ROUNDED: "ig"},
    "CAU_FIN": {BACK: "ért", FRONT_UNROUNDED: "ért", FRONT_ROUNDED: "ért"},
    "ESS_FOR": {BACK: "ként", FRONT_UNROUNDED: "ként", FRONT_ROUNDED: "ként"},
}

# Cases generated for every lemma, in paradigm order.
CASE_ORDER = [
    "NOM",
    "ACC",
    "DAT",
    "INE",
    "ILL",
    "ELA",
    "ADE",
    "ALL",
    "ABL",
    "SUP",
    "SUB",
    "DEL",
    "INS",
    "TER",
    "CAU_FIN",
    "ESS_FOR",
    "TRA",
]

# `-kor` is lexically restricted to temporal nouns, so it is opt-in per lemma
# rather than part of CASE_ORDER.
TEMPORAL_CASE = "TEMP"


def apply_suffix(stem: str, case: str) -> str:
    """Attach a harmony-selected suffix to `stem`."""
    allomorphs = SUFFIXES[case]
    harmony = harmony_class(stem)

    suffix = allomorphs[harmony]

    if suffix is None:
        suffix = allomorphs[FRONT_UNROUNDED]

    # SUP is the one generated case with a post-vocalic allomorph: after a
    # vowel-final stem the linking vowel drops (idő -> időn, not *időön).
    if case == "SUP" and stem[-1] in VOWELS:
        return stem + "n"

    return stem + suffix


def assimilate_v(stem: str, back_form: str, front_form: str) -> str:
    """Realize a `-v-` initial suffix (INS `-val/-vel`, TRA `-vá/-vé`).

    After a vowel the `v` surfaces unchanged (idő + vel -> idővel). After a
    consonant it fully assimilates to that consonant, which for Hungarian
    digraphs and trigraphs means copying the whole grapheme:

        tárgy + val -> tárggyal      (gy -> ggy, not *gyy)
        hely  + vel -> hellyel       (ly -> lly)
        eszköz + vel -> eszközzel    (z  -> zz)
    """
    harmony = harmony_class(stem)
    suffix = back_form if harmony == BACK else front_form

    if stem[-1] in VOWELS:
        return stem + suffix

    # Longest-first so trigraphs beat digraphs beat single letters.
    for grapheme in ("dzs", "cs", "dz", "gy", "ly", "ny", "sz", "ty", "zs"):
        if stem.endswith(grapheme):
            # A digraph doubles by repeating its *first* letter: gy -> ggy.
            doubled = grapheme[0] + grapheme
            return stem[: -len(grapheme)] + doubled + suffix[1:]

    return stem + stem[-1] + suffix[1:]


def instrumental(stem: str) -> str:
    return assimilate_v(stem, "val", "vel")


def translative(stem: str) -> str:
    return assimilate_v(stem, "vá", "vé")


# ---------------------------------------------------------------------------
# Stem inventory
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Stem:
    """One controlled bleached noun.

    `accusative` is always given explicitly: the linking vowel in the Hungarian
    accusative (`-t` / `-at` / `-ot` / `-et` / `-öt`) depends on the shape of
    the final consonant cluster in ways that are not worth approximating for a
    hand-picked inventory of this size.

    `overrides` covers stem alternations case by case. No lemma in the current
    inventory needs it: alternating stems were deliberately excluded (see the
    note next to `darab`). It stays because the escape hatch is what makes it
    safe to add a lemma like `ló` or `tó` later without weakening the generator
    for everything else.

    `role` records why the lemma is in the inventory. `controlled` lemmas are
    pinned by `generate_templates.py` as the fixed filler for a particular
    frame position; `filler` lemmas vary freely within their semantic class.
    """

    lemma: str
    accusative: str
    semantic_class: str
    countability: str
    animacy: str
    role: str = "filler"
    temporal: bool = False
    overrides: dict[str, str] = field(default_factory=dict)


STEMS: list[Stem] = [
    # -- animate ------------------------------------------------------------
    Stem("ember", "embert", "human", "countable", "animate", role="controlled"),
    Stem("személy", "személyt", "human", "countable", "animate"),
    Stem("csoport", "csoportot", "group", "countable", "variable"),
    Stem("szervezet", "szervezetet", "group", "countable", "variable"),
    Stem("állat", "állatot", "animal", "countable", "animate"),
    # -- inanimate objects --------------------------------------------------
    Stem(
        "tárgy",
        "tárgyat",
        "inanimate_object",
        "countable",
        "inanimate",
        role="controlled",
    ),
    # `dolog` would be the obvious third inanimate here, but it has a
    # shortening stem (dolgot, dolgon) that alternates before vowel-initial
    # suffixes only. Controlled bleached fillers should not carry that kind of
    # unpredictability, so `darab` is used instead.
    Stem("darab", "darabot", "inanimate_object", "countable", "inanimate"),
    Stem("anyag", "anyagot", "substance", "mass", "inanimate"),
    # -- instruments --------------------------------------------------------
    Stem(
        "eszköz",
        "eszközt",
        "instrument",
        "countable",
        "inanimate",
        role="controlled",
    ),
    Stem("szerszám", "szerszámot", "instrument", "countable", "inanimate"),
    # -- locations ----------------------------------------------------------
    Stem("hely", "helyet", "location", "countable", "inanimate", role="controlled"),
    Stem("terület", "területet", "location", "countable", "inanimate"),
    # -- events -------------------------------------------------------------
    Stem("esemény", "eseményt", "event", "countable", "inanimate", role="controlled"),
    Stem("folyamat", "folyamatot", "event", "countable", "inanimate"),
    # -- abstract -----------------------------------------------------------
    Stem("tény", "tényt", "abstract", "countable", "inanimate"),
    Stem("ügy", "ügyet", "abstract", "countable", "inanimate"),
    # -- temporal -----------------------------------------------------------
    Stem("idő", "időt", "time", "mass", "inanimate"),
    Stem("pillanat", "pillanatot", "time", "countable", "inanimate", temporal=True),
]


def build_paradigm(stem: Stem) -> list[dict[str, str]]:
    """Return one row per case for a single lemma."""
    rows: list[dict[str, str]] = []

    cases = list(CASE_ORDER)

    if stem.temporal:
        cases.append(TEMPORAL_CASE)

    for case in cases:
        if case in stem.overrides:
            form = stem.overrides[case]
        elif case == "NOM":
            form = stem.lemma
        elif case == "ACC":
            form = stem.accusative
        elif case == "INS":
            form = instrumental(stem.lemma)
        elif case == "TRA":
            form = translative(stem.lemma)
        elif case == TEMPORAL_CASE:
            form = stem.lemma + "kor"
        else:
            form = apply_suffix(stem.lemma, case)

        rows.append(
            {
                "lemma": stem.lemma,
                "form": form,
                "case": case,
                "number": "SG",
                "semantic_class": stem.semantic_class,
                "countability": stem.countability,
                "animacy": stem.animacy,
                "harmony": harmony_class(stem.lemma),
                "role": stem.role,
            }
        )

    return rows


FIELDNAMES = [
    "lemma",
    "form",
    "case",
    "number",
    "semantic_class",
    "countability",
    "animacy",
    "harmony",
    "role",
]


def build_csv_text() -> str:
    """Render the whole paradigm as CSV text."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=FIELDNAMES, lineterminator="\n")
    writer.writeheader()

    for stem in STEMS:
        for row in build_paradigm(stem):
            writer.writerow(row)

    return buffer.getvalue()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero if bleached_nouns.csv differs from the generator.",
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print the generated CSV instead of writing it.",
    )
    arguments = parser.parse_args()

    text = build_csv_text()
    target = Path(__file__).with_name("bleached_nouns.csv")

    if arguments.stdout:
        sys.stdout.write(text)
        return 0

    if arguments.check:
        if not target.exists():
            print(f"{target.name} does not exist; run build_bleached_nouns.py")
            return 1

        if target.read_text(encoding="utf-8") != text:
            print(
                f"{target.name} is out of date with build_bleached_nouns.py; "
                "re-run the generator"
            )
            return 1

        print(f"{target.name} is up to date")
        return 0

    target.write_text(text, encoding="utf-8")

    lemma_count = len(STEMS)
    row_count = text.count("\n") - 1
    print(f"Wrote {row_count} forms for {lemma_count} lemmas to {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
