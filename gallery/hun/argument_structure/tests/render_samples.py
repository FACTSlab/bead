#!/usr/bin/env python3
"""Render sample sentences from the template inventory without running bead.

`fill_templates.py` needs the full bead stack plus a downloaded UniMorph verb
lexicon. This script is a much smaller thing: it loads the resource CSVs,
evaluates the same constraint expressions the real resolver evaluates, and
renders one example per template using a handful of stub verbs.

It exists so that a change to `generate_templates.py` can be eyeballed in
Hungarian in about a second, and so `tests/test_templates.py` has something to
assert against that does not require the pipeline's dependencies.

Usage, from `hun/argument_structure`:

    python -m tests.render_samples
    python -m tests.render_samples --word-order neutral
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from pathlib import Path
from typing import Any, Iterator

GALLERY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GALLERY_DIR))

RESOURCES = GALLERY_DIR / "resources"


class Filler:
    """Stand-in for bead's LexicalItem, with the attributes constraints use."""

    def __init__(self, lemma: str, form: str, features: dict[str, Any]):
        self.lemma = lemma
        self.form = form
        self.features = features

    def __repr__(self) -> str:
        return f"Filler({self.form!r})"


def _coerce(value: str) -> Any:
    """Match generate_lexicons.clean_value's true/false/blank handling."""
    value = (value or "").strip()

    if not value:
        return None
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False

    return value


def load_csv(name: str, pos: str, identifier: str) -> list[Filler]:
    fillers = []

    with (RESOURCES / name).open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            lemma = (row.get(identifier) or "").strip()

            if not lemma:
                continue

            form = (row.get("form") or "").strip() or lemma
            features: dict[str, Any] = {"pos": pos}

            for key, raw in row.items():
                if key in {identifier, "form", "notes"}:
                    continue

                coerced = _coerce(raw)

                if coerced is not None:
                    features[key] = coerced

            fillers.append(Filler(lemma, form, features))

    return fillers


# A few hand-built verb forms standing in for the UniMorph lexicon. `mond`
# is included because it is the one verb here that plausibly licenses the
# hogy frame, and `történik` because the hogy frame's embedded clause needs it.
def stub_verbs() -> list[Filler]:
    def verb(lemma: str, form: str, tense: str, agreement: str) -> Filler:
        return Filler(
            lemma,
            form,
            {
                "pos": "V",
                "finiteness": "FIN",
                "mood": "IND",
                "person": "3",
                "number": "SG",
                "tense": tense,
                "object_agreement": agreement,
            },
        )

    return [
        verb("ad", "ad", "PRS", "INDF"),
        verb("ad", "adja", "PRS", "DEF"),
        verb("ad", "adott", "PST", "INDF"),
        verb("ad", "adta", "PST", "DEF"),
        verb("mond", "mond", "PRS", "INDF"),
        verb("mond", "mondja", "PRS", "DEF"),
        verb("történik", "történik", "PRS", "INDF"),
    ]


def build_lexicon() -> list[Filler]:
    return [
        *load_csv("bleached_nouns.csv", "NOUN", "lemma"),
        *load_csv("determiners.csv", "DET", "lemma"),
        *load_csv("particles.csv", "PART", "lemma"),
        *load_csv("spatial_postpositions.csv", "POSTP", "form"),
        *load_csv("complex_postpositions.csv", "POSTP", "form"),
        *stub_verbs(),
    ]


# bead evaluates constraints with its own DSL (bead/dsl/grammar.lark), not with
# Python. The two overlap enough that Python `eval` is a usable stand-in here,
# but Python accepts things the DSL does not, so a plain eval would give a
# false pass on an expression the real pipeline cannot parse.
#
# `true`/`false` are the DSL's boolean literals; they are supplied here so that
# DSL-valid expressions evaluate. `check_dsl_compatible` rejects the Python-only
# constructs, which is the other half of keeping this honest.
DSL_NAMESPACE = {"true": True, "false": False}

# Constructs Python accepts but the DSL grammar has no rule for. The grammar's
# comparison operators are == != < > <= >= in, "not in" — there is no `is`, and
# its literals are lowercase true/false with no None.
DSL_UNSUPPORTED = [
    (r"\bis\s+not\b", "`is not` (the DSL has no identity operator; use !=)"),
    (r"\bis\b(?!\s*[a-z_]*\()", "`is` (the DSL has no identity operator; use ==)"),
    (r"\bTrue\b", "`True` (the DSL literal is lowercase `true`)"),
    (r"\bFalse\b", "`False` (the DSL literal is lowercase `false`)"),
    (r"\bNone\b", "`None` (the DSL has no null literal)"),
]


def check_dsl_compatible(expression: str) -> list[str]:
    """Return descriptions of Python-only constructs in a constraint.

    An empty list means the expression uses nothing this checker knows the DSL
    cannot handle. It is a lint, not a parser: it catches the mistakes that are
    easy to make when writing constraint strings in Python source, not every
    possible grammar violation.
    """
    import re

    # String literals are DSL data, not syntax; blank them before matching so
    # a feature value that happens to contain "is" is not flagged.
    without_strings = re.sub(r"'[^']*'|\"[^\"]*\"", '""', expression)

    return [
        message
        for pattern, message in DSL_UNSUPPORTED
        if re.search(pattern, without_strings)
    ]


def _evaluate(expression: str, variables: dict) -> bool:
    problems = check_dsl_compatible(expression)

    if problems:
        raise ValueError(
            f"constraint is not valid in bead's DSL: {'; '.join(problems)}\n"
            f"  {expression}"
        )

    return bool(eval(expression, dict(DSL_NAMESPACE), variables))  # noqa: S307


def candidates(slot, lexicon: list[Filler]) -> list[Filler]:
    """Fillers satisfying every single-slot constraint on `slot`."""
    matches = []

    for filler in lexicon:
        if all(
            _evaluate(constraint.expression, {"self": filler})
            for constraint in slot.constraints
        ):
            matches.append(filler)

    return matches


def satisfies_template_constraints(template, assignment: dict[str, Filler]) -> bool:
    """Check the cross-slot constraints (article allomorphy, government)."""
    return all(
        _evaluate(constraint.expression, dict(assignment))
        for constraint in template.constraints
    )


def assignments(template, lexicon: list[Filler]) -> Iterator[dict[str, Filler]]:
    """Yield every filler combination satisfying all constraints."""
    names = list(template.slots)
    pools = [candidates(template.slots[name], lexicon) for name in names]

    if any(not pool for pool in pools):
        return

    for combination in itertools.product(*pools):
        assignment = dict(zip(names, combination, strict=True))

        if satisfies_template_constraints(template, assignment):
            yield assignment


def render(template, assignment: dict[str, Filler]) -> str:
    surfaces = {name: filler.form for name, filler in assignment.items()}
    text = template.template_string.format_map(surfaces)
    text = " ".join(text.split())

    for punctuation in (".", ",", "?", "!"):
        text = text.replace(f" {punctuation}", punctuation)

    return text


def empty_slots(template, lexicon: list[Filler]) -> list[str]:
    """Slot names with no candidate filler at all."""
    return [
        name
        for name, slot in template.slots.items()
        if not candidates(slot, lexicon)
    ]


def main() -> int:
    from generate_templates import NEUTRAL, PREVERBAL, build_templates

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--word-order", choices=[PREVERBAL, NEUTRAL], default=PREVERBAL)
    parser.add_argument(
        "--per-template", type=int, default=2, help="Examples to show per template"
    )
    arguments = parser.parse_args()

    lexicon = build_lexicon()
    templates = build_templates(word_order=arguments.word_order)

    failures = 0

    for template in templates:
        missing = empty_slots(template, lexicon)

        if missing:
            print(f"{template.name}\n    NO FILLERS for slots: {', '.join(missing)}")
            failures += 1
            continue

        examples = list(itertools.islice(assignments(template, lexicon), arguments.per_template))

        if not examples:
            print(f"{template.name}\n    NO SATISFYING COMBINATION")
            failures += 1
            continue

        print(template.name)

        for assignment in examples:
            print(f"    {render(template, assignment)}")

    print()
    print(f"{len(templates)} templates, {failures} without output")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
