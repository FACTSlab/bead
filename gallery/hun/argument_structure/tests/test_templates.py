"""Tests for the Hungarian Stage-1 template inventory.

Two things are checked here that automatic tests genuinely can establish, in
the sense of LINGUISTIC_DESIGN_NOTES.md section 6:

1. structural invariants of the template inventory — names, slots, agreement,
   controlled fillers, word order;
2. that every frame actually renders, and renders with the case, agreement and
   article the frame specifies.

None of it says anything about whether a given verb is acceptable in a given
frame. That is the experiment's job, not the generator's.

Rendering uses `tests/render_samples.py`, which resolves the real constraint
expressions against the real resource CSVs with a small stub verb lexicon, so
these tests do not need UniMorph downloaded.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import pytest

GALLERY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GALLERY_DIR))

from generate_templates import (  # noqa: E402
    ADJUNCTS,
    CONTROLLED_LEMMAS,
    FAMILIES,
    NEUTRAL,
    PREVERBAL,
    build_templates,
)

from tests.render_samples import (  # noqa: E402
    assignments,
    build_lexicon,
    candidates,
    check_dsl_compatible,
    render,
)

# The Stage-1 inventory as it stood before the Korean-parallel families were
# added. create_2afc_pairs.py refers to these by name, so they are frozen.
ORIGINAL_TEMPLATE_NAMES = {
    "subj_nom-verb.",
    "subj_nom-noun_dat-verb.",
    "subj_nom-noun_loc-verb.",
    "subj_nom-noun_inst-verb.",
    "subj_nom-noun_inst-noun_loc-verb.",
    "subj_nom-noun_inst-noun_dat-verb.",
    "subj_nom-noun_loc-noun_dat-verb.",
    "tr-indef.",
    "tr-def.",
    "tr_dat-indef.",
    "tr_dat-def.",
    "tr_loc-indef.",
    "tr_loc-def.",
    "tr_ins-indef.",
    "tr_ins-def.",
    "tr_ins_loc-indef.",
    "tr_ins_loc-def.",
    "tr_ins_dat-indef.",
    "tr_ins_dat-def.",
    "tr_loc_dat-indef.",
    "tr_loc_dat-def.",
    "subj_nom-verb-hogy_ind.",
    "subj_nom-eppen-verb_prs.",
    "subj_nom-eppen-obj_acc-indf-verb_prs.",
    "subj_nom-eppen-obj_acc-def-verb_prs.",
    "subj_nom-eppen-verb_pst.",
    "subj_nom-eppen-obj_acc-indf-verb_pst.",
    "subj_nom-eppen-obj_acc-def-verb_pst.",
}

# Original preverbal surface strings, so a word-order refactor cannot quietly
# reorder the frames the pilot data was collected on.
FROZEN_TEMPLATE_STRINGS = {
    "subj_nom-verb.": "Egy {subject} {verb}.",
    "subj_nom-noun_loc-verb.": "Egy {subject} egy {location} {verb}.",
    "tr-indef.": "Egy {subject} {determiner} {object} {verb}.",
    "tr-def.": "Egy {subject} {determiner} {object} {verb}.",
    "tr_ins_loc-indef.": (
        "Egy {subject} {determiner} {object} egy {instrument} egy {location} {verb}."
    ),
    "subj_nom-verb-hogy_ind.": (
        "Egy {subject} {verb}, hogy egy {comp_subject} {comp_verb}."
    ),
    "subj_nom-eppen-verb_prs.": "Egy {subject} {ongoing} {verb}.",
    # éppen precedes the object here; it attaches to the subject, not the verb.
    "subj_nom-eppen-obj_acc-def-verb_prs.": (
        "Egy {subject} {ongoing} {determiner} {object} {verb}."
    ),
}

VOWELS = "aáeéiíoóöőuúüű"


@pytest.fixture(scope="module")
def lexicon():
    return build_lexicon()


def templates_by_name(**kwargs):
    return {template.name: template for template in build_templates(**kwargs)}


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------


class TestInventory:
    def test_all_original_templates_survive(self) -> None:
        names = set(templates_by_name())
        missing = ORIGINAL_TEMPLATE_NAMES - names
        assert not missing, f"create_2afc_pairs.py refers to these: {sorted(missing)}"

    @pytest.mark.parametrize("name", sorted(FROZEN_TEMPLATE_STRINGS))
    def test_original_surface_strings_are_unchanged(self, name: str) -> None:
        template = templates_by_name()[name]
        assert template.template_string == FROZEN_TEMPLATE_STRINGS[name]

    def test_template_names_are_unique(self) -> None:
        templates = build_templates()
        names = [template.name for template in templates]
        assert len(names) == len(set(names))

    def test_every_transitive_frame_has_both_definiteness_realizations(self) -> None:
        """The INDF/DEF split is the core Hungarian-specific manipulation."""
        templates = build_templates()
        transitive = [
            template
            for template in templates
            if template.metadata.get("transitivity") == "transitive"
            and template.metadata.get("frame_family") != "TR-ONGOING"
        ]

        by_family: dict[str, set[str]] = {}

        for template in transitive:
            family = template.metadata["frame_family"]
            by_family.setdefault(family, set()).add(
                template.metadata["object_definiteness"]
            )

        for family, realizations in by_family.items():
            assert realizations == {"INDF", "DEF"}, (
                f"{family} has {realizations}, expected both INDF and DEF"
            )

    def test_families_can_be_selected_independently(self) -> None:
        for family in FAMILIES:
            subset = build_templates(families=[family])
            assert subset, f"family {family!r} produced no templates"
            assert len(subset) < len(build_templates())

    def test_adjunct_filter_restricts_frames(self) -> None:
        only_dative = build_templates(families=["nominal"], adjunct_filter={"dat"})
        used = {
            adjunct
            for template in only_dative
            for adjunct in template.metadata.get("adjuncts", [])
        }
        assert used <= {"dat"}

    def test_every_korean_adjunct_type_is_covered(self) -> None:
        """The point of this revision: parity with Korean's adjunct inventory."""
        templates = build_templates(families=["nominal"])
        used = {
            adjunct
            for template in templates
            for adjunct in template.metadata.get("adjuncts", [])
        }
        assert used == set(ADJUNCTS)


# ---------------------------------------------------------------------------
# Slot and agreement invariants
# ---------------------------------------------------------------------------


class TestSlotInvariants:
    def test_object_definiteness_matches_verb_agreement(self) -> None:
        """A DEF object requires the objective conjugation, and vice versa."""
        for template in build_templates():
            definiteness = template.metadata.get("object_definiteness")

            if definiteness is None:
                continue

            verb = template.slots["verb"]
            expression = " ".join(c.expression for c in verb.constraints)
            assert f"'object_agreement') == '{definiteness}'" in expression, (
                f"{template.name} has a {definiteness} object but its verb slot "
                "does not require the matching conjugation"
            )

    def test_hogy_frame_requires_definite_conjugation(self) -> None:
        """A finite direct-object hogy-clause triggers objective agreement."""
        template = templates_by_name()["subj_nom-verb-hogy_ind."]
        expression = " ".join(c.expression for c in template.slots["verb"].constraints)
        assert "'object_agreement') == 'DEF'" in expression
        assert template.metadata["matrix_object_agreement"] == "DEF"

    def test_definite_frames_carry_the_article_constraint(self) -> None:
        for template in build_templates():
            if template.metadata.get("object_definiteness") != "DEF":
                continue

            expressions = " ".join(c.expression for c in template.constraints)
            assert "determiner.form == 'az'" in expressions, (
                f"{template.name} has a definite object but no a/az constraint"
            )

    def test_controlled_slots_are_pinned_by_lemma(self) -> None:
        """The controlled baseline must not depend on a class having one member."""
        for template in build_templates():
            for slot_name, lemma in CONTROLLED_LEMMAS.items():
                slot = template.slots.get(slot_name)

                if slot is None:
                    continue

                expression = " ".join(c.expression for c in slot.constraints)
                assert f"self.lemma == '{lemma}'" in expression, (
                    f"{template.name}: slot {slot_name!r} is documented as "
                    f"controlled but is not pinned to {lemma!r}"
                )

    def test_complex_postposition_government_is_constrained(self) -> None:
        for template in build_templates(families=["complex"]):
            expressions = " ".join(c.expression for c in template.constraints)
            assert "gov_case" in expressions, (
                f"{template.name} has no government constraint, so the "
                "complement's case would not covary with the postposition"
            )

    def test_complex_postposition_complement_leaves_case_open(self) -> None:
        """Case comes from the government constraint, not the slot."""
        for template in build_templates(families=["complex"]):
            slot = template.slots["complex_postp_object"]
            expression = " ".join(c.expression for c in slot.constraints)
            assert "features.get('case')" not in expression


# ---------------------------------------------------------------------------
# Word order
# ---------------------------------------------------------------------------


class TestWordOrder:
    def test_preverbal_is_the_default(self) -> None:
        assert build_templates()[0].metadata["word_order"] == PREVERBAL

    def test_neutral_places_the_verb_before_the_object(self) -> None:
        template = templates_by_name(word_order=NEUTRAL)["tr-indef."]
        string = template.template_string
        assert string.index("{verb}") < string.index("{object}")

    def test_preverbal_places_the_object_before_the_verb(self) -> None:
        template = templates_by_name(word_order=PREVERBAL)["tr-indef."]
        string = template.template_string
        assert string.index("{object}") < string.index("{verb}")

    def test_eppen_stays_adjacent_to_the_subject(self) -> None:
        """éppen creates the ongoing reading; it must not drift rightward."""
        for word_order in (PREVERBAL, NEUTRAL):
            for template in build_templates(
                families=["ongoing"], word_order=word_order
            ):
                string = template.template_string
                assert string.startswith("Egy {subject} {ongoing}"), (
                    f"{template.name} ({word_order}): {string}"
                )

    def test_both_orders_produce_the_same_frames(self) -> None:
        assert set(templates_by_name(word_order=PREVERBAL)) == set(
            templates_by_name(word_order=NEUTRAL)
        )

    def test_word_order_is_recorded_in_metadata(self) -> None:
        for word_order in (PREVERBAL, NEUTRAL):
            for template in build_templates(word_order=word_order):
                assert template.metadata["word_order"] == word_order
                assert f"order_{word_order}" in template.tags


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


class TestRendering:
    @pytest.mark.parametrize("word_order", [PREVERBAL, NEUTRAL])
    def test_every_template_renders(self, word_order: str) -> None:
        lex = build_lexicon()

        for template in build_templates(word_order=word_order):
            for slot_name, slot in template.slots.items():
                assert candidates(slot, lex), (
                    f"{template.name}: slot {slot_name!r} has no candidate filler"
                )

            first = next(iter(assignments(template, lex)), None)
            assert first is not None, (
                f"{template.name}: no filler combination satisfies its constraints"
            )

    def test_rendered_sentences_are_well_formed(self, lexicon) -> None:
        for template in build_templates():
            for assignment in itertools.islice(assignments(template, lexicon), 5):
                text = render(template, assignment)

                assert "{" not in text and "}" not in text, text
                assert text.endswith("."), text
                assert "  " not in text, text
                assert text == text.strip(), text
                assert " ." not in text, text

                if "," in text:
                    assert ", " in text, f"missing space after comma: {text}"

    def test_definite_article_allomorphy(self, lexicon) -> None:
        """a before a consonant, az before a vowel."""
        for template in build_templates():
            if template.metadata.get("object_definiteness") != "DEF":
                continue

            for assignment in itertools.islice(assignments(template, lexicon), 20):
                determiner = assignment["determiner"].form
                noun = assignment["object"].form
                expected = "az" if noun[0].lower() in VOWELS else "a"

                assert determiner == expected, (
                    f"{template.name}: got {determiner!r} before {noun!r}"
                )

    def test_controlled_fillers_are_constant(self, lexicon) -> None:
        """hely and eszköz are the Stage-1 locative and instrument baselines."""
        expected_forms = {
            "location": "helyen",
            "instrument": "eszközzel",
            "goal": "helyre",
            "source": "helyről",
            "terminus": "helyig",
            "origin": "helytől",
        }

        for template in build_templates():
            for assignment in itertools.islice(assignments(template, lexicon), 20):
                for slot_name, expected in expected_forms.items():
                    if slot_name in assignment:
                        assert assignment[slot_name].form == expected, (
                            f"{template.name}: {slot_name} was "
                            f"{assignment[slot_name].form!r}, expected {expected!r}"
                        )

    def test_postposition_complement_bears_the_governed_case(self, lexicon) -> None:
        for template in build_templates(families=["complex"]):
            for assignment in assignments(template, lexicon):
                governed = assignment["postposition"].features["gov_case"]
                actual = assignment["complex_postp_object"].features["case"]
                assert actual == governed, (
                    f"{assignment['postposition'].form} governs {governed} "
                    f"but complement is {actual}"
                )

    def test_spatial_postposition_complement_is_caseless(self, lexicon) -> None:
        for template in build_templates(families=["spatial"]):
            for assignment in itertools.islice(assignments(template, lexicon), 20):
                assert assignment["postp_object"].features["case"] == "NOM"

    def test_open_slots_actually_vary(self, lexicon) -> None:
        """Subject and object should draw more than one form across fills."""
        template = templates_by_name()["tr-indef."]
        subjects = set()
        objects = set()

        for assignment in assignments(template, lexicon):
            subjects.add(assignment["subject"].form)
            objects.add(assignment["object"].form)

        assert len(subjects) > 1, f"subject never varies: {subjects}"
        assert len(objects) > 1, f"object never varies: {objects}"


# ---------------------------------------------------------------------------
# Generated documentation
# ---------------------------------------------------------------------------


class TestFrameReference:
    def test_frame_reference_is_up_to_date(self) -> None:
        """FRAME_REFERENCE.csv is generated; a stale copy is a silent doc bug."""
        import build_frame_reference

        on_disk = (GALLERY_DIR / "FRAME_REFERENCE.csv").read_text(encoding="utf-8")
        assert on_disk == build_frame_reference.build_csv_text(), (
            "FRAME_REFERENCE.csv is out of date; run python build_frame_reference.py"
        )

    def test_frame_reference_covers_every_template(self) -> None:
        import csv

        import build_frame_reference  # noqa: F401

        with (GALLERY_DIR / "FRAME_REFERENCE.csv").open(encoding="utf-8") as handle:
            documented = {row["template_name"] for row in csv.DictReader(handle)}

        assert documented == set(templates_by_name())


# ---------------------------------------------------------------------------
# Constraint DSL compatibility
# ---------------------------------------------------------------------------


class TestConstraintDialect:
    """Constraints are evaluated by bead's DSL, not by Python.

    The DSL's comparison operators are == != < > <= >= in / not in, and its
    boolean literals are lowercase `true` / `false`. Writing these expressions
    as Python strings makes it easy to reach for `is True` or `None`, which
    Python accepts and the DSL cannot parse — and which then fails at fill
    time, per template, rather than here.
    """

    def all_expressions(self):
        for word_order in (PREVERBAL, NEUTRAL):
            for template in build_templates(word_order=word_order):
                for constraint in template.constraints:
                    yield template.name, "template", constraint.expression

                for slot_name, slot in template.slots.items():
                    for constraint in slot.constraints:
                        yield template.name, slot_name, constraint.expression

    def test_no_python_only_constructs(self) -> None:
        problems = [
            (name, where, expression, issues)
            for name, where, expression in self.all_expressions()
            if (issues := check_dsl_compatible(expression))
        ]

        assert not problems, "\n".join(
            f"{name} [{where}]: {'; '.join(issues)}\n    {expression}"
            for name, where, expression, issues in problems
        )

    def test_boolean_features_use_lowercase_literals(self) -> None:
        for name, where, expression in self.all_expressions():
            if "stage1" in expression and "features" in expression:
                assert "== true" in expression or "== false" in expression, (
                    f"{name} [{where}] tests a boolean feature without a DSL "
                    f"boolean literal: {expression}"
                )
