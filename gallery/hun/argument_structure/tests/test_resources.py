"""Tests for the Hungarian resource inventories.

Most of these cover pure data plus pure Python — the noun paradigm generator,
the CSV/loader column contract, and the postposition tables — so they fail fast
on the kind of drift that is otherwise silent. Two tests reach into
`generate_templates` to check that the template layer's pinned lemmas and open
semantic classes line up with what the noun inventory actually provides; those
need bead importable, like the rest of the pipeline.
"""

from __future__ import annotations

import csv
import importlib.util
import sys
from pathlib import Path

import pytest

GALLERY_DIR = Path(__file__).resolve().parent.parent
RESOURCES = GALLERY_DIR / "resources"

sys.path.insert(0, str(GALLERY_DIR))


def _load_builder():
    """Import resources/build_bleached_nouns.py by path.

    It lives in `resources/`, which is a data directory rather than a package,
    so a plain import will not find it.
    """
    spec = importlib.util.spec_from_file_location(
        "build_bleached_nouns", RESOURCES / "build_bleached_nouns.py"
    )
    module = importlib.util.module_from_spec(spec)
    # @dataclass resolves field types via sys.modules[cls.__module__], so the
    # module has to be registered before exec_module runs.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


builder = _load_builder()


def read_csv(name: str) -> list[dict[str, str]]:
    with (RESOURCES / name).open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


# ---------------------------------------------------------------------------
# Vowel harmony
# ---------------------------------------------------------------------------


class TestHarmony:
    @pytest.mark.parametrize(
        ("stem", "expected"),
        [
            ("ember", builder.FRONT_UNROUNDED),
            ("hely", builder.FRONT_UNROUNDED),
            ("terület", builder.FRONT_UNROUNDED),
            ("csoport", builder.BACK),
            ("tárgy", builder.BACK),
            # A back vowel anywhere in the stem wins even when the last vowel
            # is front: szerszám is e + á.
            ("szerszám", builder.BACK),
            ("eszköz", builder.FRONT_ROUNDED),
            ("ügy", builder.FRONT_ROUNDED),
            # Rounding is decided by the last vowel, so i + ő is front rounded.
            ("idő", builder.FRONT_ROUNDED),
        ],
    )
    def test_harmony_class(self, stem: str, expected: str) -> None:
        assert builder.harmony_class(stem) == expected

    def test_vowelless_stem_is_rejected(self) -> None:
        with pytest.raises(ValueError):
            builder.harmony_class("xyz")

    @pytest.mark.parametrize(
        ("stem", "case", "expected"),
        [
            ("hely", "SUP", "helyen"),
            ("csoport", "SUP", "csoporton"),
            ("eszköz", "SUP", "eszközön"),
            # SUP loses its linking vowel after a vowel-final stem.
            ("idő", "SUP", "időn"),
            ("eszköz", "ALL", "eszközhöz"),
            ("ember", "ALL", "emberhez"),
            ("csoport", "ALL", "csoporthoz"),
            ("hely", "TER", "helyig"),
            ("hely", "SUB", "helyre"),
            ("hely", "DEL", "helyről"),
            ("hely", "ABL", "helytől"),
        ],
    )
    def test_apply_suffix(self, stem: str, case: str, expected: str) -> None:
        assert builder.apply_suffix(stem, case) == expected


class TestVAssimilation:
    @pytest.mark.parametrize(
        ("stem", "expected"),
        [
            # Digraphs double by repeating their first letter: gy -> ggy.
            ("tárgy", "tárggyal"),
            ("hely", "hellyel"),
            ("esemény", "eseménnyel"),
            ("ügy", "üggyel"),
            # Single consonants simply double.
            ("eszköz", "eszközzel"),
            ("ember", "emberrel"),
            ("szerszám", "szerszámmal"),
            ("csoport", "csoporttal"),
            # No assimilation after a vowel.
            ("idő", "idővel"),
        ],
    )
    def test_instrumental(self, stem: str, expected: str) -> None:
        assert builder.instrumental(stem) == expected

    @pytest.mark.parametrize(
        ("stem", "expected"),
        [
            ("tárgy", "tárggyá"),
            ("hely", "hellyé"),
            ("ember", "emberré"),
            ("csoport", "csoporttá"),
            ("idő", "idővé"),
        ],
    )
    def test_translative(self, stem: str, expected: str) -> None:
        assert builder.translative(stem) == expected

    def test_digraph_does_not_double_second_letter(self) -> None:
        """gy + val is tárggyal, never *tárgyval or *tárgygyal."""
        result = builder.instrumental("tárgy")
        assert "gyv" not in result
        assert "gygy" not in result


# ---------------------------------------------------------------------------
# The generated CSV
# ---------------------------------------------------------------------------


class TestBleachedNouns:
    def test_csv_is_up_to_date_with_generator(self) -> None:
        """A hand edit to bleached_nouns.csv must not silently diverge."""
        on_disk = (RESOURCES / "bleached_nouns.csv").read_text(encoding="utf-8")
        assert on_disk == builder.build_csv_text(), (
            "bleached_nouns.csv is out of date; "
            "run python resources/build_bleached_nouns.py"
        )

    def test_every_lemma_has_the_full_case_paradigm(self) -> None:
        rows = read_csv("bleached_nouns.csv")
        by_lemma: dict[str, set[str]] = {}

        for row in rows:
            by_lemma.setdefault(row["lemma"], set()).add(row["case"])

        for lemma, cases in by_lemma.items():
            missing = set(builder.CASE_ORDER) - cases
            assert not missing, f"{lemma} is missing cases: {sorted(missing)}"

    def test_forms_are_unique_within_a_lemma(self) -> None:
        rows = read_csv("bleached_nouns.csv")
        seen: set[tuple[str, str]] = set()

        for row in rows:
            key = (row["lemma"], row["case"])
            assert key not in seen, f"duplicate {key}"
            seen.add(key)

    def test_no_form_is_empty_or_padded(self) -> None:
        for row in read_csv("bleached_nouns.csv"):
            assert row["form"], f"empty form for {row['lemma']}/{row['case']}"
            assert row["form"] == row["form"].strip()

    def test_nominative_equals_the_lemma(self) -> None:
        for row in read_csv("bleached_nouns.csv"):
            if row["case"] == "NOM":
                assert row["form"] == row["lemma"]

    def test_temporal_case_is_restricted(self) -> None:
        """-kor is lexically restricted; it must not appear on every noun."""
        rows = read_csv("bleached_nouns.csv")
        temporal = {row["lemma"] for row in rows if row["case"] == "TEMP"}
        all_lemmas = {row["lemma"] for row in rows}

        assert temporal, "expected at least one lemma licensing -kor"
        assert temporal < all_lemmas, "-kor must not be generated for every lemma"

        for row in rows:
            if row["case"] == "TEMP":
                assert row["semantic_class"] == "time"

    def test_controlled_lemmas_are_present(self) -> None:
        """Template pinning depends on these lemmas existing."""
        from generate_templates import CONTROLLED_LEMMAS

        available = {row["lemma"] for row in read_csv("bleached_nouns.csv")}

        for slot_name, lemma in CONTROLLED_LEMMAS.items():
            assert lemma in available, f"{slot_name} pins missing lemma {lemma!r}"

    def test_open_classes_have_more_than_one_lemma(self) -> None:
        """A varying slot with one member is controlled by accident, not design."""
        from generate_templates import OPEN_CLASSES

        rows = read_csv("bleached_nouns.csv")

        for slot_name, classes in OPEN_CLASSES.items():
            lemmas = {row["lemma"] for row in rows if row["semantic_class"] in classes}
            assert len(lemmas) > 1, (
                f"slot {slot_name!r} draws on {classes} but only {lemmas} exist, "
                "so it is not actually varying"
            )


# ---------------------------------------------------------------------------
# CSV / loader column contract
# ---------------------------------------------------------------------------
#
# generate_lexicons.py names the columns it wants in `feature_columns` and
# silently skips any that are absent. That made three resource files lose every
# feature they declared. This test makes the contract explicit.

DECLARED_COLUMNS = {
    "bleached_nouns.csv": [
        "case", "number", "semantic_class", "countability", "animacy",
        "harmony", "role",
    ],
    "bleached_verbs.csv": [
        "tense", "semantic_class", "aspectual_class", "valency", "frame_type",
        "subject_role", "object_role", "oblique_role", "telicity",
        "lemma_vp", "infinitive_vp", "present_clause", "past_clause",
    ],
    "bleached_adjectives.csv": ["semantic_class", "gradability", "stage_vs_individual"],
    "case_markers.csv": [
        "case", "harmony_pattern", "morphophonology", "concatenable_directly",
    ],
    "determiners.csv": ["definiteness", "following_segment_type"],
    "subject_pronouns.csv": [
        "pronoun_type", "case", "person", "number", "definiteness",
    ],
    "auxiliary_verbs.csv": [
        "finiteness", "mood", "tense", "person", "number", "function",
    ],
    "preverbs.csv": ["separable", "aspectual_effect", "stage1_manipulated"],
    "particles.csv": ["function", "stage1_manipulated"],
    "spatial_postpositions.csv": [
        "postp_type", "spatial_class", "series", "gov_case", "eng_gloss", "stage1",
    ],
    "complex_postpositions.csv": [
        "postp_type", "gov_case", "semantic_class", "eng_gloss", "stage1",
    ],
}


class TestColumnContract:
    @pytest.mark.parametrize("name", sorted(DECLARED_COLUMNS))
    def test_declared_columns_exist(self, name: str) -> None:
        with (RESOURCES / name).open(encoding="utf-8") as handle:
            actual = set(csv.DictReader(handle).fieldnames or [])

        missing = [column for column in DECLARED_COLUMNS[name] if column not in actual]
        assert not missing, (
            f"{name} is missing declared columns {missing}; "
            "generate_lexicons.py would drop these features without erroring"
        )

    @pytest.mark.parametrize("name", sorted(DECLARED_COLUMNS))
    def test_generate_lexicons_declares_what_the_csv_has(self, name: str) -> None:
        """Guard the other direction too: a new column nobody reads."""
        source = (GALLERY_DIR / "generate_lexicons.py").read_text(encoding="utf-8")

        with (RESOURCES / name).open(encoding="utf-8") as handle:
            actual = list(csv.DictReader(handle).fieldnames or [])

        identifiers = {"lemma", "word", "marker", "form", "notes", "pos"}

        for column in actual:
            if column in identifiers:
                continue
            assert f'"{column}"' in source, (
                f"{name} has column {column!r} that generate_lexicons.py never reads"
            )

    @pytest.mark.parametrize("name", sorted(DECLARED_COLUMNS))
    def test_has_an_identifier_column(self, name: str) -> None:
        """load_csv_items accepts lemma, word, marker, or form."""
        with (RESOURCES / name).open(encoding="utf-8") as handle:
            actual = set(csv.DictReader(handle).fieldnames or [])

        assert actual & {"lemma", "word", "marker", "form"}, (
            f"{name} has no identifier column; load_csv_items would raise"
        )


# ---------------------------------------------------------------------------
# Postpositions
# ---------------------------------------------------------------------------

VALID_CASES = set(builder.CASE_ORDER) | {builder.TEMPORAL_CASE}


class TestPostpositions:
    def test_spatial_series_are_complete(self) -> None:
        """Each spatial class should offer the series it claims to."""
        rows = read_csv("spatial_postpositions.csv")
        by_class: dict[str, set[str]] = {}

        for row in rows:
            by_class.setdefault(row["spatial_class"], set()).add(row["series"])

        for spatial_class, series in by_class.items():
            assert "ESS" in series, f"{spatial_class} has no essive member"
            assert series <= {"ESS", "LAT", "ABL"}

    def test_spatial_postpositions_govern_nominative(self) -> None:
        """Hungarian bare postpositions take a caseless complement."""
        for row in read_csv("spatial_postpositions.csv"):
            assert row["gov_case"] == "NOM", (
                f"{row['form']} is in the bare table but governs {row['gov_case']}"
            )

    def test_governed_cases_exist_in_the_noun_paradigm(self) -> None:
        """A postposition governing a case no noun bears can never be filled."""
        noun_cases = {row["case"] for row in read_csv("bleached_nouns.csv")}

        for name in ("spatial_postpositions.csv", "complex_postpositions.csv"):
            for row in read_csv(name):
                assert row["gov_case"] in VALID_CASES, (
                    f"{row['form']} governs unknown case {row['gov_case']}"
                )
                assert row["gov_case"] in noun_cases, (
                    f"{row['form']} governs {row['gov_case']}, "
                    "which no bleached noun is inflected for"
                )

    def test_forms_are_unique_across_both_tables(self) -> None:
        spatial = {row["form"] for row in read_csv("spatial_postpositions.csv")}
        complex_ = {row["form"] for row in read_csv("complex_postpositions.csv")}
        overlap = spatial & complex_

        assert not overlap, (
            f"{sorted(overlap)} appear in both postposition tables; "
            "the postp_type feature would be ambiguous"
        )

    def test_postp_type_is_consistent(self) -> None:
        for row in read_csv("spatial_postpositions.csv"):
            assert row["postp_type"] == "spatial"

        for row in read_csv("complex_postpositions.csv"):
            assert row["postp_type"] == "complex"


# ---------------------------------------------------------------------------
# Particles
# ---------------------------------------------------------------------------


class TestParticles:
    def test_eppen_is_marked_as_manipulated(self) -> None:
        """éppen is the manipulated slot in every ongoing frame.

        It was previously flagged stage1_manipulated=false while six templates
        varied it, which made the metadata contradict the design.
        """
        rows = read_csv("particles.csv")
        eppen = [row for row in rows if row["form"] == "éppen"]

        assert eppen, "éppen missing from particles.csv"
        assert eppen[0]["stage1_manipulated"] == "true"

    def test_only_eppen_is_manipulated_in_stage1(self) -> None:
        manipulated = {
            row["form"]
            for row in read_csv("particles.csv")
            if row["stage1_manipulated"] == "true"
        }
        assert manipulated == {"éppen"}
