"""Tests that config.yaml stays in step with the rest of the pipeline.

`fill_templates.py` loads only the lexicons named in `config.yaml`. A lexicon
that `generate_lexicons.py` writes but the config never lists is loaded by
nothing, so every slot depending on it produces no candidate filler — and
`fill_templates.py` logs "No fills" and moves on. The result is a frame family
that silently generates zero sentences.

That is exactly what happened when the two postposition lexicons were added, so
the coupling is asserted here rather than left to be noticed downstream.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import yaml

GALLERY_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GALLERY_DIR))

from generate_templates import build_templates  # noqa: E402

CONFIG_PATH = GALLERY_DIR / "config.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def generated_lexicon_names() -> set[str]:
    """Lexicon file stems that generate_lexicons.py writes."""
    source = (GALLERY_DIR / "generate_lexicons.py").read_text(encoding="utf-8")
    return {
        match.removesuffix(".jsonl")
        for match in re.findall(r'lexicons_dir / "([a-z_]+\.jsonl)"', source)
    }


class TestLexiconWiring:
    def test_every_generated_lexicon_is_declared(self) -> None:
        config = load_config()
        declared = {entry["name"] for entry in config["resources"]["lexicons"]}
        generated = generated_lexicon_names()

        missing = generated - declared
        assert not missing, (
            f"generate_lexicons.py writes {sorted(missing)} but config.yaml does "
            "not list them, so fill_templates.py would never load them"
        )

    def test_every_declared_lexicon_is_generated(self) -> None:
        config = load_config()
        declared = {entry["name"] for entry in config["resources"]["lexicons"]}
        generated = generated_lexicon_names()

        missing = declared - generated
        assert not missing, (
            f"config.yaml declares {sorted(missing)} but nothing generates them; "
            "fill_templates.py raises FileNotFoundError on a missing lexicon"
        )

    def test_declared_paths_match_names(self) -> None:
        for entry in load_config()["resources"]["lexicons"]:
            expected = f"lexicons/{entry['name']}.jsonl"
            assert entry["path"] == expected, (
                f"lexicon {entry['name']!r} points at {entry['path']!r}"
            )


class TestSlotWiring:
    def test_every_template_slot_has_a_strategy(self) -> None:
        """Unlisted slots fall back to exhaustive, but silently.

        MixedFillingStrategy's default_strategy covers an unlisted slot, so a
        missing entry is not fatal. It does mean the config stops describing
        what the pipeline does, which is how slot-level intent gets lost.
        """
        configured = set(load_config()["template"]["slot_strategies"])
        used = {
            slot_name
            for template in build_templates()
            for slot_name in template.slots
        }

        missing = used - configured
        assert not missing, (
            f"templates use slots {sorted(missing)} with no slot_strategies entry"
        )

    def test_no_stale_slot_strategies(self) -> None:
        configured = set(load_config()["template"]["slot_strategies"])
        used = {
            slot_name
            for template in build_templates()
            for slot_name in template.slots
        }

        stale = configured - used
        assert not stale, (
            f"config.yaml configures slots {sorted(stale)} that no template uses"
        )


class TestContrastWiring:
    def test_batch_coverage_targets_are_producible(self) -> None:
        """Every contrast type the config requires must exist in the generator."""
        from create_2afc_pairs import SAME_VERB_CONTRASTS

        config = load_config()
        targets: set[str] = set()

        for constraint in config["lists"].get("batch_constraints", []):
            if constraint.get("type") == "coverage":
                targets.update(constraint.get("target_values", []))

        available = {spec.contrast_type for spec in SAME_VERB_CONTRASTS}
        available.add("lexical_verb")  # produced by the different-verb pairs

        missing = targets - available
        assert not missing, (
            f"config.yaml requires contrast types {sorted(missing)} that "
            "create_2afc_pairs.py never generates, so batch coverage cannot be met"
        )

    def test_contrast_specs_reference_real_templates(self) -> None:
        """A renamed template silently drops every contrast that referenced it."""
        from create_2afc_pairs import SAME_VERB_CONTRASTS

        names = {template.name for template in build_templates()}
        dangling = sorted(
            {
                template_name
                for spec in SAME_VERB_CONTRASTS
                for template_name in (spec.left_template, spec.right_template)
                if template_name not in names
            }
        )

        assert not dangling, (
            f"create_2afc_pairs.py references templates that no longer exist: "
            f"{dangling}"
        )

    def test_every_frame_has_a_controlled_contrast(self) -> None:
        """A frame with no same-verb contrast yields no controlled comparison.

        Such a frame is still filled and still appears in different-verb
        (`lexical_verb`) pairs, so nothing errors — it simply contributes no
        frame-level manipulation, which is easy to miss when adding frames.

        The lative and ablative spatial series are the known exception. Their
        natural contrast is against the essive series, which would make
        `postposition` the manipulated slot; `shared_slots` in
        create_2afc_pairs.py currently treats `postposition` as shared material
        that must match, so those pairs would all be rejected. Contrasting the
        series means excluding `postposition` there first.
        """
        from create_2afc_pairs import SAME_VERB_CONTRASTS

        known_uncovered = {
            "subj_nom-spostp_lat-verb.",
            "subj_nom-spostp_abl-verb.",
            "tr_spostp_lat-indef.",
            "tr_spostp_lat-def.",
            "tr_spostp_abl-indef.",
            "tr_spostp_abl-def.",
        }

        contrasted = {
            template_name
            for spec in SAME_VERB_CONTRASTS
            for template_name in (spec.left_template, spec.right_template)
        }
        uncovered = {
            template.name
            for template in build_templates()
            if template.name not in contrasted
        }

        assert uncovered == known_uncovered, (
            "frames without a same-verb contrast changed; "
            f"unexpected: {sorted(uncovered - known_uncovered)}, "
            f"newly covered: {sorted(known_uncovered - uncovered)}"
        )
