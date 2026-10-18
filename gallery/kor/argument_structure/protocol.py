"""Protocol declarations for the Korean argument-structure gallery.

The 2AFC acceptability question (어느 문장이 더 자연스러운가요?) is declared
once in ``config.yaml`` under the ``protocol:`` section and materialized
here as a live :class:`~bead.protocol.AnnotationProtocol`, mirroring the
English gallery's ``protocol.py``. ``create_concrete_template_pairs.py``
records the anchor name on every pair, ``generate_deployment.py`` builds
its item template from the protocol, and the active-learning scripts read
the response space from it, so the prompt, response options and drift
thresholds have a single source of truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import yaml

from bead.config.protocol import ProtocolConfig
from bead.protocol import AnnotationProtocol, QuestionFamily, SemanticAnchor
from bead.protocol.drift import DriftScore, StructuralDriftValidator

ACCEPTABILITY_ANCHOR_NAME: Final = "acceptability"


def load_protocol_config(config_path: Path | str = "config.yaml") -> ProtocolConfig:
    """Parse the ``protocol:`` section of ``config_path`` into a ProtocolConfig."""
    data = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    return ProtocolConfig.model_validate(data["protocol"])


def build_protocol(config_path: Path | str = "config.yaml") -> AnnotationProtocol:
    """Build the live :class:`AnnotationProtocol` from ``config_path``."""
    return load_protocol_config(config_path).build()


def acceptability_family(
    protocol: AnnotationProtocol,
) -> QuestionFamily:
    """Return the acceptability family from ``protocol``."""
    return protocol.family_by_name(ACCEPTABILITY_ANCHOR_NAME)


def acceptability_anchor(protocol: AnnotationProtocol) -> SemanticAnchor:
    """Return the acceptability anchor from ``protocol``."""
    return acceptability_family(protocol).anchor


def check_canonical_prompt(config_path: Path | str = "config.yaml") -> DriftScore:
    """Run the structural drift checks on the anchor's own canonical prompt.

    bead applies drift checks to realizations of a prompt, not to the
    canonical prompt, so building the protocol never checks it. This runs the
    same structural checks (``protocol.drift``: min_length, trailing ``?``,
    required keywords) on the canonical prompt itself.
    """
    drift = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))[
        "protocol"
    ].get("drift", {})
    validator = StructuralDriftValidator(
        min_length=drift.get("min_length", 15),
        require_question_mark=drift.get("require_question_mark", True),
        keyword_case_sensitive=drift.get("keyword_case_sensitive", False),
    )
    anchor = acceptability_anchor(build_protocol(config_path))
    return validator.validate(anchor.canonical_prompt, anchor, None)  # type: ignore[arg-type]
