"""Tests for read-only predictor protocols, registration, and conformance."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from bead.items import Item, ItemTemplate, PresentationSpec, TaskSpec
from bead.protocol import ResponseEncoding, ScaleType
from bead.serving import (
    ConformanceError,
    ModelIdentity,
    NLICapability,
    NLICase,
    PersistablePredictor,
    PredictionCase,
    PredictorRegistry,
    ProbabilisticPredictor,
    ServingPredictor,
    TaskCompatibility,
    run_predictor_conformance,
)
from bead.serving.protocols import NLIProbabilities


class ExternalPredictor:
    """Structural predictor with no bead implementation base class."""

    def __init__(self, *, invalid_probabilities: bool = False) -> None:
        self.model_name = "external-test-model"
        self.model_version = "1.2.3"
        self.response_encoding = ResponseEncoding(
            name="inference",
            n_levels=2,
            scale_type=ScaleType.BINARY,
            labels=("no", "yes"),
        )
        self.supported_task_types = ("binary",)
        self.invalid_probabilities = invalid_probabilities

    def validate_item_compatibility(
        self, item: Item, item_template: ItemTemplate
    ) -> None:
        if item.item_template_id != item_template.id:
            raise ValueError("item and template do not match")
        if item_template.task_type not in self.supported_task_types:
            raise ValueError(f"unsupported task type: {item_template.task_type}")

    def predict_proba(
        self,
        items: list[Item],
        participant_ids: list[str] | None = None,
    ) -> np.ndarray:
        if participant_ids is not None and len(participant_ids) != len(items):
            raise ValueError("participant_ids length does not match items")
        row = [0.8, 0.8] if self.invalid_probabilities else [0.25, 0.75]
        return np.asarray([row for _item in items], dtype=np.float64)

    def save(self, path: str | Path) -> None:
        artifact_path = Path(path)
        artifact_path.mkdir(parents=True, exist_ok=True)
        state = {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "invalid_probabilities": self.invalid_probabilities,
        }
        (artifact_path / "predictor.json").write_text(
            json.dumps(state), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> ExternalPredictor:
        state = json.loads((Path(path) / "predictor.json").read_text(encoding="utf-8"))
        predictor = cls(invalid_probabilities=state["invalid_probabilities"])
        predictor.model_name = state["model_name"]
        predictor.model_version = state["model_version"]
        return predictor

    def compute_nli(self, premise: str, hypothesis: str) -> NLIProbabilities:
        return {
            "entailment": 0.7,
            "neutral": 0.2,
            "contradiction": 0.1,
        }


@pytest.fixture
def binary_case() -> PredictionCase:
    template = ItemTemplate(
        name="binary-inference",
        judgment_type="inference",
        task_type="binary",
        task_spec=TaskSpec(prompt="Does the hypothesis follow?", options=("no", "yes")),
        presentation_spec=PresentationSpec(),
    )
    item = Item(
        item_template_id=template.id,
        rendered_elements={},
    )
    return PredictionCase(item=item, item_template=template)


def test_external_predictor_satisfies_independent_protocols() -> None:
    predictor = ExternalPredictor()

    assert isinstance(predictor, ModelIdentity)
    assert isinstance(predictor, TaskCompatibility)
    assert isinstance(predictor, ProbabilisticPredictor)
    assert isinstance(predictor, PersistablePredictor)
    assert isinstance(predictor, NLICapability)
    assert isinstance(predictor, ServingPredictor)


def test_registry_accepts_structural_loading_factory(tmp_path: Path) -> None:
    original = ExternalPredictor()
    artifact_path = tmp_path / "native-artifact"
    original.save(artifact_path)

    registry = PredictorRegistry[ServingPredictor]()
    registry.register("external", ExternalPredictor.load)
    restored = registry.load("external", artifact_path)

    assert registry.list_factories() == ("external",)
    assert isinstance(restored, ExternalPredictor)
    assert restored.model_name == original.model_name
    assert restored.model_version == original.model_version


def test_registry_reports_unknown_factory(tmp_path: Path) -> None:
    registry = PredictorRegistry[ServingPredictor]()

    with pytest.raises(KeyError, match="Unknown predictor factory"):
        registry.load("missing", tmp_path / "artifact")


def test_full_conformance_suite(binary_case: PredictionCase) -> None:
    predictor = ExternalPredictor()

    report = run_predictor_conformance(
        predictor,
        prediction_cases=[binary_case],
        response_encoding=predictor.response_encoding,
        nli_cases=[NLICase("A dog runs.", "An animal moves.")],
    )

    assert report.model_name == "external-test-model"
    assert report.model_version == "1.2.3"
    assert report.n_items == 1
    assert report.n_levels == 2
    assert report.n_nli_cases == 1
    assert report.checks == (
        "identity",
        "task_compatibility",
        "predict_proba",
        "persistence",
        "nli",
    )


def test_conformance_rejects_unnormalized_probabilities(
    binary_case: PredictionCase,
) -> None:
    predictor = ExternalPredictor(invalid_probabilities=True)

    with pytest.raises(ConformanceError, match="do not sum to one"):
        run_predictor_conformance(
            predictor,
            prediction_cases=[binary_case],
            response_encoding=predictor.response_encoding,
            nli_cases=[NLICase("A dog runs.", "An animal moves.")],
        )


def test_conformance_rejects_encoding_task_mismatch(
    binary_case: PredictionCase,
) -> None:
    predictor = ExternalPredictor()
    ordinal_encoding = ResponseEncoding(
        name="rating",
        n_levels=3,
        scale_type=ScaleType.ORDINAL,
        labels=("low", "middle", "high"),
    )
    predictor.response_encoding = ordinal_encoding

    with pytest.raises(ConformanceError, match="maps to 'ordinal_scale'"):
        run_predictor_conformance(
            predictor,
            prediction_cases=[binary_case],
            response_encoding=ordinal_encoding,
            nli_cases=[NLICase("A dog runs.", "An animal moves.")],
        )
