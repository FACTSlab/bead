"""Reusable conformance checks for third-party serving predictors."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from bead.items.item import Item
from bead.items.item_template import ItemTemplate
from bead.items.validation import validate_item
from bead.protocol.encoding import ResponseEncoding
from bead.protocol.items import scale_type_to_task_type
from bead.serving.protocols import (
    ModelIdentity,
    NLICapability,
    NLIProbabilities,
    PersistablePredictor,
    ProbabilisticPredictor,
    ServingPredictor,
    TaskCompatibility,
)

__all__ = [
    "ConformanceError",
    "ConformanceReport",
    "NLICase",
    "PredictionCase",
    "run_predictor_conformance",
]

type _ServingCapability = (
    ModelIdentity
    | TaskCompatibility
    | ProbabilisticPredictor
    | PersistablePredictor
    | NLICapability
)


class ConformanceError(AssertionError):
    """A serving predictor violated its public capability contract."""


@dataclass(frozen=True, slots=True)
class PredictionCase:
    """A real bead item paired with the template that defines its task."""

    item: Item
    item_template: ItemTemplate


@dataclass(frozen=True, slots=True)
class NLICase:
    """A premise-hypothesis pair used to exercise NLI scoring."""

    premise: str
    hypothesis: str


@dataclass(frozen=True, slots=True)
class ConformanceReport:
    """Summary returned after every conformance check passes."""

    model_name: str
    model_version: str
    n_items: int
    n_levels: int
    n_nli_cases: int
    checks: tuple[str, ...]


def _fail(message: str) -> ConformanceError:
    return ConformanceError(message)


def _validate_identity(predictor: ServingPredictor) -> None:
    if not predictor.model_name.strip():
        raise _fail("model_name must be a non-empty string")
    if not predictor.model_version.strip():
        raise _fail("model_version must be a non-empty string")


def _same_response_encoding(left: ResponseEncoding, right: ResponseEncoding) -> bool:
    """Return whether two encodings assign the same response semantics."""
    left_poles = (
        left.semantic_poles.as_tuple() if left.semantic_poles is not None else None
    )
    right_poles = (
        right.semantic_poles.as_tuple() if right.semantic_poles is not None else None
    )
    return (
        left.name == right.name
        and left.n_levels == right.n_levels
        and left.scale_type == right.scale_type
        and left.labels == right.labels
        and left_poles == right_poles
    )


def _validate_prediction_matrix(
    probabilities: np.ndarray | list[list[float]],
    *,
    n_items: int,
    response_encoding: ResponseEncoding,
    source: str,
) -> None:
    if not isinstance(probabilities, np.ndarray):
        raise _fail(f"{source} must return numpy.ndarray")

    expected_shape = (n_items, response_encoding.n_levels)
    if probabilities.shape != expected_shape:
        raise _fail(
            f"{source} returned shape {probabilities.shape}; expected {expected_shape} "
            f"for encoding {response_encoding.name!r}"
        )

    try:
        numeric = probabilities.astype(np.float64, copy=False)
    except (TypeError, ValueError) as error:
        raise _fail(f"{source} returned non-numeric probabilities: {error}") from error

    if not np.isfinite(numeric).all():
        raise _fail(f"{source} returned non-finite probabilities")
    if ((numeric < 0.0) | (numeric > 1.0)).any():
        raise _fail(f"{source} returned probabilities outside [0, 1]")
    if not np.allclose(numeric.sum(axis=1), 1.0, rtol=1e-6, atol=1e-8):
        raise _fail(f"{source} returned rows that do not sum to one")


def _validate_nli(
    probabilities: NLIProbabilities,
    *,
    source: str,
) -> None:
    expected_keys = {"entailment", "neutral", "contradiction"}
    if set(probabilities) != expected_keys:
        raise _fail(
            f"{source} returned NLI keys {set(probabilities)}; expected {expected_keys}"
        )
    values = tuple(probabilities[key] for key in sorted(expected_keys))
    try:
        numeric_values = tuple(float(value) for value in values)
    except (TypeError, ValueError) as error:
        raise _fail(f"{source} returned non-numeric NLI probabilities") from error
    if not all(isfinite(value) for value in numeric_values):
        raise _fail(f"{source} returned non-finite NLI probabilities")
    if not all(0.0 <= value <= 1.0 for value in numeric_values):
        raise _fail(f"{source} returned NLI probabilities outside [0, 1]")
    if not np.isclose(sum(numeric_values), 1.0, rtol=1e-6, atol=1e-8):
        raise _fail(f"{source} returned NLI probabilities that do not sum to one")


def _nli_as_array(probabilities: NLIProbabilities) -> np.ndarray:
    return np.asarray(
        [
            probabilities["entailment"],
            probabilities["neutral"],
            probabilities["contradiction"],
        ],
        dtype=np.float64,
    )


def run_predictor_conformance(
    predictor: _ServingCapability,
    *,
    prediction_cases: Sequence[PredictionCase],
    response_encoding: ResponseEncoding,
    nli_cases: Sequence[NLICase],
) -> ConformanceReport:
    """Run the published serving-predictor conformance suite.

    The suite verifies structural capability composition, identity, task and
    item compatibility, ``ResponseEncoding`` column semantics, probability
    shape and normalization, optional participant identifiers, native artifact
    save/load equality, and three-way NLI probabilities.

    Parameters
    ----------
    predictor : ServingPredictor
        Structural implementation of all serving capabilities. It does not
        need to inherit from a bead class.
    prediction_cases : Sequence[PredictionCase]
        Non-empty sequence of real :class:`Item` and :class:`ItemTemplate`
        pairs accepted by the predictor.
    response_encoding : ResponseEncoding
        Encoding whose ordered labels name the columns returned by
        ``predict_proba``.
    nli_cases : Sequence[NLICase]
        Non-empty sequence of premise-hypothesis pairs.

    Returns
    -------
    ConformanceReport
        Summary of the checks that passed.

    Raises
    ------
    ConformanceError
        If any capability violates the serving contract.

    Examples
    --------
    >>> report = run_predictor_conformance(  # doctest: +SKIP
    ...     predictor,
    ...     prediction_cases=[PredictionCase(item, item_template)],
    ...     response_encoding=encoding,
    ...     nli_cases=[NLICase("A dog runs.", "An animal moves.")],
    ... )
    >>> report.checks  # doctest: +SKIP
    ('identity', 'task_compatibility', 'predict_proba', 'persistence', 'nli')
    """
    if not isinstance(predictor, ServingPredictor):
        raise _fail("predictor does not implement every ServingPredictor capability")
    if not prediction_cases:
        raise _fail("prediction_cases must contain at least one case")
    if not nli_cases:
        raise _fail("nli_cases must contain at least one case")
    if not _same_response_encoding(predictor.response_encoding, response_encoding):
        raise _fail("predictor.response_encoding does not match response_encoding")

    _validate_identity(predictor)

    expected_task_type = scale_type_to_task_type(response_encoding.scale_type)
    items: list[Item] = []
    for index, case in enumerate(prediction_cases):
        item_errors = validate_item(case.item, case.item_template)
        if item_errors:
            raise _fail(
                f"prediction_cases[{index}] is not a valid Item/ItemTemplate pair: "
                f"{'; '.join(item_errors)}"
            )
        if case.item_template.task_type != expected_task_type:
            raise _fail(
                f"prediction_cases[{index}] uses task type "
                f"{case.item_template.task_type!r}, but encoding "
                f"{response_encoding.name!r} maps to {expected_task_type!r}"
            )
        if case.item_template.task_type not in predictor.supported_task_types:
            raise _fail(
                f"prediction_cases[{index}] uses unsupported task type "
                f"{case.item_template.task_type!r}"
            )
        try:
            predictor.validate_item_compatibility(case.item, case.item_template)
        except (TypeError, ValueError) as error:
            raise _fail(
                f"prediction_cases[{index}] failed compatibility validation: {error}"
            ) from error
        items.append(case.item)

    baseline = predictor.predict_proba(items)
    _validate_prediction_matrix(
        baseline,
        n_items=len(items),
        response_encoding=response_encoding,
        source="predict_proba(items)",
    )

    participant_ids = [f"bead-conformance-{index}" for index in range(len(items))]
    participant_predictions = predictor.predict_proba(items, participant_ids)
    _validate_prediction_matrix(
        participant_predictions,
        n_items=len(items),
        response_encoding=response_encoding,
        source="predict_proba(items, participant_ids)",
    )

    baseline_nli: list[NLIProbabilities] = []
    for index, case in enumerate(nli_cases):
        scores = predictor.compute_nli(case.premise, case.hypothesis)
        _validate_nli(scores, source=f"compute_nli case {index}")
        baseline_nli.append(scores)

    with TemporaryDirectory(prefix="bead-predictor-conformance-") as directory:
        artifact_path = Path(directory) / "artifact"
        predictor.save(artifact_path)
        if not artifact_path.exists():
            raise _fail("save(path) did not create a native artifact")
        restored = type(predictor).load(artifact_path)

        if restored.model_name != predictor.model_name:
            raise _fail("load(path) changed model_name")
        if restored.model_version != predictor.model_version:
            raise _fail("load(path) changed model_version")
        if set(restored.supported_task_types) != set(predictor.supported_task_types):
            raise _fail("load(path) changed supported_task_types")
        if not _same_response_encoding(restored.response_encoding, response_encoding):
            raise _fail("load(path) changed response_encoding")

        for index, case in enumerate(prediction_cases):
            try:
                restored.validate_item_compatibility(case.item, case.item_template)
            except (TypeError, ValueError) as error:
                raise _fail(
                    f"load(path) changed compatibility for prediction_cases[{index}]: "
                    f"{error}"
                ) from error

        restored_predictions = restored.predict_proba(items)
        _validate_prediction_matrix(
            restored_predictions,
            n_items=len(items),
            response_encoding=response_encoding,
            source="loaded predict_proba(items)",
        )
        if not np.allclose(baseline, restored_predictions, rtol=1e-6, atol=1e-8):
            raise _fail("save/load changed predict_proba output")

        for index, (case, expected) in enumerate(
            zip(nli_cases, baseline_nli, strict=True)
        ):
            actual = restored.compute_nli(case.premise, case.hypothesis)
            _validate_nli(actual, source=f"loaded compute_nli case {index}")
            if not np.allclose(
                _nli_as_array(expected),
                _nli_as_array(actual),
                rtol=1e-6,
                atol=1e-8,
            ):
                raise _fail("save/load changed compute_nli output")

    return ConformanceReport(
        model_name=predictor.model_name,
        model_version=predictor.model_version,
        n_items=len(items),
        n_levels=response_encoding.n_levels,
        n_nli_cases=len(nli_cases),
        checks=(
            "identity",
            "task_compatibility",
            "predict_proba",
            "persistence",
            "nli",
        ),
    )
