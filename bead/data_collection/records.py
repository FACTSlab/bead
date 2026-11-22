"""Bridge from JATOS results to bead annotation records.

JATOS returns experimental results as nested JSON: each study run
contains a ``data`` array of trial objects, each carrying the
metadata serialized by
:func:`bead.deployment.jspsych.trials._serialize_item_metadata` and a
jsPsych ``response`` field.

This module is the single canonical conversion from that
representation into :class:`~bead.evaluation.AnnotationRecord`
instances, the input shape consumed by every reliability,
inter-annotator-agreement, and conditional-observation check in bead.
There is no other path from raw JATOS output to bead records.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from bead.data.base import JsonValue
from bead.data_collection.result_stream import parse_jatos_result_stream
from bead.evaluation.reliability import AnnotationRecord


def _coerce_response_label(response: JsonValue) -> str:
    """Normalize a jsPsych ``response`` value to a string label.

    Parameters
    ----------
    response : JsonValue
        The raw value emitted by jsPsych. For binary, categorical, and
        forced-choice tasks this is already a string label; for
        ordinal, magnitude, and similar numeric tasks it is an int or
        float; jsPsych may also wrap the response in an object with a
        ``"response"`` key for some plugin variants.

    Returns
    -------
    str
        String form suitable for :class:`AnnotationRecord.response_label`.
    """
    if isinstance(response, str):
        return response
    if isinstance(response, bool):
        return "true" if response else "false"
    if isinstance(response, int | float):
        return str(response)
    if isinstance(response, dict) and "response" in response:
        inner = response["response"]
        if isinstance(inner, str):
            return inner
        return str(inner)
    return str(response)


def _annotator_id(
    result: Mapping[str, Any],
    *,
    annotator_id_key: str,
) -> str | None:
    """Extract the annotator id from a JATOS result envelope.

    Looks first in the URL query parameters (``urlQueryParameters``)
    for the configured key (typically ``"PROLIFIC_PID"``), then falls
    back to the JATOS-assigned ``worker_id``.
    """
    url_params = result.get("urlQueryParameters") or result.get("url_params")
    if isinstance(url_params, Mapping) and annotator_id_key in url_params:
        candidate = url_params[annotator_id_key]
        if isinstance(candidate, str) and candidate:
            return candidate
    participant_id = result.get("prolific_pid") or result.get("participant_id")
    if isinstance(participant_id, str) and participant_id:
        return participant_id
    worker_id = (
        result.get("worker_id")
        or result.get("workerId")
        or result.get("jatos_worker_id")
    )
    if isinstance(worker_id, str) and worker_id:
        return worker_id
    if isinstance(worker_id, int):
        return str(worker_id)
    return None


def jatos_results_to_annotation_records(
    results: Iterable[Mapping[str, Any]],
    *,
    annotator_id_key: str = "PROLIFIC_PID",
) -> tuple[AnnotationRecord, ...]:
    """Convert a sequence of JATOS results to :class:`AnnotationRecord`s.

    Each JATOS result is expected to be the dict shape returned by
    :class:`~bead.data_collection.JATOSDataCollector` (a study run
    with a ``data`` field carrying jsPsych trial dicts). Trials that
    lack ``item_id`` or ``template_name`` are silently skipped, since
    they correspond to non-question trials such as instructions or
    consent.

    Parameters
    ----------
    results : Iterable[Mapping[str, Any]]
        JATOS result envelopes with an aggregate array or appended JSONL.
    annotator_id_key : str, optional
        Query-parameter key carrying the annotator identifier.
        Defaults to ``"PROLIFIC_PID"``.

    Returns
    -------
    tuple[AnnotationRecord, ...]
        One record per scalar response or answered named field. Named field
        questions use template_name.field_name. Missing answers are skipped.
    """
    records: list[AnnotationRecord] = []
    for result in results:
        payload = result.get("data")
        stream = (
            parse_jatos_result_stream(payload)
            if isinstance(payload, str | list | tuple)
            else None
        )
        annotator_id = _annotator_id(result, annotator_id_key=annotator_id_key)
        metadata = result.get("metadata")
        if annotator_id is None and stream is not None and stream.session is not None:
            annotator_id = _annotator_id(
                stream.session, annotator_id_key=annotator_id_key
            )
        if annotator_id is None and isinstance(metadata, Mapping):
            annotator_id = _annotator_id(metadata, annotator_id_key=annotator_id_key)
        if annotator_id is None:
            continue
        trials = stream.trials if stream is not None else payload
        if not isinstance(trials, list | tuple):
            continue
        for trial in trials:
            if (
                not isinstance(trial, Mapping)
                or trial.get("annotation_eligible") is False
            ):
                continue
            item_id = trial.get("item_id")
            question_name = trial.get("template_name")
            if not isinstance(item_id, str) or not isinstance(question_name, str):
                continue
            fields = trial.get("response_field_names")
            responses = trial.get("responses")
            if isinstance(fields, list | tuple) and isinstance(responses, Mapping):
                named_responses = (
                    (f"{question_name}.{name}", responses.get(name))
                    for name in fields
                    if isinstance(name, str) and name
                )
            else:
                named_responses = ((question_name, trial.get("response")),)
            for name, response in named_responses:
                if response is None or (
                    isinstance(responses, Mapping) and response == ""
                ):
                    continue
                records.append(
                    AnnotationRecord(
                        annotator_id=annotator_id,
                        item_id=item_id,
                        question_name=name,
                        response_label=_coerce_response_label(response),
                    )
                )
    return tuple(records)
