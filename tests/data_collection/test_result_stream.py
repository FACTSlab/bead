"""Tests for JATOS archive loading and partial trial normalization."""

import json
import zipfile

from bead.data_collection import (
    jatos_results_to_annotation_records,
    load_jatos_result_archive,
    parse_jatos_result_stream,
)


def test_archived_partial_stream_keeps_duplicates_and_named_answers(tmp_path) -> None:
    records = [
        {"record_type": "session", "seq": 0, "prolific_pid": "study-id"},
        {
            "record_type": "trial",
            "seq": 3,
            "item_id": "stable-item",
            "template_name": "duration",
            "response_field_names": ["preposition", "numeral", "unit"],
            "responses": {"preposition": "in", "numeral": None, "unit": "millenia"},
        },
        {"record_type": "plan", "seq": 1, "trials": []},
        {"record_type": "event", "seq": 4, "event": "pagehide"},
    ]
    delivered = [*records, records[1]]
    raw = "\n".join(json.dumps(record) for record in delivered) + '\n{"truncated"'
    path = tmp_path / "results.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(
            "metadata.json",
            json.dumps(
                {
                    "data": [
                        {
                            "studyId": 2,
                            "studyResults": [
                                {
                                    "id": 6,
                                    "workerId": 9,
                                    "componentResults": [
                                        {
                                            "id": 11,
                                            "componentId": 1,
                                            "componentState": "ABORTED",
                                        },
                                        {
                                            "id": 12,
                                            "componentId": 1,
                                            "componentState": "STARTED",
                                        },
                                    ],
                                }
                            ],
                        }
                    ],
                }
            ),
        )
        archive.writestr("study_result_6/comp_result_11/data.txt", raw)
    results = load_jatos_result_archive(path)
    assert results[0]["data"] == raw
    assert results[1]["data"] is None
    stream = parse_jatos_result_stream(raw)
    assert stream is not None
    assert stream.records == tuple(delivered)
    assert stream.missing_seq_ranges == ((2, 2),)
    assert stream.malformed_line_numbers == (6,)
    assert stream.terminal is None
    assert stream.trials == (records[1],)
    assert [
        (a.annotator_id, a.item_id, a.question_name, a.response_label)
        for a in jatos_results_to_annotation_records(results)
    ] == [
        ("study-id", "stable-item", "duration.preposition", "in"),
        ("study-id", "stable-item", "duration.unit", "millenia"),
    ]
