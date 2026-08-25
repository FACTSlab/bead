#!/usr/bin/env python3
"""Build a balanced human-review sheet from generated Hungarian stimuli.

Reviewers distinguish verb-frame mismatch from unrelated generator, morphology,
or filler errors.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

LABELS = (
    "natural",
    "acceptable_but_marked",
    "semantically_odd",
    "verb_frame_mismatch",
    "generator_error",
    "uncertain",
)


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_frame_reference(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as f:
        return {row["template_name"]: row for row in csv.DictReader(f)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="filled_templates/generic_frames_filled.jsonl")
    parser.add_argument("--output", default="validation/human_review.csv")
    parser.add_argument("--per-template", type=int, default=3)
    parser.add_argument("--frame-reference", default="FRAME_REFERENCE.csv")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    frame_reference = load_frame_reference(Path(args.frame_reference))
    by_template: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_template[row.get("template_name", "UNKNOWN")].append(row)

    selected = []
    for template in sorted(by_template):
        selected.extend(by_template[template][: args.per_template])

    fields = [
        "review_id", "hungarian_sentence", "frame_example_translation",
        "verb", "template", "frame_description", "human_label", "confidence_1_to_5",
        "generator_error_type", "notes", "reviewer", "review_date",
    ]
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(selected, 1):
            verb = row.get("slot_fillers", {}).get("verb", {}).get("lemma", "")
            template_name = row.get("template_name", "")
            ref = frame_reference.get(template_name, {})
            writer.writerow({
                "review_id": f"HUN-{i:04d}",
                "hungarian_sentence": row.get("rendered_text", ""),
                "frame_example_translation": ref.get("english_translation", ""),
                "verb": verb,
                "template": template_name,
                "frame_description": ref.get("frame_description", ""),
            })

    print(f"Wrote {len(selected)} review rows to {out}")
    print("Allowed human_label values:", ", ".join(LABELS))


if __name__ == "__main__":
    main()
