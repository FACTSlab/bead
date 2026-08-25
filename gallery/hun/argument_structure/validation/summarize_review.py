"""Summarize a completed Hungarian human-review sheet."""
from __future__ import annotations
import csv
from collections import Counter
from pathlib import Path

PATH = Path(__file__).with_name("human_review.csv")
VALID = {
    "natural", "acceptable_but_marked", "semantically_odd",
    "verb_frame_mismatch", "generator_error", "uncertain",
}
CONF = {"high", "medium", "low"}

with PATH.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

labels = Counter()
bad = []
for row in rows:
    label = row["human_label"].strip()
    conf = row["confidence"].strip()
    if not label:
        labels["unreviewed"] += 1
        continue
    if label not in VALID:
        bad.append((row["review_id"], f"invalid label: {label}"))
    if conf and conf not in CONF:
        bad.append((row["review_id"], f"invalid confidence: {conf}"))
    labels[label] += 1

print(f"Rows: {len(rows)}")
for key, value in labels.most_common():
    print(f"{key}: {value}")

if bad:
    print("\nProblems:")
    for rid, msg in bad:
        print(f"- {rid}: {msg}")
    raise SystemExit(1)
