"""Report missing/duplicate recorded attempts; never rewrite or impute rewards."""

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path


def audit(rows, models, tasks, attempts=5):
    cells = collections.defaultdict(list)
    seen = set()
    duplicates = []
    for row in rows:
        model, task = row["model"], row["task"]
        if model not in models or task not in tasks:
            raise ValueError(f"Unexpected model/task: {model}/{task}")
        identity = (model, task, row["replicate"])
        if not identity[2]:
            raise ValueError("Attempt identity is required")
        if identity in seen:
            duplicates.append(identity)
        seen.add(identity)
        if row["reward"] and not math.isfinite(float(row["reward"])):
            raise ValueError(
                "Reward must be finite; fractional task scores are retained"
            )
        cells[model, task].append(row)
    incomplete = []
    for model in models:
        for task in tasks:
            recorded = cells[model, task]
            scored = sum(row["reward"] != "" for row in recorded)
            if len(recorded) != attempts or scored != attempts:
                incomplete.append(
                    {
                        "model": model,
                        "task": task,
                        "recorded": len(recorded),
                        "scored": scored,
                        "missing_rows": max(0, attempts - len(recorded)),
                        "unscored_ids": [
                            r["replicate"] for r in recorded if not r["reward"]
                        ],
                    }
                )
    return {
        "expected_attempts": len(models) * len(tasks) * attempts,
        "recorded_attempts": sum(len(value) for value in cells.values()),
        "duplicates": duplicates,
        "incomplete_cells": incomplete,
        "complete_counts_only": not incomplete and not duplicates,
        "caveat": "Counts do not validate image provenance, grading correctness, or variant comparability.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", required=True, type=Path)
    parser.add_argument(
        "--inventory",
        required=True,
        type=Path,
        help="JSON with explicit models and tasks lists",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    raw = args.csv.read_bytes()
    with args.csv.open(newline="") as source:
        result = audit(csv.DictReader(source), inventory["models"], inventory["tasks"])
    result["source_sha256"] = hashlib.sha256(raw).hexdigest()
    with args.output.open("x") as target:
        json.dump(result, target, indent=2)


if __name__ == "__main__":
    main()
