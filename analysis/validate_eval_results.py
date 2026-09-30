#!/usr/bin/env python3
"""Validate that evaluation JSON files are complete and internally consistent."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def validate_result(path: Path, expected_val_n: int) -> None:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    required = {
        "dataset",
        "val_n",
        "num_problems",
        "total_solutions",
        "average_at_n_pct",
        "results",
    }
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"{path}: missing fields {missing}")
    if payload["val_n"] != expected_val_n:
        raise ValueError(f"{path}: val_n={payload['val_n']}, expected {expected_val_n}")
    if len(payload["results"]) != payload["num_problems"]:
        raise ValueError(
            f"{path}: {len(payload['results'])} result rows, expected {payload['num_problems']}"
        )
    if payload["total_solutions"] != payload["num_problems"] * expected_val_n:
        raise ValueError(f"{path}: inconsistent total_solutions")

    for index, result in enumerate(payload["results"]):
        generations = result.get("generations")
        if not isinstance(generations, list) or len(generations) != expected_val_n:
            raise ValueError(
                f"{path}: result {index} has {len(generations) if isinstance(generations, list) else 0} "
                f"generations, expected {expected_val_n}"
            )
        num_correct = result.get("num_correct")
        if not isinstance(num_correct, int) or not 0 <= num_correct <= expected_val_n:
            raise ValueError(f"{path}: invalid num_correct at result {index}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expected-val-n", type=int, required=True)
    parser.add_argument("paths", type=Path, nargs="+")
    args = parser.parse_args()

    for path in args.paths:
        validate_result(path, args.expected_val_n)


if __name__ == "__main__":
    main()
