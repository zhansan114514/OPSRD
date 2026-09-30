#!/usr/bin/env python3
"""Require the official OPSD evaluation to improve over the base macro average."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_scores(directory: Path) -> dict[str, float]:
    scores = {}
    for path in directory.glob("*.json"):
        with path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        scores[payload["dataset"]] = float(payload["average_at_n_pct"])
    return scores


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--minimum-improvement", type=float, default=0.0)
    args = parser.parse_args()

    base = load_scores(args.base)
    candidate = load_scores(args.candidate)
    datasets = sorted(base.keys() & candidate.keys())
    if not datasets:
        raise ValueError("Base and candidate evaluations have no shared datasets.")

    base_macro = sum(base[dataset] for dataset in datasets) / len(datasets)
    candidate_macro = sum(candidate[dataset] for dataset in datasets) / len(datasets)
    improvement = candidate_macro - base_macro
    print(
        f"official reproduction: base={base_macro:.4f}, candidate={candidate_macro:.4f}, "
        f"improvement={improvement:+.4f} percentage points"
    )
    if improvement <= args.minimum_improvement:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
