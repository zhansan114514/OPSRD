from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from summarize_multiseed import summarize_multiseed  # noqa: E402


DATASETS = ("aime24", "aime25", "hmmt25")


def make_run(correct: int) -> dict[str, dict]:
    return {
        dataset: {
            "average_at_n_pct": 100 * correct / 12,
            "results": [
                {"problem_id": str(problem_id), "num_correct": correct, "val_n": 12}
                for problem_id in range(10)
            ],
        }
        for dataset in DATASETS
    }


class MultiseedSummaryTests(unittest.TestCase):
    def test_consistent_improvement_has_positive_hierarchical_interval(self) -> None:
        report = summarize_multiseed(
            make_run(5),
            {"seed42": make_run(6), "seed43": make_run(6), "seed44": make_run(6)},
            bootstrap_samples=1_000,
            seed=2026,
        )
        macro = report["aggregate"]["macro"]
        self.assertEqual(macro["positive_seed_count"], 3)
        self.assertGreater(macro["difference_hierarchical_ci95_pct"][0], 0)

    def test_seed_variability_is_reported(self) -> None:
        report = summarize_multiseed(
            make_run(5),
            {"seed42": make_run(4), "seed43": make_run(5), "seed44": make_run(6)},
            bootstrap_samples=1_000,
            seed=2026,
        )
        macro = report["aggregate"]["macro"]
        self.assertGreater(macro["candidate_std_pct"], 0)
        lower, upper = macro["difference_hierarchical_ci95_pct"]
        self.assertLessEqual(lower, 0)
        self.assertGreaterEqual(upper, 0)


if __name__ == "__main__":
    unittest.main()
