from __future__ import annotations

import sys
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from check_role_signal_gate import evaluate_gate  # noqa: E402


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


class RoleSignalGateTests(unittest.TestCase):
    def test_gate_passes_macro_and_neutral_control_criterion(self) -> None:
        result = evaluate_gate(
            make_run(5),
            make_run(6),
            make_run(5),
            min_macro_improvement_pct=2.0,
            bootstrap_samples=1_000,
            seed=2026,
        )
        self.assertTrue(result["passed"])
        self.assertTrue(result["criteria"]["macro_and_control"])

    def test_gate_passes_taskwise_positive_interval_criterion(self) -> None:
        result = evaluate_gate(
            make_run(5),
            make_run(6),
            make_run(7),
            min_macro_improvement_pct=20.0,
            bootstrap_samples=1_000,
            seed=2026,
        )
        self.assertTrue(result["passed"])
        self.assertFalse(result["criteria"]["macro_and_control"])
        self.assertTrue(result["criteria"]["taskwise_and_paired"])

    def test_gate_rejects_no_signal(self) -> None:
        result = evaluate_gate(
            make_run(5),
            make_run(5),
            make_run(4),
            min_macro_improvement_pct=2.0,
            bootstrap_samples=1_000,
            seed=2026,
        )
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()
