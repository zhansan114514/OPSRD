from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))

from compare_training_rollouts import compare  # noqa: E402


class RolloutComparisonTests(unittest.TestCase):
    def test_pairs_same_user_message_across_system_prompt_change(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            reference = root / "reference"
            candidate = root / "candidate"
            reference.mkdir()
            candidate.mkdir()
            user = "<|im_start|>user\nProblem<|im_end|>"
            reference_payload = {
                "generations": [{"prompt": user, "completion": "A \\boxed{1}"}]
            }
            candidate_payload = {
                "generations": [
                    {
                        "prompt": "<|im_start|>system\nRole<|im_end|>\n" + user,
                        "completion": "B \\boxed{1}",
                    }
                ]
            }
            (reference / "generations_step_5.json").write_text(
                json.dumps(reference_payload), encoding="utf-8"
            )
            (candidate / "generations_step_5.json").write_text(
                json.dumps(candidate_payload), encoding="utf-8"
            )

            result = compare(reference, candidate)

        self.assertEqual(result["paired_outputs"], 1)
        self.assertEqual(result["same_user_message"], 1)
        self.assertEqual(result["identical_completion"], 0)
        self.assertEqual(result["reference"]["boxed_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
