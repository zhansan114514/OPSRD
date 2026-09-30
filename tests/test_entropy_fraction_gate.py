import unittest

from analysis.check_entropy_fraction_gate import score_first_n, screen


def make_summary(dataset, outcomes):
    return {
        "dataset": dataset,
        "val_n": len(outcomes[0]),
        "results": [
            {
                "problem_id": index,
                "generations": [{"correct": value} for value in problem],
            }
            for index, problem in enumerate(outcomes)
        ],
    }


class EntropyFractionGateTests(unittest.TestCase):
    def test_reference_is_sliced_to_candidate_sample_count(self):
        candidate = make_summary("aime24", [[True, True], [True, False]])
        reference = make_summary(
            "aime24",
            [[True, False, True, True], [False, False, True, True]],
        )
        report = screen({"aime24": candidate}, {"aime24": reference}, threshold=20.0)

        self.assertEqual(score_first_n(reference, 2), 25.0)
        self.assertEqual(report["difference_pp"], 50.0)
        self.assertTrue(report["promote"])

    def test_threshold_is_inclusive(self):
        candidate = make_summary("aime24", [[True, False]])
        reference = make_summary("aime24", [[False, False]])
        report = screen({"aime24": candidate}, {"aime24": reference}, threshold=50.0)
        self.assertTrue(report["promote"])


if __name__ == "__main__":
    unittest.main()
