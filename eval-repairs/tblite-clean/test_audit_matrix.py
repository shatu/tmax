import unittest

from audit_matrix import audit


class MatrixTest(unittest.TestCase):
    def test_fractional_rewards_are_valid_and_nonfinite_are_rejected(self):
        row = {"model": "m", "task": "t", "replicate": "trial", "reward": "0.375"}
        self.assertTrue(audit([row], ["m"], ["t"], 1)["complete_counts_only"])
        for reward in ("nan", "inf", "-inf"):
            with self.subTest(reward=reward), self.assertRaises(ValueError):
                audit([{**row, "reward": reward}], ["m"], ["t"], 1)

    def test_missing_cells_and_unscored_are_not_zeros(self):
        rows = [{"model": "m", "task": "t", "replicate": "1", "reward": ""}]
        result = audit(rows, ["m"], ["t", "other"], 1)
        self.assertEqual(result["expected_attempts"], 2)
        self.assertEqual(len(result["incomplete_cells"]), 2)
        self.assertEqual(result["incomplete_cells"][0]["scored"], 0)
        self.assertEqual(result["incomplete_cells"][1]["missing_rows"], 1)

    def test_duplicate_is_not_completion(self):
        row = {"model": "m", "task": "t", "replicate": "1", "reward": "0"}
        result = audit([row, row], ["m"], ["t"], 2)
        self.assertFalse(result["complete_counts_only"])
        self.assertEqual(len(result["duplicates"]), 1)
        with self.assertRaises(ValueError):
            audit([row], ["unexpected"], ["t"])


if __name__ == "__main__":
    unittest.main()
