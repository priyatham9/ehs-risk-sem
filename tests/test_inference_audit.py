"""Checks for the simulation's estimands, seed isolation and failure accounting."""
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from simulations.study_05_inference_audit import (
    BASE_BETA, Cell, model_for, oracle_ols, rate_fields, replicate,
    run_cell, seed_for, wilson,
)


class TestInferenceAudit(unittest.TestCase):
    def test_null_preserves_pair_mean_and_other_coefficients(self):
        for pair, indexes in [("positive", (0, 1)), ("negative", (2, 3))]:
            model = model_for(Cell(pair, .75, 1), "null_validation")
            beta = model.beta[4, :4]
            i, j = indexes
            self.assertEqual(beta[i], beta[j])
            self.assertAlmostEqual(beta[i] + beta[j], BASE_BETA[i] + BASE_BETA[j])
            np.testing.assert_equal(np.delete(beta, indexes), np.delete(BASE_BETA, indexes))
            self.assertAlmostEqual(model.implied_latent_correlation()[4, 4], 1)

    def test_streams_are_separate_and_reproducible(self):
        cell = Cell("positive", .75, 1)
        seeds = {seed_for(12, cell, stage, r)
                 for stage in ("null_calibration", "null_validation", "alternative")
                 for r in range(100)}
        self.assertEqual(len(seeds), 300)
        self.assertEqual(seed_for(12, cell, "alternative", 7), seed_for(12, cell, "alternative", 7))

    def test_oracle_matches_augmented_design_ols(self):
        rng = np.random.default_rng(4)
        eta = rng.normal(size=(50, 5))
        contrast = np.array([1, -1, 0, 0])
        estimate, se = oracle_ols(eta, contrast)
        x = np.column_stack([np.ones(50), eta[:, :4]])
        b = np.linalg.lstsq(x, eta[:, 4], rcond=None)[0]
        c = np.r_[0, contrast]
        s2 = np.sum((eta[:, 4] - x @ b)**2) / 45
        self.assertAlmostEqual(estimate, c @ b)
        self.assertAlmostEqual(se, np.sqrt(s2 * c @ np.linalg.inv(x.T @ x) @ c))

    def test_fit_failure_does_not_zero_impute_or_lose_oracle(self):
        with patch("simulations.study_05_inference_audit.fit", side_effect=ValueError("injected failure")):
            row = replicate(Cell("positive", .75, .5), "alternative", 0, 71)
        self.assertFalse(row["observed_valid"])
        self.assertFalse(row["observed_reject"])
        self.assertNotIn("observed_estimate", row)
        self.assertTrue(np.isfinite(row["oracle_estimate"]))
        self.assertIn("injected failure", row["error"])

    def test_checkpoint_resume_does_not_duplicate_replicates(self):
        with tempfile.TemporaryDirectory() as directory:
            args = (Cell("positive", .75, .5), "alternative", 2, 71, directory)
            first = run_cell(args)
            second = run_cell(args)
            self.assertEqual(first["new_reps"], 2)
            self.assertEqual(second["new_reps"], 0)
            with self.assertRaisesRegex(ValueError, "identity mismatch"):
                run_cell((args[0], args[1], 2, 72, directory))

    def test_probability_denominators_and_boundary_intervals(self):
        row = rate_fields("test", [True, False, False, True])
        self.assertEqual(row["test"], .5)
        self.assertEqual(row["test_denominator"], 4)
        self.assertGreater(wilson(0, 100)[1], 0)
        self.assertLess(wilson(100, 100)[0], 1)
        self.assertTrue(np.isnan(rate_fields("empty", [])["empty"]))

    def test_actual_estimator_reproducible(self):
        a = replicate(Cell("positive", .75, .5), "alternative", 3, 11)
        b = replicate(Cell("positive", .75, .5), "alternative", 3, 11)
        self.assertEqual(a, b)
        self.assertTrue(a["observed_valid"])
        self.assertAlmostEqual(a["true_difference"], .15)


if __name__ == "__main__":
    unittest.main()
