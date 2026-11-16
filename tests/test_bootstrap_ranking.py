import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from simulations.study_06_bootstrap_ranking import interval_fields, paired_bootstrap_contrasts
from simulations.study_05_inference_audit import Cell, model_for
from simulations.dgp import generate_dataset
from ehs_risk_sem.model import ModelSpec


class TestBootstrapRanking(unittest.TestCase):
    def test_contrast_is_paired_within_replicate(self):
        # Strong shared motion cancels in the contrast; marginal SEs would not.
        returns = []
        for x in (1, 2, 3, 4):
            returns.append(SimpleNamespace(
                paths={"Risk": SimpleNamespace(beta=np.array([x, x + .5, 0, 0]), r_squared=.2)},
                phi_repaired=False, measurement={"x": SimpleNamespace(heywood=False, converged=True)}))
        with patch("simulations.study_06_bootstrap_ranking.fit", side_effect=returns):
            draws, failed, improper, nonconverged = paired_bootstrap_contrasts(
                None, pd.DataFrame({"x": range(20)}), np.array([1, -1, 0, 0]), 4, 7)
        np.testing.assert_equal(draws, [-.5] * 4)
        self.assertEqual((failed, improper, nonconverged), (0, 0, 0))

    def test_omitting_unused_fit_indices_preserves_exact_estimates(self):
        for loading in (.5, .75, .9):
            model = model_for(Cell("positive", loading, .5), "alternative")
            data, _ = generate_dataset(model, 385, 371)
            spec = ModelSpec(latents=model.indicator_names, paths=model.paths())
            args = (spec, data, np.array([1, -1, 0, 0]), 5, 317)
            full = paired_bootstrap_contrasts(*args, evaluate_fit_indices=True)
            fast = paired_bootstrap_contrasts(*args, evaluate_fit_indices=False)
            self.assertEqual(full, fast)

    def test_percentile_null_rejection_and_truth_coverage(self):
        fields = interval_fields(.4, np.linspace(.1, .7, 399), .4)
        self.assertTrue(fields["percentile_reject"])
        self.assertTrue(fields["percentile_coverage"])
        fields = interval_fields(0, np.linspace(-.5, .5, 399), 0)
        self.assertFalse(fields["percentile_reject"])
        self.assertTrue(fields["percentile_coverage"])

    def test_failed_bootstraps_counted_not_zero_imputed(self):
        with patch("simulations.study_06_bootstrap_ranking.fit", side_effect=ValueError("injected")):
            draws, failed, _, _ = paired_bootstrap_contrasts(
                None, pd.DataFrame({"x": range(20)}), np.array([1, -1, 0, 0]), 4, 7)
        self.assertEqual(draws, [])
        self.assertEqual(failed, 4)


if __name__ == "__main__":
    unittest.main()
