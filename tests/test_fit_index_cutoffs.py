"""Tests for tools/fit_index_cutoffs.py (SIMULATION; fitted model taken as correct)."""

from __future__ import annotations

import io
import os
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import fit_index_cutoffs as fc  # noqa: E402
from ehs_risk_sem.model import fit  # noqa: E402

DATA = pd.read_csv(os.path.join(ROOT, "synthetic", "survey_style.csv"), comment="#")
SPEC = fc.spec_from_columns(DATA, "risk")
RES = fit(SPEC, DATA)


class TestModelFromFit(unittest.TestCase):
    def test_structure_is_carried_over(self):
        m = fc.model_from_fit(RES)
        self.assertEqual(m.latent_names, SPEC.latent_names())
        self.assertEqual(m.paths(), {k: list(v) for k, v in SPEC.paths.items()})
        np.testing.assert_allclose(m.beta, RES.beta_matrix)
        for k in m.latent_names:
            self.assertTrue(np.all((m.loadings[k] > 0) & (m.loadings[k] < 1)))

    def test_spec_from_columns_rejects_unknown_outcome(self):
        with self.assertRaises(ValueError):
            fc.spec_from_columns(DATA, "nope")


class TestSimulation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = fc.model_from_fit(RES)
        cls.a = fc.simulate_indices(cls.model, SPEC, 300, reps=20, seed=7)
        cls.b = fc.simulate_indices(cls.model, SPEC, 300, reps=20, seed=7)

    def test_seeded_reproducibility(self):
        pd.testing.assert_frame_equal(self.a, self.b)
        c = fc.simulate_indices(self.model, SPEC, 300, reps=20, seed=8)
        self.assertFalse(self.a["rmsea"].equals(c["rmsea"]))

    def test_correct_model_chi_square_centred_near_df(self):
        # Under a correct model E[chi2] = df; allow Monte Carlo and
        # limited-information slack (the package's chi2 is an upper bound).
        df = RES.fit_indices.df
        self.assertLess(abs(self.a["chi_square"].mean() - df), 0.2 * df)

    def test_cutoff_table_directions_and_failure_rates(self):
        t = fc.cutoff_table(self.a, 0.05).set_index("index")
        x = self.a["cfi"].to_numpy()
        self.assertAlmostEqual(t.loc["cfi", "cutoff"], np.quantile(x, 0.05))
        self.assertAlmostEqual(t.loc["rmsea", "cutoff"], np.quantile(self.a["rmsea"], 0.95))
        self.assertAlmostEqual(
            t.loc["srmr", "fixed_cutoff_failure_rate"], np.mean(self.a["srmr"] > 0.08)
        )
        self.assertTrue(np.isnan(t.loc["tli", "fixed_cutoff"]))

    def test_cutoff_table_observed_percentile_and_failed_fits_excluded(self):
        sims = pd.DataFrame(
            {
                "chi_square": [1.0, 2.0, 3.0, 4.0, np.nan],
                "cfi": [0.9, 0.92, 0.94, 0.96, np.nan],
                "tli": [0.9, 0.92, 0.94, 0.96, np.nan],
                "rmsea": [0.1, 0.08, 0.06, 0.04, np.nan],
                "srmr": [0.1, 0.09, 0.08, 0.07, np.nan],
                "proper": [True] * 4 + [False],
                "ok": [True] * 4 + [False],
            }
        )
        t = fc.cutoff_table(sims, 0.25, {"cfi": 0.93}).set_index("index")
        self.assertEqual(t.loc["cfi", "n_reps_used"], 4)
        self.assertAlmostEqual(t.loc["cfi", "observed_percentile"], 0.5)
        self.assertAlmostEqual(t.loc["cfi", "fixed_cutoff_failure_rate"], 0.75)
        with self.assertRaises(ValueError):
            fc.cutoff_table(sims, 0.6)

    def test_cutoff_moves_with_sample_size(self):
        small = fc.simulate_indices(self.model, SPEC, 150, reps=20, seed=3)
        large = fc.simulate_indices(self.model, SPEC, 1200, reps=20, seed=3)
        self.assertGreater(small["srmr"].mean(), large["srmr"].mean())


class TestCli(unittest.TestCase):
    def test_cli_runs_and_is_deterministic(self):
        outs = []
        for _ in range(2):
            buf = io.StringIO()
            with redirect_stdout(buf):
                self.assertEqual(fc.main(["--reps", "5", "--n", "250", "--seed", "1"]), 0)
            outs.append(buf.getvalue())
        self.assertEqual(outs[0], outs[1])
        self.assertIn("Says nothing about power", outs[0])

    def test_cli_bad_inputs_exit_2(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(fc.main(["--outcome", "nope"]), 2)
            self.assertEqual(fc.main(["--alpha", "0.9", "--reps", "5"]), 2)
            self.assertEqual(fc.main(["--data", "/nonexistent.csv"]), 2)


class TestCommittedTables(unittest.TestCase):
    """The README quotes these committed tables."""

    def load(self, name):
        return pd.read_csv(os.path.join(ROOT, "results", name)).set_index("index")

    def test_row_regenerates_from_code(self):
        sims = fc.simulate_indices(fc.model_from_fit(RES), SPEC, 200, reps=500)
        fresh = fc.cutoff_table(sims, 0.05)
        committed = pd.read_csv(os.path.join(ROOT, "results", "fit_index_cutoffs_n200.csv"))
        pd.testing.assert_frame_equal(
            fresh.drop(columns=["observed", "observed_percentile"]),
            committed.drop(columns=["observed", "observed_percentile"]),
            rtol=1e-6,
            check_dtype=False,
        )

    def test_cutoffs_looser_at_small_n(self):
        small, large = self.load("fit_index_cutoffs_n200.csv"), self.load("fit_index_cutoffs_n1200.csv")
        self.assertLess(small.loc["cfi", "cutoff"], large.loc["cfi", "cutoff"])
        self.assertGreater(small.loc["rmsea", "cutoff"], large.loc["rmsea", "cutoff"])
        self.assertGreater(small.loc["srmr", "cutoff"], large.loc["srmr", "cutoff"])


if __name__ == "__main__":
    unittest.main()
