"""Tests for tools/sample_size_planner.py (SIMULATION; known generating model)."""

from __future__ import annotations

import io
import os
import sys
import unittest
from contextlib import redirect_stderr, redirect_stdout

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "tools"))

import sample_size_planner as sp  # noqa: E402


class TestPlanner(unittest.TestCase):
    def test_build_model_rejects_impossible_inputs(self):
        with self.assertRaises(ValueError):
            sp.build_model([0.9, 0.9], 0.75, 3, 0.35)  # R^2 >= 1
        with self.assertRaises(ValueError):
            sp.build_model([0.3], 1.0, 3, 0.3)
        with self.assertRaises(ValueError):
            sp.build_model([0.3], 0.7, 2, 0.3)

    def test_plan_is_deterministic_and_precision_improves_with_n(self):
        model = sp.build_model(sp.DEFAULT_BETAS, 0.75, 3, 0.35)
        a = sp.plan(model, [100, 1000], reps=15, seed=1)
        b = sp.plan(model, [100, 1000], reps=15, seed=1)
        pd.testing.assert_frame_equal(a, b)
        self.assertGreater(a.loading_sd[0], a.loading_sd[1])
        self.assertGreater(a.path_sd[0], a.path_sd[1])

    def test_recommend_picks_smallest_passing_n_or_none(self):
        df = pd.DataFrame(
            {
                "n": [100, 200, 300],
                "proper_solution_rate": [0.80, 0.97, 1.0],
                "loading_sd": [0.09, 0.06, 0.04],
                "path_sd": [0.3, 0.2, 0.1],
            }
        )
        self.assertEqual(sp.recommend(df), 300)
        self.assertEqual(sp.recommend(df, max_loading_sd=0.07), 200)
        self.assertEqual(sp.recommend(df, max_loading_sd=0.07, max_path_sd=0.15), 300)
        self.assertIsNone(sp.recommend(df, max_loading_sd=0.01))

    def test_cli_reports_none_without_extrapolating(self):
        out = io.StringIO()
        with redirect_stdout(out):
            code = sp.main(["--grid", "100", "--reps", "10", "--max-loading-sd", "0.001"])
        self.assertEqual(code, 0)
        self.assertIn("nothing is extrapolated", out.getvalue())

    def test_cli_bad_model_exits_2(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(sp.main(["--loading", "1.5", "--reps", "5"]), 2)


class TestCommittedPlannerResults(unittest.TestCase):
    """The README quotes recommended N from these committed tables."""

    def load(self, name):
        return pd.read_csv(os.path.join(ROOT, "results", name))

    def test_recommended_n_matches_readme_claims(self):
        self.assertEqual(sp.recommend(self.load("planner_loading075.csv")), 200)
        self.assertEqual(sp.recommend(self.load("planner_loading050.csv")), 1000)

    def test_row_regenerates_from_code(self):
        committed = self.load("planner_loading075.csv")
        committed = committed[committed.n == 200].reset_index(drop=True)
        model = sp.build_model(sp.DEFAULT_BETAS, 0.75, 3, 0.35)
        fresh = sp.plan(model, [200], reps=300)
        pd.testing.assert_frame_equal(fresh, committed, rtol=1e-6, check_dtype=False)


if __name__ == "__main__":
    unittest.main()
