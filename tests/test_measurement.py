"""Tests for the measurement model."""

from __future__ import annotations

import unittest

import numpy as np

from ehs_risk_sem.compat import silence_accelerate_matmul_warnings
from ehs_risk_sem.measurement import (
    average_variance_extracted,
    composite_reliability,
    cronbach_alpha,
    factor_score_determinacy,
    factor_scores,
    fit_congeneric_block,
    fornell_larcker,
    guttman_indeterminacy_bound,
    htmt,
    principal_axis_factor,
)

silence_accelerate_matmul_warnings()


class TestPrincipalAxisFactor(unittest.TestCase):
    def test_recovers_loadings_from_a_population_correlation_matrix(self) -> None:
        """Exact population input, so the answer should be exact."""
        lam = np.array([0.9, 0.8, 0.7, 0.6])
        r = np.outer(lam, lam)
        np.fill_diagonal(r, 1.0)
        loadings, comm, converged, _, heywood = principal_axis_factor(r, n_factors=1)
        self.assertTrue(converged)
        self.assertFalse(heywood)
        for est, truth in zip(loadings[:, 0], lam):
            self.assertAlmostEqual(abs(est), truth, places=5)
        for c, truth in zip(comm, lam**2):
            self.assertAlmostEqual(c, truth, places=5)

    def test_two_factor_extraction(self) -> None:
        lam = np.zeros((6, 2))
        lam[:3, 0] = 0.8
        lam[3:, 1] = 0.8
        phi = np.array([[1.0, 0.4], [0.4, 1.0]])
        r = lam @ phi @ lam.T
        np.fill_diagonal(r, 1.0)
        loadings, _, converged, _, _ = principal_axis_factor(r, n_factors=2)
        self.assertTrue(converged)
        self.assertEqual(loadings.shape, (6, 2))
        # Communalities should approximately reproduce the generating values.
        comm = np.sum(loadings**2, axis=1)
        for c in comm:
            self.assertAlmostEqual(c, 0.64, delta=0.02)

    def test_sign_is_fixed(self) -> None:
        lam = np.array([0.8, 0.8, 0.8])
        r = np.outer(lam, lam)
        np.fill_diagonal(r, 1.0)
        loadings, _, _, _, _ = principal_axis_factor(r, n_factors=1)
        self.assertTrue(np.all(loadings[:, 0] > 0))

    def test_rejects_bad_factor_count(self) -> None:
        r = np.eye(3)
        with self.assertRaises(ValueError):
            principal_axis_factor(r, n_factors=0)
        with self.assertRaises(ValueError):
            principal_axis_factor(r, n_factors=3)


class TestReliability(unittest.TestCase):
    def test_omega_closed_form(self) -> None:
        lam = np.array([0.8, 0.7, 0.6])
        theta = 1.0 - lam**2
        expected = lam.sum() ** 2 / (lam.sum() ** 2 + theta.sum())
        self.assertAlmostEqual(composite_reliability(lam, theta), expected, places=12)

    def test_alpha_equals_omega_under_tau_equivalence(self) -> None:
        """Equal loadings is exactly the case where alpha is not a lower bound."""
        lam = np.full(4, 0.75)
        theta = 1.0 - lam**2
        r = np.outer(lam, lam)
        np.fill_diagonal(r, 1.0)
        self.assertAlmostEqual(
            cronbach_alpha(r), composite_reliability(lam, theta), places=8
        )

    def test_alpha_below_omega_under_unequal_loadings(self) -> None:
        lam = np.array([0.9, 0.7, 0.4])
        theta = 1.0 - lam**2
        r = np.outer(lam, lam)
        np.fill_diagonal(r, 1.0)
        self.assertLess(cronbach_alpha(r), composite_reliability(lam, theta))

    def test_ave(self) -> None:
        lam = np.array([0.8, 0.6])
        self.assertAlmostEqual(average_variance_extracted(lam), 0.5, places=12)


class TestFactorScoreIndeterminacy(unittest.TestCase):
    def test_determinacy_formula(self) -> None:
        lam = np.full(3, 0.8)
        theta = 1.0 - lam**2
        s = float(np.sum(lam**2 / theta))
        self.assertAlmostEqual(
            factor_score_determinacy(lam, theta), np.sqrt(s / (1 + s)), places=12
        )

    def test_guttman_bound(self) -> None:
        self.assertAlmostEqual(guttman_indeterminacy_bound(0.9), 0.62, places=12)
        self.assertAlmostEqual(guttman_indeterminacy_bound(1.0), 1.0, places=12)

    def test_determinacy_increases_with_loading_strength(self) -> None:
        prev = -1.0
        for lam_value in (0.4, 0.6, 0.8, 0.95):
            lam = np.full(3, lam_value)
            rho = factor_score_determinacy(lam, 1.0 - lam**2)
            self.assertGreater(rho, prev)
            prev = rho

    def test_regression_and_bartlett_scores_differ_but_correlate(self) -> None:
        rng = np.random.default_rng(4)
        n = 5000
        latent = rng.standard_normal(n)
        lam = np.array([0.85, 0.65, 0.45])
        x = np.column_stack(
            [lam[i] * latent + np.sqrt(1 - lam[i] ** 2) * rng.standard_normal(n) for i in range(3)]
        )
        sol = fit_congeneric_block(x, ["a", "b", "c"], "L", [0, 1, 2])
        reg = factor_scores(x, sol.loadings, sol.uniquenesses, method="regression")
        bart = factor_scores(x, sol.loadings, sol.uniquenesses, method="bartlett")
        corr = float(np.corrcoef(reg, bart)[0, 1])
        self.assertGreater(corr, 0.99)
        # They are proportional for a single factor, so the standardized scores
        # coincide; the indeterminacy that matters is the Guttman bound, not
        # the choice between these two scoring rules.
        self.assertLess(sol.guttman_bound, 1.0)

    def test_unknown_scoring_method_rejected(self) -> None:
        rng = np.random.default_rng(5)
        x = rng.standard_normal((100, 3))
        with self.assertRaises(ValueError):
            factor_scores(x, np.full(3, 0.7), np.full(3, 0.51), method="nonsense")


class TestBlockFitting(unittest.TestCase):
    def test_single_indicator_is_rejected(self) -> None:
        rng = np.random.default_rng(6)
        x = rng.standard_normal((200, 3))
        with self.assertRaises(ValueError) as ctx:
            fit_congeneric_block(x, ["only"], "L", [0])
        self.assertIn("at least", str(ctx.exception))

    def test_two_indicators_warn(self) -> None:
        rng = np.random.default_rng(7)
        latent = rng.standard_normal(1000)
        x = np.column_stack(
            [0.8 * latent + 0.6 * rng.standard_normal(1000) for _ in range(2)]
        )
        sol = fit_congeneric_block(x, ["a", "b"], "L", [0, 1])
        self.assertTrue(any("three-indicator" in w for w in sol.warnings))

    def test_mismatched_index_length_rejected(self) -> None:
        rng = np.random.default_rng(8)
        x = rng.standard_normal((100, 4))
        with self.assertRaises(ValueError):
            fit_congeneric_block(x, ["a", "b", "c"], "L", [0, 1])


class TestDiscriminantValidity(unittest.TestCase):
    def test_fornell_larcker_flags_high_correlation(self) -> None:
        names = ["A", "B"]
        ave = [0.50, 0.50]
        phi = np.array([[1.0, 0.95], [0.95, 1.0]])
        result = fornell_larcker(ave, phi, names)
        self.assertEqual(len(result), 1)
        self.assertFalse(result[0][4])

    def test_fornell_larcker_passes_when_distinct(self) -> None:
        result = fornell_larcker([0.60, 0.60], np.array([[1.0, 0.3], [0.3, 1.0]]), ["A", "B"])
        self.assertTrue(result[0][4])

    def test_htmt_near_one_for_identical_constructs(self) -> None:
        rng = np.random.default_rng(9)
        n = 4000
        latent = rng.standard_normal(n)
        cols = [0.8 * latent + 0.6 * rng.standard_normal(n) for _ in range(6)]
        x = np.column_stack(cols)
        matrix = htmt(x, [[0, 1, 2], [3, 4, 5]], ["A", "B"])
        self.assertGreater(matrix[0, 1], 0.90)

    def test_htmt_low_for_unrelated_constructs(self) -> None:
        rng = np.random.default_rng(10)
        n = 4000
        l1, l2 = rng.standard_normal(n), rng.standard_normal(n)
        cols = [0.8 * l1 + 0.6 * rng.standard_normal(n) for _ in range(3)]
        cols += [0.8 * l2 + 0.6 * rng.standard_normal(n) for _ in range(3)]
        matrix = htmt(np.column_stack(cols), [[0, 1, 2], [3, 4, 5]], ["A", "B"])
        self.assertLess(matrix[0, 1], 0.15)


if __name__ == "__main__":
    unittest.main()
