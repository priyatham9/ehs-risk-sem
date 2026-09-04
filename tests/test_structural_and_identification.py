"""Tests for the structural solver, identification rules and the model spec."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from ehs_risk_sem.compat import silence_accelerate_matmul_warnings
from ehs_risk_sem.identification import (
    check_identification,
    count_free_parameters,
    is_recursive,
    t_rule,
)
from ehs_risk_sem.model import ModelSpec, fit, implied_covariance
from ehs_risk_sem.structural import (
    disattenuate,
    effect_decomposition,
    indirect_effect_delta_se,
    latent_correlations,
    solve_paths,
)
from simulations.dgp import generate_dataset, make_four_factor_model

silence_accelerate_matmul_warnings()


class TestDisattenuation(unittest.TestCase):
    def test_exact_correction(self) -> None:
        self.assertAlmostEqual(disattenuate(0.40, 0.64, 1.0), 0.5, places=12)
        self.assertAlmostEqual(disattenuate(0.36, 0.81, 0.64), 0.5, places=12)

    def test_perfect_reliability_is_the_identity(self) -> None:
        self.assertAlmostEqual(disattenuate(0.42, 1.0, 1.0), 0.42, places=12)

    def test_out_of_range_result_is_reported_not_clipped(self) -> None:
        value = disattenuate(0.60, 0.30, 0.30)
        self.assertGreater(value, 1.0)
        phi, warnings = latent_correlations(
            np.array([[1.0, 0.60], [0.60, 1.0]]), [0.30, 0.30], ["A", "B"]
        )
        self.assertGreater(abs(phi[0, 1]), 1.0)
        self.assertTrue(any("outside [-1, 1]" in w for w in warnings))


class TestSolvePaths(unittest.TestCase):
    def test_recovers_coefficients_from_a_population_matrix(self) -> None:
        """Given the exact latent correlation matrix, the solution is exact."""
        beta = np.array([0.45, 0.30, -0.25, -0.20])
        exo = np.full((4, 4), 0.35)
        np.fill_diagonal(exo, 1.0)
        phi = np.eye(5)
        phi[:4, :4] = exo
        phi[:4, 4] = phi[4, :4] = exo @ beta
        names = ["A", "B", "C", "D", "Y"]
        est = solve_paths(phi, names, "Y", ["A", "B", "C", "D"], n_obs=1000)
        for value, truth in zip(est.beta, beta):
            self.assertAlmostEqual(value, truth, places=10)
        self.assertAlmostEqual(est.r_squared, float(beta @ exo @ beta), places=10)
        self.assertAlmostEqual(est.disturbance_variance, 1 - est.r_squared, places=12)

    def test_orthogonal_predictors_give_vif_one(self) -> None:
        phi = np.eye(3)
        phi[0, 2] = phi[2, 0] = 0.5
        phi[1, 2] = phi[2, 1] = 0.3
        est = solve_paths(phi, ["A", "B", "Y"], "Y", ["A", "B"], n_obs=500)
        self.assertTrue(np.allclose(est.vif, 1.0))
        self.assertAlmostEqual(est.beta[0], 0.5, places=12)
        self.assertAlmostEqual(est.beta[1], 0.3, places=12)

    def test_collinear_predictors_raise_vif_warning(self) -> None:
        phi = np.eye(3)
        phi[0, 1] = phi[1, 0] = 0.95
        phi[0, 2] = phi[2, 0] = 0.5
        phi[1, 2] = phi[2, 1] = 0.5
        est = solve_paths(phi, ["A", "B", "Y"], "Y", ["A", "B"], n_obs=500)
        self.assertTrue(any("variance inflation" in w for w in est.warnings))

    def test_standard_errors_shrink_with_sample_size(self) -> None:
        phi = np.eye(2)
        phi[0, 1] = phi[1, 0] = 0.4
        small = solve_paths(phi, ["A", "Y"], "Y", ["A"], n_obs=100)
        large = solve_paths(phi, ["A", "Y"], "Y", ["A"], n_obs=10000)
        self.assertGreater(small.se[0], large.se[0])
        # Residual df is n - k where k = 2 (intercept + one predictor), so the
        # ratio is sqrt((10000 - 2) / (100 - 2)), not sqrt((n-1) / (n-1)).
        self.assertAlmostEqual(small.se[0] / large.se[0], np.sqrt(9998 / 98), delta=0.02)

    def test_no_predictors_rejected(self) -> None:
        with self.assertRaises(ValueError):
            solve_paths(np.eye(2), ["A", "Y"], "Y", [], n_obs=100)


class TestEffectDecomposition(unittest.TestCase):
    def test_chain_total_effect(self) -> None:
        b = np.zeros((3, 3))
        b[1, 0] = 0.5  # X -> M
        b[2, 1] = 0.4  # M -> Y
        b[2, 0] = 0.2  # X -> Y
        effects = effect_decomposition(b, ["X", "M", "Y"])
        self.assertAlmostEqual(effects["total"][2, 0], 0.2 + 0.5 * 0.4, places=12)
        self.assertAlmostEqual(effects["indirect"][2, 0], 0.5 * 0.4, places=12)
        self.assertAlmostEqual(effects["direct"][2, 0], 0.2, places=12)

    def test_no_indirect_effect_without_a_mediator(self) -> None:
        b = np.zeros((2, 2))
        b[1, 0] = 0.6
        effects = effect_decomposition(b, ["X", "Y"])
        self.assertAlmostEqual(effects["indirect"][1, 0], 0.0, places=12)

    def test_delta_method_se(self) -> None:
        se = indirect_effect_delta_se(0.5, 0.4, 0.05, 0.06)
        expected = np.sqrt(0.4**2 * 0.05**2 + 0.5**2 * 0.06**2)
        self.assertAlmostEqual(se, expected, places=12)


class TestIdentificationRules(unittest.TestCase):
    def test_t_rule(self) -> None:
        self.assertTrue(t_rule(15, 40))
        self.assertTrue(t_rule(3, 6))
        self.assertFalse(t_rule(3, 7))

    def test_free_parameter_count(self) -> None:
        # 5 latents x 3 indicators, 4 exogenous, 4 paths:
        # 15 loadings + 15 uniquenesses + 6 exogenous covariances + 4 paths = 40
        self.assertEqual(count_free_parameters([3, 3, 3, 3, 3], 4, 4), 40)

    def test_recursive_detection(self) -> None:
        self.assertTrue(is_recursive({"Y": ["X"]}, ["X", "Y"]))
        self.assertTrue(is_recursive({"M": ["X"], "Y": ["X", "M"]}, ["X", "M", "Y"]))
        self.assertFalse(is_recursive({"Y": ["X"], "X": ["Y"]}, ["X", "Y"]))

    def test_check_identification_reports_df(self) -> None:
        blocks = {f"L{i}": [f"v{i}{j}" for j in range(3)] for i in range(5)}
        paths = {"L4": ["L0", "L1", "L2", "L3"]}
        report = check_identification(blocks, paths, np.eye(15))
        self.assertEqual(report.n_observed, 15)
        self.assertEqual(report.n_moments, 120)
        self.assertEqual(report.n_free_parameters, 40)
        self.assertEqual(report.df, 80)
        self.assertTrue(report.necessary_condition_met)

    def test_singular_covariance_is_an_error(self) -> None:
        blocks = {"A": ["a1", "a2", "a3"], "B": ["b1", "b2", "b3"]}
        cov = np.eye(6)
        cov[0, 1] = cov[1, 0] = 1.0
        report = check_identification(blocks, {"B": ["A"]}, cov)
        self.assertFalse(report.necessary_condition_met)
        self.assertTrue(any("singular" in e for e in report.errors))

    def test_feedback_loop_is_an_error(self) -> None:
        blocks = {"A": ["a1", "a2", "a3"], "B": ["b1", "b2", "b3"]}
        report = check_identification(blocks, {"B": ["A"], "A": ["B"]}, np.eye(6))
        self.assertFalse(report.recursive)
        self.assertFalse(report.necessary_condition_met)

    def test_single_indicator_latent_is_an_error(self) -> None:
        blocks = {"A": ["a1"], "B": ["b1", "b2", "b3"]}
        report = check_identification(blocks, {"B": ["A"]}, np.eye(4))
        self.assertFalse(report.necessary_condition_met)


class TestModelSpec(unittest.TestCase):
    def test_cross_loading_rejected(self) -> None:
        spec = ModelSpec(
            latents={"A": ["x1", "x2", "x3"], "B": ["x3", "x4", "x5"]},
            paths={"B": ["A"]},
        )
        with self.assertRaises(ValueError) as ctx:
            spec.validate()
        self.assertIn("both", str(ctx.exception))

    def test_unknown_path_target_rejected(self) -> None:
        spec = ModelSpec(latents={"A": ["x1", "x2", "x3"]}, paths={"Z": ["A"]})
        with self.assertRaises(ValueError):
            spec.validate()

    def test_topological_order(self) -> None:
        spec = ModelSpec(
            latents={
                "X": ["x1", "x2"],
                "M": ["m1", "m2"],
                "Y": ["y1", "y2"],
            },
            paths={"M": ["X"], "Y": ["X", "M"]},
        )
        order = spec.topological_order()
        self.assertLess(order.index("X"), order.index("M"))
        self.assertLess(order.index("M"), order.index("Y"))

    def test_missing_columns_rejected(self) -> None:
        spec = ModelSpec(
            latents={"A": ["a1", "a2", "a3"], "B": ["b1", "b2", "b3"]},
            paths={"B": ["A"]},
        )
        frame = pd.DataFrame(np.random.default_rng(1).standard_normal((100, 3)),
                             columns=["a1", "a2", "a3"])
        with self.assertRaises(ValueError) as ctx:
            fit(spec, frame)
        self.assertIn("missing indicator columns", str(ctx.exception))

    def test_listwise_deletion_is_reported(self) -> None:
        model = make_four_factor_model()
        data, _ = generate_dataset(model, 800, seed=99)
        data = data.copy()
        data.iloc[:40, 0] = np.nan
        spec = ModelSpec(
            latents={n: model.indicator_names[n] for n in model.latent_names},
            paths=model.paths(),
        )
        res = fit(spec, data)
        self.assertEqual(res.n_obs, 760)
        self.assertTrue(any("listwise deletion" in w for w in res.warnings))


class TestImpliedCovariance(unittest.TestCase):
    def test_implied_matrix_reproduces_a_population_model(self) -> None:
        spec = ModelSpec(
            latents={"A": ["a1", "a2", "a3"], "B": ["b1", "b2", "b3"]},
            paths={"B": ["A"]},
        )
        lam = {"A": np.full(3, 0.8), "B": np.full(3, 0.8)}
        theta = {k: 1.0 - v**2 for k, v in lam.items()}
        beta = np.array([[0.0, 0.0], [0.6, 0.0]])
        sigma, phi = implied_covariance(
            spec, lam, theta, beta, np.array([[1.0]]), np.array([1.0, 1 - 0.36])
        )
        self.assertTrue(np.allclose(np.diag(phi), 1.0, atol=1e-12))
        self.assertAlmostEqual(phi[0, 1], 0.6, places=12)
        self.assertTrue(np.allclose(np.diag(sigma), 1.0, atol=1e-12))
        # Cross-block indicator correlation is lambda * phi * lambda.
        self.assertAlmostEqual(sigma[0, 3], 0.8 * 0.6 * 0.8, places=12)


if __name__ == "__main__":
    unittest.main()
