"""Ground-truth recovery: does the estimator return the parameters that made the data?

These are the tests that matter most. Every other test checks that a formula
computes what it says it computes; these check that the whole procedure, run on
data whose generating parameters are known, returns those parameters.

Sample sizes here are large by design. Recovery is an asymptotic claim, and a
test that passed at n = 200 would be testing luck.
"""

from __future__ import annotations

import unittest

import numpy as np

from ehs_risk_sem.compat import silence_accelerate_matmul_warnings
from ehs_risk_sem.measurement import composite_reliability, fit_congeneric_block
from ehs_risk_sem.model import ModelSpec, fit
from simulations.dgp import TrueModel, generate_dataset, make_four_factor_model

silence_accelerate_matmul_warnings()

TRUE_BETA = np.array([0.45, 0.30, -0.25, -0.20])


def _spec(model: TrueModel) -> ModelSpec:
    return ModelSpec(
        latents={n: model.indicator_names[n] for n in model.latent_names},
        paths=model.paths(),
    )


class TestMeasurementRecovery(unittest.TestCase):
    def test_loadings_recovered(self) -> None:
        model = make_four_factor_model(loading=0.80)
        data, _ = generate_dataset(model, 20_000, seed=11)
        sol = fit_congeneric_block(
            data.to_numpy(), model.indicator_names["UnsafeActs"], "UnsafeActs", [0, 1, 2]
        )
        for lam in sol.loadings:
            self.assertAlmostEqual(lam, 0.80, delta=0.02)
        self.assertFalse(sol.heywood)
        self.assertTrue(sol.converged)

    def test_composite_reliability_matches_population_value(self) -> None:
        lam = np.full(3, 0.80)
        theta = 1.0 - lam**2
        population = composite_reliability(lam, theta)
        model = make_four_factor_model(loading=0.80)
        data, _ = generate_dataset(model, 20_000, seed=12)
        sol = fit_congeneric_block(
            data.to_numpy(), model.indicator_names["UnsafeActs"], "UnsafeActs", [0, 1, 2]
        )
        self.assertAlmostEqual(sol.omega, population, delta=0.02)

    def test_unequal_loadings_recovered(self) -> None:
        names = ["A", "B"]
        model = TrueModel(
            latent_names=names,
            loadings={"A": np.array([0.85, 0.70, 0.55]), "B": np.full(3, 0.75)},
            beta=np.array([[0.0, 0.0], [0.40, 0.0]]),
            exogenous_corr=np.array([[1.0]]),
            indicator_names={"A": ["a1", "a2", "a3"], "B": ["b1", "b2", "b3"]},
        )
        data, _ = generate_dataset(model, 30_000, seed=13)
        sol = fit_congeneric_block(data.to_numpy(), ["a1", "a2", "a3"], "A", [0, 1, 2])
        for est, truth in zip(sol.loadings, (0.85, 0.70, 0.55)):
            self.assertAlmostEqual(est, truth, delta=0.03)


class TestStructuralRecovery(unittest.TestCase):
    def test_four_predictor_paths_recovered(self) -> None:
        model = make_four_factor_model(beta=tuple(TRUE_BETA))
        data, _ = generate_dataset(model, 40_000, seed=21)
        est = fit(_spec(model), data).paths["Risk"]
        for value, truth in zip(est.beta, TRUE_BETA):
            self.assertAlmostEqual(value, truth, delta=0.02)

    def test_r_squared_recovered(self) -> None:
        model = make_four_factor_model(beta=tuple(TRUE_BETA))
        exo = model.exogenous_corr
        expected = float(TRUE_BETA @ exo @ TRUE_BETA)
        data, _ = generate_dataset(model, 40_000, seed=22)
        est = fit(_spec(model), data).paths["Risk"]
        self.assertAlmostEqual(est.r_squared, expected, delta=0.02)

    def test_estimator_is_unbiased_across_replications(self) -> None:
        """Averaged over replications, the estimate sits on the true value."""
        model = make_four_factor_model(beta=tuple(TRUE_BETA))
        spec = _spec(model)
        draws = []
        for r in range(40):
            data, _ = generate_dataset(model, 4000, seed=3000 + r)
            draws.append(fit(spec, data).paths["Risk"].beta)
        mean = np.mean(np.array(draws), axis=0)
        mcse = np.std(np.array(draws), axis=0, ddof=1) / np.sqrt(len(draws))
        for m, truth, se in zip(mean, TRUE_BETA, mcse):
            self.assertLess(abs(m - truth), max(4.0 * se, 0.01))

    def test_mediation_effects_decompose(self) -> None:
        """Total effect equals direct plus the traced indirect effect."""
        a, b, direct = 0.55, 0.45, 0.20
        model = TrueModel(
            latent_names=["X", "M", "Y"],
            loadings={k: np.full(3, 0.80) for k in ("X", "M", "Y")},
            beta=np.array([[0, 0, 0], [a, 0, 0], [direct, b, 0]], dtype=float),
            exogenous_corr=np.array([[1.0]]),
            indicator_names={
                "X": ["x1", "x2", "x3"],
                "M": ["m1", "m2", "m3"],
                "Y": ["y1", "y2", "y3"],
            },
        )
        data, _ = generate_dataset(model, 30_000, seed=31)
        res = fit(_spec(model), data)
        names = res.latent_names()
        iy, ix = names.index("Y"), names.index("X")
        total = res.effects["total"][iy, ix]
        self.assertAlmostEqual(total, direct + a * b, delta=0.03)
        self.assertAlmostEqual(
            res.effects["direct"][iy, ix] + res.effects["indirect"][iy, ix],
            total,
            places=10,
        )


class TestFitOnCorrectModel(unittest.TestCase):
    def test_correct_model_fits_well(self) -> None:
        model = make_four_factor_model()
        data, _ = generate_dataset(model, 5000, seed=41)
        res = fit(_spec(model), data)
        f = res.fit_indices
        self.assertGreater(f.cfi, 0.98)
        self.assertLess(f.rmsea, 0.03)
        self.assertLess(f.srmr, 0.05)
        self.assertEqual(f.df, 80)

    def test_equivalent_models_have_identical_fit(self) -> None:
        """The central negative result, asserted as a test."""
        model = TrueModel(
            latent_names=["Cause", "Effect"],
            loadings={"Cause": np.full(3, 0.80), "Effect": np.full(3, 0.80)},
            beta=np.array([[0.0, 0.0], [0.5, 0.0]]),
            exogenous_corr=np.array([[1.0]]),
            indicator_names={
                "Cause": ["c1", "c2", "c3"],
                "Effect": ["e1", "e2", "e3"],
            },
        )
        data, _ = generate_dataset(model, 3000, seed=51)
        latents = {"Cause": ["c1", "c2", "c3"], "Effect": ["e1", "e2", "e3"]}
        forward = fit(ModelSpec(latents=latents, paths={"Effect": ["Cause"]}), data)
        reverse = fit(ModelSpec(latents=latents, paths={"Cause": ["Effect"]}), data)
        self.assertAlmostEqual(
            forward.fit_indices.chi_square, reverse.fit_indices.chi_square, places=9
        )
        self.assertAlmostEqual(forward.fit_indices.cfi, reverse.fit_indices.cfi, places=9)
        self.assertAlmostEqual(
            forward.fit_indices.rmsea, reverse.fit_indices.rmsea, places=9
        )
        self.assertAlmostEqual(forward.fit_indices.srmr, reverse.fit_indices.srmr, places=9)
        self.assertAlmostEqual(
            float(forward.paths["Effect"].beta[0]),
            float(reverse.paths["Cause"].beta[0]),
            places=9,
        )


class TestAnalyticStandardErrorsUnderstate(unittest.TestCase):
    """The documented weakness of the two-step estimator, asserted as a test.

    If a future change made the analytic standard errors correct, this test
    would fail, and that failure would be the signal to update the README's
    claim rather than to relax the test.
    """

    def test_analytic_se_is_smaller_than_empirical_sd(self) -> None:
        model = make_four_factor_model()
        spec = _spec(model)
        betas, ses = [], []
        for r in range(60):
            data, _ = generate_dataset(model, 1000, seed=6000 + r)
            est = fit(spec, data).paths["Risk"]
            betas.append(est.beta)
            ses.append(est.se)
        empirical = np.std(np.array(betas), axis=0, ddof=1)
        analytic = np.mean(np.array(ses), axis=0)
        self.assertTrue(np.all(analytic < empirical))
        ratio = float(np.mean(analytic / empirical))
        self.assertLess(ratio, 0.95)
        self.assertGreater(ratio, 0.40)


if __name__ == "__main__":
    unittest.main()
