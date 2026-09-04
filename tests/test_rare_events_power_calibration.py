"""Tests for the three modules the README's headline numbers depend on.

The arithmetic in :mod:`ehs_risk_sem.rare_events`, the sample-size results in
:mod:`ehs_risk_sem.power` and the scoring rules in
:mod:`ehs_risk_sem.calibration` are quoted directly in ``README.md``. Anything
quoted there should be pinned by a test, so that a later change to the code
cannot silently invalidate the document.

Where a closed form exists the test compares against it computed independently
inside the test rather than by calling the function under test. Where none
exists, the test compares against Monte Carlo.
"""

from __future__ import annotations

import unittest

import numpy as np

from ehs_risk_sem.calibration import (
    auc,
    brier_decomposition,
    brier_score,
    calibration_intercept_slope,
    expected_calibration_error,
    net_benefit,
    reliability_diagram,
)
from ehs_risk_sem.power import (
    min_n_for_rmsea_power,
    min_n_to_distinguish,
    path_se,
    rmsea_power,
    se_of_difference,
)
from ehs_risk_sem.rare_events import (
    alerts_per_true_event,
    events_for_relative_precision,
    ppv_from_sens_spec,
    probability_per_shift,
    rate_per_hour,
    shifts_per_event,
    worker_shifts_for_calibration,
)


class TestRareEventArithmetic(unittest.TestCase):
    """Every value here is checkable by hand, which is the point of the module."""

    def test_rate_conversion(self) -> None:
        # 2.3 cases per 200,000 hours.
        self.assertAlmostEqual(rate_per_hour(2.3), 2.3 / 200_000.0, places=15)
        self.assertAlmostEqual(rate_per_hour(2.3), 1.15e-05, places=15)

    def test_probability_per_shift(self) -> None:
        self.assertAlmostEqual(probability_per_shift(2.3, 8.0), 9.2e-05, places=12)
        self.assertAlmostEqual(probability_per_shift(2.3, 12.0), 1.38e-04, places=12)

    def test_shifts_per_event_is_the_reciprocal(self) -> None:
        self.assertAlmostEqual(shifts_per_event(2.3, 8.0), 1.0 / 9.2e-05, places=6)
        self.assertAlmostEqual(shifts_per_event(2.3, 8.0), 10869.565217, places=4)

    def test_ppv_matches_bayes_computed_independently(self) -> None:
        sens, spec, prev = 0.80, 0.95, 9.2e-05
        expected = (sens * prev) / (sens * prev + (1.0 - spec) * (1.0 - prev))
        self.assertAlmostEqual(ppv_from_sens_spec(sens, spec, prev), expected, places=15)
        # The README quotes this figure.
        self.assertAlmostEqual(ppv_from_sens_spec(sens, spec, prev), 0.00146997, places=8)

    def test_alerts_per_true_event_is_one_over_ppv(self) -> None:
        prev = probability_per_shift(2.3, 8.0)
        for sens, spec, expected in [
            (0.80, 0.95, 680.285),
            (0.80, 0.99, 136.857),
            (0.90, 0.999, 13.0762),
        ]:
            got = alerts_per_true_event(sens, spec, prev)
            self.assertAlmostEqual(got, 1.0 / ppv_from_sens_spec(sens, spec, prev), places=6)
            self.assertAlmostEqual(got, expected, places=2)

    def test_ppv_degenerates_to_prevalence_for_a_useless_test(self) -> None:
        """Sensitivity equal to the false-positive rate carries no information."""
        prev = 0.01
        self.assertAlmostEqual(ppv_from_sens_spec(0.30, 0.70, prev), prev, places=12)

    def test_events_required_is_independent_of_the_rate(self) -> None:
        """(z/rel)^2 events, whatever the base rate. Only the exposure changes."""
        expected = (1.959963985 / 0.10) ** 2
        self.assertAlmostEqual(events_for_relative_precision(0.10), expected, places=6)
        self.assertAlmostEqual(events_for_relative_precision(0.10), 384.1459, places=3)

    def test_worker_shifts_for_calibration(self) -> None:
        out = worker_shifts_for_calibration(2.3, 8.0, 0.10)
        self.assertAlmostEqual(out["base_rate_per_shift"], 9.2e-05, places=12)
        # ~4.18 million worker-shifts, as quoted in the README.
        self.assertAlmostEqual(out["worker_shifts_required"] / 1e6, 4.1755, places=3)
        self.assertAlmostEqual(out["expected_events"], 384.11, places=1)

    def test_longer_shifts_need_fewer_shifts_for_the_same_precision(self) -> None:
        eight = worker_shifts_for_calibration(2.3, 8.0, 0.10)["worker_shifts_required"]
        twelve = worker_shifts_for_calibration(2.3, 12.0, 0.10)["worker_shifts_required"]
        self.assertLess(twelve, eight)
        # Nearly the same worker-hours either way. Not exactly: the required
        # count carries a (1 - p) factor and p differs between the two shift
        # lengths, which moves the ratio by about 5e-5.
        self.assertAlmostEqual(eight * 8.0 / (twelve * 12.0), 1.0, places=4)


class TestPower(unittest.TestCase):
    def test_power_increases_with_n(self) -> None:
        previous = -1.0
        for n in (50, 100, 200, 400, 800):
            value = rmsea_power(df=80, n=n)
            self.assertGreaterEqual(value, previous)
            previous = value

    def test_power_is_between_zero_and_one(self) -> None:
        for df in (10, 80, 200):
            for n in (30, 500, 20_000):
                value = rmsea_power(df=df, n=n)
                self.assertGreaterEqual(value, 0.0)
                self.assertLessEqual(value, 1.0)

    def test_min_n_inverts_rmsea_power(self) -> None:
        """The returned N attains the target and N-1 does not."""
        for df in (20, 50, 80, 100):
            n = min_n_for_rmsea_power(df, target_power=0.80)
            self.assertIsNotNone(n)
            assert n is not None
            self.assertGreaterEqual(rmsea_power(df, n), 0.80)
            self.assertLess(rmsea_power(df, n - 1), 0.80)

    def test_more_degrees_of_freedom_need_fewer_observations(self) -> None:
        values = [min_n_for_rmsea_power(df) for df in (20, 50, 80, 100)]
        for a, b in zip(values, values[1:]):
            assert a is not None and b is not None
            self.assertGreater(a, b)

    def test_path_se_scales_as_one_over_root_n(self) -> None:
        a = path_se(r_squared=0.5, vif=1.9, n=1000, n_predictors=4)
        b = path_se(r_squared=0.5, vif=1.9, n=4000, n_predictors=4)
        # Denominator is (n - k - 1), not n, so the ratio is sqrt(3995/995).
        self.assertAlmostEqual(a / b, np.sqrt(3995.0 / 995.0), places=12)
        self.assertAlmostEqual(a / b, 2.0, delta=0.01)

    def test_path_se_grows_with_collinearity(self) -> None:
        low = path_se(r_squared=0.5, vif=1.0, n=500, n_predictors=4)
        high = path_se(r_squared=0.5, vif=4.0, n=500, n_predictors=4)
        self.assertAlmostEqual(high / low, 2.0, places=6)

    def test_distinguishing_close_coefficients_needs_more_data(self) -> None:
        """The README's central sample-size claim, as an ordering."""
        corr = np.full((4, 4), 0.35)
        np.fill_diagonal(corr, 1.0)
        far = min_n_to_distinguish(0.45, 0.30, corr, r_squared=0.288, i=0, j=1)
        near = min_n_to_distinguish(-0.25, -0.20, corr, r_squared=0.288, i=2, j=3)
        self.assertIsNotNone(far)
        self.assertIsNotNone(near)
        assert far is not None and near is not None
        self.assertGreater(near, far)
        self.assertGreater(near, 5000)

    def test_se_of_difference_shrinks_with_n(self) -> None:
        corr = np.full((4, 4), 0.35)
        np.fill_diagonal(corr, 1.0)
        a = se_of_difference(corr, 0.288, 250, 0, 1)
        b = se_of_difference(corr, 0.288, 1000, 0, 1)
        self.assertGreater(a, b)
        self.assertAlmostEqual(a / b, 2.0, delta=0.05)


class TestCalibration(unittest.TestCase):
    def test_brier_score_is_mean_squared_error(self) -> None:
        y = np.array([0.0, 1.0, 1.0, 0.0])
        p = np.array([0.1, 0.9, 0.6, 0.3])
        self.assertAlmostEqual(brier_score(y, p), float(np.mean((p - y) ** 2)), places=15)

    def test_brier_decomposition_reconstructs_the_score(self) -> None:
        """Murphy: Brier = reliability - resolution + uncertainty + binning gap.

        The three-term identity is exact only for a discrete forecast. With
        continuous predictions grouped into bins there is a residual from
        within-bin spread, which this implementation reports as
        ``identity_gap`` rather than absorbing silently.
        """
        rng = np.random.default_rng(11)
        p = rng.uniform(0.01, 0.99, size=5000)
        y = (rng.uniform(size=5000) < p).astype(float)
        out = brier_decomposition(y, p, n_bins=10)
        rebuilt = (
            out["reliability"]
 - out["resolution"]
            + out["uncertainty"]
            + out["identity_gap"]
        )
        self.assertAlmostEqual(rebuilt, out["brier"], places=10)
        # The gap is a binning artefact and should be small.
        self.assertLess(abs(out["identity_gap"]), 0.01)

    def test_uncertainty_term_depends_only_on_the_outcome(self) -> None:
        """It is p_bar*(1-p_bar), so two different models share it."""
        rng = np.random.default_rng(3)
        y = (rng.uniform(size=4000) < 0.2).astype(float)
        base = float(np.mean(y))
        a = brier_decomposition(y, rng.uniform(size=4000))
        b = brier_decomposition(y, np.full(4000, 0.2))
        self.assertAlmostEqual(a["uncertainty"], base * (1 - base), places=12)
        self.assertAlmostEqual(b["uncertainty"], base * (1 - base), places=12)

    def test_perfectly_calibrated_predictions_give_intercept_zero_slope_one(self) -> None:
        rng = np.random.default_rng(5)
        p = rng.uniform(0.05, 0.95, size=40000)
        y = (rng.uniform(size=40000) < p).astype(float)
        out = calibration_intercept_slope(y, p)
        self.assertAlmostEqual(out["calibration_intercept"], 0.0, delta=0.06)
        self.assertAlmostEqual(out["calibration_slope"], 1.0, delta=0.06)

    def test_inflated_predictions_are_detected(self) -> None:
        """A model trained on a rebalanced sample overpredicts.

        The intercept is a level shift on the logit scale with ``logit(p)`` held
        as an offset, so systematic OVERprediction drives it negative: the
        observed rate sits below what was predicted.
        """
        rng = np.random.default_rng(6)
        p_true = rng.uniform(0.001, 0.02, size=20000)
        y = (rng.uniform(size=20000) < p_true).astype(float)
        inflated = (50.0 * p_true) / (50.0 * p_true + (1.0 - p_true))

        honest = calibration_intercept_slope(y, p_true)["calibration_intercept"]
        overpredicted = calibration_intercept_slope(y, inflated)["calibration_intercept"]

        self.assertLess(abs(honest), 0.35)
        self.assertLess(overpredicted, -2.0)
        # Roughly the log-odds multiplier that was applied.
        self.assertAlmostEqual(overpredicted, -np.log(50.0), delta=0.6)

    def test_auc_is_rank_based_and_invariant_to_monotone_transforms(self) -> None:
        rng = np.random.default_rng(8)
        y = (rng.uniform(size=3000) < 0.3).astype(float)
        p = np.clip(0.3 + 0.4 * y + rng.normal(0, 0.3, size=3000), 1e-6, 1 - 1e-6)
        base = auc(y, p)
        # A strictly increasing map must not change the ordering, hence not the AUC.
        self.assertAlmostEqual(auc(y, p**3), base, places=12)
        self.assertGreater(base, 0.5)

    def test_auc_of_a_perfect_and_a_useless_ranking(self) -> None:
        y = np.array([0.0, 0.0, 1.0, 1.0])
        self.assertAlmostEqual(auc(y, np.array([0.1, 0.2, 0.8, 0.9])), 1.0, places=12)
        self.assertAlmostEqual(auc(y, np.array([0.9, 0.8, 0.2, 0.1])), 0.0, places=12)
        self.assertAlmostEqual(auc(y, np.array([0.5, 0.5, 0.5, 0.5])), 0.5, places=12)

    def test_auc_is_insensitive_to_calibration(self) -> None:
        """The README's claim that ranking survives what calibration does not."""
        rng = np.random.default_rng(9)
        y = (rng.uniform(size=5000) < 0.01).astype(float)
        p = np.clip(rng.uniform(0.001, 0.05, size=5000) + 0.02 * y, 1e-9, 1 - 1e-9)
        # Multiply the odds by 80. Strictly monotone, never saturates, and it
        # is what rebalancing the training data does to the predictions.
        inflated = (80.0 * p) / (80.0 * p + (1.0 - p))
        self.assertGreater(float(np.mean(inflated)), 20.0 * float(np.mean(p)))
        self.assertAlmostEqual(auc(y, inflated), auc(y, p), places=12)

    def test_reliability_diagram_bins_cover_every_observation(self) -> None:
        rng = np.random.default_rng(12)
        p = rng.uniform(size=2000)
        y = (rng.uniform(size=2000) < p).astype(float)
        table = reliability_diagram(y, p, n_bins=10)
        self.assertEqual(int(table["n"].sum()), 2000)

    def test_ece_is_near_zero_when_calibrated_and_large_when_not(self) -> None:
        rng = np.random.default_rng(13)
        p = rng.uniform(0.05, 0.95, size=30000)
        y = (rng.uniform(size=30000) < p).astype(float)
        good = expected_calibration_error(y, p)["ece"]
        bad = expected_calibration_error(y, np.clip(p * 0.2, 1e-9, 1 - 1e-9))["ece"]
        self.assertLess(good, 0.02)
        self.assertGreater(bad, 0.20)

    def test_net_benefit_of_treating_nobody_is_zero(self) -> None:
        rng = np.random.default_rng(14)
        y = (rng.uniform(size=1000) < 0.1).astype(float)
        self.assertAlmostEqual(net_benefit(y, np.zeros(1000), threshold=0.2), 0.0, places=12)

    def test_net_benefit_of_treating_everyone_matches_closed_form(self) -> None:
        rng = np.random.default_rng(15)
        y = (rng.uniform(size=2000) < 0.1).astype(float)
        t = 0.2
        prevalence = float(np.mean(y))
        expected = prevalence - (1.0 - prevalence) * (t / (1.0 - t))
        self.assertAlmostEqual(
            net_benefit(y, np.ones(2000), threshold=t), expected, places=10
        )


if __name__ == "__main__":
    unittest.main()
