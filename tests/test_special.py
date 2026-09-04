"""Tests for the special functions.

Two kinds of check. Where an exact or textbook value exists, compare against
it. Where none does -- the noncentral chi-square in particular -- compare
against Monte Carlo, which is an independent computation rather than a
restatement of the same code.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from ehs_risk_sem.special import (
    chi2_cdf,
    chi2_ppf,
    chi2_sf,
    gammainc_lower_reg,
    ncx2_cdf,
    norm_cdf,
    norm_ppf,
)


class TestIncompleteGamma(unittest.TestCase):
    def test_matches_closed_form_for_integer_shape(self) -> None:
        # P(n, x) = 1 - exp(-x) * sum_{k<n} x^k / k!  for integer n.
        for a in (1, 2, 3, 5, 8):
            for x in (0.1, 0.5, 1.0, 3.0, 7.5, 20.0):
                closed = 1.0 - math.exp(-x) * sum(x**k / math.factorial(k) for k in range(a))
                self.assertAlmostEqual(gammainc_lower_reg(a, x), closed, places=11)

    def test_half_integer_shape_matches_erf(self) -> None:
        # P(1/2, x) = erf(sqrt(x))
        for x in (0.01, 0.25, 1.0, 4.0, 12.0):
            self.assertAlmostEqual(
                gammainc_lower_reg(0.5, x), math.erf(math.sqrt(x)), places=11
            )

    def test_boundaries(self) -> None:
        self.assertEqual(gammainc_lower_reg(2.0, 0.0), 0.0)
        self.assertAlmostEqual(gammainc_lower_reg(2.0, 200.0), 1.0, places=12)
        with self.assertRaises(ValueError):
            gammainc_lower_reg(0.0, 1.0)
        with self.assertRaises(ValueError):
            gammainc_lower_reg(1.0, -1.0)


class TestChiSquare(unittest.TestCase):
    def test_known_critical_values(self) -> None:
        # Standard table values for the upper 5% point.
        known = {1: 3.841459, 2: 5.991465, 5: 11.070498, 10: 18.307038, 20: 31.410433}
        for df, crit in known.items():
            self.assertAlmostEqual(chi2_ppf(0.95, df), crit, places=4)
            self.assertAlmostEqual(chi2_cdf(crit, df), 0.95, places=6)

    def test_cdf_and_sf_are_complementary(self) -> None:
        for df in (1, 3, 17, 80):
            for x in (0.5, 3.0, 20.0, 100.0):
                self.assertAlmostEqual(chi2_cdf(x, df) + chi2_sf(x, df), 1.0, places=12)

    def test_ppf_inverts_cdf(self) -> None:
        for df in (2, 9, 40):
            for p in (0.01, 0.25, 0.5, 0.9, 0.99):
                self.assertAlmostEqual(chi2_cdf(chi2_ppf(p, df), df), p, places=9)


class TestNoncentralChiSquare(unittest.TestCase):
    def test_reduces_to_central_at_zero_noncentrality(self) -> None:
        for df in (1, 4, 25):
            for x in (0.5, 5.0, 40.0):
                self.assertAlmostEqual(ncx2_cdf(x, df, 0.0), chi2_cdf(x, df), places=12)

    def test_agrees_with_monte_carlo(self) -> None:
        """Independent check: simulate the distribution and compare.

        400,000 draws gives a Monte Carlo standard error under 0.0008 on a
        probability, so a tolerance of 0.004 is roughly five standard errors.
        """
        rng = np.random.default_rng(20260903)
        cases = [(5, 3.0, 10.0), (20, 20.0, 40.0), (100, 33.5, 124.0), (100, 85.76, 150.0)]
        for df, nc, x in cases:
            z = rng.standard_normal((400_000, df))
            z[:, 0] += math.sqrt(nc)
            draws = (z**2).sum(axis=1)
            mc = float(np.mean(draws <= x))
            self.assertAlmostEqual(ncx2_cdf(x, df, nc), mc, delta=0.004)

    def test_monotone_in_noncentrality(self) -> None:
        prev = 1.1
        for nc in (0.0, 1.0, 5.0, 20.0, 60.0):
            value = ncx2_cdf(30.0, 10, nc)
            self.assertLess(value, prev)
            prev = value


class TestNormal(unittest.TestCase):
    def test_known_quantiles(self) -> None:
        self.assertAlmostEqual(norm_ppf(0.975), 1.959963985, places=7)
        self.assertAlmostEqual(norm_ppf(0.95), 1.644853627, places=7)
        self.assertAlmostEqual(norm_ppf(0.5), 0.0, places=10)

    def test_cdf_symmetry(self) -> None:
        for x in (0.1, 1.0, 2.5, 4.0):
            self.assertAlmostEqual(norm_cdf(x) + norm_cdf(-x), 1.0, places=12)

    def test_round_trip(self) -> None:
        for p in (0.001, 0.1, 0.5, 0.8, 0.999):
            self.assertAlmostEqual(norm_cdf(norm_ppf(p)), p, places=10)


if __name__ == "__main__":
    unittest.main()
