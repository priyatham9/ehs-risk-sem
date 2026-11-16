import unittest
import numpy as np
from simulations.study_08_bootstrap_null_validation import (
    binomial_check, exact_binomial_interval, paired_summary, prefix_interval,
)


class TestBootstrapNullValidation(unittest.TestCase):
    def test_exact_interval_against_independent_beta_reference(self):
        # Pinned independent beta.ppf references, not a SciPy dependency.
        low, high = exact_binomial_interval(23, 300)
        self.assertAlmostEqual(low, .04921878387567447, delta=2e-12)
        self.assertAlmostEqual(high, .11281811536048231, delta=2e-12)
        self.assertLess(low, .05)
        self.assertGreater(high, .05)
        self.assertEqual(exact_binomial_interval(0, 10)[0], 0)
        self.assertEqual(exact_binomial_interval(10, 10)[1], 1)

    def test_exact_binomial_probability_ordering(self):
        result = binomial_check(0, 10, .5)
        self.assertAlmostEqual(result['two_sided_probability_ordering_p'], 2 / 1024)

    def test_prefix_uses_attempt_positions_including_failed_refits(self):
        draws = [None] * 30 + list(np.linspace(-.2, .2, 969))
        first = prefix_interval(0, draws, 0, 399)
        full = prefix_interval(0, draws, 0, 999)
        self.assertEqual(first['used'], 369)
        self.assertFalse(first['valid'])
        self.assertEqual(full['used'], 969)
        self.assertTrue(full['valid'])

    def test_paired_difference_and_discordance(self):
        a = [False, False, True, True] * 100
        b = [False, True, False, True] * 100
        row = paired_summary(a, b)
        self.assertEqual(row['b999_minus_b399'], 0)
        self.assertEqual(row['b999_only_reject'], 100)
        self.assertEqual(row['b399_only_reject'], 100)
        self.assertEqual(row['both_reject'], 100)
        self.assertLess(row['paired_bootstrap_low'], 0)
        self.assertGreater(row['paired_bootstrap_high'], 0)


if __name__ == '__main__':
    unittest.main()
