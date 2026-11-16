"""Independent pinned SciPy 1.17 reference values; no SciPy runtime dependency."""
import math
import unittest

from simulations.inference_distributions import (
    normal_cdf, normal_ppf, normal_sf, student_t_cdf, student_t_critical_975,
)


class TestInferenceDistributions(unittest.TestCase):
    def test_t_critical_values_match_independent_references(self):
        references = {1: 12.706204736174694, 2: 4.302652729749462,
                      5: 2.5705818356363146, 30: 2.0422724563012378,
                      380: 1.9662263966111804, 765: 1.963069821326746,
                      1535: 1.961510636707333, 3437: 1.960654439197212,
                      6879: 1.9603089011835837, 13763: 1.9601363652944335}
        for df, reference in references.items():
            self.assertAlmostEqual(student_t_critical_975(df), reference, delta=2e-9)

    def test_cauchy_cdf_closed_form_and_symmetry(self):
        for value in (-20, -3, -.1, 0, .1, 3, 20):
            self.assertAlmostEqual(student_t_cdf(value, 1), .5 + math.atan(value) / math.pi, places=12)
            self.assertAlmostEqual(student_t_cdf(value, 7) + student_t_cdf(-value, 7), 1)
        with self.assertRaises(ValueError):
            student_t_critical_975(0)

    def test_normal_quantile_and_tail(self):
        self.assertAlmostEqual(normal_ppf(.975), 1.959963984540054, places=14)
        self.assertAlmostEqual(normal_cdf(1.959963984540054), .975, places=14)
        self.assertAlmostEqual(normal_sf(1.959963984540054), .025, places=14)
        self.assertGreater(normal_sf(10), 0)


if __name__ == "__main__":
    unittest.main()
