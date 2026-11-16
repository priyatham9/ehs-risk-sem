import unittest
import numpy as np
from simulations.study_05_inference_audit import Cell, model_for
from simulations.study_07_information_bound import (
    LATENT_PAIRS, bound_for, contrast_from_phi, contrast_gradient,
    expected_information, measurement_covariance, measurement_derivatives,
    standardized_latent_bound,
)


class TestInformationBound(unittest.TestCase):
    def setUp(self):
        self.phi = model_for(Cell("positive", .75, 1), "alternative").implied_latent_correlation()
        self.contrast = np.array([1, -1, 0, 0])

    def test_all_covariance_derivatives_match_finite_difference(self):
        loadings, uniquenesses = np.full(15, .75), np.full(15, 1 - .75**2)
        analytic = measurement_derivatives(loadings, self.phi)
        h = 1e-6
        for j in range(40):
            plus, minus = [loadings.copy(), uniquenesses.copy(), self.phi.copy()], [loadings.copy(), uniquenesses.copy(), self.phi.copy()]
            if j < 15:
                plus[0][j] += h
                minus[0][j] -= h
            elif j < 30:
                plus[1][j - 15] += h
                minus[1][j - 15] -= h
            else:
                a, b = LATENT_PAIRS[j - 30]
                plus[2][a, b] += h
                plus[2][b, a] += h
                minus[2][a, b] -= h
                minus[2][b, a] -= h
            numeric = (measurement_covariance(*plus) - measurement_covariance(*minus)) / (2 * h)
            np.testing.assert_allclose(numeric, analytic[j], atol=1e-9)

    def test_contrast_gradient_matches_finite_difference(self):
        gradient = contrast_gradient(self.phi, self.contrast)
        for index, (i, j) in enumerate(LATENT_PAIRS):
            delta = np.zeros((5, 5))
            delta[i, j] = delta[j, i] = 1e-6
            numeric = (contrast_from_phi(self.phi + delta, self.contrast) -
                       contrast_from_phi(self.phi - delta, self.contrast)) / 2e-6
            self.assertAlmostEqual(gradient[index], numeric, places=8)

    def test_information_matches_direct_trace(self):
        covariance = measurement_covariance(np.full(15, .75), np.full(15, .4375), self.phi)
        derivative = measurement_derivatives(np.full(15, .75), self.phi)
        info = expected_information(covariance, derivative)
        inverse = np.linalg.inv(covariance)
        for i, j in [(0, 0), (12, 33), (31, 39)]:
            self.assertAlmostEqual(info[i, j], .5 * np.trace(inverse @ derivative[i] @ inverse @ derivative[j]))
        self.assertGreater(np.linalg.eigvalsh(info).min(), 0)

    def test_better_measurement_and_known_loadings_cannot_worsen_bound(self):
        unknown = [bound_for(self.phi, loading, self.contrast) for loading in (.5, .75, .9)]
        self.assertGreater(unknown[0], unknown[1])
        self.assertGreater(unknown[1], unknown[2])
        self.assertLessEqual(bound_for(self.phi, .75, self.contrast, True), unknown[1] + 1e-9)
        self.assertGreater(unknown[2], standardized_latent_bound(self.phi, self.contrast))

    def test_near_perfect_measurement_approaches_standardized_latent_bound(self):
        observed = bound_for(self.phi, .999, self.contrast)
        latent = standardized_latent_bound(self.phi, self.contrast)
        self.assertAlmostEqual(observed / latent, 1, delta=.005)


if __name__ == "__main__":
    unittest.main()
