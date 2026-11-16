"""Independent numerical marker-identification check of study_07.

Specification supplied by a skeptical reviewer independently of the analytic
unit-latent-variance derivation. Marker indicators have loading 1; latent
variances/covariances are free. All 40 covariance and standardized-contrast
parameter derivatives are central finite differences. No analytic derivative or
Fisher helper from study_07 is reused; only final variance values are compared.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from simulations.study_05_inference_audit import Cell, PAIRS, model_for
from simulations.study_07_information_bound import bound_for

FREE_LOADINGS = [i for i in range(15) if i % 3 != 0]
TRIANGLE = [(i, j) for i in range(5) for j in range(i, 5)]


def marker_covariance_and_contrast(parameters, contrast):
    loadings = np.zeros((15, 5))
    loadings[np.arange(15), np.arange(15) // 3] = 1
    loadings[FREE_LOADINGS, np.array(FREE_LOADINGS) // 3] = parameters[:10]
    latent = np.zeros((5, 5))
    for value, (i, j) in zip(parameters[25:], TRIANGLE):
        latent[i, j] = latent[j, i] = value
    sigma = loadings @ latent @ loadings.T + np.diag(parameters[10:25])
    raw_beta = np.linalg.solve(latent[:4, :4], latent[:4, 4])
    standard_beta = raw_beta * np.sqrt(np.diag(latent)[:4] / latent[4, 4])
    return sigma, float(contrast @ standard_beta)


def numerical_marker_bound(phi, loading, contrast, step=1e-5):
    parameters = np.r_[np.ones(10), np.full(15, 1 - loading**2),
                       [loading**2 * phi[i, j] for i, j in TRIANGLE]]
    covariance, _ = marker_covariance_and_contrast(parameters, contrast)
    derivative, gradient = [], []
    for k in range(40):
        perturbation = np.zeros(40)
        perturbation[k] = step
        plus_cov, plus_target = marker_covariance_and_contrast(parameters + perturbation, contrast)
        minus_cov, minus_target = marker_covariance_and_contrast(parameters - perturbation, contrast)
        derivative.append((plus_cov - minus_cov) / (2 * step))
        gradient.append((plus_target - minus_target) / (2 * step))
    inverse = np.linalg.inv(covariance)
    information = np.empty((40, 40))
    for a in range(40):
        for b in range(40):
            information[a, b] = .5 * np.trace(inverse @ derivative[a] @ inverse @ derivative[b])
    gradient = np.array(gradient)
    variance = float(gradient @ np.linalg.solve(information, gradient))
    return variance, int(np.linalg.matrix_rank(information)), float(np.linalg.cond(information))


def validation_table():
    rows = []
    for pair in PAIRS:
        for loading in (.5, .75, .9, .99):
            cell = Cell(pair, loading, 1.)
            phi = model_for(cell, "alternative").implied_latent_correlation()
            contrast = np.zeros(4)
            i, j, _ = PAIRS[pair]
            contrast[i], contrast[j] = 1, -1
            marker, rank, condition = numerical_marker_bound(phi, loading, contrast)
            analytic = bound_for(phi, loading, contrast)
            rows.append({"pair": pair, "loading": loading,
                         "analytic_unit_variance_parameterization": analytic,
                         "numerical_marker_parameterization": marker,
                         "relative_difference": abs(marker / analytic - 1),
                         "information_rank": rank, "information_condition": condition,
                         "finite_difference_step": 1e-5})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    table = validation_table()
    assert table.relative_difference.max() < 1e-6
    assert (table.information_rank == 40).all()
    destination = Path("results/information_bound/marker_validation.csv")
    destination.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination, index=False)
    print(table.to_string(index=False))
