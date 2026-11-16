"""Gaussian full-information benchmark for standardized contrast precision.

This is an expected-information calculation under a correctly specified normal
congeneric covariance model, not an empirical fit or a promise of finite-sample
power. Forty free parameters: 15 loadings, 15 uniquenesses, and 10 latent
correlations, with latent variances fixed to 1. That is the same 80-df covariance
model as the package's five-factor/four-predictor example. The structural part is
saturated among five factors, so its beta vector is a smooth transformation of
the ten latent correlations.

Expected information per observation: I_ab = tr(Sigma^-1 Sigma_a Sigma^-1
Sigma_b)/2. Contrast variance bound = gradient' I^-1 gradient / N. Unknown
loadings and uniquenesses are nuisance parameters and remain in the inversion.
The known-loadings variant fixes only loadings, not residual variances. A latent
oracle bound includes random standardization uncertainty, unlike raw-unit OLS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from simulations.study_05_inference_audit import Cell, PAIRS, model_for

LATENT_PAIRS = [(i, j) for i in range(5) for j in range(i + 1, 5)]


def measurement_covariance(loadings, uniquenesses, phi):
    lam = np.zeros((15, 5))
    lam[np.arange(15), np.arange(15) // 3] = loadings
    return lam @ phi @ lam.T + np.diag(uniquenesses)


def measurement_derivatives(loadings, phi):
    lam = np.zeros((15, 5))
    lam[np.arange(15), np.arange(15) // 3] = loadings
    derivatives = []
    for i in range(15):
        delta = np.zeros_like(lam)
        delta[i, i // 3] = 1
        derivatives.append(delta @ phi @ lam.T + lam @ phi @ delta.T)
    for i in range(15):
        delta = np.zeros((15, 15))
        delta[i, i] = 1
        derivatives.append(delta)
    for i, j in LATENT_PAIRS:
        delta = np.zeros((5, 5))
        delta[i, j] = delta[j, i] = 1
        derivatives.append(lam @ delta @ lam.T)
    return np.array(derivatives)


def contrast_from_phi(phi, contrast):
    return float(contrast @ np.linalg.solve(phi[:4, :4], phi[:4, 4]))


def contrast_gradient(phi, contrast):
    inverse = np.linalg.inv(phi[:4, :4])
    beta = inverse @ phi[:4, 4]
    values = []
    for i, j in LATENT_PAIRS:
        delta = np.zeros((5, 5))
        delta[i, j] = delta[j, i] = 1
        values.append(float(contrast @ inverse @ (delta[:4, 4] - delta[:4, :4] @ beta)))
    return np.array(values)


def expected_information(covariance, derivatives):
    inverse = np.linalg.inv(covariance)
    scaled = np.einsum("ij,ajk->aik", inverse, derivatives)
    return .5 * np.einsum("aij,bji->ab", scaled, scaled)


def bound_for(phi, loading, contrast, known_loadings=False):
    loadings = np.full(15, loading)
    covariance = measurement_covariance(loadings, 1 - loadings**2, phi)
    derivatives = measurement_derivatives(loadings, phi)
    gradient = np.r_[np.zeros(30), contrast_gradient(phi, contrast)]
    if known_loadings:
        derivatives, gradient = derivatives[15:], gradient[15:]
    information = expected_information(covariance, derivatives)
    return float(gradient @ np.linalg.solve(information, gradient))


def standardized_latent_bound(phi, contrast):
    # Covariance parametrized by five latent SDs and ten correlations. Retaining
    # SD nuisance parameters is essential: treating sample SDs as known is wrong.
    derivatives = []
    for i in range(5):
        delta = np.zeros((5, 5))
        delta[i, :] += phi[i, :]
        delta[:, i] += phi[:, i]
        derivatives.append(delta)
    for i, j in LATENT_PAIRS:
        delta = np.zeros((5, 5))
        delta[i, j] = delta[j, i] = 1
        derivatives.append(delta)
    info = expected_information(phi, np.array(derivatives))
    gradient = np.r_[np.zeros(5), contrast_gradient(phi, contrast)]
    return float(gradient @ np.linalg.solve(info, gradient))


def normal_power(delta, variance, n):
    noncentrality = abs(delta) * np.sqrt(n / variance)
    cutoff = norm.ppf(.975)
    return float(norm.sf(cutoff - noncentrality) + norm.cdf(-cutoff - noncentrality))


def table():
    rows = []
    for pair in PAIRS:
        for loading in (.5, .75, .9):
            cell = Cell(pair, loading, 1.)
            i, j, threshold = PAIRS[pair]
            contrast = np.zeros(4)
            contrast[i], contrast[j] = 1, -1
            for stage in ("null_validation", "alternative"):
                model = model_for(cell, stage)
                phi = model.implied_latent_correlation()
                delta = contrast_from_phi(phi, contrast)
                variance = bound_for(phi, loading, contrast)
                known_variance = bound_for(phi, loading, contrast, known_loadings=True)
                latent_variance = standardized_latent_bound(phi, contrast)
                beta = model.beta[4, :4]
                raw_variance = (1 - beta @ phi[:4, :4] @ beta) * contrast @ np.linalg.solve(phi[:4, :4], contrast)
                row = {"pair": pair, "loading": loading, "stage": stage,
                       "true_difference": delta, "advertised_n": threshold,
                       "full_information_variance_times_n": variance,
                       "known_loadings_variance_times_n": known_variance,
                       "standardized_latent_variance_times_n": latent_variance,
                       "raw_oracle_ols_variance_times_n": float(raw_variance),
                       "variance_inflation_vs_standardized_latent": variance / latent_variance}
                if stage == "alternative":
                    row.update({"asymptotic_n_for_80pct_power": int(np.ceil(
                        (norm.ppf(.975) + norm.ppf(.8))**2 * variance / delta**2)),
                        "asymptotic_power_at_advertised_n": normal_power(delta, variance, threshold),
                        "asymptotic_power_at_double_n": normal_power(delta, variance, 2 * threshold)})
                rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/information_bound"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    table().to_csv(args.out / "summary.csv", index=False)
    manifest = {"analysis": "Gaussian expected Fisher information, asymptotic theoretical benchmark",
                "empirical_data": False, "free_parameters": 40,
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "numpy": np.__version__}
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    main()
