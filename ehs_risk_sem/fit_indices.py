"""Global fit indices, and an explicit statement of what they cannot do.

Every index here is computed from the maximum-likelihood discrepancy between
the observed covariance matrix ``S`` and the model-implied matrix ``Sigma``:

``F_ML = log|Sigma| + tr(S Sigma^-1) - log|S| - p``

with ``chi2 = (n - 1) F_ML``.

Two caveats travel with the numbers and are attached to the result object so
they cannot be dropped when the table is copied into a paper.

1. This package estimates by a limited-information two-step procedure, not by
   minimizing ``F_ML``. The indices are therefore evaluated at the two-step
   estimates, and the chi-square is an upper bound on what a full-information
   ML fit of the same model would produce. Fit is if anything understated, not
   flattered.

2. Fit does not identify a causal structure. For any fitted model there
   generally exist equivalent models -- including models with arrows reversed
 -- that reproduce the same covariance matrix exactly and therefore have
   identical chi-square, CFI, RMSEA and SRMR (MacCallum, Wegener, Uchino &
   Fabrigar 1993). ``simulations/study_03_misspecification.py`` demonstrates
   this on generated data. A well-fitting model can also be substantively
   wrong, omit important variables, and predict poorly (Tomarken & Waller
   2003).

The conventional cutoffs (CFI >= .95, RMSEA <= .06, SRMR <= .08) come from Hu
& Bentler (1999) and are reported for comparability, with the caveat that they
were derived under specific conditions and were never meant as universal
pass/fail thresholds (Marsh, Hau & Wen 2004; McNeish & Wolf 2023).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .linalg import logdet, safe_inverse
from .special import chi2_ppf, chi2_sf, ncx2_cdf, bisect

__all__ = ["FitIndices", "ml_discrepancy", "srmr", "compute_fit_indices", "rmsea_ci"]

HU_BENTLER_CFI = 0.95
HU_BENTLER_RMSEA = 0.06
HU_BENTLER_SRMR = 0.08


@dataclass
class FitIndices:
    """Global fit statistics for a fitted model."""

    chi_square: float
    df: int
    p_value: float
    n_obs: int
    f_ml: float
    cfi: float
    tli: float
    rmsea: float
    rmsea_lower: float
    rmsea_upper: float
    p_close_fit: float
    srmr: float
    aic: float
    bic: float
    n_free_parameters: int
    baseline_chi_square: float
    baseline_df: int
    estimator: str = "limited-information two-step (not ML minimization)"
    notes: List[str] = field(default_factory=list)

    def meets_conventional_cutoffs(self) -> bool:
        """Whether the Hu & Bentler (1999) cutoffs are met.

        Reported because reviewers ask for it. It is not evidence that the
        model is correct and this method's name is deliberately not
        ``is_good_fit``.
        """
        return (
            self.cfi >= HU_BENTLER_CFI
            and self.rmsea <= HU_BENTLER_RMSEA
            and self.srmr <= HU_BENTLER_SRMR
        )

    def summary(self) -> str:
        lines = [
            "Global fit",
            f"  estimator                 : {self.estimator}",
            f"  chi-square({self.df})      = {self.chi_square:.3f}, p = {self.p_value:.4f}",
            f"  CFI                       = {self.cfi:.3f}",
            f"  TLI                       = {self.tli:.3f}",
            f"  RMSEA                     = {self.rmsea:.3f} "
            f"[90% CI {self.rmsea_lower:.3f}, {self.rmsea_upper:.3f}]",
            f"  p(RMSEA <= .05)           = {self.p_close_fit:.4f}",
            f"  SRMR                      = {self.srmr:.3f}",
            f"  AIC                       = {self.aic:.2f}",
            f"  BIC                       = {self.bic:.2f}",
        ]
        for n in self.notes:
            lines.append(f"  NOTE    : {n}")
        return "\n".join(lines)


def ml_discrepancy(s: np.ndarray, sigma: np.ndarray) -> float:
    """Maximum-likelihood discrepancy between observed and implied matrices.

    ``F = log|Sigma| + tr(S Sigma^-1) - log|S| - p``, which is zero when
    ``Sigma == S`` and positive otherwise.
    """
    s_arr = np.asarray(s, dtype=float)
    sig = np.asarray(sigma, dtype=float)
    if s_arr.shape != sig.shape:
        raise ValueError("S and Sigma must have the same shape")
    p = s_arr.shape[0]
    inv_sigma, _ = safe_inverse(sig)
    value = logdet(sig) + float(np.trace(s_arr @ inv_sigma)) - logdet(s_arr) - p
    return max(value, 0.0)


def srmr(s: np.ndarray, sigma: np.ndarray) -> float:
    """Standardized root mean square residual.

    Root mean square of the residuals between the observed and implied
    correlation matrices, over the lower triangle including the diagonal.
    """
    s_arr = np.asarray(s, dtype=float)
    sig = np.asarray(sigma, dtype=float)
    ds = np.sqrt(np.diag(s_arr))
    dsig = np.sqrt(np.diag(sig))
    rs = s_arr / np.outer(ds, ds)
    rsig = sig / np.outer(dsig, dsig)
    resid = rs - rsig
    idx = np.tril_indices(resid.shape[0])
    return float(np.sqrt(np.mean(resid[idx] ** 2)))


def rmsea_ci(
    chi_square: float, df: int, n_obs: int, level: float = 0.90
) -> tuple:
    """Confidence interval for RMSEA by inverting the noncentral chi-square.

    Finds noncentrality parameters ``lam_lo`` and ``lam_hi`` such that the
    noncentral chi-square CDF at the observed statistic equals the upper and
    lower tail probabilities, then converts each to the RMSEA scale via
    ``sqrt(lam / (df * (n - 1)))``.
    """
    if df <= 0 or n_obs <= 1:
        return (float("nan"), float("nan"))
    alpha = 1.0 - level
    upper_tail = 1.0 - alpha / 2.0
    lower_tail = alpha / 2.0

    def solve(target: float) -> float:
        if ncx2_cdf(chi_square, df, 0.0) < target:
            return 0.0
        hi = max(chi_square, 1.0)
        while ncx2_cdf(chi_square, df, hi) > target and hi < 1e8:
            hi *= 2.0
        return bisect(lambda lam: ncx2_cdf(chi_square, df, lam) - target, 0.0, hi, tol=1e-8)

    lam_lo = solve(upper_tail)
    lam_hi = solve(lower_tail)
    scale = df * (n_obs - 1)
    return (
        float(np.sqrt(max(lam_lo, 0.0) / scale)),
        float(np.sqrt(max(lam_hi, 0.0) / scale)),
    )


def compute_fit_indices(
    s: np.ndarray,
    sigma: np.ndarray,
    n_obs: int,
    df: int,
    n_free_parameters: int,
    estimator: str = "limited-information two-step (not ML minimization)",
    extra_notes: Optional[List[str]] = None,
) -> FitIndices:
    """Compute the full set of fit indices for a fitted model.

    The baseline (null) model for CFI and TLI is the independence model: all
    observed variances free, all covariances zero.
    """
    s_arr = np.asarray(s, dtype=float)
    sig = np.asarray(sigma, dtype=float)
    p = s_arr.shape[0]
    n = int(n_obs)

    f_ml = ml_discrepancy(s_arr, sig)
    chi2 = (n - 1) * f_ml

    baseline_sigma = np.diag(np.diag(s_arr))
    f_base = ml_discrepancy(s_arr, baseline_sigma)
    chi2_base = (n - 1) * f_base
    df_base = p * (p - 1) // 2

    d_model = max(chi2 - df, 0.0)
    d_base = max(chi2_base - df_base, 0.0)
    cfi = 1.0 if d_base <= 0 else float(1.0 - d_model / d_base)
    cfi = float(min(max(cfi, 0.0), 1.0))

    if df_base > 0 and df > 0 and chi2_base > 0:
        ratio_base = chi2_base / df_base
        tli = (ratio_base - chi2 / df) / (ratio_base - 1.0) if ratio_base != 1.0 else 1.0
    else:
        tli = float("nan")
    if np.isfinite(tli):
        tli = float(min(max(tli, 0.0), 1.0))

    rmsea = float(np.sqrt(max(chi2 - df, 0.0) / (df * (n - 1)))) if df > 0 else float("nan")
    lo, hi = rmsea_ci(chi2, df, n) if df > 0 else (float("nan"), float("nan"))

    if df > 0:
        lam_close = 0.05**2 * df * (n - 1)
        p_close = float(1.0 - ncx2_cdf(chi2, df, lam_close))
    else:
        p_close = float("nan")

    p_value = float(chi2_sf(chi2, df)) if df > 0 else float("nan")
    aic = chi2 + 2.0 * n_free_parameters
    bic = chi2 + np.log(n) * n_free_parameters

    notes = [
        "Fit indices are evaluated at limited-information two-step estimates, "
        "not at the ML minimum. The chi-square is an upper bound on the "
        "full-information ML chi-square for the same model.",
        "Equivalent models with different causal structures reproduce these "
        "same indices exactly. Fit is not evidence for the direction of any arrow.",
    ]
    if extra_notes:
        notes.extend(extra_notes)

    return FitIndices(
        chi_square=float(chi2),
        df=int(df),
        p_value=p_value,
        n_obs=n,
        f_ml=float(f_ml),
        cfi=cfi,
        tli=float(tli),
        rmsea=rmsea,
        rmsea_lower=lo,
        rmsea_upper=hi,
        p_close_fit=p_close,
        srmr=srmr(s_arr, sig),
        aic=float(aic),
        bic=float(bic),
        n_free_parameters=int(n_free_parameters),
        baseline_chi_square=float(chi2_base),
        baseline_df=int(df_base),
        estimator=estimator,
        notes=notes,
    )


def chi_square_critical(df: int, alpha: float = 0.05) -> float:
    """Critical value of the central chi-square at level ``alpha``."""
    return chi2_ppf(1.0 - alpha, df)
