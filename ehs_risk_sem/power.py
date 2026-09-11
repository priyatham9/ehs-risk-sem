"""Sample size and power, treated as three separate questions.

Safety-science SEM papers routinely justify their sample size with a single
sentence ("N = 250 exceeds the recommended minimum"). There is no single
minimum, and the three requirements below bind at very different sample sizes:

1. **Global fit power.** Can the test of close fit detect a model that is
   wrong by a stated amount? This is the MacCallum, Browne & Sugawara (1996)
   RMSEA calculation, implemented in :func:`rmsea_power` and inverted in
   :func:`min_n_for_rmsea_power`. For a model with many degrees of freedom
   this is satisfied at surprisingly small N -- and it is usually *not* the
   binding constraint.

2. **Precision of the structural coefficients.** Can the individual paths be
   estimated tightly enough to say anything about their relative size? This is
   :func:`path_se` and :func:`min_n_for_path_se`, and it binds far later.
   :func:`min_n_to_distinguish` answers the specific question of how large a
   sample is needed to tell one coefficient from another in the same model.

3. **Event count.** If the outcome is a real incident rather than a survey
   item, neither of the above governs: the number of events does. See
   :mod:`ehs_risk_sem.rare_events`.

The distinction matters because a paper can pass (1) with a few hundred cases,
report a coefficient ordering, and have no basis at all for that ordering.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from .linalg import safe_inverse
from .special import bisect, chi2_ppf, ncx2_cdf

__all__ = [
    "rmsea_power",
    "min_n_for_rmsea_power",
    "noncentrality_from_rmsea",
    "path_se",
    "min_n_for_path_se",
    "se_of_difference",
    "min_n_to_distinguish",
]


def noncentrality_from_rmsea(rmsea: float, df: int, n: int) -> float:
    """Noncentrality parameter implied by an RMSEA value.

    ``lambda = (n - 1) * df * rmsea^2``, following MacCallum, Browne &
    Sugawara (1996).
    """
    return float((n - 1) * df * rmsea**2)


def rmsea_power(
    df: int,
    n: int,
    rmsea_null: float = 0.05,
    rmsea_alt: float = 0.08,
    alpha: float = 0.05,
) -> float:
    """Power of the test of close fit.

    The null hypothesis is ``RMSEA <= rmsea_null``. The critical value is taken
    from the noncentral chi-square with the noncentrality implied by
    ``rmsea_null``, and power is evaluated under ``rmsea_alt``.

    Handles both directions: when ``rmsea_alt > rmsea_null`` this is power to
    reject close fit for a model that is not close-fitting; when
    ``rmsea_alt < rmsea_null`` it is power for the test of not-close fit.
    """
    if df <= 0:
        raise ValueError("df must be positive")
    if n <= 1:
        raise ValueError("n must exceed 1")
    lam_null = noncentrality_from_rmsea(rmsea_null, df, n)
    lam_alt = noncentrality_from_rmsea(rmsea_alt, df, n)

    if rmsea_alt >= rmsea_null:
        crit = _ncx2_quantile(1.0 - alpha, df, lam_null)
        return float(1.0 - ncx2_cdf(crit, df, lam_alt))
    crit = _ncx2_quantile(alpha, df, lam_null)
    return float(ncx2_cdf(crit, df, lam_alt))


def _ncx2_quantile(p: float, df: int, nc: float) -> float:
    """Quantile of the noncentral chi-square, by bisection on its CDF."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    hi = max(chi2_ppf(min(p + (1 - p) / 2, 0.999999), df) + nc * 4.0, df + nc + 10.0)
    while ncx2_cdf(hi, df, nc) < p and hi < 1e10:
        hi *= 2.0
    return bisect(lambda x: ncx2_cdf(x, df, nc) - p, 1e-9, hi, tol=1e-8)


def min_n_for_rmsea_power(
    df: int,
    target_power: float = 0.80,
    rmsea_null: float = 0.05,
    rmsea_alt: float = 0.08,
    alpha: float = 0.05,
    n_max: int = 200_000,
) -> Optional[int]:
    """Smallest N reaching ``target_power`` for the test of close fit.

    Implements the procedure described by MacCallum, Browne & Sugawara (1996):
    reject close fit when the chi-square exceeds the ``1 - alpha`` quantile of
    a noncentral chi-square with noncentrality ``(N-1)*df*rmsea_null^2``, and
    evaluate power against noncentrality ``(N-1)*df*rmsea_alt^2``.

    Accuracy. The values returned here have been checked against the printed
    tables in MacCallum, Browne & Sugawara (1996), transcribed from the article
    itself rather than from a secondary reproduction; see
    ``tests/test_maccallum_1996_tables.py``, which pins all 52 cells of their
    Table 4 (this function), all 50 cells of their Table 5, and all 105 cells
    of their Table 2 (:func:`rmsea_power`). Power agrees to within 0.001, the
    precision at which the paper prints it. Sample sizes agree exactly at every
    df from 6 to 100 except for occasional one-unit differences, which are a
    property of their interval-halving search convention rather than of the
    procedure; the largest absolute disagreement anywhere in Table 4 is 6
    observations at df = 2, where Nmin is 3,488. The underlying noncentral
    chi-square is separately validated against Monte Carlo in
    ``tests/test_special.py``.

    One consequence worth knowing before quoting a cell: at df = 80 this
    function returns 153 where the published table prints 154. Power at N = 153
    is 0.8002, so 153 is the smallest integer attaining the target, and both
    values are defensible. Quote whichever you can source.

    Returns ``None`` if ``n_max`` is reached without attaining the target.
    """
    lo, hi = 5, 200
    while hi < n_max and rmsea_power(df, hi, rmsea_null, rmsea_alt, alpha) < target_power:
        lo = hi
        hi *= 2
    if hi >= n_max:
        if rmsea_power(df, n_max, rmsea_null, rmsea_alt, alpha) < target_power:
            return None
        hi = n_max
    while lo < hi:
        mid = (lo + hi) // 2
        if rmsea_power(df, mid, rmsea_null, rmsea_alt, alpha) >= target_power:
            hi = mid
        else:
            lo = mid + 1
    return int(lo)


def path_se(r_squared: float, vif: float, n: int, n_predictors: int) -> float:
    """Asymptotic standard error of a standardized path coefficient.

    ``SE = sqrt((1 - R^2) * VIF / (n - k - 1))``

    ``VIF`` is the diagonal element of the inverse predictor correlation
    matrix. This expression conditions on the predictor correlations; in a
    latent-variable model those correlations are themselves estimated, so the
    real standard error is larger. ``simulations/study_01_sample_size.py``
    measures the gap.
    """
    dof = max(n - n_predictors - 1, 1)
    return float(np.sqrt(max(1.0 - r_squared, 0.0) * vif / dof))


def min_n_for_path_se(
    target_se: float, r_squared: float, vif: float, n_predictors: int
) -> int:
    """Smallest N attaining a target standard error on a path coefficient."""
    if target_se <= 0:
        raise ValueError("target_se must be positive")
    needed = (1.0 - r_squared) * vif / target_se**2
    return int(np.ceil(needed + n_predictors + 1))


def se_of_difference(
    predictor_corr: np.ndarray, r_squared: float, n: int, i: int, j: int
) -> float:
    """Standard error of the difference between two coefficients in one model.

    ``Var(b_i - b_j) = (1 - R^2)/(n - k - 1) * [Rinv_ii + Rinv_jj - 2 Rinv_ij]``

    Using the correct covariance term matters: when two predictors are
    positively correlated their coefficient estimates are negatively
    correlated, and ignoring that understates the uncertainty in their
    difference.
    """
    r = np.asarray(predictor_corr, dtype=float)
    inv, _ = safe_inverse(r)
    k = r.shape[0]
    dof = max(n - k - 1, 1)
    var = (1.0 - r_squared) / dof * (inv[i, i] + inv[j, j] - 2.0 * inv[i, j])
    return float(np.sqrt(max(var, 0.0)))


def min_n_to_distinguish(
    beta_i: float,
    beta_j: float,
    predictor_corr: np.ndarray,
    r_squared: float,
    i: int = 0,
    j: int = 1,
    power: float = 0.80,
    alpha: float = 0.05,
    n_max: int = 2_000_000,
) -> Optional[int]:
    """Sample size needed to detect that two coefficients differ.

    Answers the question a reader actually has when a paper reports paths of
    0.45 and 0.30 and describes the first as the dominant driver: is the
    sample large enough to distinguish them at all?

    Uses a two-sided z test on the difference at the given power.
    """
    from .special import norm_ppf

    delta = abs(float(beta_i) - float(beta_j))
    if delta == 0.0:
        return None
    z_alpha = norm_ppf(1.0 - alpha / 2.0)
    z_beta = norm_ppf(power)
    lo, hi = 10, 1000
    while hi < n_max:
        se = se_of_difference(predictor_corr, r_squared, hi, i, j)
        if delta / se >= z_alpha + z_beta:
            break
        lo = hi
        hi *= 2
    if hi >= n_max:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        se = se_of_difference(predictor_corr, r_squared, mid, i, j)
        if delta / se >= z_alpha + z_beta:
            hi = mid
        else:
            lo = mid + 1
    return int(lo)


def equicorrelated_matrix(k: int, rho: float) -> np.ndarray:
    """Helper: a ``k x k`` correlation matrix with all off-diagonals equal to ``rho``."""
    m = np.full((k, k), float(rho))
    np.fill_diagonal(m, 1.0)
    return m
