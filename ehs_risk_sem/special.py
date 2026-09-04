"""Special functions needed for SEM inference, implemented on stdlib + numpy.

SciPy is not a dependency of this package, so the distribution functions that
fit indices and power analysis require are implemented here directly:

* regularized lower incomplete gamma, via the series / continued-fraction pair
  given in Press et al., *Numerical Recipes* (the standard textbook algorithm);
* central chi-square CDF and quantile function;
* noncentral chi-square CDF, as a Poisson mixture of central chi-square CDFs --
  needed for the RMSEA confidence interval and for MacCallum-style power
  analysis, both of which are inversions of a noncentral chi-square;
* standard normal CDF and quantile function.

Accuracy targets are roughly 1e-10 relative on the central functions and 1e-8
on the noncentral chi-square. `tests/test_special.py` checks these against
values computed independently.
"""

from __future__ import annotations

import math
from typing import Callable

import numpy as np

__all__ = [
    "gammaln",
    "gammainc_lower_reg",
    "gammainc_upper_reg",
    "chi2_cdf",
    "chi2_sf",
    "chi2_ppf",
    "ncx2_cdf",
    "norm_cdf",
    "norm_ppf",
    "bisect",
]

_MAX_ITER = 1000
_EPS = 3.0e-16
_FPMIN = 1.0e-300


def gammaln(x: float) -> float:
    """Natural log of the gamma function. Thin wrapper over ``math.lgamma``."""
    return math.lgamma(x)


def _gser(a: float, x: float) -> float:
    """Series representation of the regularized lower incomplete gamma P(a, x).

    Converges quickly for ``x < a + 1``.
    """
    if x <= 0.0:
        return 0.0
    ap = a
    total = 1.0 / a
    delta = total
    for _ in range(_MAX_ITER):
        ap += 1.0
        delta *= x / ap
        total += delta
        if abs(delta) < abs(total) * _EPS:
            break
    return total * math.exp(-x + a * math.log(x) - gammaln(a))


def _gcf(a: float, x: float) -> float:
    """Continued-fraction representation of the regularized upper gamma Q(a, x).

    Converges quickly for ``x >= a + 1``. Modified Lentz algorithm.
    """
    b = x + 1.0 - a
    c = 1.0 / _FPMIN
    d = 1.0 / b
    h = d
    for i in range(1, _MAX_ITER + 1):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < _FPMIN:
            d = _FPMIN
        c = b + an / c
        if abs(c) < _FPMIN:
            c = _FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < _EPS:
            break
    return math.exp(-x + a * math.log(x) - gammaln(a)) * h


def gammainc_lower_reg(a: float, x: float) -> float:
    """Regularized lower incomplete gamma function ``P(a, x)``.

    Parameters
 ----------
    a
        Shape parameter, must be strictly positive.
    x
        Upper limit of integration, must be non-negative.

    Returns
 -------
    float
        ``P(a, x) = gamma(a, x) / Gamma(a)`` in [0, 1].
    """
    if a <= 0.0:
        raise ValueError("a must be > 0")
    if x < 0.0:
        raise ValueError("x must be >= 0")
    if x == 0.0:
        return 0.0
    if x < a + 1.0:
        return _gser(a, x)
    return 1.0 - _gcf(a, x)


def gammainc_upper_reg(a: float, x: float) -> float:
    """Regularized upper incomplete gamma function ``Q(a, x) = 1 - P(a, x)``."""
    return 1.0 - gammainc_lower_reg(a, x)


def chi2_cdf(x: float, df: float) -> float:
    """CDF of the central chi-square distribution with ``df`` degrees of freedom."""
    if df <= 0:
        raise ValueError("df must be > 0")
    if x <= 0.0:
        return 0.0
    return gammainc_lower_reg(df / 2.0, x / 2.0)


def chi2_sf(x: float, df: float) -> float:
    """Survival function (upper tail p-value) of the central chi-square."""
    if df <= 0:
        raise ValueError("df must be > 0")
    if x <= 0.0:
        return 1.0
    return gammainc_upper_reg(df / 2.0, x / 2.0)


def bisect(
    func: Callable[[float], float],
    lo: float,
    hi: float,
    tol: float = 1e-12,
    max_iter: int = 200,
) -> float:
    """Bisection root finder for a monotone function on a bracketing interval.

    Used for quantile functions. Assumes ``func(lo)`` and ``func(hi)`` straddle
    zero; the caller is responsible for supplying a valid bracket.
    """
    f_lo = func(lo)
    f_hi = func(hi)
    if f_lo == 0.0:
        return lo
    if f_hi == 0.0:
        return hi
    if f_lo * f_hi > 0.0:
        raise ValueError(
            f"bracket does not straddle a root: f({lo})={f_lo}, f({hi})={f_hi}"
        )
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        f_mid = func(mid)
        if f_mid == 0.0 or (hi - lo) < tol:
            return mid
        if f_lo * f_mid < 0.0:
            hi = mid
            f_hi = f_mid
        else:
            lo = mid
            f_lo = f_mid
    return 0.5 * (lo + hi)


def chi2_ppf(p: float, df: float) -> float:
    """Quantile function of the central chi-square distribution.

    Solved by bisection on :func:`chi2_cdf`, which is monotone in ``x``.
    """
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    hi = max(df * 2.0, 1.0)
    while chi2_cdf(hi, df) < p:
        hi *= 2.0
        if hi > 1e12:
            break
    return bisect(lambda x: chi2_cdf(x, df) - p, 0.0, hi)


def ncx2_cdf(x: float, df: float, nc: float, tol: float = 1e-12) -> float:
    """CDF of the noncentral chi-square distribution.

    Evaluated as the Poisson mixture

    ``F(x; df, nc) = sum_j Pois(j; nc/2) * F_central(x; df + 2j)``

    summing outward from the Poisson mode so that the dominant terms are added
    first. This is the representation used for RMSEA confidence intervals and
    for the MacCallum, Browne & Sugawara (1996) power calculation.

    Parameters
 ----------
    x
        Quantile, non-negative.
    df
        Degrees of freedom, positive.
    nc
        Noncentrality parameter, non-negative. ``nc = 0`` reduces to the
        central chi-square.
    tol
        Terms whose Poisson weight is below ``tol`` and which lie beyond the
        mode are dropped.
    """
    if df <= 0:
        raise ValueError("df must be > 0")
    if nc < 0:
        raise ValueError("nc must be >= 0")
    if x <= 0.0:
        return 0.0
    if nc == 0.0:
        return chi2_cdf(x, df)

    half_nc = nc / 2.0
    mode = int(half_nc)
    total = 0.0

    # Walk up from the Poisson mode.
    j = mode
    while j < mode + 10000:
        log_w = -half_nc + j * math.log(half_nc) - gammaln(j + 1.0)
        w = math.exp(log_w)
        term = w * chi2_cdf(x, df + 2.0 * j)
        total += term
        if w < tol and j > mode:
            break
        j += 1

    # Walk down from just below the mode.
    j = mode - 1
    while j >= 0:
        log_w = -half_nc + j * math.log(half_nc) - gammaln(j + 1.0)
        w = math.exp(log_w)
        total += w * chi2_cdf(x, df + 2.0 * j)
        if w < tol:
            break
        j -= 1

    return min(max(total, 0.0), 1.0)


def norm_cdf(x: float) -> float:
    """Standard normal CDF, via ``math.erf``."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def norm_ppf(p: float) -> float:
    """Standard normal quantile function, by bisection on :func:`norm_cdf`."""
    if not 0.0 < p < 1.0:
        raise ValueError("p must be in (0, 1)")
    return bisect(lambda x: norm_cdf(x) - p, -40.0, 40.0)


def vectorized_norm_cdf(x: np.ndarray) -> np.ndarray:
    """Elementwise standard normal CDF over a numpy array."""
    arr = np.asarray(x, dtype=float)
    flat = np.array([norm_cdf(float(v)) for v in arr.ravel()], dtype=float)
    return flat.reshape(arr.shape)
