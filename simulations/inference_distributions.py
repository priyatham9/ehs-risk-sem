"""Standard-library distribution functions used by the inference audit.

Student-t CDF is evaluated through the regularized incomplete beta function,
using the standard continued-fraction recurrence (Numerical Recipes, section
6.4). Quantiles are monotone bisection inversions, cached by degrees of freedom.
Pinned independent reference values cover the study's grid and small-df cases.
The SEM estimator itself continues to use ehs_risk_sem.special unchanged.
"""
from functools import lru_cache
import math
from statistics import NormalDist


def normal_cdf(x):
    return .5 * math.erfc(-x / math.sqrt(2))


def normal_sf(x):
    return .5 * math.erfc(x / math.sqrt(2))


def normal_ppf(p):
    return NormalDist().inv_cdf(p)


def _beta_fraction(a, b, x):
    qab, qap, qam = a + b, a + 1, a - 1
    tiny = 1e-300
    c = 1.0
    d = 1 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1 / d
    result = d
    for m in range(1, 1001):
        m2 = 2 * m
        coefficient = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + coefficient * d
        c = 1 + coefficient / c
        if abs(d) < tiny:
            d = tiny
        if abs(c) < tiny:
            c = tiny
        d = 1 / d
        result *= d * c
        coefficient = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + coefficient * d
        c = 1 + coefficient / c
        if abs(d) < tiny:
            d = tiny
        if abs(c) < tiny:
            c = tiny
        d = 1 / d
        change = d * c
        result *= change
        if abs(change - 1) < 3e-15:
            return result
    raise ArithmeticError("Incomplete-beta continued fraction did not converge")


def _regularized_beta(x, a, b):
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    factor = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                      + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1) / (a + b + 2):
        return factor * _beta_fraction(a, b, x) / a
    return 1 - factor * _beta_fraction(b, a, 1 - x) / b


def student_t_cdf(value, df):
    if df <= 0:
        raise ValueError("Student-t degrees of freedom must be positive")
    beta = _regularized_beta(df / (df + value**2), df / 2, .5)
    return 1 - beta / 2 if value >= 0 else beta / 2


@lru_cache(maxsize=None)
def student_t_critical_975(df):
    if df <= 0:
        raise ValueError("Student-t degrees of freedom must be positive")
    low, high = 0., 2.
    while student_t_cdf(high, df) < .975:
        high *= 2
    for _ in range(80):
        mid = (low + high) / 2
        if student_t_cdf(mid, df) < .975:
            low = mid
        else:
            high = mid
    return (low + high) / 2
