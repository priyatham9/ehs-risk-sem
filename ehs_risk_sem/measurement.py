"""The measurement half of the model: indicators to latent variables.

A latent variable in SEM is defined by its measurement model. Take the
indicators away and the object is gone: there are no degrees of freedom, no
reliability estimate, no convergent or discriminant validity evidence, and no
test of measurement invariance. That is the single most important thing this
module exists to enforce -- every latent variable here must name at least two
observed indicators, and the code refuses to proceed otherwise.

Extraction is by principal-axis factoring: iteratively re-estimate the
communalities on the diagonal of the correlation matrix and take the leading
eigenvectors of the reduced matrix. For a one-factor block this is a congeneric
measurement model with freely estimated loadings.

Two quantities computed here are load-bearing for the honesty argument made in
the README:

* composite reliability (omega), which is what the latent correlations are
  disattenuated by in :mod:`ehs_risk_sem.structural`;
* factor score determinacy, and the Guttman lower bound ``2*rho^2 - 1`` on the
  correlation between two equally valid sets of factor scores for the same
  factor. When that bound is well below 1, a per-unit "risk score" built from
  factor scores is not uniquely determined by the model, however well the model
  fits.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

from .linalg import corr_matrix, safe_inverse, standardize

__all__ = [
    "FactorSolution",
    "principal_axis_factor",
    "fit_congeneric_block",
    "composite_reliability",
    "cronbach_alpha",
    "average_variance_extracted",
    "factor_scores",
    "factor_score_determinacy",
    "guttman_indeterminacy_bound",
    "fornell_larcker",
    "htmt",
]

MIN_INDICATORS = 2
RECOMMENDED_INDICATORS = 3


@dataclass
class FactorSolution:
    """Estimated single-factor (congeneric) measurement model for one block.

    Attributes
 ----------
    name
        Latent variable name.
    indicators
        Column names of the indicators, in the order the loadings are given.
    loadings
        Standardized loadings, one per indicator. The latent variable is scaled
        to unit variance, so these are correlations between indicator and
        factor under the model.
    uniquenesses
        Indicator error variances in the standardized metric, ``1 - loading^2``
        under a congeneric model with unit-variance factor.
    communalities
        ``loading^2``: the share of each indicator's variance explained.
    omega
        Composite (congeneric) reliability of the unit-weighted sum.
    alpha
        Cronbach's alpha, reported alongside omega because alpha is a lower
        bound that only equals reliability under tau-equivalence.
    ave
        Average variance extracted, used for Fornell-Larcker discriminant
        validity.
    determinacy
        Correlation between the factor and its regression-method factor score.
    guttman_bound
        ``2 * determinacy^2 - 1``: the minimum correlation between two equally
        valid sets of factor scores.
    heywood
        True if any communality exceeded 1 during estimation and was capped --
        a sign of an improper solution that must be reported, not hidden.
    converged, n_iter
        Convergence status of the communality iteration.
    """

    name: str
    indicators: List[str]
    loadings: np.ndarray
    uniquenesses: np.ndarray
    communalities: np.ndarray
    omega: float
    alpha: float
    ave: float
    determinacy: float
    guttman_bound: float
    heywood: bool
    converged: bool
    n_iter: int
    warnings: List[str] = field(default_factory=list)

    def n_indicators(self) -> int:
        return len(self.indicators)


def principal_axis_factor(
    r: np.ndarray,
    n_factors: int = 1,
    max_iter: int = 200,
    tol: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray, bool, int, bool]:
    """Extract factors from a correlation matrix by principal-axis factoring.

    The diagonal of ``r`` is replaced by communality estimates (initialized at
    the squared multiple correlation of each variable with the others), the
    reduced matrix is eigendecomposed, loadings are read off the leading
    eigenvectors, and the communalities are updated. Iterate to convergence.

    Parameters
 ----------
    r
        Correlation matrix, ``(p, p)``.
    n_factors
        Number of factors to extract.
    max_iter, tol
        Controls on the communality iteration.

    Returns
 -------
    loadings
        ``(p, n_factors)`` loading matrix.
    communalities
        ``(p,)`` final communality estimates.
    converged
        Whether the communality iteration met ``tol``.
    n_iter
        Iterations used.
    heywood
        Whether any communality had to be capped at just under 1.
    """
    corr = np.asarray(r, dtype=float).copy()
    p = corr.shape[0]
    if corr.shape[0] != corr.shape[1]:
        raise ValueError("r must be square")
    if n_factors < 1 or n_factors >= p:
        raise ValueError("n_factors must be at least 1 and less than p")

    inv, _ = safe_inverse(corr)
    smc = 1.0 - 1.0 / np.clip(np.diag(inv), 1e-12, None)
    communalities = np.clip(smc, 0.001, 0.999)

    heywood = False
    converged = False
    n_iter = 0
    loadings = np.zeros((p, n_factors))
    for n_iter in range(1, max_iter + 1):
        reduced = corr.copy()
        np.fill_diagonal(reduced, communalities)
        vals, vecs = np.linalg.eigh(reduced)
        order = np.argsort(vals)[::-1][:n_factors]
        top_vals = np.clip(vals[order], 0.0, None)
        loadings = vecs[:, order] * np.sqrt(top_vals)
        new_comm = np.sum(loadings**2, axis=1)
        if np.any(new_comm > 1.0):
            heywood = True
        new_comm = np.clip(new_comm, 0.001, 0.999)
        delta = float(np.max(np.abs(new_comm - communalities)))
        communalities = new_comm
        if delta < tol:
            converged = True
            break

    # Orient each factor so the majority of its loadings are positive. Sign is
    # arbitrary in factor analysis; fixing it makes results comparable across
    # replications in the simulation studies.
    for j in range(n_factors):
        if np.sum(loadings[:, j]) < 0:
            loadings[:, j] *= -1.0

    return loadings, communalities, converged, n_iter, heywood


def composite_reliability(loadings: np.ndarray, uniquenesses: np.ndarray) -> float:
    """Congeneric composite reliability (McDonald's omega) of a unit-weighted sum.

    ``omega = (sum lambda)^2 / ((sum lambda)^2 + sum theta)``

    This is the quantity the observed composite correlations are divided by in
    order to recover the latent correlations, so any error in it propagates
    directly into the structural coefficients.
    """
    lam = np.asarray(loadings, dtype=float).ravel()
    theta = np.asarray(uniquenesses, dtype=float).ravel()
    num = float(np.sum(lam)) ** 2
    den = num + float(np.sum(theta))
    if den <= 0:
        return float("nan")
    return num / den


def cronbach_alpha(r: np.ndarray) -> float:
    """Cronbach's alpha computed from a correlation matrix (standardized alpha).

    Reported next to omega only so a reader can see the gap. Alpha equals
    reliability only under tau-equivalence (all loadings equal); otherwise it
    is a lower bound and understates reliability, which would over-correct a
    disattenuated correlation.
    """
    corr = np.asarray(r, dtype=float)
    k = corr.shape[0]
    if k < 2:
        return float("nan")
    off = corr[np.triu_indices(k, k=1)]
    mean_r = float(np.mean(off))
    return (k * mean_r) / (1.0 + (k - 1) * mean_r)


def average_variance_extracted(loadings: np.ndarray) -> float:
    """Mean squared standardized loading: the share of indicator variance the
    factor accounts for. Used for the Fornell-Larcker discriminant check."""
    lam = np.asarray(loadings, dtype=float).ravel()
    return float(np.mean(lam**2))


def factor_score_determinacy(loadings: np.ndarray, uniquenesses: np.ndarray) -> float:
    """Correlation between a factor and its regression-method factor score.

    For a single factor with unit variance,
    ``rho = sqrt(sum(lambda_j^2 / theta_j) / (1 + sum(lambda_j^2 / theta_j)))``.
    """
    lam = np.asarray(loadings, dtype=float).ravel()
    theta = np.clip(np.asarray(uniquenesses, dtype=float).ravel(), 1e-12, None)
    s = float(np.sum(lam**2 / theta))
    return float(np.sqrt(s / (1.0 + s)))


def guttman_indeterminacy_bound(determinacy: float) -> float:
    """Minimum correlation between two equally valid sets of factor scores.

    ``2 * rho^2 - 1``. At ``rho = 0.90`` this is 0.62: two analysts can compute
    factor scores that are equally consistent with the same fitted model and
    correlate only 0.62 with each other. That is a property of the model, not
    of the estimation, and it is why a per-unit latent "risk score" cannot be
    treated as a determinate quantity.
    """
    return 2.0 * float(determinacy) ** 2 - 1.0


def fit_congeneric_block(
    data: np.ndarray,
    indicators: Sequence[str],
    name: str,
    column_index: Optional[Sequence[int]] = None,
) -> FactorSolution:
    """Fit a one-factor measurement model to a block of indicators.

    Parameters
 ----------
    data
        Observed data, ``(n_obs, n_vars)``. Standardized internally.
    indicators
        Names of the indicators in this block, for reporting.
    name
        Latent variable name.
    column_index
        Column positions of the indicators in ``data``. Defaults to the first
        ``len(indicators)`` columns, which is almost never what a caller wants,
        so :class:`ehs_risk_sem.model.ModelSpec` always supplies it explicitly.
    """
    arr = np.asarray(data, dtype=float)
    idx = list(column_index) if column_index is not None else list(range(len(indicators)))
    if len(idx) != len(indicators):
        raise ValueError("column_index and indicators have different lengths")
    if len(idx) < MIN_INDICATORS:
        raise ValueError(
            f"latent variable '{name}' has {len(idx)} indicator(s). A latent "
            f"variable requires at least {MIN_INDICATORS} indicators to be "
            "identified without fixing its error variance a priori. A "
            "single-indicator 'latent variable' is an observed variable with "
            "an unestimated reliability, and calling it latent does not make "
            "it one."
        )

    block = standardize(arr[:, idx])
    corr = corr_matrix(block)
    loadings_mat, communalities, converged, n_iter, heywood = principal_axis_factor(
        corr, n_factors=1
    )
    loadings = loadings_mat[:, 0]
    uniquenesses = np.clip(1.0 - loadings**2, 1e-6, None)

    warnings: List[str] = []
    if len(idx) < RECOMMENDED_INDICATORS:
        warnings.append(
            f"'{name}' has {len(idx)} indicators. The three-indicator rule "
            "(Bollen 1989) gives a sufficient condition for identification of "
            "a one-factor model in isolation; with two indicators the factor "
            "is identified only through its correlations with other factors."
        )
    if heywood:
        warnings.append(
            f"'{name}' produced a communality at or above 1 during estimation "
            "(a Heywood case). The solution is improper and the loadings "
            "should not be interpreted."
        )
    if not converged:
        warnings.append(f"'{name}': communality iteration did not converge.")

    determinacy = factor_score_determinacy(loadings, uniquenesses)
    return FactorSolution(
        name=name,
        indicators=list(indicators),
        loadings=loadings,
        uniquenesses=uniquenesses,
        communalities=loadings**2,
        omega=composite_reliability(loadings, uniquenesses),
        alpha=cronbach_alpha(corr),
        ave=average_variance_extracted(loadings),
        determinacy=determinacy,
        guttman_bound=guttman_indeterminacy_bound(determinacy),
        heywood=heywood,
        converged=converged,
        n_iter=n_iter,
        warnings=warnings,
    )


def factor_scores(
    data: np.ndarray,
    loadings: np.ndarray,
    uniquenesses: np.ndarray,
    method: str = "regression",
) -> np.ndarray:
    """Compute factor scores for one factor.

    Parameters
 ----------
    data
        Indicator block, ``(n_obs, p)``. Standardized internally.
    loadings, uniquenesses
        From a fitted :class:`FactorSolution`.
    method
        ``"regression"`` (Thurstone/Thomson) or ``"bartlett"``. The two give
        different scores for the same fitted model; see
        :func:`guttman_indeterminacy_bound`.
    """
    block = standardize(np.asarray(data, dtype=float))
    lam = np.asarray(loadings, dtype=float).reshape(-1, 1)
    theta = np.clip(np.asarray(uniquenesses, dtype=float).ravel(), 1e-8, None)

    if method == "regression":
        sigma = lam @ lam.T + np.diag(theta)
        inv_sigma, _ = safe_inverse(sigma)
        weights = inv_sigma @ lam
    elif method == "bartlett":
        inv_theta = np.diag(1.0 / theta)
        # lam is (p, 1), so this product is (1, 1). Index it rather than
        # calling float() on the array: numpy deprecated the implicit
        # ndim > 0 to scalar conversion in 1.25 and will raise on it.
        middle = float((lam.T @ inv_theta @ lam)[0, 0])
        weights = inv_theta @ lam / middle
    else:
        raise ValueError("method must be 'regression' or 'bartlett'")

    scores = block @ weights
    return scores.ravel()


def fornell_larcker(
    ave_by_factor: Sequence[float], phi: np.ndarray, names: Sequence[str]
) -> List[Tuple[str, str, float, float, bool]]:
    """Fornell-Larcker discriminant validity check for every factor pair.

    Returns tuples ``(factor_a, factor_b, sqrt_ave_min, |phi|, passes)`` where
    the criterion passes when the smaller of the two square-root AVEs exceeds
    the absolute latent correlation.
    """
    out: List[Tuple[str, str, float, float, bool]] = []
    k = len(names)
    ave = np.asarray(ave_by_factor, dtype=float)
    for i in range(k):
        for j in range(i + 1, k):
            bound = float(min(np.sqrt(ave[i]), np.sqrt(ave[j])))
            r = abs(float(phi[i, j]))
            out.append((names[i], names[j], bound, r, bound > r))
    return out


def htmt(
    data: np.ndarray,
    blocks: Sequence[Sequence[int]],
    names: Sequence[str],
) -> np.ndarray:
    """Heterotrait-monotrait ratio of correlations.

    For each pair of factors, the mean absolute cross-block indicator
    correlation divided by the geometric mean of the two within-block mean
    absolute correlations. Values approaching 1 mean the two "constructs" are
    not empirically distinguishable, whatever they are called.
    """
    arr = standardize(np.asarray(data, dtype=float))
    corr = corr_matrix(arr)
    k = len(blocks)
    out = np.full((k, k), np.nan)
    mono = []
    for b in blocks:
        idx = list(b)
        if len(idx) < 2:
            mono.append(np.nan)
            continue
        sub = corr[np.ix_(idx, idx)]
        vals = np.abs(sub[np.triu_indices(len(idx), k=1)])
        mono.append(float(np.mean(vals)))
    for i in range(k):
        for j in range(i + 1, k):
            cross = np.abs(corr[np.ix_(list(blocks[i]), list(blocks[j]))])
            hetero = float(np.mean(cross))
            denom = np.sqrt(mono[i] * mono[j]) if np.isfinite(mono[i] * mono[j]) else np.nan
            value = hetero / denom if denom and denom > 0 else np.nan
            out[i, j] = out[j, i] = value
    np.fill_diagonal(out, 1.0)
    return out
