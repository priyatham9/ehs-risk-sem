"""The structural half of the model: paths among latent variables.

Estimation is limited-information and closed-form. Given the measurement model
from :mod:`ehs_risk_sem.measurement`:

1. Form unit-weighted composites of each indicator block.
2. Correct the composite correlations for attenuation using the composite
   reliabilities, giving an estimate of the latent correlation matrix ``Phi``.
   Under a congeneric measurement model this correction is exact in
   expectation, which is why the estimator recovers the generating parameters
   in the simulation studies.
3. Solve the standardized normal equations on ``Phi`` for each endogenous
   latent variable.

This is a two-step estimator in the sense of Anderson & Gerbing (1988): the
measurement model is established first, and the structural paths are read off
afterwards. Two consequences are stated here rather than buried:

* the analytic standard errors below treat ``Phi`` as known when it is
  estimated, so they are too small. ``simulations/study_01_sample_size.py``
  measures how much too small, and :func:`bootstrap_paths` in
  :mod:`ehs_risk_sem.model` gives an interval that does not have this problem;
* nothing in the arithmetic makes the coefficients causal. They are causal
  effects only relative to an assumed graph, and the graph is an assumption the
  analyst supplies. See ``README.md``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .linalg import safe_inverse
from .special import norm_cdf, norm_ppf

__all__ = [
    "PathEstimates",
    "disattenuate",
    "latent_correlations",
    "solve_paths",
    "effect_decomposition",
    "indirect_effect_delta_se",
]


@dataclass
class PathEstimates:
    """Standardized structural coefficients for one endogenous latent variable.

    Attributes
 ----------
    outcome
        Name of the endogenous latent variable.
    predictors
        Names of its direct predictors, matching the order of ``beta``.
    beta
        Standardized path coefficients. These apply to z-scores of the latent
        variables, not to raw indicator values.
    se
        Analytic standard errors conditional on ``Phi``. Understated; see the
        module docstring.
    r_squared
        Proportion of the endogenous latent's variance explained.
    disturbance_variance
        ``1 - r_squared`` in the standardized metric.
    vif
        Variance inflation factor per predictor.
    n_obs
        Sample size used.
    """

    outcome: str
    predictors: List[str]
    beta: np.ndarray
    se: np.ndarray
    r_squared: float
    disturbance_variance: float
    vif: np.ndarray
    n_obs: int
    warnings: List[str] = field(default_factory=list)

    def z_values(self) -> np.ndarray:
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.se > 0, self.beta / self.se, np.nan)

    def p_values(self) -> np.ndarray:
        return np.array(
            [
                np.nan if not np.isfinite(v) else 2.0 * (1.0 - norm_cdf(abs(float(v))))
                for v in self.z_values()
            ]
        )

    def confidence_intervals(self, level: float = 0.95) -> np.ndarray:
        """Wald intervals. Too narrow, for the reason given in the module docstring."""
        z = norm_ppf(0.5 + level / 2.0)
        lo = self.beta - z * self.se
        hi = self.beta + z * self.se
        return np.column_stack([lo, hi])


def disattenuate(r_observed: float, reliability_x: float, reliability_y: float) -> float:
    """Correct an observed correlation for attenuation due to measurement error.

    ``rho = r / sqrt(rel_x * rel_y)``

    The correction can push the estimate outside [-1, 1] when the observed
    correlation is high relative to the reliabilities. That is not a numerical
    artefact to be clipped away quietly: it means the congeneric measurement
    model is inconsistent with the observed data, usually because two
    "constructs" are not distinct. The value is returned uncapped and the
    caller decides what to say about it.
    """
    denom = np.sqrt(float(reliability_x) * float(reliability_y))
    if denom <= 0:
        return float("nan")
    return float(r_observed) / float(denom)


def latent_correlations(
    composite_corr: np.ndarray,
    reliabilities: Sequence[float],
    names: Sequence[str],
) -> Tuple[np.ndarray, List[str]]:
    """Disattenuate a composite correlation matrix into a latent correlation matrix.

    Returns the matrix and a list of warning strings. Off-diagonal entries whose
    absolute value exceeds 1 after correction are reported by name and left in
    place; :func:`ehs_risk_sem.model.fit` decides whether to repair the matrix
    and says so in its output when it does.
    """
    corr = np.asarray(composite_corr, dtype=float)
    rel = np.asarray(reliabilities, dtype=float)
    k = corr.shape[0]
    phi = np.eye(k)
    warnings: List[str] = []
    for i in range(k):
        for j in range(i + 1, k):
            value = disattenuate(corr[i, j], rel[i], rel[j])
            phi[i, j] = phi[j, i] = value
            if abs(value) > 1.0:
                warnings.append(
                    f"disattenuated correlation between '{names[i]}' and "
                    f"'{names[j]}' is {value:.3f}, outside [-1, 1]. The two "
                    "blocks are not empirically distinct given their "
                    "reliabilities: the measurement model is misspecified, or "
                    "these are one construct rather than two."
                )
            elif abs(value) > 0.90:
                warnings.append(
                    f"disattenuated correlation between '{names[i]}' and "
                    f"'{names[j]}' is {value:.3f}. Discriminant validity is "
                    "doubtful at this level."
                )
    return phi, warnings


def solve_paths(
    phi: np.ndarray,
    names: Sequence[str],
    outcome: str,
    predictors: Sequence[str],
    n_obs: int,
) -> PathEstimates:
    """Solve the standardized normal equations for one endogenous latent variable.

    ``beta = R_xx^-1 r_xy`` where ``R_xx`` is the predictor block of ``Phi`` and
    ``r_xy`` the predictor-outcome column.

    Standard errors use ``SE(b_j) = sqrt((1 - R^2) * (R_xx^-1)_jj / (n - k - 1))``,
    the usual standardized-regression expression. It conditions on ``Phi``.
    """
    name_list = list(names)
    idx_y = name_list.index(outcome)
    idx_x = [name_list.index(p) for p in predictors]
    k = len(idx_x)
    if k == 0:
        raise ValueError(f"'{outcome}' has no predictors")

    phi_arr = np.asarray(phi, dtype=float)
    r_xx = phi_arr[np.ix_(idx_x, idx_x)]
    r_xy = phi_arr[np.ix_(idx_x, [idx_y])].ravel()

    inv_rxx, warn = safe_inverse(r_xx)
    warnings: List[str] = []
    if warn is not None:
        warnings.append(
            f"predictor correlation matrix for '{outcome}' is ill-conditioned: {warn}"
        )

    beta = inv_rxx @ r_xy
    r_squared = float(beta @ r_xy)
    if r_squared > 1.0:
        warnings.append(
            f"R^2 for '{outcome}' is {r_squared:.3f}, above 1. The "
            "disattenuated latent correlation matrix is not positive "
            "semidefinite; the structural solution is improper."
        )
    disturbance = 1.0 - r_squared

    dof = max(n_obs - k - 1, 1)
    var_scale = max(disturbance, 1e-12) / dof
    se = np.sqrt(np.clip(var_scale * np.diag(inv_rxx), 0.0, None))
    vif = np.clip(np.diag(inv_rxx), 0.0, None)

    if np.any(vif > 5.0):
        high = [predictors[i] for i in np.where(vif > 5.0)[0]]
        warnings.append(
            "variance inflation above 5 for: " + ", ".join(high) + ". The "
            "individual coefficients are poorly separated from one another; "
            "reporting their relative magnitudes will not be reliable."
        )

    return PathEstimates(
        outcome=outcome,
        predictors=list(predictors),
        beta=beta,
        se=se,
        r_squared=r_squared,
        disturbance_variance=disturbance,
        vif=vif,
        n_obs=n_obs,
        warnings=warnings,
    )


def effect_decomposition(
    beta_matrix: np.ndarray, names: Sequence[str]
) -> Dict[str, np.ndarray]:
    """Decompose a recursive path model into direct, total and indirect effects.

    ``beta_matrix[i, j]`` is the direct effect of latent ``j`` on latent ``i``.
    For a recursive (acyclic) system the total effects are
    ``(I - B)^-1 - I`` and the indirect effects are total minus direct.

    This matters for interpretation. A coefficient in a model that also contains
    a mediator is a direct effect with the mediated pathway conditioned away.
    Reporting four such coefficients side by side as though they were
    comparable "contributions to risk" is the Table 2 fallacy (Westreich &
    Greenland 2013), and total effects are usually the quantity a reader has in
    mind.
    """
    b = np.asarray(beta_matrix, dtype=float)
    k = b.shape[0]
    if b.shape[0] != b.shape[1]:
        raise ValueError("beta_matrix must be square")
    identity = np.eye(k)
    inv, warn = safe_inverse(identity - b)
    if warn is not None:
        raise ValueError(
            "cannot invert (I - B); the path model may contain a feedback loop, "
            "which this estimator does not support"
        )
    total = inv - identity
    return {"direct": b, "total": total, "indirect": total - b, "names": np.array(list(names))}


def indirect_effect_delta_se(
    a: float, b: float, se_a: float, se_b: float
) -> float:
    """Delta-method standard error of the product ``a * b``.

    ``SE = sqrt(b^2 * SE_a^2 + a^2 * SE_b^2)``

    Provided for completeness, with a warning attached: the sampling
    distribution of a product is skewed, so a symmetric Wald interval built
    from this quantity has poor coverage in small samples. More importantly, a
    mediation coefficient estimated this way is not a causal mechanism estimate
    without assumptions that observational data cannot check (Bullock, Green &
    Ha 2010).
    """
    return float(np.sqrt(b**2 * se_a**2 + a**2 * se_b**2))
