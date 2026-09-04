"""Logistic regression by iteratively reweighted least squares.

Needed in two places: the calibration slope (a logistic regression of the
outcome on the logit of the predicted probability) and the rare-event
simulation study, where the point is to show what happens to a fitted risk
model when the outcome is a recordable injury rather than a survey item.

The King & Zeng (2001) small-sample bias correction is implemented because the
whole rare-event section of this repository turns on the fact that ordinary
maximum-likelihood logistic regression underestimates the probability of rare
events. Having the correction available makes that claim checkable rather than
rhetorical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .linalg import safe_inverse
from .special import norm_cdf

__all__ = ["LogisticFit", "logistic_irls", "king_zeng_correction", "predict_proba"]


def _sigmoid(eta: np.ndarray) -> np.ndarray:
    """Numerically stable logistic function."""
    out = np.empty_like(eta, dtype=float)
    pos = eta >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-eta[pos]))
    exp_eta = np.exp(eta[~pos])
    out[~pos] = exp_eta / (1.0 + exp_eta)
    return out


@dataclass
class LogisticFit:
    """Result of a logistic regression fit.

    Attributes
 ----------
    coef
        Estimated coefficients, including the intercept in position 0 when one
        was requested.
    se
        Asymptotic standard errors from the inverse observed information.
    loglik
        Maximized log-likelihood.
    n_obs, n_events
        Sample size and number of positive outcomes. Reported together because
        the second, not the first, governs how much a rare-event model can
        actually learn.
    converged, n_iter
        IRLS convergence status.
    separation_warning
        Set when the fitted probabilities are numerically 0 or 1 for every
        observation in some region, which indicates (quasi-)complete
        separation. The coefficients are then not finite in the limit and the
        standard errors are meaningless.
    """

    coef: np.ndarray
    se: np.ndarray
    loglik: float
    n_obs: int
    n_events: int
    converged: bool
    n_iter: int
    separation_warning: Optional[str] = None
    ridge: float = 0.0
    warnings: list = field(default_factory=list)

    def z_values(self) -> np.ndarray:
        """Wald z statistics. Meaningless if ``separation_warning`` is set."""
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.where(self.se > 0, self.coef / self.se, np.nan)

    def p_values(self) -> np.ndarray:
        """Two-sided Wald p-values."""
        z = self.z_values()
        return np.array(
            [
                np.nan if not np.isfinite(v) else 2.0 * (1.0 - norm_cdf(abs(float(v))))
                for v in z
            ]
        )


def logistic_irls(
    x: np.ndarray,
    y: np.ndarray,
    add_intercept: bool = True,
    ridge: float = 0.0,
    max_iter: int = 100,
    tol: float = 1e-10,
) -> LogisticFit:
    """Fit a logistic regression by IRLS (Fisher scoring).

    Parameters
 ----------
    x
        Design matrix of shape ``(n_obs, n_pred)``, without an intercept
        column unless ``add_intercept`` is False.
    y
        Binary outcome, coded 0/1.
    add_intercept
        Prepend a column of ones.
    ridge
        Optional L2 penalty on the non-intercept coefficients. A small ridge
        keeps the algorithm finite under separation; when it is non-zero the
        standard errors are conditional on the penalty and should not be
        interpreted as unpenalized asymptotic errors.
    max_iter, tol
        Convergence controls on the change in the coefficient vector.
    """
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(-1, 1)
    y_arr = np.asarray(y, dtype=float).ravel()
    if y_arr.shape[0] != x_arr.shape[0]:
        raise ValueError("x and y have different numbers of rows")
    unique = np.unique(y_arr)
    if not np.all(np.isin(unique, (0.0, 1.0))):
        raise ValueError("y must be coded 0/1")

    if add_intercept:
        design = np.hstack([np.ones((x_arr.shape[0], 1)), x_arr])
    else:
        design = x_arr

    n_obs, n_par = design.shape
    beta = np.zeros(n_par)
    penalty = np.eye(n_par) * ridge
    if add_intercept and ridge > 0:
        penalty[0, 0] = 0.0

    converged = False
    n_iter = 0
    warnings: list = []
    for n_iter in range(1, max_iter + 1):
        eta = design @ beta
        mu = _sigmoid(eta)
        w = np.clip(mu * (1.0 - mu), 1e-12, None)
        z = eta + (y_arr - mu) / w
        wx = design * w[:, None]
        hessian = design.T @ wx + penalty
        rhs = wx.T @ z
        inv_hess, warn = safe_inverse(hessian)
        if warn is not None:
            warnings.append(f"iteration {n_iter}: {warn}")
        new_beta = inv_hess @ rhs
        delta = float(np.max(np.abs(new_beta - beta)))
        beta = new_beta
        if delta < tol:
            converged = True
            break

    eta = design @ beta
    mu = _sigmoid(eta)
    eps = 1e-12
    loglik = float(
        np.sum(y_arr * np.log(np.clip(mu, eps, 1.0)) + (1 - y_arr) * np.log(np.clip(1 - mu, eps, 1.0)))
    )

    w = np.clip(mu * (1.0 - mu), 1e-12, None)
    information = design.T @ (design * w[:, None]) + penalty
    cov, warn = safe_inverse(information)
    if warn is not None:
        warnings.append(warn)
    se = np.sqrt(np.clip(np.diag(cov), 0.0, None))

    separation = None
    if np.max(np.abs(beta)) > 25.0 or float(np.min(w)) < 1e-10:
        separation = (
            "fitted probabilities are numerically 0 or 1 for some observations; "
            "this indicates quasi-complete separation. Coefficients and standard "
            "errors from this fit should not be interpreted."
        )

    return LogisticFit(
        coef=beta,
        se=se,
        loglik=loglik,
        n_obs=n_obs,
        n_events=int(np.sum(y_arr)),
        converged=converged,
        n_iter=n_iter,
        separation_warning=separation,
        ridge=ridge,
        warnings=warnings,
    )


def king_zeng_correction(
    x: np.ndarray, y: np.ndarray, fit: LogisticFit, add_intercept: bool = True
) -> np.ndarray:
    """Apply the King & Zeng (2001) rare-event bias correction to a fit.

    The correction subtracts an estimate of the O(1/n) bias of the ML
    estimator:

    ``bias = (X' W X)^-1 X' W xi``, with ``xi_i = 0.5 * Q_ii * ((1 + w1) p_i - w1)``

    where ``Q = X (X' W X)^-1 X'`` and ``w1 = 1`` in the no-prior-correction
    case used here.

    Returns the corrected coefficient vector. The reduction in bias is small in
    absolute terms but concentrated in the intercept, which is exactly the
    parameter that sets the level of the predicted probabilities -- so it moves
    calibration, not discrimination.

    Citation: King, G., & Zeng, L. (2001). Logistic regression in rare events
    data. *Political Analysis*, 9(2), 137-163. Verified against the journal
    record and the authors' copy of the article.
    """
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(-1, 1)
    if add_intercept:
        design = np.hstack([np.ones((x_arr.shape[0], 1)), x_arr])
    else:
        design = x_arr

    eta = design @ fit.coef
    p = _sigmoid(eta)
    w = np.clip(p * (1.0 - p), 1e-12, None)
    wx = design * w[:, None]
    information = design.T @ wx
    inv_info, _ = safe_inverse(information)
    q_diag = np.einsum("ij,jk,ik->i", design, inv_info, design)
    xi = 0.5 * q_diag * (2.0 * p - 1.0)
    bias = inv_info @ (design.T @ (w * xi))
    return fit.coef - bias


def predict_proba(
    x: np.ndarray, coef: np.ndarray, add_intercept: bool = True
) -> np.ndarray:
    """Predicted probabilities for a fitted logistic model."""
    x_arr = np.asarray(x, dtype=float)
    if x_arr.ndim == 1:
        x_arr = x_arr.reshape(-1, 1)
    if add_intercept:
        x_arr = np.hstack([np.ones((x_arr.shape[0], 1)), x_arr])
    return _sigmoid(x_arr @ np.asarray(coef, dtype=float))
