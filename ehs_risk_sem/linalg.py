"""Matrix helpers used across the estimator.

Nothing here is exotic. It exists so that the numerical behaviour of the
estimator -- in particular how it handles a covariance matrix that is singular
or nearly so -- is written down in one place rather than scattered through the
model code.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

__all__ = [
    "standardize",
    "cov_matrix",
    "corr_matrix",
    "is_symmetric",
    "smallest_eigenvalue",
    "condition_number",
    "is_positive_definite",
    "safe_inverse",
    "nearest_positive_definite",
    "cov_to_corr",
    "vech",
    "logdet",
]


def standardize(x: np.ndarray, ddof: int = 1) -> np.ndarray:
    """Return ``x`` with each column centred and scaled to unit variance.

    Columns with zero variance are returned centred but unscaled, and a
    ``ValueError`` is raised only if every column is constant, because a
    constant indicator is a data problem the caller should see rather than a
    numerical edge case to be smoothed over.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 2:
        raise ValueError("expected a 2-d array of shape (n_obs, n_vars)")
    centred = arr - arr.mean(axis=0, keepdims=True)
    sd = centred.std(axis=0, ddof=ddof, keepdims=True)
    zero = sd == 0.0
    if bool(np.all(zero)):
        raise ValueError("all columns are constant; nothing to standardize")
    sd = np.where(zero, 1.0, sd)
    return centred / sd


def cov_matrix(x: np.ndarray, ddof: int = 1) -> np.ndarray:
    """Sample covariance matrix of the columns of ``x``."""
    arr = np.asarray(x, dtype=float)
    return np.cov(arr, rowvar=False, ddof=ddof)


def corr_matrix(x: np.ndarray) -> np.ndarray:
    """Sample correlation matrix of the columns of ``x``."""
    arr = np.asarray(x, dtype=float)
    return np.corrcoef(arr, rowvar=False)


def cov_to_corr(sigma: np.ndarray) -> np.ndarray:
    """Convert a covariance matrix to the corresponding correlation matrix."""
    s = np.sqrt(np.diag(sigma))
    if np.any(s <= 0):
        raise ValueError("covariance matrix has a non-positive diagonal entry")
    outer = np.outer(s, s)
    return sigma / outer


def is_symmetric(a: np.ndarray, tol: float = 1e-8) -> bool:
    """True if ``a`` is square and symmetric to within ``tol``."""
    arr = np.asarray(a, dtype=float)
    return arr.ndim == 2 and arr.shape[0] == arr.shape[1] and bool(
        np.allclose(arr, arr.T, atol=tol)
    )


def smallest_eigenvalue(a: np.ndarray) -> float:
    """Smallest eigenvalue of a symmetric matrix."""
    return float(np.min(np.linalg.eigvalsh(np.asarray(a, dtype=float))))


def condition_number(a: np.ndarray) -> float:
    """Ratio of largest to smallest eigenvalue of a symmetric matrix.

    Returns ``inf`` for a singular matrix. Used as an empirical-identification
    diagnostic: a very large condition number on the observed covariance matrix
    means the data carry little independent information about some directions
    in parameter space, whatever the algebraic identification rules say.
    """
    eig = np.linalg.eigvalsh(np.asarray(a, dtype=float))
    lo = float(np.min(np.abs(eig)))
    hi = float(np.max(np.abs(eig)))
    if lo == 0.0:
        return float("inf")
    return hi / lo


def is_positive_definite(a: np.ndarray, tol: float = 1e-10) -> bool:
    """True if all eigenvalues of the symmetric matrix ``a`` exceed ``tol``."""
    return smallest_eigenvalue(a) > tol


def safe_inverse(
    a: np.ndarray, ridge: float = 0.0, warn_condition: float = 1e10
) -> Tuple[np.ndarray, Optional[str]]:
    """Invert a symmetric matrix, optionally with a ridge, reporting trouble.

    Returns the inverse and a warning string (or ``None``). The warning is
    returned rather than raised so that a caller running thousands of
    simulation replications can count problem cases instead of aborting.
    """
    arr = np.asarray(a, dtype=float)
    warning: Optional[str] = None
    if ridge > 0.0:
        arr = arr + ridge * np.eye(arr.shape[0])
    cond = condition_number(arr)
    if not np.isfinite(cond) or cond > warn_condition:
        warning = (
            f"matrix is ill-conditioned (condition number {cond:.3e}); "
            "the inverse is numerically unreliable"
        )
        return np.linalg.pinv(arr), warning
    return np.linalg.inv(arr), warning


def nearest_positive_definite(a: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    """Project a symmetric matrix onto the positive-definite cone.

    Eigenvalues below ``floor`` are raised to ``floor``. This is a repair, not
    an estimate; anything using it should say so. In this package it is used
    only when a disattenuated latent correlation matrix comes back indefinite,
    which is itself a finding worth reporting (see
    :mod:`ehs_risk_sem.diagnostics`).
    """
    arr = np.asarray(a, dtype=float)
    arr = 0.5 * (arr + arr.T)
    vals, vecs = np.linalg.eigh(arr)
    vals = np.maximum(vals, floor)
    repaired = vecs @ np.diag(vals) @ vecs.T
    return 0.5 * (repaired + repaired.T)


def vech(a: np.ndarray) -> np.ndarray:
    """Half-vectorization: the lower triangle of a symmetric matrix, including
    the diagonal, as a 1-d array. Its length ``p(p+1)/2`` is the number of
    non-redundant observed moments available to identify a model."""
    arr = np.asarray(a, dtype=float)
    idx = np.tril_indices(arr.shape[0])
    return arr[idx]


def logdet(a: np.ndarray) -> float:
    """Log determinant of a symmetric positive-definite matrix.

    Raises ``ValueError`` when the matrix is not positive definite, because the
    maximum-likelihood discrepancy function is undefined in that case and
    silently returning ``-inf`` would hide a model that has failed.
    """
    sign, value = np.linalg.slogdet(np.asarray(a, dtype=float))
    if sign <= 0:
        raise ValueError("matrix is not positive definite; log determinant undefined")
    return float(value)
