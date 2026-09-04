"""Data-generating processes for the simulation studies.

Every dataset produced here is synthetic and generated from a known
ground-truth model. That is the point: a simulation study is the only way to
ask whether an estimator recovers a parameter, because it is the only setting
in which the parameter is known.

Nothing generated here is an empirical finding about workplace injuries, and no
number produced from it should be presented as one. The one real-world quantity
used to choose a realistic base rate is the US private-industry total
recordable case rate, cited and sourced at its definition in
:mod:`ehs_risk_sem.rare_events`.

The generating model is

``eta = B eta + Gamma xi + zeta``,  ``x = Lambda eta + delta``

restricted to the recursive case, which is what the estimator in
:mod:`ehs_risk_sem.model` supports. Latents are standardized, so ``Lambda``
entries are correlations between indicator and factor and disturbance variances
are ``1 - R^2``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "TrueModel",
    "make_four_factor_model",
    "generate_latent",
    "generate_indicators",
    "generate_dataset",
    "generate_binary_outcome",
    "generate_zero_inflated_count",
    "generate_formative_block",
]


@dataclass
class TrueModel:
    """A known generating model.

    Attributes
 ----------
    latent_names
        Ordered latent variable names.
    loadings
        Mapping from latent name to the vector of standardized loadings for its
        indicators.
    beta
        ``(k, k)`` matrix of direct effects; ``beta[i, j]`` is the effect of
        latent ``j`` on latent ``i``. Must be acyclic.
    exogenous_corr
        Correlation matrix among the exogenous latents, in the order they
        appear in ``latent_names``.
    indicator_names
        Mapping from latent name to indicator column names.
    """

    latent_names: List[str]
    loadings: Dict[str, np.ndarray]
    beta: np.ndarray
    exogenous_corr: np.ndarray
    indicator_names: Dict[str, List[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.indicator_names:
            self.indicator_names = {
                name: [f"{name.lower()}_{i+1}" for i in range(len(self.loadings[name]))]
                for name in self.latent_names
            }

    def exogenous(self) -> List[str]:
        endo = {self.latent_names[i] for i in range(len(self.latent_names)) if np.any(self.beta[i] != 0)}
        return [n for n in self.latent_names if n not in endo]

    def endogenous(self) -> List[str]:
        return [n for n in self.latent_names if n not in self.exogenous()]

    def paths(self) -> Dict[str, List[str]]:
        """Structural paths in the form :class:`ehs_risk_sem.model.ModelSpec` expects."""
        out: Dict[str, List[str]] = {}
        for i, target in enumerate(self.latent_names):
            preds = [
                self.latent_names[j]
                for j in range(len(self.latent_names))
                if self.beta[i, j] != 0
            ]
            if preds:
                out[target] = preds
        return out

    def true_beta_for(self, outcome: str) -> Dict[str, float]:
        i = self.latent_names.index(outcome)
        return {
            self.latent_names[j]: float(self.beta[i, j])
            for j in range(len(self.latent_names))
            if self.beta[i, j] != 0
        }

    def implied_latent_correlation(self) -> np.ndarray:
        """Population latent correlation matrix implied by ``beta`` and ``exogenous_corr``."""
        k = len(self.latent_names)
        exo = self.exogenous()
        exo_idx = [self.latent_names.index(n) for n in exo]
        psi = np.zeros((k, k))
        psi[np.ix_(exo_idx, exo_idx)] = self.exogenous_corr
        inv_ib = np.linalg.inv(np.eye(k) - self.beta)
        # Solve for disturbance variances that leave every latent standardized.
        for i, name in enumerate(self.latent_names):
            if name in exo:
                continue
            row = inv_ib[i].copy()
            row[i] = 0.0
            explained = float(row @ psi @ row.T)
            psi[i, i] = max(1.0 - explained, 1e-8)
        phi = inv_ib @ psi @ inv_ib.T
        return 0.5 * (phi + phi.T)


def make_four_factor_model(
    beta: Sequence[float] = (0.45, 0.30, -0.25, -0.20),
    exogenous_correlation: float = 0.35,
    loading: float = 0.75,
    n_indicators: int = 3,
    names: Optional[Sequence[str]] = None,
) -> TrueModel:
    """The four-predictor safety model used throughout the simulation studies.

    Four exogenous latents predict one endogenous latent. The default
    coefficients are the illustrative values ``0.45, 0.30, -0.25, -0.20`` that
    circulate in practitioner writing on safety risk scoring. They are used
    here as a *generating* model so that the studies can ask what would have to
    be true for an estimator to return them, and what an analyst could and
    could not conclude if it did.

    Note the arithmetic that makes those particular values suspicious as
    estimates: their absolute values sum to exactly 1.20 and every one lands on
    a round twentieth. Estimated standardized coefficients over correlated
    predictors have no reason to do that.
    """
    if names is None:
        names = (
            "UnsafeActs",
            "OperationalStress",
            "SystemCondition",
            "SafetyResponseCapability",
            "Risk",
        )
    names = list(names)
    if len(names) != 5:
        raise ValueError("expected four predictor names plus one outcome name")
    if len(beta) != 4:
        raise ValueError("expected four path coefficients")

    k = 5
    b = np.zeros((k, k))
    for j in range(4):
        b[4, j] = float(beta[j])

    exo_corr = np.full((4, 4), float(exogenous_correlation))
    np.fill_diagonal(exo_corr, 1.0)

    loadings = {n: np.full(n_indicators, float(loading)) for n in names}
    model = TrueModel(
        latent_names=names, loadings=loadings, beta=b, exogenous_corr=exo_corr
    )

    r2 = float(np.array(beta) @ exo_corr @ np.array(beta))
    if r2 >= 1.0:
        raise ValueError(
            f"the requested coefficients imply R^2 = {r2:.3f} >= 1 at "
            f"exogenous correlation {exogenous_correlation}. No standardized "
            "model can produce them."
        )
    return model


def generate_latent(model: TrueModel, n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw ``n`` observations of the latent variables from the generating model.

    Returns an ``(n, k)`` array whose columns are standardized in population
    (sample means and variances will vary).
    """
    k = len(model.latent_names)
    exo = model.exogenous()
    exo_idx = [model.latent_names.index(n_) for n_ in exo]

    eta = np.zeros((n, k))
    chol = np.linalg.cholesky(model.exogenous_corr)
    eta[:, exo_idx] = rng.standard_normal((n, len(exo_idx))) @ chol.T

    phi = model.implied_latent_correlation()
    order = _topological_order(model.beta)
    for i in order:
        name = model.latent_names[i]
        if name in exo:
            continue
        preds = np.where(model.beta[i] != 0)[0]
        explained = eta[:, preds] @ model.beta[i, preds]
        var_explained = float(
            model.beta[i, preds] @ phi[np.ix_(preds, preds)] @ model.beta[i, preds]
        )
        disturbance_sd = np.sqrt(max(1.0 - var_explained, 1e-8))
        eta[:, i] = explained + disturbance_sd * rng.standard_normal(n)
    return eta


def _topological_order(beta: np.ndarray) -> List[int]:
    """Indices ordered so every predictor precedes its outcome."""
    k = beta.shape[0]
    remaining = set(range(k))
    order: List[int] = []
    while remaining:
        ready = [i for i in remaining if not (set(np.where(beta[i] != 0)[0]) & remaining)]
        if not ready:
            raise ValueError("beta matrix contains a cycle")
        for i in ready:
            order.append(i)
            remaining.discard(i)
    return order


def generate_indicators(
    model: TrueModel, eta: np.ndarray, rng: np.random.Generator
) -> pd.DataFrame:
    """Generate observed indicators from latent scores and the loading matrix."""
    cols: Dict[str, np.ndarray] = {}
    for j, name in enumerate(model.latent_names):
        lam = np.asarray(model.loadings[name], dtype=float)
        uniq = np.sqrt(np.clip(1.0 - lam**2, 1e-8, None))
        for m, ind_name in enumerate(model.indicator_names[name]):
            cols[ind_name] = lam[m] * eta[:, j] + uniq[m] * rng.standard_normal(eta.shape[0])
    return pd.DataFrame(cols)


def generate_dataset(
    model: TrueModel, n: int, seed: Optional[int] = None
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Generate one synthetic dataset.

    Returns the observed indicator DataFrame and the ``(n, k)`` latent scores.
    The latent scores are returned so that studies can compare an estimator
    against the truth; they are never available in real data, and every study
    that uses them says so.
    """
    rng = np.random.default_rng(seed)
    eta = generate_latent(model, n, rng)
    return generate_indicators(model, eta, rng), eta


def generate_binary_outcome(
    latent: np.ndarray,
    coefficients: Sequence[float],
    base_rate: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a rare binary outcome from latent predictors.

    The intercept is solved numerically so the realized marginal probability
    matches ``base_rate``, which lets a study hold the base rate fixed while
    varying the strength of the predictors.

    Returns ``(y, p_true)``: the realized 0/1 outcome and the true probability
    for each observation. Having ``p_true`` makes it possible to distinguish
    "the model is miscalibrated" from "the outcome is noisy", which is
    impossible with real data.
    """
    x = np.asarray(latent, dtype=float)
    coefs = np.asarray(coefficients, dtype=float)
    lp = x @ coefs

    lo, hi = -30.0, 30.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        p = 1.0 / (1.0 + np.exp(-(mid + lp)))
        if float(np.mean(p)) < base_rate:
            lo = mid
        else:
            hi = mid
    intercept = 0.5 * (lo + hi)
    p_true = 1.0 / (1.0 + np.exp(-(intercept + lp)))
    y = (rng.random(x.shape[0]) < p_true).astype(float)
    return y, p_true


def generate_zero_inflated_count(
    latent: np.ndarray,
    count_coefficients: Sequence[float],
    zero_coefficients: Sequence[float],
    base_mean: float,
    zero_inflation: float,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a zero-inflated Poisson count outcome.

    Two processes: a structural-zero process (an establishment that would report
    zero whatever happened, whether because nothing happened or because nothing
    was recorded) and a Poisson count process. The observed zeros are a mixture
    of the two, which is exactly why a spike at zero in reported injury counts
    cannot be read directly as an absence of injuries.

    Returns ``(counts, mu, p_structural_zero)``.
    """
    x = np.asarray(latent, dtype=float)
    lp_count = x @ np.asarray(count_coefficients, dtype=float)
    lp_zero = x @ np.asarray(zero_coefficients, dtype=float)

    mu = base_mean * np.exp(lp_count - float(np.mean(lp_count)))
    logit_base = np.log(zero_inflation / (1 - zero_inflation)) if 0 < zero_inflation < 1 else 0.0
    p_zero = 1.0 / (1.0 + np.exp(-(logit_base + lp_zero - float(np.mean(lp_zero)))))

    structural = rng.random(x.shape[0]) < p_zero
    counts = rng.poisson(mu).astype(float)
    counts[structural] = 0.0
    return counts, mu, p_zero


def generate_formative_block(
    n: int,
    n_indicators: int,
    weights: Sequence[float],
    rng: np.random.Generator,
    indicator_correlation: float = 0.05,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Generate a formative construct and its indicators.

    The indicators are drawn first and the construct is formed from them, so
    the causal direction runs indicator to construct. The indicators are only
    weakly correlated with one another, which is normal for a formative
    construct and fatal for a reflective measurement model fitted to them.

    Returns the construct scores and the indicator DataFrame.
    """
    corr = np.full((n_indicators, n_indicators), float(indicator_correlation))
    np.fill_diagonal(corr, 1.0)
    chol = np.linalg.cholesky(corr)
    x = rng.standard_normal((n, n_indicators)) @ chol.T
    w = np.asarray(weights, dtype=float)
    construct = x @ w
    construct = (construct - construct.mean()) / construct.std(ddof=1)
    frame = pd.DataFrame(
        {f"form_{i+1}": x[:, i] for i in range(n_indicators)}
    )
    return construct, frame
