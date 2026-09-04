"""Calibration assessment for probability predictions.

A latent "risk score" is not a risk. It has no probability scale, no link
function, no exposure denominator and no time window, so its calibration is not
poor -- it is undefined. This module exists for the step after that gap is
closed: once a model produces ``P(event | unit, window)``, these are the
measures that say whether the number means what it claims to mean.

Discrimination (AUC) is included, but it is not the headline. At the base rates
that occupational injury prediction actually operates at, a model can have
excellent AUC and still be useless or harmful for decisions; see
:mod:`ehs_risk_sem.rare_events` for the arithmetic and Van Calster et al.
(2019) for the argument.

Implemented:

* Brier score and its reliability / resolution / uncertainty partition;
* reliability diagram data on equal-width or equal-count bins;
* calibration-in-the-large, calibration intercept and calibration slope, which
  are the weak-calibration level of the Van Calster et al. (2016) hierarchy;
* expected and maximum calibration error;
* AUC by the rank formula, with ties handled;
* net benefit and decision curves, so a model can be judged at the thresholds
  an organization would actually act on rather than in the abstract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .glm import logistic_irls

__all__ = [
    "CalibrationResult",
    "brier_score",
    "brier_decomposition",
    "reliability_diagram",
    "calibration_in_the_large",
    "calibration_intercept_slope",
    "expected_calibration_error",
    "auc",
    "confusion_at_threshold",
    "net_benefit",
    "decision_curve",
    "assess_calibration",
]

_EPS = 1e-12


def _validate(y: np.ndarray, p: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    y_arr = np.asarray(y, dtype=float).ravel()
    p_arr = np.asarray(p, dtype=float).ravel()
    if y_arr.shape != p_arr.shape:
        raise ValueError("y and p must have the same length")
    if not np.all(np.isin(np.unique(y_arr), (0.0, 1.0))):
        raise ValueError("y must be coded 0/1")
    if np.any(p_arr < 0) or np.any(p_arr > 1):
        raise ValueError("p must lie in [0, 1]")
    return y_arr, p_arr


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    """Mean squared error of probability predictions."""
    y_arr, p_arr = _validate(y, p)
    return float(np.mean((p_arr - y_arr) ** 2))


def brier_decomposition(
    y: np.ndarray, p: np.ndarray, n_bins: int = 10, strategy: str = "quantile"
) -> dict:
    """Partition the Brier score into reliability, resolution and uncertainty.

    ``Brier = reliability - resolution + uncertainty``

    * reliability: mean squared gap between predicted probability and observed
      frequency within a bin. Lower is better; zero means perfectly calibrated
      at this binning.
    * resolution: how far bin frequencies depart from the base rate. Higher is
      better; zero means the model separates nothing.
    * uncertainty: ``base_rate * (1 - base_rate)``, a property of the outcome,
      not of the model. At a base rate near zero it dominates, which is why a
      very low Brier score at a low base rate says almost nothing.

    The partition is binning-dependent, and the residual returned as
    ``identity_gap`` reports how far the three terms are from reproducing the
    Brier score exactly.

    Citation: Murphy, A. H. (1973). A new vector partition of the probability
    score. *Journal of Applied Meteorology*, 12(4), 595-600. Verified against
    the journal record; that article is where the reliability / resolution /
    uncertainty partition of the Brier score originates.
    """
    y_arr, p_arr = _validate(y, p)
    n = y_arr.size
    base = float(np.mean(y_arr))
    edges = _bin_edges(p_arr, n_bins, strategy)
    idx = np.clip(np.digitize(p_arr, edges[1:-1], right=True), 0, len(edges) - 2)

    reliability = 0.0
    resolution = 0.0
    for b in range(len(edges) - 1):
        mask = idx == b
        n_b = int(np.sum(mask))
        if n_b == 0:
            continue
        p_bar = float(np.mean(p_arr[mask]))
        o_bar = float(np.mean(y_arr[mask]))
        reliability += n_b * (p_bar - o_bar) ** 2
        resolution += n_b * (o_bar - base) ** 2
    reliability /= n
    resolution /= n
    uncertainty = base * (1.0 - base)
    bs = brier_score(y_arr, p_arr)
    return {
        "brier": bs,
        "reliability": reliability,
        "resolution": resolution,
        "uncertainty": uncertainty,
        "identity_gap": bs - (reliability - resolution + uncertainty),
        "n_bins": len(edges) - 1,
        "strategy": strategy,
        "base_rate": base,
    }


def _bin_edges(p: np.ndarray, n_bins: int, strategy: str) -> np.ndarray:
    if strategy == "uniform":
        return np.linspace(0.0, 1.0, n_bins + 1)
    if strategy == "quantile":
        qs = np.linspace(0.0, 1.0, n_bins + 1)
        edges = np.quantile(p, qs)
        edges[0] = min(edges[0], 0.0)
        edges[-1] = max(edges[-1], 1.0)
        return np.unique(edges)
    raise ValueError("strategy must be 'uniform' or 'quantile'")


def reliability_diagram(
    y: np.ndarray, p: np.ndarray, n_bins: int = 10, strategy: str = "quantile"
) -> pd.DataFrame:
    """Data for a reliability diagram: mean prediction against observed frequency.

    Returns one row per bin with the bin bounds, count, mean predicted
    probability, observed frequency and a Wilson interval for the observed
    frequency. The interval is included because at low base rates a bin can
    easily contain zero events, and a plotted point with no uncertainty
    attached invites overreading.
    """
    y_arr, p_arr = _validate(y, p)
    edges = _bin_edges(p_arr, n_bins, strategy)
    idx = np.clip(np.digitize(p_arr, edges[1:-1], right=True), 0, len(edges) - 2)
    rows = []
    for b in range(len(edges) - 1):
        mask = idx == b
        n_b = int(np.sum(mask))
        if n_b == 0:
            continue
        obs = float(np.mean(y_arr[mask]))
        lo, hi = _wilson_interval(int(np.sum(y_arr[mask])), n_b)
        rows.append(
            {
                "bin": b,
                "lower_edge": float(edges[b]),
                "upper_edge": float(edges[b + 1]),
                "n": n_b,
                "n_events": int(np.sum(y_arr[mask])),
                "mean_predicted": float(np.mean(p_arr[mask])),
                "observed_frequency": obs,
                "obs_ci_low": lo,
                "obs_ci_high": hi,
            }
        )
    return pd.DataFrame(rows)


def _wilson_interval(successes: int, n: int, z: float = 1.959963985) -> Tuple[float, float]:
    """Wilson score interval for a binomial proportion.

    Used instead of the normal approximation because bins at a low base rate
    routinely contain zero or one event, where the normal interval is degenerate.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    phat = successes / n
    denom = 1.0 + z**2 / n
    centre = (phat + z**2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(phat * (1 - phat) / n + z**2 / (4 * n**2))
    return (float(max(centre - half, 0.0)), float(min(centre + half, 1.0)))


def calibration_in_the_large(y: np.ndarray, p: np.ndarray) -> dict:
    """Mean predicted probability against observed event rate.

    The weakest calibration requirement: if these two disagree, the model is
    wrong about the overall level of risk regardless of how well it ranks.
    """
    y_arr, p_arr = _validate(y, p)
    mean_p = float(np.mean(p_arr))
    obs = float(np.mean(y_arr))
    return {
        "mean_predicted": mean_p,
        "observed_rate": obs,
        "difference": mean_p - obs,
        "ratio": mean_p / obs if obs > 0 else float("nan"),
    }


def calibration_intercept_slope(y: np.ndarray, p: np.ndarray) -> dict:
    """Calibration intercept and slope from logistic recalibration.

    The slope comes from regressing the outcome on ``logit(p)``. A slope below
    1 means the predictions are too extreme (overfitted); above 1 means too
    compressed. The intercept is estimated with ``logit(p)`` as an offset, so
    it measures the remaining level shift once the slope is held at 1.

    Perfect calibration is intercept 0 and slope 1.
    """
    y_arr, p_arr = _validate(y, p)
    clipped = np.clip(p_arr, _EPS, 1 - _EPS)
    lp = np.log(clipped / (1 - clipped))

    slope_fit = logistic_irls(lp.reshape(-1, 1), y_arr, add_intercept=True)
    slope = float(slope_fit.coef[1])
    slope_se = float(slope_fit.se[1])

    # Intercept with the linear predictor as a fixed offset: fit an
    # intercept-only logistic model to the residual on the logit scale.
    intercept = _offset_intercept(y_arr, lp)

    return {
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "slope_se": slope_se,
        "slope_converged": slope_fit.converged,
        "separation_warning": slope_fit.separation_warning,
    }


def _offset_intercept(y: np.ndarray, offset: np.ndarray, max_iter: int = 200) -> float:
    """Maximum-likelihood intercept in a logistic model with a fixed offset."""
    a = 0.0
    for _ in range(max_iter):
        eta = a + offset
        mu = 1.0 / (1.0 + np.exp(-np.clip(eta, -500, 500)))
        grad = float(np.sum(y - mu))
        hess = -float(np.sum(mu * (1 - mu)))
        if abs(hess) < 1e-12:
            break
        step = grad / hess
        a_new = a - step
        if abs(a_new - a) < 1e-12:
            a = a_new
            break
        a = a_new
    return float(a)


def expected_calibration_error(
    y: np.ndarray, p: np.ndarray, n_bins: int = 10, strategy: str = "quantile"
) -> dict:
    """Expected and maximum calibration error over bins.

    ECE is the sample-weighted mean absolute gap between predicted probability
    and observed frequency; MCE is the largest such gap. Both depend on the
    binning, which is reported alongside them.
    """
    diagram = reliability_diagram(y, p, n_bins=n_bins, strategy=strategy)
    if diagram.empty:
        return {"ece": float("nan"), "mce": float("nan"), "n_bins": 0}
    gaps = (diagram["mean_predicted"] - diagram["observed_frequency"]).abs()
    weights = diagram["n"] / diagram["n"].sum()
    return {
        "ece": float(np.sum(weights * gaps)),
        "mce": float(np.max(gaps)),
        "n_bins": int(len(diagram)),
        "strategy": strategy,
    }


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """Area under the ROC curve, by the Mann-Whitney rank formula with ties.

    Returns NaN when either class is empty.
    """
    y_arr, p_arr = _validate(y, p)
    pos = p_arr[y_arr == 1]
    neg = p_arr[y_arr == 0]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(p_arr, kind="mergesort")
    ranks = np.empty(p_arr.size, dtype=float)
    sorted_p = p_arr[order]
    i = 0
    while i < sorted_p.size:
        j = i
        while j + 1 < sorted_p.size and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        avg_rank = 0.5 * ((i + 1) + (j + 1))
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1
    r_pos = float(np.sum(ranks[y_arr == 1]))
    n_pos = float(pos.size)
    n_neg = float(neg.size)
    return (r_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def confusion_at_threshold(y: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    """Counts and derived rates when predictions above ``threshold`` trigger action.

    ``alerts_per_true_event`` is included explicitly because it is the number
    an operations manager experiences and the number that decides whether a
    system gets ignored.
    """
    y_arr, p_arr = _validate(y, p)
    flagged = p_arr >= threshold
    tp = int(np.sum(flagged & (y_arr == 1)))
    fp = int(np.sum(flagged & (y_arr == 0)))
    fn = int(np.sum(~flagged & (y_arr == 1)))
    tn = int(np.sum(~flagged & (y_arr == 0)))
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "sensitivity": sens,
        "specificity": spec,
        "ppv": ppv,
        "npv": npv,
        "n_alerts": tp + fp,
        "alerts_per_true_event": (tp + fp) / tp if tp else float("inf"),
    }


def net_benefit(y: np.ndarray, p: np.ndarray, threshold: float) -> float:
    """Net benefit of acting on predictions above ``threshold``.

    ``NB = TP/n - (FP/n) * (t / (1 - t))``

    The threshold encodes the exchange rate between a missed event and a
    needless intervention: acting at threshold ``t`` says one true positive is
    worth ``(1-t)/t`` false positives. Comparing net benefit against the
    treat-all and treat-none strategies is the decision-analytic evaluation
    that Steyerberg et al. (2010) put alongside discrimination and calibration.
    """
    y_arr, p_arr = _validate(y, p)
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be in (0, 1)")
    n = y_arr.size
    flagged = p_arr >= threshold
    tp = float(np.sum(flagged & (y_arr == 1)))
    fp = float(np.sum(flagged & (y_arr == 0)))
    return tp / n - (fp / n) * (threshold / (1.0 - threshold))


def decision_curve(
    y: np.ndarray, p: np.ndarray, thresholds: Optional[np.ndarray] = None
) -> pd.DataFrame:
    """Net benefit of the model, of treating all, and of treating none.

    A model is worth deploying at a given threshold only where its net benefit
    exceeds both reference strategies. At low base rates that window is often
    empty, which is the point of computing it.
    """
    y_arr, p_arr = _validate(y, p)
    if thresholds is None:
        thresholds = np.linspace(0.001, 0.20, 40)
    base = float(np.mean(y_arr))
    rows = []
    for t in np.asarray(thresholds, dtype=float):
        if not 0.0 < t < 1.0:
            continue
        nb_model = net_benefit(y_arr, p_arr, float(t))
        nb_all = base - (1.0 - base) * (t / (1.0 - t))
        rows.append(
            {
                "threshold": float(t),
                "net_benefit_model": nb_model,
                "net_benefit_treat_all": nb_all,
                "net_benefit_treat_none": 0.0,
                "model_is_best": nb_model > max(nb_all, 0.0),
            }
        )
    return pd.DataFrame(rows)


@dataclass
class CalibrationResult:
    """Bundle of calibration, discrimination and decision measures."""

    n_obs: int
    n_events: int
    base_rate: float
    brier: dict
    in_the_large: dict
    intercept_slope: dict
    ece: dict
    auc: float
    diagram: pd.DataFrame
    warnings: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            "Calibration",
            f"  n = {self.n_obs}, events = {self.n_events}, base rate = {self.base_rate:.6f}",
            f"  Brier                 = {self.brier['brier']:.6f}",
            f"    reliability         = {self.brier['reliability']:.6f}  (lower is better)",
            f"    resolution          = {self.brier['resolution']:.6f}  (higher is better)",
            f"    uncertainty         = {self.brier['uncertainty']:.6f}  (outcome property)",
            f"  mean predicted        = {self.in_the_large['mean_predicted']:.6f}",
            f"  observed rate         = {self.in_the_large['observed_rate']:.6f}",
            f"  calibration intercept = {self.intercept_slope['calibration_intercept']:+.4f}"
            "  (0 is perfect)",
            f"  calibration slope     = {self.intercept_slope['calibration_slope']:.4f}"
            "  (1 is perfect)",
            f"  ECE                   = {self.ece['ece']:.6f}",
            f"  AUC                   = {self.auc:.4f}",
        ]
        for w in self.warnings:
            lines.append(f"  WARNING : {w}")
        return "\n".join(lines)


def assess_calibration(
    y: np.ndarray, p: np.ndarray, n_bins: int = 10, strategy: str = "quantile"
) -> CalibrationResult:
    """Run the full calibration assessment and flag rare-event problems."""
    y_arr, p_arr = _validate(y, p)
    n_events = int(np.sum(y_arr))
    base = float(np.mean(y_arr))
    warnings: List[str] = []
    if n_events < 100:
        warnings.append(
            f"only {n_events} events. Calibration cannot be estimated with "
            "useful precision below roughly 100 events, and the reliability "
            "diagram will be dominated by sampling noise."
        )
    if base < 0.01:
        warnings.append(
            f"base rate is {base:.6f}. The Brier score is dominated by the "
            "uncertainty term at this rate and comparing Brier scores across "
            "settings with different base rates is not meaningful."
        )
    return CalibrationResult(
        n_obs=int(y_arr.size),
        n_events=n_events,
        base_rate=base,
        brier=brier_decomposition(y_arr, p_arr, n_bins=n_bins, strategy=strategy),
        in_the_large=calibration_in_the_large(y_arr, p_arr),
        intercept_slope=calibration_intercept_slope(y_arr, p_arr),
        ece=expected_calibration_error(y_arr, p_arr, n_bins=n_bins, strategy=strategy),
        auc=auc(y_arr, p_arr),
        diagram=reliability_diagram(y_arr, p_arr, n_bins=n_bins, strategy=strategy),
        warnings=warnings,
    )
