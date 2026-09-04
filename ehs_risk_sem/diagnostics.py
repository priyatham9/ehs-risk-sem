"""Assumption checks, stated as tests rather than left implicit.

Each function here corresponds to an assumption that the estimator in
:mod:`ehs_risk_sem.model` actually relies on. The intent is that a fitted model
comes with a list of assumptions and a measured status for each, rather than a
fit table and silence.

Two of these deserve a note.

``harman_single_factor`` is included and immediately labelled as inadequate.
It is the check most often cited in organizational research as evidence against
common method bias, and it does not work: a single factor accounting for less
than half the variance is not evidence that method variance is absent
(Podsakoff, MacKenzie, Lee & Podsakoff 2003; Podsakoff et al. 2024). It is
implemented so that a user who is going to report it anyway sees the warning
attached to the number.

``formative_indicator_check`` is a heuristic, not a test. Whether a construct
is reflective or formative is a question about what causes what, and no
correlation pattern settles it. What the function can do is flag the specific
symptom: a block whose indicators barely correlate with each other is unlikely
to be a reflective construct, and modelling it as one biases the structural
estimates (Jarvis, MacKenzie & Podsakoff 2003; MacKenzie, Podsakoff & Jarvis
2005).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd

from .linalg import corr_matrix, safe_inverse, standardize
from .measurement import principal_axis_factor
from .special import chi2_sf, norm_cdf

__all__ = [
    "AssumptionReport",
    "bartlett_sphericity",
    "kmo",
    "mardia",
    "vif",
    "harman_single_factor",
    "formative_indicator_check",
    "linearity_check",
    "run_all_checks",
]


def bartlett_sphericity(data: np.ndarray) -> dict:
    """Bartlett's test that the correlation matrix is an identity matrix.

    A necessary precondition for factor analysis to be sensible: if the
    indicators are mutually uncorrelated there is no common factor to extract.
    Rejecting the null is a very low bar and passing it proves nothing.
    """
    arr = standardize(np.asarray(data, dtype=float))
    n, p = arr.shape
    r = corr_matrix(arr)
    sign, logdet_r = np.linalg.slogdet(r)
    if sign <= 0:
        return {
            "chi_square": float("nan"),
            "df": p * (p - 1) // 2,
            "p_value": float("nan"),
            "note": "correlation matrix is singular; the statistic is undefined",
        }
    stat = -((n - 1) - (2 * p + 5) / 6.0) * logdet_r
    df = p * (p - 1) // 2
    return {
        "chi_square": float(stat),
        "df": int(df),
        "p_value": float(chi2_sf(stat, df)),
        "note": "",
    }


def kmo(data: np.ndarray) -> dict:
    """Kaiser-Meyer-Olkin measure of sampling adequacy.

    The ratio of squared correlations to squared correlations plus squared
    partial correlations, overall and per variable. Low values mean the
    correlations between variables are largely explained by other variables,
    so a common-factor model is a poor description.
    """
    arr = standardize(np.asarray(data, dtype=float))
    r = corr_matrix(arr)
    inv, warn = safe_inverse(r)
    d = np.sqrt(np.diag(inv))
    partial = -inv / np.outer(d, d)
    np.fill_diagonal(partial, 0.0)
    r_off = r.copy()
    np.fill_diagonal(r_off, 0.0)
    num = float(np.sum(r_off**2))
    den = num + float(np.sum(partial**2))
    overall = num / den if den > 0 else float("nan")
    per_var = []
    for j in range(r.shape[0]):
        a = float(np.sum(r_off[j] ** 2))
        b = float(np.sum(partial[j] ** 2))
        per_var.append(a / (a + b) if (a + b) > 0 else float("nan"))
    return {"overall": overall, "per_variable": np.array(per_var), "warning": warn}


def mardia(data: np.ndarray) -> dict:
    """Mardia's multivariate skewness and kurtosis tests.

    The chi-square reference distribution used by every fit index in this
    package assumes multivariate normal indicators. Departures inflate the
    chi-square and deflate the standard errors, so a rejected normality test
    means the reported fit statistics are optimistic about the model and the
    standard errors are optimistic about precision.
    """
    arr = np.asarray(data, dtype=float)
    n, p = arr.shape
    centred = arr - arr.mean(axis=0, keepdims=True)
    cov = centred.T @ centred / n
    inv_cov, _ = safe_inverse(cov)
    mahal = centred @ inv_cov @ centred.T

    b1 = float(np.sum(mahal**3) / (n**2))
    skew_stat = n * b1 / 6.0
    skew_df = p * (p + 1) * (p + 2) / 6.0

    b2 = float(np.mean(np.diag(mahal) ** 2))
    expected = p * (p + 2)
    var_b2 = 8.0 * p * (p + 2) / n
    kurt_z = (b2 - expected) / np.sqrt(var_b2)

    return {
        "skewness_b1p": b1,
        "skewness_chi_square": float(skew_stat),
        "skewness_df": float(skew_df),
        "skewness_p": float(chi2_sf(skew_stat, skew_df)),
        "kurtosis_b2p": b2,
        "kurtosis_expected": float(expected),
        "kurtosis_z": float(kurt_z),
        "kurtosis_p": float(2.0 * (1.0 - norm_cdf(abs(kurt_z)))),
    }


def vif(data: np.ndarray, names: Sequence[str]) -> pd.DataFrame:
    """Variance inflation factor for each column against all the others."""
    arr = standardize(np.asarray(data, dtype=float))
    r = corr_matrix(arr)
    inv, _ = safe_inverse(r)
    return pd.DataFrame(
        {"variable": list(names), "vif": np.clip(np.diag(inv), 0.0, None)}
    )


def harman_single_factor(data: np.ndarray) -> dict:
    """Share of variance explained by the first unrotated factor.

    Reported with its own refutation attached. This is not a valid test for
    common method bias: it has low power, no defensible threshold, and passing
    it does not rule out method variance. If method effects are a concern, the
    design must address them; a post-hoc statistic cannot.
    """
    arr = standardize(np.asarray(data, dtype=float))
    r = corr_matrix(arr)
    loadings, _, _, _, _ = principal_axis_factor(r, n_factors=1)
    explained = float(np.sum(loadings[:, 0] ** 2) / r.shape[0])
    return {
        "first_factor_variance_share": explained,
        "note": (
            "Harman's single-factor test is not a valid test of common method "
            "bias. It is reported only because reviewers ask for it. A value "
            "below 0.50 is not evidence that method variance is absent "
            "(Podsakoff et al. 2003; Podsakoff et al. 2024)."
        ),
    }


def formative_indicator_check(
    data: np.ndarray, blocks: Dict[str, Sequence[int]], threshold: float = 0.30
) -> pd.DataFrame:
    """Flag blocks whose indicators are too weakly intercorrelated to be reflective.

    A reflective measurement model says the latent variable causes its
    indicators, so the indicators must correlate. A formative construct is
    defined by its indicators -- equipment age, overdue preventive maintenance
    and alarm rate jointly constitute "system condition" and need not correlate
    at all. Modelling the second as though it were the first is a specification
    error with substantial bias in the structural parameters.

    This function reports the mean absolute within-block correlation. Below
    ``threshold`` it flags the block. That flag is a prompt to think about the
    direction of measurement, not a verdict: the question is causal and cannot
    be answered from a correlation matrix.
    """
    arr = standardize(np.asarray(data, dtype=float))
    r = corr_matrix(arr)
    rows = []
    for name, idx in blocks.items():
        idx_list = list(idx)
        if len(idx_list) < 2:
            continue
        sub = r[np.ix_(idx_list, idx_list)]
        off = np.abs(sub[np.triu_indices(len(idx_list), k=1)])
        mean_r = float(np.mean(off))
        rows.append(
            {
                "latent": name,
                "n_indicators": len(idx_list),
                "mean_abs_within_correlation": mean_r,
                "min_abs_within_correlation": float(np.min(off)),
                "possibly_formative": mean_r < threshold,
            }
        )
    return pd.DataFrame(rows)


def linearity_check(x: np.ndarray, y: np.ndarray) -> dict:
    """Incremental variance explained by a quadratic term over a linear fit.

    The structural model is linear in the latent variables. If a squared term
    adds materially to the fit, the linear coefficient is a summary of a
    nonlinear relation and should not be read as a constant effect.
    """
    x_arr = np.asarray(x, dtype=float).ravel()
    y_arr = np.asarray(y, dtype=float).ravel()
    xz = (x_arr - x_arr.mean()) / x_arr.std(ddof=1)
    yz = (y_arr - y_arr.mean()) / y_arr.std(ddof=1)

    def r2(design: np.ndarray) -> float:
        d = np.hstack([np.ones((design.shape[0], 1)), design])
        coef, *_ = np.linalg.lstsq(d, yz, rcond=None)
        resid = yz - d @ coef
        return float(1.0 - np.sum(resid**2) / np.sum((yz - yz.mean()) ** 2))

    r2_lin = r2(xz.reshape(-1, 1))
    r2_quad = r2(np.column_stack([xz, xz**2]))
    return {
        "r2_linear": r2_lin,
        "r2_quadratic": r2_quad,
        "increment": r2_quad - r2_lin,
        "flag": (r2_quad - r2_lin) > 0.01,
    }


@dataclass
class AssumptionReport:
    """Collected assumption checks with a plain-language status for each."""

    checks: Dict[str, dict] = field(default_factory=dict)
    tables: Dict[str, pd.DataFrame] = field(default_factory=dict)
    flags: List[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = ["Assumption checks"]
        b = self.checks.get("bartlett", {})
        if b:
            lines.append(
                f"  Bartlett sphericity : chi2({b['df']}) = {b['chi_square']:.1f}, "
                f"p = {b['p_value']:.3g}"
            )
        k = self.checks.get("kmo", {})
        if k:
            lines.append(f"  KMO overall         : {k['overall']:.3f}")
        m = self.checks.get("mardia", {})
        if m:
            lines.append(
                f"  Mardia skewness     : chi2 = {m['skewness_chi_square']:.1f}, "
                f"p = {m['skewness_p']:.3g}"
            )
            lines.append(
                f"  Mardia kurtosis     : z = {m['kurtosis_z']:.2f}, "
                f"p = {m['kurtosis_p']:.3g}"
            )
        h = self.checks.get("harman", {})
        if h:
            lines.append(
                f"  Harman first factor : {h['first_factor_variance_share']:.3f} "
                "of variance (not a valid CMB test)"
            )
        for f in self.flags:
            lines.append(f"  FLAG    : {f}")
        return "\n".join(lines)


def run_all_checks(
    data: pd.DataFrame, blocks: Dict[str, Sequence[str]]
) -> AssumptionReport:
    """Run every assumption check for a model specification.

    Parameters
 ----------
    data
        Wide DataFrame of indicators.
    blocks
        Mapping from latent name to indicator column names.
    """
    indicators: List[str] = []
    for inds in blocks.values():
        indicators.extend(list(inds))
    arr = data[indicators].dropna().to_numpy(dtype=float)
    idx_blocks = {}
    offset = 0
    for name, inds in blocks.items():
        idx_blocks[name] = list(range(offset, offset + len(list(inds))))
        offset += len(list(inds))

    report = AssumptionReport()
    report.checks["bartlett"] = bartlett_sphericity(arr)
    report.checks["kmo"] = kmo(arr)
    report.checks["mardia"] = mardia(arr)
    report.checks["harman"] = harman_single_factor(arr)
    report.tables["vif"] = vif(arr, indicators)
    report.tables["formative"] = formative_indicator_check(arr, idx_blocks)

    if report.checks["kmo"]["overall"] < 0.60:
        report.flags.append(
            f"KMO is {report.checks['kmo']['overall']:.3f}. The correlation "
            "structure is not well suited to common-factor analysis."
        )
    if report.checks["mardia"]["kurtosis_p"] < 0.05:
        report.flags.append(
            "Mardia's kurtosis test rejects multivariate normality. The "
            "chi-square and its p-value are inflated and the standard errors "
            "are understated; consider a robust correction, which this "
            "package does not implement."
        )
    formative = report.tables["formative"]
    if not formative.empty:
        for _, row in formative[formative["possibly_formative"]].iterrows():
            report.flags.append(
                f"'{row['latent']}' has mean within-block correlation "
                f"{row['mean_abs_within_correlation']:.3f}. If its indicators "
                "constitute the construct rather than reflect it, the "
                "reflective model fitted here is misspecified."
            )
    high_vif = report.tables["vif"]
    for _, row in high_vif[high_vif["vif"] > 10].iterrows():
        report.flags.append(
            f"indicator '{row['variable']}' has VIF {row['vif']:.1f}; it is "
            "nearly collinear with the others."
        )
    return report
