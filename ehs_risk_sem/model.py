"""Model specification, estimation and reporting.

A model is declared as a set of latent variables, each naming its observed
indicators, plus a set of directed structural paths among the latents. Nothing
can be estimated until every latent has indicators and the identification
checks in :mod:`ehs_risk_sem.identification` pass.

Estimation proceeds in the order Anderson & Gerbing (1988) recommend:

1. fit and evaluate each block's measurement model;
2. form composites, disattenuate their correlations into a latent correlation
   matrix;
3. solve the structural paths;
4. rebuild the model-implied covariance matrix and evaluate global fit.

The result object carries its warnings with it. A fitted model with a Heywood
case, an out-of-range disattenuated correlation, or an R-squared above 1 is
returned rather than raised, but it is returned marked, and
:meth:`SEMResults.summary` prints the marks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .fit_indices import FitIndices, compute_fit_indices
from .identification import IdentificationReport, check_identification, count_free_parameters
from .linalg import corr_matrix, is_positive_definite, nearest_positive_definite, safe_inverse, standardize
from .measurement import (
    FactorSolution,
    fit_congeneric_block,
    fornell_larcker,
    htmt,
)
from .structural import PathEstimates, effect_decomposition, latent_correlations, solve_paths

__all__ = ["ModelSpec", "SEMResults", "fit", "bootstrap_paths", "implied_covariance"]


@dataclass(frozen=True)
class ModelSpec:
    """Declaration of a structural equation model.

    Parameters
 ----------
    latents
        Ordered mapping from latent variable name to its indicator column
        names. Order determines the order of rows and columns in ``Phi``.
    paths
        Mapping from each endogenous latent name to the latents that point
        directly into it. Latents absent from this mapping are exogenous.

    Example
 -------
    >>> spec = ModelSpec(
    ...     latents={
    ...         "OperationalStress": ["ot_hours", "backlog", "staffing_gap"],
    ...         "SystemCondition": ["equip_age", "pm_overdue", "alarm_rate"],
    ...         "IncidentRate": ["trir", "dart", "near_miss"],
    ...     },
    ...     paths={"IncidentRate": ["OperationalStress", "SystemCondition"]},
    ... )
    """

    latents: Dict[str, Sequence[str]]
    paths: Dict[str, Sequence[str]] = field(default_factory=dict)

    def latent_names(self) -> List[str]:
        return list(self.latents)

    def endogenous(self) -> List[str]:
        return [name for name in self.latents if name in self.paths]

    def exogenous(self) -> List[str]:
        return [name for name in self.latents if name not in self.paths]

    def all_indicators(self) -> List[str]:
        out: List[str] = []
        for inds in self.latents.values():
            out.extend(list(inds))
        return out

    def n_paths(self) -> int:
        return int(sum(len(list(v)) for v in self.paths.values()))

    def validate(self) -> None:
        """Check internal consistency before any data are touched."""
        seen: Dict[str, str] = {}
        for name, inds in self.latents.items():
            if not list(inds):
                raise ValueError(f"latent '{name}' declares no indicators")
            for ind in inds:
                if ind in seen:
                    raise ValueError(
                        f"indicator '{ind}' is assigned to both '{seen[ind]}' and "
                        f"'{name}'. Cross-loadings are not supported by this "
                        "estimator; each indicator must load on exactly one factor."
                    )
                seen[ind] = name
        for target, sources in self.paths.items():
            if target not in self.latents:
                raise ValueError(f"path target '{target}' is not a declared latent")
            for s in sources:
                if s not in self.latents:
                    raise ValueError(f"path source '{s}' is not a declared latent")

    def topological_order(self) -> List[str]:
        """Latent names ordered so that every predictor precedes its outcome."""
        names = self.latent_names()
        incoming = {n: set(self.paths.get(n, [])) for n in names}
        ordered: List[str] = []
        remaining = set(names)
        while remaining:
            ready = [n for n in names if n in remaining and not (incoming[n] & remaining)]
            if not ready:
                raise ValueError("structural model contains a cycle")
            for n in ready:
                ordered.append(n)
                remaining.discard(n)
        return ordered


def implied_covariance(
    spec: ModelSpec,
    loadings_by_latent: Dict[str, np.ndarray],
    uniquenesses_by_latent: Dict[str, np.ndarray],
    beta_matrix: np.ndarray,
    exogenous_phi: np.ndarray,
    disturbance_variances: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rebuild the model-implied indicator covariance matrix.

    ``Sigma = Lambda Phi Lambda' + Theta`` with
    ``Phi = (I - B)^-1 Psi (I - B)^-T``, where ``Psi`` holds the exogenous
    latent covariances and the endogenous disturbance variances.

    Returns ``(Sigma, Phi)``. ``Phi`` should have a unit diagonal; departures
    from it indicate an improper solution and are reported by :func:`fit`.
    """
    names = spec.latent_names()
    k = len(names)
    indicators = spec.all_indicators()
    p = len(indicators)

    lam = np.zeros((p, k))
    theta = np.zeros(p)
    row = 0
    for j, name in enumerate(names):
        block = list(spec.latents[name])
        lam[row : row + len(block), j] = loadings_by_latent[name]
        theta[row : row + len(block)] = uniquenesses_by_latent[name]
        row += len(block)

    exo = spec.exogenous()
    exo_idx = [names.index(n) for n in exo]
    psi = np.zeros((k, k))
    psi[np.ix_(exo_idx, exo_idx)] = exogenous_phi
    for j, name in enumerate(names):
        if name not in exo:
            psi[j, j] = disturbance_variances[j]

    identity = np.eye(k)
    inv_ib, warn = safe_inverse(identity - beta_matrix)
    if warn is not None:
        raise ValueError("cannot invert (I - B); the structural model is degenerate")
    phi = inv_ib @ psi @ inv_ib.T
    phi = 0.5 * (phi + phi.T)

    sigma = lam @ phi @ lam.T + np.diag(theta)
    return 0.5 * (sigma + sigma.T), phi


@dataclass
class SEMResults:
    """Everything a fitted model produced, including what went wrong."""

    spec: ModelSpec
    n_obs: int
    measurement: Dict[str, FactorSolution]
    composite_correlations: np.ndarray
    phi_estimated: np.ndarray
    phi_implied: np.ndarray
    phi_repaired: bool
    paths: Dict[str, PathEstimates]
    beta_matrix: np.ndarray
    effects: Dict[str, np.ndarray]
    observed_correlation: np.ndarray
    implied_correlation: np.ndarray
    fit_indices: FitIndices
    identification: IdentificationReport
    discriminant_validity: List[Tuple[str, str, float, float, bool]]
    htmt_matrix: np.ndarray
    warnings: List[str] = field(default_factory=list)

    def latent_names(self) -> List[str]:
        return self.spec.latent_names()

    def coefficient_table(self) -> pd.DataFrame:
        """Structural coefficients as a DataFrame.

        The table deliberately includes a ``interpretation`` column reminding
        the reader that these are direct effects conditional on the other
        predictors of the same outcome. Reporting them side by side as
        comparable contributions is the Table 2 fallacy.
        """
        rows = []
        for outcome, est in self.paths.items():
            ci = est.confidence_intervals()
            for i, pred in enumerate(est.predictors):
                rows.append(
                    {
                        "outcome": outcome,
                        "predictor": pred,
                        "beta_standardized": float(est.beta[i]),
                        "se_analytic": float(est.se[i]),
                        "ci_low": float(ci[i, 0]),
                        "ci_high": float(ci[i, 1]),
                        "z": float(est.z_values()[i]),
                        "p_value": float(est.p_values()[i]),
                        "vif": float(est.vif[i]),
                        "r_squared_outcome": est.r_squared,
                        "interpretation": "direct effect, conditional on other predictors",
                    }
                )
        return pd.DataFrame(rows)

    def measurement_table(self) -> pd.DataFrame:
        """Loadings, uniquenesses and reliability, one row per latent variable."""
        rows = []
        for name, sol in self.measurement.items():
            rows.append(
                {
                    "latent": name,
                    "n_indicators": sol.n_indicators(),
                    "min_loading": float(np.min(sol.loadings)),
                    "max_loading": float(np.max(sol.loadings)),
                    "omega_composite_reliability": sol.omega,
                    "cronbach_alpha": sol.alpha,
                    "ave": sol.ave,
                    "factor_score_determinacy": sol.determinacy,
                    "guttman_indeterminacy_bound": sol.guttman_bound,
                    "heywood_case": sol.heywood,
                }
            )
        return pd.DataFrame(rows)

    def summary(self) -> str:
        """Human-readable report, warnings included."""
        lines: List[str] = []
        lines.append("=" * 72)
        lines.append("Structural equation model")
        lines.append("=" * 72)
        lines.append(f"observations : {self.n_obs}")
        lines.append(f"latents      : {', '.join(self.latent_names())}")
        lines.append("")
        lines.append(self.identification.summary())
        lines.append("")
        lines.append("Measurement model")
        for name, sol in self.measurement.items():
            lines.append(
                f"  {name}: {sol.n_indicators()} indicators, "
                f"omega={sol.omega:.3f}, alpha={sol.alpha:.3f}, AVE={sol.ave:.3f}"
            )
            lines.append(
                f"      loadings: " + ", ".join(f"{v:.3f}" for v in sol.loadings)
            )
            lines.append(
                f"      factor score determinacy {sol.determinacy:.3f}; two equally "
                f"valid score sets can correlate as low as {sol.guttman_bound:.3f}"
            )
        lines.append("")
        lines.append("Structural model (standardized direct effects)")
        for outcome, est in self.paths.items():
            ci = est.confidence_intervals()
            lines.append(f"  {outcome}  (R^2 = {est.r_squared:.3f})")
            for i, pred in enumerate(est.predictors):
                lines.append(
                    f"      {pred:<28s} beta = {est.beta[i]:+.3f}  "
                    f"SE = {est.se[i]:.3f}  95% CI [{ci[i,0]:+.3f}, {ci[i,1]:+.3f}]  "
                    f"VIF = {est.vif[i]:.2f}"
                )
        lines.append("")
        lines.append(self.fit_indices.summary())
        if self.warnings:
            lines.append("")
            lines.append("Warnings")
            for w in self.warnings:
                lines.append(f" - {w}")
        lines.append("")
        lines.append(
            "Analytic standard errors condition on the estimated latent "
            "correlation matrix and are too small. Use bootstrap_paths() for "
            "intervals that account for measurement-model uncertainty."
        )
        return "\n".join(lines)


def _composites(data: pd.DataFrame, spec: ModelSpec) -> np.ndarray:
    """Unit-weighted standardized composites, one column per latent."""
    cols = []
    for name in spec.latent_names():
        block = standardize(data[list(spec.latents[name])].to_numpy(dtype=float))
        cols.append(block.mean(axis=1))
    return np.column_stack(cols)


def fit(
    spec: ModelSpec,
    data: pd.DataFrame,
    repair_improper_phi: bool = True,
) -> SEMResults:
    """Estimate a model.

    Parameters
 ----------
    spec
        Model declaration.
    data
        Wide DataFrame with one column per indicator. Rows with any missing
        indicator are dropped, and the number dropped is reported as a warning;
        this package does not implement FIML or multiple imputation, and
        pretending listwise deletion is harmless would be dishonest.
    repair_improper_phi
        When the disattenuated latent correlation matrix is indefinite, project
        it onto the positive-definite cone so that estimation can continue. The
        repair is always recorded in ``SEMResults.phi_repaired`` and in the
        warnings, because an indefinite ``Phi`` is a finding about the model,
        not a numerical nuisance.

    Raises
 ------
    ValueError
        If the model fails the necessary identification conditions.
    """
    spec.validate()
    indicators = spec.all_indicators()
    missing = [c for c in indicators if c not in data.columns]
    if missing:
        raise ValueError(f"data is missing indicator columns: {missing}")

    frame = data[indicators]
    n_before = len(frame)
    frame = frame.dropna()
    n_obs = len(frame)
    warnings: List[str] = []
    if n_obs < n_before:
        warnings.append(
            f"listwise deletion dropped {n_before - n_obs} of {n_before} rows "
            f"({100.0 * (n_before - n_obs) / max(n_before, 1):.1f}%). Complete-case "
            "analysis is unbiased only under missing-completely-at-random."
        )
    if n_obs < 50:
        warnings.append(
            f"n = {n_obs}. Asymptotic standard errors and the chi-square "
            "reference distribution are not trustworthy at this sample size."
        )

    observed = corr_matrix(standardize(frame.to_numpy(dtype=float)))

    ident = check_identification(
        {k: list(v) for k, v in spec.latents.items()},
        {k: list(v) for k, v in spec.paths.items()},
        observed,
    )
    if not ident.necessary_condition_met:
        raise ValueError(
            "model failed the necessary identification conditions:\n"
            + ident.summary()
        )
    warnings.extend(ident.warnings)

    names = spec.latent_names()
    measurement: Dict[str, FactorSolution] = {}
    offset = 0
    blocks_idx: List[List[int]] = []
    for name in names:
        block = list(spec.latents[name])
        idx = list(range(offset, offset + len(block)))
        blocks_idx.append(idx)
        offset += len(block)
        sol = fit_congeneric_block(
            frame.to_numpy(dtype=float), block, name, column_index=idx
        )
        measurement[name] = sol
        warnings.extend(sol.warnings)

    comp = _composites(frame, spec)
    comp_corr = corr_matrix(comp)
    reliabilities = [measurement[n].omega for n in names]
    phi_est, phi_warn = latent_correlations(comp_corr, reliabilities, names)
    warnings.extend(phi_warn)

    phi_repaired = False
    phi_use = phi_est
    if not is_positive_definite(phi_est):
        if repair_improper_phi:
            phi_use = nearest_positive_definite(phi_est)
            phi_repaired = True
            warnings.append(
                "the disattenuated latent correlation matrix was not positive "
                "definite and has been projected onto the positive-definite "
                "cone so that estimation could proceed. This is a repair, not "
                "an estimate: the structural coefficients below rest on a "
                "modified correlation matrix."
            )
        else:
            raise ValueError(
                "disattenuated latent correlation matrix is not positive definite"
            )

    paths: Dict[str, PathEstimates] = {}
    k = len(names)
    beta_matrix = np.zeros((k, k))
    disturbances = np.ones(k)
    for outcome, predictors in spec.paths.items():
        est = solve_paths(phi_use, names, outcome, list(predictors), n_obs)
        paths[outcome] = est
        warnings.extend(est.warnings)
        i = names.index(outcome)
        for j_name, b in zip(est.predictors, est.beta):
            beta_matrix[i, names.index(j_name)] = b
        disturbances[i] = max(est.disturbance_variance, 1e-8)

    exo = spec.exogenous()
    exo_idx = [names.index(n) for n in exo]
    exo_phi = phi_use[np.ix_(exo_idx, exo_idx)] if exo_idx else np.zeros((0, 0))

    loadings_by = {n: measurement[n].loadings for n in names}
    uniq_by = {n: measurement[n].uniquenesses for n in names}
    sigma, phi_implied = implied_covariance(
        spec, loadings_by, uniq_by, beta_matrix, exo_phi, disturbances
    )

    diag_dev = float(np.max(np.abs(np.diag(phi_implied) - 1.0)))
    if diag_dev > 0.05:
        warnings.append(
            f"the model-implied latent correlation matrix has a diagonal that "
            f"departs from 1 by up to {diag_dev:.3f}. The standardized solution "
            "is internally inconsistent."
        )

    n_free = count_free_parameters(
        [len(list(v)) for v in spec.latents.values()], len(exo), spec.n_paths()
    )
    p = len(indicators)
    df = p * (p + 1) // 2 - n_free
    fit_idx = compute_fit_indices(observed, sigma, n_obs, df, n_free)

    effects = effect_decomposition(beta_matrix, names)
    disc = fornell_larcker([measurement[n].ave for n in names], phi_use, names)
    for a, b, bound, r, ok in disc:
        if not ok:
            warnings.append(
                f"Fornell-Larcker discriminant validity fails for '{a}' and "
                f"'{b}': shared variance ({r:.3f}) exceeds sqrt(AVE) "
                f"({bound:.3f})."
            )
    ht = htmt(frame.to_numpy(dtype=float), blocks_idx, names)

    return SEMResults(
        spec=spec,
        n_obs=n_obs,
        measurement=measurement,
        composite_correlations=comp_corr,
        phi_estimated=phi_est,
        phi_implied=phi_implied,
        phi_repaired=phi_repaired,
        paths=paths,
        beta_matrix=beta_matrix,
        effects=effects,
        observed_correlation=observed,
        implied_correlation=sigma,
        fit_indices=fit_idx,
        identification=ident,
        discriminant_validity=disc,
        htmt_matrix=ht,
        warnings=warnings,
    )


def bootstrap_paths(
    spec: ModelSpec,
    data: pd.DataFrame,
    n_boot: int = 500,
    seed: Optional[int] = None,
    level: float = 0.95,
) -> pd.DataFrame:
    """Nonparametric bootstrap of the structural coefficients.

    Resamples rows with replacement and refits the entire model -- measurement
    model included -- so the resulting interval carries the uncertainty in the
    estimated loadings and reliabilities that the analytic standard errors
    ignore.

    Returns a DataFrame with the point estimate, the bootstrap standard error,
    and percentile interval bounds. Replications that fail to fit are counted
    in the ``n_failed`` column rather than being silently discarded.
    """
    rng = np.random.default_rng(seed)
    base = fit(spec, data)
    keys: List[Tuple[str, str]] = []
    for outcome, est in base.paths.items():
        for pred in est.predictors:
            keys.append((outcome, pred))

    draws: Dict[Tuple[str, str], List[float]] = {kk: [] for kk in keys}
    n_failed = 0
    n = len(data)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = data.iloc[idx].reset_index(drop=True)
        try:
            res = fit(spec, sample)
        except Exception:  # noqa: BLE001 - a failed replication is data, not a crash
            n_failed += 1
            continue
        for outcome, pred in keys:
            est = res.paths[outcome]
            j = est.predictors.index(pred)
            draws[(outcome, pred)].append(float(est.beta[j]))

    alpha = 1.0 - level
    rows = []
    for outcome, pred in keys:
        vals = np.array(draws[(outcome, pred)], dtype=float)
        est = base.paths[outcome]
        j = est.predictors.index(pred)
        rows.append(
            {
                "outcome": outcome,
                "predictor": pred,
                "beta": float(est.beta[j]),
                "se_analytic": float(est.se[j]),
                "se_bootstrap": float(np.std(vals, ddof=1)) if vals.size > 1 else np.nan,
                "ci_low": float(np.quantile(vals, alpha / 2)) if vals.size else np.nan,
                "ci_high": float(np.quantile(vals, 1 - alpha / 2)) if vals.size else np.nan,
                "n_boot_used": int(vals.size),
                "n_failed": n_failed,
            }
        )
    return pd.DataFrame(rows)
