"""Identification checks, run before anything is estimated.

Identification is a property of the model and the scale-setting choice, not of
the data. A model can be unidentified and still produce a number, which is why
these checks run first and why :func:`ehs_risk_sem.model.fit` refuses to
proceed when the necessary condition fails.

Rules implemented:

* the t-rule (necessary, not sufficient): free parameters must not exceed
  ``p(p+1)/2``, the number of non-redundant observed moments;
* the scale-setting requirement: every latent variable needs a metric. This
  package fixes all latent variances to 1, so every loading is free;
* the two-indicator and three-indicator rules for one-factor blocks;
* recursiveness: the structural model must be acyclic, since the estimator
  solves ``(I - B)^-1`` in closed form and has no instruments with which to
  identify a feedback loop;
* an empirical check on the observed covariance matrix, because a model that is
  algebraically identified can still be empirically underidentified when the
  data are nearly collinear.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence

import numpy as np

from .linalg import condition_number, smallest_eigenvalue

__all__ = [
    "IdentificationReport",
    "count_free_parameters",
    "t_rule",
    "is_recursive",
    "check_identification",
]


@dataclass
class IdentificationReport:
    """Outcome of the identification checks.

    ``necessary_condition_met`` is the gate. ``sufficient`` is deliberately
    absent: no general sufficient condition for an arbitrary SEM is checked
    here, and claiming one would be false.
    """

    n_observed: int
    n_moments: int
    n_free_parameters: int
    df: int
    t_rule_passed: bool
    recursive: bool
    scale_setting: str
    per_factor_indicator_counts: Dict[str, int]
    smallest_eigenvalue: float
    condition_number: float
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def necessary_condition_met(self) -> bool:
        return self.t_rule_passed and self.recursive and not self.errors

    def summary(self) -> str:
        lines = [
            "Identification",
            f"  observed variables            : {self.n_observed}",
            f"  non-redundant moments p(p+1)/2: {self.n_moments}",
            f"  free parameters               : {self.n_free_parameters}",
            f"  degrees of freedom            : {self.df}",
            f"  scale setting                 : {self.scale_setting}",
            f"  t-rule (necessary)            : {'pass' if self.t_rule_passed else 'FAIL'}",
            f"  recursive structural model    : {'yes' if self.recursive else 'NO'}",
            f"  smallest eigenvalue of S      : {self.smallest_eigenvalue:.3e}",
            f"  condition number of S         : {self.condition_number:.3e}",
        ]
        for e in self.errors:
            lines.append(f"  ERROR   : {e}")
        for w in self.warnings:
            lines.append(f"  WARNING : {w}")
        return "\n".join(lines)


def count_free_parameters(
    indicator_counts: Sequence[int], n_exogenous: int, n_paths: int
) -> int:
    """Count free parameters under this package's parameterization.

    With every latent variance fixed to 1:

    * one free loading per indicator: ``sum(indicator_counts)``;
    * one free error variance per indicator: ``sum(indicator_counts)``;
    * free covariances among the exogenous latents: ``n_exo * (n_exo - 1) / 2``;
    * one coefficient per structural path: ``n_paths``.

    Disturbance variances of the endogenous latents are *not* free: with the
    latent standardized, the disturbance variance is ``1 - R^2``, determined by
    the paths.
    """
    p = int(np.sum(indicator_counts))
    n_exo = int(n_exogenous)
    return 2 * p + n_exo * (n_exo - 1) // 2 + int(n_paths)


def t_rule(n_observed: int, n_free: int) -> bool:
    """Necessary condition: free parameters must not exceed observed moments."""
    return n_free <= n_observed * (n_observed + 1) // 2


def is_recursive(paths: Dict[str, Sequence[str]], latents: Sequence[str]) -> bool:
    """True if the directed graph implied by ``paths`` is acyclic.

    ``paths`` maps each endogenous latent to the list of latents pointing into
    it. Depth-first search with a colour marking.
    """
    adj: Dict[str, List[str]] = {name: [] for name in latents}
    for target, sources in paths.items():
        for s in sources:
            adj.setdefault(s, []).append(target)
    colour: Dict[str, int] = {name: 0 for name in adj}

    def visit(node: str) -> bool:
        if colour.get(node, 0) == 1:
            return False
        if colour.get(node, 0) == 2:
            return True
        colour[node] = 1
        for nxt in adj.get(node, []):
            if not visit(nxt):
                return False
        colour[node] = 2
        return True

    return all(visit(node) for node in list(adj))


def check_identification(
    blocks: Dict[str, Sequence[str]],
    paths: Dict[str, Sequence[str]],
    sample_covariance: np.ndarray,
    min_indicators: int = 2,
    recommended_indicators: int = 3,
) -> IdentificationReport:
    """Run every identification check and return a report.

    Parameters
 ----------
    blocks
        Mapping from latent variable name to its indicator names.
    paths
        Mapping from endogenous latent name to its direct predictors.
    sample_covariance
        Observed covariance (or correlation) matrix of the indicators, used
        only for the empirical checks.
    """
    latents = list(blocks)
    counts = {name: len(list(inds)) for name, inds in blocks.items()}
    p = int(sum(counts.values()))
    endogenous = set(paths)
    exogenous = [name for name in latents if name not in endogenous]
    n_paths = int(sum(len(list(v)) for v in paths.values()))

    n_free = count_free_parameters(list(counts.values()), len(exogenous), n_paths)
    moments = p * (p + 1) // 2
    passed = t_rule(p, n_free)
    recursive = is_recursive({k: list(v) for k, v in paths.items()}, latents)

    errors: List[str] = []
    warnings: List[str] = []

    if not passed:
        errors.append(
            f"t-rule violated: {n_free} free parameters exceed {moments} "
            "non-redundant observed moments. The model cannot be identified as "
            "specified."
        )
    if not recursive:
        errors.append(
            "the structural model contains a feedback loop. A non-recursive "
            "model requires instrumental variables for identification; this "
            "estimator does not support that and will not guess at one."
        )
    for name, count in counts.items():
        if count < min_indicators:
            errors.append(
                f"latent '{name}' has {count} indicator(s). At least "
                f"{min_indicators} are required. A single-indicator latent is "
                "identified only if its error variance is fixed a priori from "
                "a known reliability, which this package does not do silently."
            )
        elif count < recommended_indicators:
            warnings.append(
                f"latent '{name}' has {count} indicators. Identification then "
                "depends on its correlations with other factors; the block is "
                "not identified in isolation."
            )
    for target, sources in paths.items():
        if target not in blocks:
            errors.append(f"path target '{target}' is not a declared latent variable")
        for s in sources:
            if s not in blocks:
                errors.append(f"path source '{s}' is not a declared latent variable")
            if s == target:
                errors.append(f"latent '{target}' is specified as a predictor of itself")

    cov = np.asarray(sample_covariance, dtype=float)
    lam_min = smallest_eigenvalue(cov)
    cond = condition_number(cov)
    if lam_min <= 1e-10:
        errors.append(
            f"the observed covariance matrix is singular (smallest eigenvalue "
            f"{lam_min:.3e}). At least one indicator is a linear combination "
            "of the others."
        )
    elif cond > 1e6:
        warnings.append(
            f"the observed covariance matrix has condition number {cond:.3e}. "
            "The model may be empirically underidentified even though it "
            "satisfies the algebraic rules: some parameter combinations are "
            "barely constrained by these data."
        )

    return IdentificationReport(
        n_observed=p,
        n_moments=moments,
        n_free_parameters=n_free,
        df=moments - n_free,
        t_rule_passed=passed,
        recursive=recursive,
        scale_setting="all latent variances fixed to 1 (standardized solution)",
        per_factor_indicator_counts=counts,
        smallest_eigenvalue=lam_min,
        condition_number=cond,
        errors=errors,
        warnings=warnings,
    )
