"""Simulation-based (Monte Carlo) fit-index cutoffs for a fitted model structure.

SIMULATION. Data are generated from the model you fitted; nothing here is an
empirical claim about any workforce.

Fixed rule-of-thumb cutoffs (CFI >= .95, RMSEA <= .06, SRMR <= .08; Hu &
Bentler 1999) were derived under specific conditions. This tool complements
them with cutoffs for *your* model structure and sample size, in the spirit of
simulation-based "dynamic" cutoffs (McNeish & Wolf 2023). Procedure:

1. Fit the model to your data with ``ehs_risk_sem.fit``.
2. Treat the fitted loadings, structural paths and exogenous latent
   correlations as the population model (a "correct model" null).
3. Simulate ``--reps`` datasets of size ``--n`` (default: the observed N)
   from it, seeded, and fit each with the same two-step estimator.
4. Report the empirical distribution of the indices the package already
   computes (chi-square, CFI, TLI, RMSEA, SRMR) and the cutoff at the chosen
   tail: the ``--alpha`` quantile for CFI and TLI (low values are bad), the
   ``1 - alpha`` quantile for chi-square, RMSEA and SRMR (high values are bad).
   Also reported: the share of correct-model replications that FAIL each fixed
   Hu & Bentler cutoff, and where the observed value sits in the simulated
   distribution.

Limits (what this does NOT cover): the null is a correctly specified model, so
it says how much sampling noise alone moves the indices. It does not give the
distribution under misspecification, so it cannot say how large an index must be
to flag a particular misspecification (power). Loadings and paths are treated as
known although they were estimated (plug-in), errors are multivariate normal,
the estimator is the limited-information two-step estimator rather than ML, and
estimation error in the fitted model is not propagated. Replications where the
fit raises an error are counted and excluded; Heywood cases and repaired latent
correlation matrices are retained and counted. Monte Carlo error applies: with
R replications a tail quantile is itself an estimate.

    python3 tools/fit_index_cutoffs.py --reps 500 --out results/fit_index_cutoffs_n1200.csv

Library use: ``model_from_fit(res)`` -> ``TrueModel``; ``simulate_indices(...)``
-> DataFrame, one row per replication; ``cutoff_table(...)`` -> summary.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ehs_risk_sem.compat import silence_accelerate_matmul_warnings  # noqa: E402
from ehs_risk_sem.fit_indices import (  # noqa: E402
    HU_BENTLER_CFI,
    HU_BENTLER_RMSEA,
    HU_BENTLER_SRMR,
)
from ehs_risk_sem.model import ModelSpec, SEMResults, fit  # noqa: E402
from simulations.dgp import TrueModel, generate_dataset  # noqa: E402

INDICES = ("chi_square", "cfi", "tli", "rmsea", "srmr")
LOW_IS_BAD = {"cfi", "tli"}
FIXED_CUTOFF = {
    "cfi": (">=", HU_BENTLER_CFI),
    "rmsea": ("<=", HU_BENTLER_RMSEA),
    "srmr": ("<=", HU_BENTLER_SRMR),
}
LOADING_BOUNDS = (0.05, 0.99)


def model_from_fit(res: SEMResults) -> TrueModel:
    """Population model whose parameters are the fitted estimates (plug-in)."""
    spec = res.spec
    names = spec.latent_names()
    loadings: Dict[str, np.ndarray] = {
        n: np.clip(np.abs(res.measurement[n].loadings), *LOADING_BOUNDS) for n in names
    }
    beta = np.array(res.beta_matrix, dtype=float)
    exo = [i for i, n in enumerate(names) if n in spec.exogenous()]
    if any(not np.any(beta[i]) for i in range(len(names)) if i not in exo):
        raise ValueError("an endogenous latent has all-zero estimated paths")
    corr = np.array(res.phi_estimated, dtype=float)[np.ix_(exo, exo)]
    corr = 0.5 * (corr + corr.T)
    if np.linalg.eigvalsh(corr).min() <= 1e-8:
        raise ValueError("estimated exogenous latent correlations are not positive definite")
    return TrueModel(
        latent_names=names,
        loadings=loadings,
        beta=beta,
        exogenous_corr=corr,
        indicator_names={n: list(spec.latents[n]) for n in names},
    )


def simulate_indices(
    model: TrueModel, spec: ModelSpec, n: int, reps: int = 500, seed: int = 20261027
) -> pd.DataFrame:
    """Fit ``reps`` simulated datasets of size ``n``; one row per attempted replication.

    Columns: the five indices, ``proper`` (no Heywood case, no repaired latent
    correlation matrix) and ``ok`` (fit completed). Failed fits have NaN indices.
    """
    silence_accelerate_matmul_warnings()
    rows = []
    for r in range(reps):
        data, _ = generate_dataset(model, n, seed=seed + r)
        try:
            res = fit(spec, data)
        except Exception:  # noqa: BLE001
            rows.append({**{k: np.nan for k in INDICES}, "proper": False, "ok": False})
            continue
        fi = res.fit_indices
        heywood = any(s.heywood for s in res.measurement.values())
        rows.append(
            {
                **{k: float(getattr(fi, k)) for k in INDICES},
                "proper": not (heywood or res.phi_repaired),
                "ok": True,
            }
        )
    return pd.DataFrame(rows)


def cutoff_table(
    sims: pd.DataFrame, alpha: float = 0.05, observed: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """Summarise simulated indices: mean, SD, simulation-based cutoff, fixed-cutoff failure rate.

    ``cutoff`` is the ``alpha`` quantile for CFI/TLI and the ``1 - alpha``
    quantile otherwise. ``observed_percentile`` is the share of simulated values
    at or below the observed value (when ``observed`` is given).
    """
    if not 0 < alpha < 0.5:
        raise ValueError("alpha must be in (0, 0.5)")
    ok = sims[sims["ok"]]
    rows = []
    for k in INDICES:
        x = ok[k].to_numpy(dtype=float)
        x = x[np.isfinite(x)]
        if len(x) == 0:
            continue
        q = alpha if k in LOW_IS_BAD else 1.0 - alpha
        row = {
            "index": k,
            "n_reps_used": len(x),
            "mean": float(np.mean(x)),
            "sd": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
            "p05": float(np.quantile(x, 0.05)),
            "p50": float(np.quantile(x, 0.50)),
            "p95": float(np.quantile(x, 0.95)),
            "tail": f"{q:.3f} quantile",
            "cutoff": float(np.quantile(x, q)),
            "fixed_cutoff": np.nan,
            "fixed_cutoff_failure_rate": np.nan,
            "observed": np.nan,
            "observed_percentile": np.nan,
        }
        if k in FIXED_CUTOFF:
            op, c = FIXED_CUTOFF[k]
            fails = x < c if op == ">=" else x > c
            row["fixed_cutoff"] = c
            row["fixed_cutoff_failure_rate"] = float(np.mean(fails))
        if observed and k in observed:
            row["observed"] = float(observed[k])
            row["observed_percentile"] = float(np.mean(x <= observed[k]))
        rows.append(row)
    return pd.DataFrame(rows)


def spec_from_columns(data: pd.DataFrame, outcome: str) -> ModelSpec:
    """Latent ``X`` = columns named ``x_1, x_2, ...``; every other latent predicts ``outcome``."""
    groups: Dict[str, list] = {}
    for c in data.columns:
        if c.startswith("latent_") or "_" not in c:
            continue
        groups.setdefault(c.rsplit("_", 1)[0], []).append(c)
    if outcome not in groups:
        raise ValueError(f"outcome '{outcome}' is not an indicator prefix; found {sorted(groups)}")
    latents = {k: groups[k] for k in groups}
    return ModelSpec(latents=latents, paths={outcome: [k for k in latents if k != outcome]})


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data", default=os.path.join(ROOT, "synthetic", "survey_style.csv"))
    p.add_argument("--outcome", default="risk", help="indicator-column prefix of the outcome latent")
    p.add_argument("--n", type=int, default=None, help="simulated N (default: rows of --data)")
    p.add_argument("--reps", type=int, default=500)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=20261027)
    p.add_argument("--out", help="write the cutoff table to this CSV")
    a = p.parse_args(argv)
    try:
        data = pd.read_csv(a.data, comment="#")
        spec = spec_from_columns(data, a.outcome)
        res = fit(spec, data)
        model = model_from_fit(res)
        if not 0 < a.alpha < 0.5 or a.reps < 2:
            raise ValueError("need 0 < alpha < 0.5 and reps >= 2")
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    n = a.n or res.n_obs
    sims = simulate_indices(model, spec, n, a.reps, a.seed)
    # The observed value is only comparable to a simulation at the observed N.
    obs = {k: getattr(res.fit_indices, k) for k in INDICES} if n == res.n_obs else None
    tab = cutoff_table(sims, a.alpha, obs)
    print(f"Model: {len(spec.latents)} latents, df = {res.fit_indices.df}; "
          f"simulated N = {n}; {int(sims['ok'].sum())}/{a.reps} fits completed, "
          f"{int((sims['ok'] & sims['proper']).sum())} proper solutions; seed {a.seed}.")
    show = ["index", "mean", "sd", "p05", "p50", "p95", "cutoff", "fixed_cutoff",
            "fixed_cutoff_failure_rate", "observed", "observed_percentile"]
    print(tab[show].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\nCutoff = alpha quantile for CFI/TLI, 1-alpha quantile for the others, under the")
    print("fitted model taken as correct. Says nothing about power against misspecification.")
    if a.out:
        tab.to_csv(a.out, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
