"""Monte Carlo sample-size planner for SEM safety-climate surveys.

SIMULATION. Data are generated from the model you describe; nothing here is an
empirical claim about any workforce.

For each candidate sample size the tool simulates ``--reps`` surveys from a
known model (``--predictors`` correlated exogenous latents driving one outcome
latent, ``--indicators`` items per latent, one common standardized loading),
fits each with the package's two-step estimator (``ehs_risk_sem.fit``), and
records

* proper-solution rate: fit completed, no Heywood case, no repaired latent
  correlation matrix, outcome R^2 <= 1;
* loading stability: empirical SD of the estimated loadings across replications
  (pooled over indicators) and mean absolute loading error;
* fit: share of replications meeting the Hu & Bentler (1999) cutoffs
  (``FitIndices.meets_conventional_cutoffs``). Reported, not used as a
  criterion: the repository's studies show good fit says little about whether
  paths are precise;
* path precision: empirical SD of the estimated standardized paths.

The recommended N is the smallest N on the grid at which the proper-solution
rate is at least ``--min-proper`` AND the loading SD is at most
``--max-loading-sd`` (and, if ``--max-path-sd`` is given, the path SD is at most
that). The defaults (0.95 and 0.05) are conventions of this tool, not
published standards; change them to your own tolerance. If no grid point
meets the criteria the tool says so rather than extrapolating.

Limits: the estimate comes from ``--reps`` replications, so it carries Monte
Carlo error; all indicators share one loading, errors are normal and
uncorrelated, and the estimator is limited-information two-step, not ML.

    python3 tools/sample_size_planner.py --loading 0.75 --out results/planner_default.csv

Usage as a library: ``plan(...)`` returns a DataFrame; ``recommend(df, ...)``
returns the recommended N or None.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, Sequence

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from ehs_risk_sem.compat import silence_accelerate_matmul_warnings  # noqa: E402
from ehs_risk_sem.model import ModelSpec, fit  # noqa: E402
from simulations.dgp import TrueModel, generate_dataset  # noqa: E402

DEFAULT_GRID = (100, 150, 200, 300, 500, 750, 1000, 2000)
DEFAULT_BETAS = (0.45, 0.30, -0.25, -0.20)


def build_model(
    betas: Sequence[float], loading: float, n_indicators: int, exo_corr: float
) -> TrueModel:
    """m exogenous latents (common correlation) predicting one outcome latent."""
    m = len(betas)
    names = [f"Predictor{i + 1}" for i in range(m)] + ["Outcome"]
    b = np.zeros((m + 1, m + 1))
    b[m, :m] = betas
    corr = np.full((m, m), float(exo_corr))
    np.fill_diagonal(corr, 1.0)
    r2 = float(np.array(betas) @ corr @ np.array(betas))
    if r2 >= 1.0:
        raise ValueError(f"betas imply R^2 = {r2:.3f} >= 1 at exogenous correlation {exo_corr}")
    if not 0 < loading < 1:
        raise ValueError("loading must be in (0, 1)")
    if n_indicators < 3:
        raise ValueError("use at least 3 indicators per latent")
    loadings = {n: np.full(n_indicators, float(loading)) for n in names}
    return TrueModel(latent_names=names, loadings=loadings, beta=b, exogenous_corr=corr)


def plan(
    model: TrueModel, grid: Sequence[int] = DEFAULT_GRID, reps: int = 200, seed: int = 20261027
) -> pd.DataFrame:
    """Simulate ``reps`` surveys at each N in ``grid``; one row per N."""
    silence_accelerate_matmul_warnings()
    spec = ModelSpec(
        latents={n: model.indicator_names[n] for n in model.latent_names}, paths=model.paths()
    )
    true_loading = {n: model.loadings[n] for n in model.latent_names}
    outcome = model.endogenous()[0]
    true_beta = np.array(list(model.true_beta_for(outcome).values()))
    rows = []
    for n in grid:
        proper = fit_ok = 0
        load_est, path_est = [], []
        for r in range(reps):
            data, _ = generate_dataset(model, n, seed=seed + 100003 * n + r)
            try:
                res = fit(spec, data)
            except Exception:  # noqa: BLE001
                continue
            est = res.paths[outcome]
            heywood = any(s.heywood for s in res.measurement.values())
            ok = not (heywood or res.phi_repaired or est.r_squared > 1.0)
            proper += ok
            fit_ok += bool(res.fit_indices.meets_conventional_cutoffs())
            load_est.append(np.concatenate([res.measurement[k].loadings for k in model.latent_names]))
            path_est.append(est.beta)
        used = len(load_est)
        truth = np.concatenate([true_loading[k] for k in model.latent_names])
        L = np.array(load_est)
        P = np.array(path_est)
        rows.append(
            {
                "n": n,
                "reps": reps,
                "completed_fits": used,
                "proper_solution_rate": proper / reps,
                "conventional_fit_rate": fit_ok / reps,
                "loading_sd": float(np.mean(np.std(L, axis=0, ddof=1))) if used > 1 else np.nan,
                "loading_mean_abs_error": float(np.mean(np.abs(L - truth))) if used else np.nan,
                "path_sd": float(np.mean(np.std(P, axis=0, ddof=1))) if used > 1 else np.nan,
                "true_betas": " ".join(f"{b:g}" for b in true_beta),
            }
        )
    return pd.DataFrame(rows)


def recommend(
    df: pd.DataFrame,
    min_proper: float = 0.95,
    max_loading_sd: float = 0.05,
    max_path_sd: Optional[float] = None,
) -> Optional[int]:
    """Smallest N meeting every stated criterion, or None if no grid point does."""
    ok = (df["proper_solution_rate"] >= min_proper) & (df["loading_sd"] <= max_loading_sd)
    if max_path_sd is not None:
        ok &= df["path_sd"] <= max_path_sd
    hit = df.loc[ok, "n"]
    return int(hit.min()) if len(hit) else None


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--betas", type=float, nargs="+", default=list(DEFAULT_BETAS))
    p.add_argument("--loading", type=float, default=0.75)
    p.add_argument("--indicators", type=int, default=3)
    p.add_argument("--exo-corr", type=float, default=0.35)
    p.add_argument("--grid", type=int, nargs="+", default=list(DEFAULT_GRID))
    p.add_argument("--reps", type=int, default=200)
    p.add_argument("--seed", type=int, default=20261027)
    p.add_argument("--min-proper", type=float, default=0.95)
    p.add_argument("--max-loading-sd", type=float, default=0.05)
    p.add_argument("--max-path-sd", type=float, default=None)
    p.add_argument("--out", help="write the table to this CSV")
    a = p.parse_args(argv)
    try:
        model = build_model(a.betas, a.loading, a.indicators, a.exo_corr)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    df = plan(model, a.grid, a.reps, a.seed)
    print(df.drop(columns=["true_betas"]).to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    n = recommend(df, a.min_proper, a.max_loading_sd, a.max_path_sd)
    crit = f"proper-solution rate >= {a.min_proper}, loading SD <= {a.max_loading_sd}"
    if a.max_path_sd is not None:
        crit += f", path SD <= {a.max_path_sd}"
    if n is None:
        print(f"\nNo N on the grid meets: {crit}. Extend --grid; nothing is extrapolated.")
    else:
        print(f"\nSmallest N on the grid meeting {crit}: {n} ({a.reps} replications per N).")
    print("Simulation under your stated model; Monte Carlo error applies.")
    if a.out:
        df.to_csv(a.out, index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
