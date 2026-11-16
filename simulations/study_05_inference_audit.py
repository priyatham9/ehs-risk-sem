"""Simulation audit of advertised coefficient-ranking sample sizes.

Uses the unchanged public two-step estimator. Oracle OLS uses the *raw* latent
scores, which have known population unit variances; rescaling them by random
sample SDs would introduce an extra source of uncertainty absent from ordinary
OLS SEs. The observed estimator returns standardized population coefficients.
Both target the same population contrast, but only the oracle knows latent units.

Independent streams: null_calibration, null_validation, alternative. A simulated
null cutoff is an oracle diagnostic, not a procedure available to a practitioner.
Each row retains actual fit failure/improper/convergence status. Failed fits never
become zero estimates. Main rejection rates use all attempted replicates (failure
is non-rejection); valid-fit conditional rates are separately labeled.

Run with BLAS threads capped, e.g. OPENBLAS_NUM_THREADS=1 python -m
simulations.study_05_inference_audit --out results/inference_audit --workers 2.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from simulations.inference_distributions import student_t_critical_975

from ehs_risk_sem.model import ModelSpec, fit
from simulations.dgp import generate_dataset, make_four_factor_model

BASE_BETA = np.array([0.45, 0.30, -0.25, -0.20])
PAIRS = {"positive": (0, 1, 770), "negative": (2, 3, 6884)}
STAGES = {"null_calibration": 0, "null_validation": 1, "alternative": 2}
Z975 = 1.959963984540054


@dataclass(frozen=True)
class Cell:
    pair: str
    loading: float
    multiplier: float

    @property
    def n(self):
        return round(PAIRS[self.pair][2] * self.multiplier)

    @property
    def key(self):
        return f"{self.pair}_l{round(self.loading * 100):02d}_n{self.n}"


def model_for(cell, stage):
    beta = BASE_BETA.copy()
    i, j, _ = PAIRS[cell.pair]
    if stage.startswith("null"):
        beta[i] = beta[j] = (beta[i] + beta[j]) / 2
    return make_four_factor_model(beta=beta, loading=cell.loading)


def seed_for(seed, cell, stage, rep):
    """Stable stream identity, independent of scheduling and checkpoint resume."""
    return np.random.SeedSequence([
        seed, list(PAIRS).index(cell.pair), round(cell.loading * 1000),
        cell.n, STAGES[stage], rep,
    ]).generate_state(1, dtype=np.uint64)[0].item()


def oracle_ols(eta, contrast):
    x = eta[:, :4] - eta[:, :4].mean(axis=0)
    y = eta[:, 4] - eta[:, 4].mean()
    inv = np.linalg.inv(x.T @ x)
    beta = inv @ x.T @ y
    residual = y - x @ beta
    covariance = (residual @ residual / (len(y) - 5)) * inv
    return float(contrast @ beta), float(np.sqrt(contrast @ covariance @ contrast))


def observed_contrast(result, contrast):
    est = result.paths["Risk"]
    names = result.latent_names()
    idx = [names.index(p) for p in est.predictors]
    inverse = np.linalg.inv(result.phi_implied[np.ix_(idx, idx)])
    covariance = max(1 - est.r_squared, 1e-12) / (result.n_obs - 5) * inverse
    return float(contrast @ est.beta), float(np.sqrt(contrast @ covariance @ contrast))


def replicate(cell, stage, rep, seed):
    model = model_for(cell, stage)
    i, j, _ = PAIRS[cell.pair]
    contrast = np.zeros(4)
    contrast[i], contrast[j] = 1, -1
    beta = model.beta[4, :4]
    replicate_seed = seed_for(seed, cell, stage, rep)
    data, eta = generate_dataset(model, cell.n, replicate_seed)
    oracle_estimate, oracle_se = oracle_ols(eta, contrast)
    row = {
        **asdict(cell), "cell": cell.key, "n": cell.n, "stage": stage,
        "rep": rep, "seed": replicate_seed, "true_difference": float(contrast @ beta),
        "true_r_squared": float(beta @ model.exogenous_corr @ beta),
        "oracle_estimate": oracle_estimate, "oracle_se": oracle_se,
        "oracle_z": oracle_estimate / oracle_se,
        "oracle_reject": bool(abs(oracle_estimate / oracle_se) > student_t_critical_975(cell.n - 5)),
        "observed_completed": False, "observed_valid": False,
        "observed_proper": False, "observed_converged": False,
        "observed_reject": False, "fit_good": False,
    }
    spec = ModelSpec(latents=model.indicator_names, paths=model.paths())
    try:
        res = fit(spec, data)
        estimate, se = observed_contrast(res, contrast)
        valid = bool(np.isfinite(estimate) and np.isfinite(se) and se > 0)
        proper = not (res.phi_repaired or res.paths["Risk"].r_squared > 1 or
                      any(s.heywood for s in res.measurement.values()))
        row.update({
            "observed_completed": True, "observed_valid": valid,
            "observed_proper": bool(proper),
            "observed_converged": all(s.converged for s in res.measurement.values()),
            "observed_estimate": estimate, "observed_se": se,
            "observed_z": estimate / se if valid else None,
            "observed_reject": bool(valid and abs(estimate / se) > Z975),
            "fit_good": bool(res.fit_indices.meets_conventional_cutoffs()),
            "cfi": res.fit_indices.cfi, "rmsea": res.fit_indices.rmsea,
            "srmr": res.fit_indices.srmr, "chi_square": res.fit_indices.chi_square,
            "phi_repaired": bool(res.phi_repaired),
            "heywood": any(s.heywood for s in res.measurement.values()),
        })
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def run_cell(args):
    cell, stage, reps, seed, out = args
    path = Path(out) / f"{cell.key}_{stage}.jsonl"
    prior = []
    if path.exists():
        prior = [json.loads(line) for line in path.read_text().splitlines()]
        for rep, row in enumerate(prior):
            if row["rep"] != rep or row["seed"] != seed_for(seed, cell, stage, rep):
                raise ValueError(f"Checkpoint identity mismatch: {path}")
        if len(prior) > reps:
            raise ValueError(f"Checkpoint exceeds requested reps: {path}")
    started = time.perf_counter()
    with path.open("a") as stream:
        for rep in range(len(prior), reps):
            row = replicate(cell, stage, rep, seed)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            if rep % 25 == 0:
                stream.flush()
    return {"cell": cell.key, "stage": stage, "reps": reps,
            "new_reps": reps - len(prior), "seconds": time.perf_counter() - started}


def wilson(successes, n):
    if n == 0:
        return (float("nan"), float("nan"))
    p = successes / n
    d = 1 + Z975**2 / n
    center = (p + Z975**2 / (2 * n)) / d
    half = Z975 * np.sqrt(p * (1 - p) / n + Z975**2 / (4 * n**2)) / d
    return float(center - half), float(center + half)


def rate_fields(name, values):
    values = np.asarray(values, dtype=bool)
    n, k = len(values), int(values.sum())
    p = k / n if n else float("nan")
    lo, hi = wilson(k, n)
    return {name: p, f"{name}_count": k, f"{name}_denominator": n,
            f"{name}_mcse": np.sqrt(p * (1 - p) / n) if n else float("nan"),
            f"{name}_low": lo, f"{name}_high": hi}


def calibrated_uncertainty(calibration_z, evaluation_z, rng, reps=2000):
    """Two independent sample bootstrap; includes simulation cutoff uncertainty."""
    rates = []
    for _ in range(reps):
        cutoff = np.quantile(rng.choice(calibration_z, len(calibration_z)), .95, method="higher")
        evaluation = rng.choice(evaluation_z, len(evaluation_z))
        rates.append(np.mean(evaluation > cutoff))
    return np.quantile(rates, [.025, .975]).tolist()


def summarize(out, uncertainty_reps=2000):
    frames = [pd.read_json(p, lines=True) for p in sorted(Path(out).glob("*.jsonl"))]
    df = pd.concat(frames, ignore_index=True)
    rows = []
    for cell_key, cell_df in df.groupby("cell", sort=True):
        cal = cell_df[cell_df.stage == "null_calibration"]
        cal_z = cal.loc[cal.observed_valid, "observed_z"].abs().to_numpy()
        cutoff = float(np.quantile(cal_z, .95, method="higher"))
        for stage in ("null_validation", "alternative"):
            block = cell_df[cell_df.stage == stage]
            valid = block.observed_valid.to_numpy(bool)
            # Failure/nonfinite test statistics are non-rejections in all-attempt rates.
            z = block.observed_z.abs().fillna(-np.inf).to_numpy()
            calibrated = z > cutoff
            good = block.fit_good.to_numpy(bool)
            truth = float(block.true_difference.iloc[0])
            estimate = block.observed_estimate.to_numpy()
            row = {k: block[k].iloc[0] for k in ("cell", "pair", "loading", "multiplier", "n", "true_difference", "true_r_squared")}
            row.update({"stage": stage, "attempts": len(block),
                        "null_calibration_attempts": len(cal), "null_calibration_valid": len(cal_z),
                        "null_calibrated_critical_z": cutoff,
                        "observed_mean_estimate": float(block.observed_estimate.mean()),
                        "observed_empirical_sd": float(block.observed_estimate.std()),
                        "observed_mean_naive_se": float(block.observed_se.mean()),
                        "oracle_empirical_sd": float(block.oracle_estimate.std()),
                        "oracle_mean_se": float(block.oracle_se.mean()),
                        "mean_cfi": float(block.cfi.mean()), "mean_rmsea": float(block.rmsea.mean()),
                        "mean_srmr": float(block.srmr.mean())})
            for name, vals in {
                "observed_completed_rate": block.observed_completed,
                "observed_proper_rate": block.observed_proper,
                "observed_converged_rate": block.observed_converged,
                "fit_good_rate": good,
                "oracle_rejection_rate": block.oracle_reject,
                "observed_naive_rejection_rate": block.observed_reject,
                "observed_calibrated_rejection_rate": calibrated,
                "observed_calibrated_rejection_valid_only": calibrated[valid],
                "observed_naive_coverage": valid & (np.abs(estimate - truth) <= Z975 * block.observed_se.to_numpy()),
            }.items():
                row.update(rate_fields(name, vals))
            digest = int.from_bytes(hashlib.sha256(f"{cell_key}:{stage}".encode()).digest()[:4], "little")
            low, high = calibrated_uncertainty(cal_z, z, np.random.default_rng(digest), uncertainty_reps)
            row.update({"calibrated_two_sample_bootstrap_low": low,
                        "calibrated_two_sample_bootstrap_high": high})
            if stage == "alternative":
                wrong = estimate * truth < 0
                row.update(rate_fields("wrong_ranking_given_good_fit", wrong[good & valid]))
                row.update(rate_fields("wrong_ranking", wrong[valid]))
                delta = block.oracle_reject.to_numpy(float) - calibrated.astype(float)
                row.update({"oracle_minus_calibrated_rejection": float(delta.mean()),
                            "paired_difference_mcse_conditional_cutoff": float(delta.std(ddof=1) / np.sqrt(len(delta)))})
            rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(Path(out) / "summary.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/inference_audit"))
    parser.add_argument("--reps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=2026111605)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.summarize_only:
        summarize(args.out)
        return
    cells = [Cell(pair, loading, multiplier) for pair in PAIRS
             for loading in (.5, .75, .9) for multiplier in (.5, 1., 2.)]
    metadata = {"design_version": 1, "reps_per_stage_cell": args.reps, "seed": args.seed,
                "cells": [dict(asdict(c), n=c.n) for c in cells], "stages": list(STAGES),
                "numpy": np.__version__, "python": platform.python_version(),
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "dependency_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(Path("ehs_risk_sem").glob("*.py")) + [Path("simulations/dgp.py"), Path("simulations/inference_distributions.py")]},
                "simulation_only": True}
    manifest = args.out / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != metadata:
        raise ValueError("Manifest differs; use a fresh output directory for a changed design or implementation")
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    jobs = [(c, stage, args.reps, args.seed, str(args.out)) for c in cells for stage in STAGES]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for progress in pool.map(run_cell, jobs):
            print(json.dumps(progress), flush=True)
    summarize(args.out)


if __name__ == "__main__":
    main()
