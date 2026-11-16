"""Implementable contrast bootstrap follow-up to the simulation-only null audit.

Resample whole survey rows and refit the public estimator, including measurement
loadings/reliability. Compute each contrast *within* a bootstrap replicate;
subtracting marginal confidence intervals would ignore coefficient covariance.
Four independent cells: two advertised N thresholds, null and alternative,
loading .75. Percentile and normal-with-bootstrap-SE intervals are evaluated
against known truth and zero. All are pointwise, without multiplicity correction.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from ehs_risk_sem.model import ModelSpec, fit
from simulations.dgp import generate_dataset
from simulations.study_05_inference_audit import (
    Cell, PAIRS, Z975, model_for, rate_fields, replicate, seed_for,
)


def paired_bootstrap_contrasts(spec, data, contrast, n_boot, seed):
    rng = np.random.default_rng(seed)
    estimates = []
    failed = improper = nonconverged = 0
    for _ in range(n_boot):
        sample = data.iloc[rng.integers(0, len(data), size=len(data))].reset_index(drop=True)
        try:
            result = fit(spec, sample)
            value = float(contrast @ result.paths["Risk"].beta)
            if not np.isfinite(value):
                failed += 1
                continue
            estimates.append(value)
            improper += bool(result.phi_repaired or result.paths["Risk"].r_squared > 1 or
                             any(s.heywood for s in result.measurement.values()))
            nonconverged += not all(s.converged for s in result.measurement.values())
        except (ValueError, np.linalg.LinAlgError, FloatingPointError):
            failed += 1
    return estimates, failed, improper, nonconverged


def interval_fields(estimate, draws, truth):
    draws = np.asarray(draws)
    se = float(np.std(draws, ddof=1))
    low, high = np.quantile(draws, [.025, .975])
    normal_low, normal_high = estimate - Z975 * se, estimate + Z975 * se
    return {
        "bootstrap_se": se, "percentile_low": float(low), "percentile_high": float(high),
        "normal_low": normal_low, "normal_high": normal_high,
        "percentile_reject": bool(low > 0 or high < 0),
        "normal_reject": bool(normal_low > 0 or normal_high < 0),
        "percentile_coverage": bool(low <= truth <= high),
        "normal_coverage": bool(normal_low <= truth <= normal_high),
    }


def bootstrap_replicate(cell, stage, rep, seed, n_boot):
    row = replicate(cell, stage, rep, seed)
    row.update({"n_boot": n_boot, "bootstrap_valid": False,
                "percentile_reject": False, "normal_reject": False,
                "percentile_coverage": False, "normal_coverage": False})
    if not row["observed_valid"]:
        return row
    model = model_for(cell, stage)
    data, _ = generate_dataset(model, cell.n, row["seed"])
    spec = ModelSpec(latents=model.indicator_names, paths=model.paths())
    i, j, _ = PAIRS[cell.pair]
    contrast = np.zeros(4)
    contrast[i], contrast[j] = 1, -1
    boot_seed = np.random.SeedSequence([row["seed"], 606]).generate_state(1, dtype=np.uint64)[0].item()
    draws, failed, improper, nonconverged = paired_bootstrap_contrasts(spec, data, contrast, n_boot, boot_seed)
    row.update({"bootstrap_seed": boot_seed, "bootstrap_draws": draws,
                "bootstrap_failed": failed, "bootstrap_improper": improper,
                "bootstrap_nonconverged": nonconverged})
    if len(draws) >= max(20, .95 * n_boot):
        row["bootstrap_valid"] = True
        row.update(interval_fields(row["observed_estimate"], draws, row["true_difference"]))
    return row


def run_cell(args):
    cell, stage, reps, seed, n_boot, out = args
    path = Path(out) / f"{cell.key}_{stage}.jsonl"
    prior = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    for rep, row in enumerate(prior):
        if row["rep"] != rep or row["seed"] != seed_for(seed, cell, stage, rep) or row["n_boot"] != n_boot:
            raise ValueError(f"Checkpoint identity mismatch: {path}")
    if len(prior) > reps:
        raise ValueError("Checkpoint has more replicates than requested")
    start = time.perf_counter()
    with path.open("a") as stream:
        for rep in range(len(prior), reps):
            row = bootstrap_replicate(cell, stage, rep, seed, n_boot)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            if (rep + 1) % 10 == 0:
                print(json.dumps({"cell": cell.key, "stage": stage, "completed": rep + 1,
                                  "seconds": time.perf_counter() - start}), flush=True)
    return {"cell": cell.key, "stage": stage, "reps": reps, "seconds": time.perf_counter() - start}


def summarize(out):
    rows = []
    for path in sorted(Path(out).glob("*.jsonl")):
        block = pd.read_json(path, lines=True)
        row = {k: block[k].iloc[0] for k in ("cell", "pair", "loading", "n", "stage", "true_difference", "n_boot")}
        row.update({"attempts": len(block), "mean_bootstrap_se": block.bootstrap_se.mean(),
                    "empirical_sd": block.observed_estimate.std(),
                    "mean_naive_se": block.observed_se.mean(),
                    "bootstrap_attempts": int(block.n_boot.sum()),
                    "bootstrap_failed": int(block.bootstrap_failed.sum()),
                    "bootstrap_improper": int(block.bootstrap_improper.sum()),
                    "bootstrap_nonconverged": int(block.bootstrap_nonconverged.sum())})
        for name in ("oracle_reject", "observed_reject", "bootstrap_valid", "percentile_reject", "normal_reject",
                     "percentile_coverage", "normal_coverage", "observed_proper", "observed_converged", "fit_good"):
            row.update(rate_fields(name + "_rate", block[name]))
        rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(Path(out) / "summary.csv", index=False)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/bootstrap_ranking"))
    parser.add_argument("--reps", type=int, default=300)
    parser.add_argument("--n-boot", type=int, default=399)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=2026111606)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    metadata = {"design_version": 1, "outer_reps": args.reps, "inner_reps": args.n_boot,
                "seed": args.seed, "loading": .75, "simulation_only": True,
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "dependencies_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in sorted(Path("ehs_risk_sem").glob("*.py")) +
                    [Path("simulations/dgp.py"), Path("simulations/study_05_inference_audit.py")]}}
    manifest = args.out / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text()) != metadata:
        raise ValueError("Changed manifest: use a fresh output directory")
    manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    jobs = [(Cell(pair, .75, 1.), stage, args.reps, args.seed, args.n_boot, str(args.out))
            for pair in PAIRS for stage in ("null_validation", "alternative")]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for result in pool.map(run_cell, jobs):
            print(json.dumps(result), flush=True)
    summarize(args.out)


if __name__ == "__main__":
    main()
