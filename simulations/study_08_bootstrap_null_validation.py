"""Fresh-seed null validation prompted by the original bootstrap's 23/300 result.

Prospective protocol is written before this follow-up computes any main draws.
This is an exploratory follow-up motivated by observed calibration uncertainty,
not a retrospective claim that the original study was preregistered.

Fixed design: negative-pair equality null, N=6884, loading=.75; 1000 independent
outer surveys; 999 row-resampling refits per survey. The first 399 attempted
resamples and all 999 are compared on the same surveys. Failed attempts retain
positions, so the 399-prefix is not replaced with the first 399 successful fits.

Run --prepare first, then --run. --pilot uses a separate seed and ten surveys
only to estimate runtime. Original studies and output directories are untouched.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time
from unittest.mock import patch

import numpy as np
import pandas as pd

from ehs_risk_sem.model import ModelSpec, fit
from simulations.dgp import generate_dataset
from simulations.inference_distributions import _regularized_beta
from simulations.study_05_inference_audit import Cell, model_for, rate_fields, replicate, seed_for
from simulations.study_06_bootstrap_ranking import interval_fields

CELL = Cell("negative", .75, 1.)
STAGE = "null_validation"
MAIN_SEED = 2026111608
PILOT_SEED = 2026111609
OUTER_REPS = 1000
INNER_REPS = 999


def beta_quantile(probability, a, b):
    low, high = 0., 1.
    for _ in range(100):
        mid = (low + high) / 2
        if _regularized_beta(mid, a, b) < probability:
            low = mid
        else:
            high = mid
    return (low + high) / 2


def exact_binomial_interval(k, n):
    return (0. if k == 0 else beta_quantile(.025, k, n - k + 1),
            1. if k == n else beta_quantile(.975, k + 1, n - k))


def binomial_check(k, n, null_probability=.05):
    probabilities = np.array([
        math.exp(math.lgamma(n + 1) - math.lgamma(j + 1) - math.lgamma(n - j + 1)
                 + j * math.log(null_probability) + (n - j) * math.log1p(-null_probability))
        for j in range(n + 1)])
    lower, upper = exact_binomial_interval(k, n)
    return {"successes": k, "attempts": n, "rate": k / n,
            "exact_cp_low": lower, "exact_cp_high": upper,
            "one_sided_upper_p": float(probabilities[k:].sum()),
            "two_sided_probability_ordering_p": float(probabilities[probabilities <= probabilities[k] * (1 + 1e-12)].sum())}


def prefix_interval(estimate, draws, truth, count):
    prefix = draws[:count]
    finite = [value for value in prefix if value is not None]
    row = {"attempted": len(prefix), "used": len(finite),
           "valid": len(finite) >= max(20, .95 * count),
           "percentile_reject": False, "normal_reject": False,
           "percentile_coverage": False, "normal_coverage": False}
    if row["valid"]:
        row.update(interval_fields(estimate, finite, truth))
    return row


def one_replication(rep, seed):
    row = replicate(CELL, STAGE, rep, seed)
    row.update({"bootstrap_draws": [], "bootstrap_failed": 0,
                "bootstrap_improper": 0, "bootstrap_nonconverged": 0})
    if not row["observed_valid"]:
        row["b399"] = prefix_interval(0., [], 0., 399)
        row["b999"] = prefix_interval(0., [], 0., 999)
        return row
    model = model_for(CELL, STAGE)
    data, _ = generate_dataset(model, CELL.n, row["seed"])
    spec = ModelSpec(latents=model.indicator_names, paths=model.paths())
    contrast = np.array([0., 0., 1., -1.])
    boot_seed = np.random.SeedSequence([row["seed"], 808]).generate_state(1, dtype=np.uint64)[0].item()
    row["bootstrap_seed"] = boot_seed
    rng = np.random.default_rng(boot_seed)
    with patch("ehs_risk_sem.model.compute_fit_indices", return_value=None):
        for _ in range(INNER_REPS):
            sample = data.iloc[rng.integers(0, len(data), size=len(data))].reset_index(drop=True)
            try:
                result = fit(spec, sample)
                value = float(contrast @ result.paths["Risk"].beta)
                if not np.isfinite(value):
                    raise FloatingPointError("nonfinite bootstrap contrast")
                row["bootstrap_draws"].append(value)
                row["bootstrap_improper"] += bool(result.phi_repaired or result.paths["Risk"].r_squared > 1 or
                                                   any(s.heywood for s in result.measurement.values()))
                row["bootstrap_nonconverged"] += not all(s.converged for s in result.measurement.values())
            except (ValueError, np.linalg.LinAlgError, FloatingPointError):
                row["bootstrap_draws"].append(None)
                row["bootstrap_failed"] += 1
    row["b399"] = prefix_interval(row["observed_estimate"], row["bootstrap_draws"], row["true_difference"], 399)
    row["b999"] = prefix_interval(row["observed_estimate"], row["bootstrap_draws"], row["true_difference"], 999)
    return row


def protocol():
    files = sorted(Path("ehs_risk_sem").glob("*.py")) + [
        Path("simulations/dgp.py"), Path("simulations/inference_distributions.py"),
        Path("simulations/study_05_inference_audit.py"), Path("simulations/study_06_bootstrap_ranking.py")]
    return {
        "schema": "sem-bootstrap-null-validation-v1", "simulation_only": True,
        "motivation": "Exploratory follow-up to original negative-null 23/300 rejection; Wilson and exact intervals differ at nominal .05. This follow-up protocol is prospective, not a claim of original-study preregistration.",
        "primary_estimand": "Null rejection probability of the 999-resample percentile interval at nominal alpha .05.",
        "secondary_estimand": "Paired B999 minus first-B399 rejection difference on the same fresh outer surveys.",
        "pair": "negative", "loading": .75, "n": 6884, "stage": STAGE,
        "null_coefficients": [.45, .30, -.225, -.225], "indicators_per_factor": 3,
        "predictor_correlation": .35, "outer_reps": OUTER_REPS, "inner_reps": INNER_REPS,
        "prefix_resamples": 399, "seed": MAIN_SEED, "pilot_seed": PILOT_SEED,
        "stopping_rule": "Exactly 1000 independent outer surveys; no outcome-dependent early stopping or sample-size extension.",
        "intervals": "Pointwise 95% Wilson and exact Clopper-Pearson rates; paired empirical-bootstrap difference interval from 20000 multinomial resamples.",
        "failure_rule": "Failed outer tests count as non-rejections in all-attempt denominators. Inner failures retain positions; a prefix needs at least95% valid refits. Report all failures, improper fits and nonconvergence.",
        "diagnostic_optimization": "Suppress unused fit indices only inside resamples; all measurement and structural estimation remains the public fit function.",
        "chunks": [[start, start + 250] for start in range(0, OUTER_REPS, 250)],
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dependency_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        "numpy": np.__version__, "pandas": pd.__version__,
    }


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    path = out / "protocol.json"
    plan = protocol()
    if path.exists():
        saved = json.loads(path.read_text())
        assert {key: saved[key] for key in plan} == plan, "Protocol changed; use a new follow-up directory"
        return saved
    plan["recorded_before_main_computation_utc"] = datetime.now(timezone.utc).isoformat()
    path.write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def run_chunk(args):
    start, stop, seed, out = args
    path = Path(out) / f"chunk_{start:04d}_{stop:04d}.jsonl"
    prior = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    assert len(prior) <= stop - start
    for offset, row in enumerate(prior):
        rep = start + offset
        assert row["rep"] == rep and row["seed"] == seed_for(seed, CELL, STAGE, rep)
    began = time.perf_counter()
    with path.open("a") as stream:
        for rep in range(start + len(prior), stop):
            row = one_replication(rep, seed)
            stream.write(json.dumps(row, allow_nan=False) + "\n")
            stream.flush()
            if (rep - start + 1) % 10 == 0:
                print(json.dumps({"chunk_start": start, "completed": rep - start + 1,
                                  "seconds": time.perf_counter() - began}), flush=True)
    return {"start": start, "stop": stop, "new_reps": stop - start - len(prior),
            "seconds": time.perf_counter() - began}


def paired_summary(a, b):
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    difference = b.astype(float) - a.astype(float)
    counts = np.array([np.sum(~a & ~b), np.sum(~a & b), np.sum(a & ~b), np.sum(a & b)])
    rng = np.random.default_rng(2026111681)
    bootstrap = rng.multinomial(len(a), counts / len(a), size=20000)
    sampled_difference = (bootstrap[:, 1] - bootstrap[:, 2]) / len(a)
    lower, upper = np.quantile(sampled_difference, [.025, .975])
    return {"b999_minus_b399": float(difference.mean()),
            "paired_mcse": float(difference.std(ddof=1) / np.sqrt(len(a))),
            "paired_bootstrap_low": float(lower), "paired_bootstrap_high": float(upper),
            "neither_reject": int(counts[0]), "b999_only_reject": int(counts[1]),
            "b399_only_reject": int(counts[2]), "both_reject": int(counts[3]),
            "attempts": len(a)}


def summarize(out):
    paths = sorted(out.glob("chunk_*.jsonl"))
    draws = [json.loads(line) for path in paths for line in path.read_text().splitlines()]
    assert len(draws) == OUTER_REPS and sorted(row["rep"] for row in draws) == list(range(OUTER_REPS))
    assert len({row["seed"] for row in draws}) == OUTER_REPS
    for row in draws:
        assert row["seed"] == seed_for(MAIN_SEED, CELL, STAGE, row["rep"])
        if row["observed_valid"]:
            assert len(row["bootstrap_draws"]) == INNER_REPS
            assert sum(value is None for value in row["bootstrap_draws"]) == row["bootstrap_failed"]
    # Verify independence from every original main and bootstrap dataset.
    previous = set()
    for directory in (Path("results/inference_audit"), Path("results/bootstrap_ranking")):
        for path in directory.glob("*.jsonl"):
            previous.update(json.loads(line)["seed"] for line in path.read_text().splitlines())
    assert not previous.intersection(row["seed"] for row in draws)
    rows = []
    for count in (399, 999):
        block = [row[f"b{count}"] for row in draws]
        rejects = [row["percentile_reject"] for row in block]
        row = {"resamples": count, "pair": "negative", "loading": .75, "n": CELL.n,
               "stage": STAGE, "attempts": len(draws)}
        row.update(rate_fields("false_positive_rate", rejects))
        row.update(rate_fields("normal_false_positive_rate", [r["normal_reject"] for r in block]))
        row.update(rate_fields("coverage_rate", [r["percentile_coverage"] for r in block]))
        row.update(rate_fields("valid_rate", [r["valid"] for r in block]))
        low, high = exact_binomial_interval(sum(rejects), len(rejects))
        row.update({"exact_cp_low": low, "exact_cp_high": high})
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "summary.csv", index=False)
    paired = paired_summary([r["b399"]["percentile_reject"] for r in draws], [r["b999"]["percentile_reject"] for r in draws])
    (out / "paired.json").write_text(json.dumps(paired, indent=2) + "\n")
    verification = {"status": "passed", "outer_reps": len(draws),
        "bootstrap_refits": sum(len(r["bootstrap_draws"]) for r in draws),
        "bootstrap_failed": sum(r["bootstrap_failed"] for r in draws),
        "bootstrap_improper": sum(r["bootstrap_improper"] for r in draws),
        "bootstrap_nonconverged": sum(r["bootstrap_nonconverged"] for r in draws),
        "outer_valid": sum(r["observed_valid"] for r in draws),
        "outer_proper": sum(r["observed_proper"] for r in draws),
        "outer_converged": sum(r["observed_converged"] for r in draws),
        "no_seed_overlap_with_original_studies": True,
        "checkpoint_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        "summary_sha256": hashlib.sha256((out / "summary.csv").read_bytes()).hexdigest()}
    (out / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/bootstrap_null_validation"))
    parser.add_argument("--workers", type=int, default=4)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--prepare", action="store_true")
    actions.add_argument("--run", action="store_true")
    actions.add_argument("--pilot", action="store_true")
    actions.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare(args.out), indent=2))
        return
    assert (args.out / "protocol.json").exists(), "Record the prospective protocol with --prepare before computation"
    plan = prepare(args.out)
    if args.pilot:
        pilot = args.out / "runtime_pilot"
        pilot.mkdir(exist_ok=True)
        result = run_chunk((0, 10, PILOT_SEED, str(pilot)))
        result["projected_single_worker_seconds_for_main"] = result["seconds"] * OUTER_REPS / max(result["new_reps"], 1)
        (pilot / "timing.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result), flush=True)
        return
    if args.summarize_only:
        summarize(args.out)
        return
    jobs = [(start, stop, MAIN_SEED, str(args.out)) for start, stop in plan["chunks"]]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for progress in pool.map(run_chunk, jobs):
            print(json.dumps(progress), flush=True)
    summarize(args.out)


if __name__ == "__main__":
    main()
