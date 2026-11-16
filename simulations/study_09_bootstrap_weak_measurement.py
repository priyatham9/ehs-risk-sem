"""Prospective null stress test at the original weakest-measurement design cell.

This single condition is selected from the original fixed study, where loading
.50 and N385 produced measurement nonconvergence. It is an implementation stress
test, not a search for an unfavorable condition. The estimator and bootstrap
procedure are imported unchanged from studies05/06. Pilot outcomes are discarded;
only elapsed time determines the fixed main sample size before outcome inspection.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from simulations.study_05_inference_audit import Cell, rate_fields, seed_for
from simulations.study_06_bootstrap_ranking import bootstrap_replicate
from simulations.study_08_bootstrap_null_validation import exact_binomial_interval, paired_summary

CELL = Cell("positive", .50, .5)
STAGE = "null_validation"
MAIN_SEED = 20261116091
PILOT_SEED = 20261116092
INNER_REPS = 399
PILOT_REPS = 5
PREVIOUS = ("results/inference_audit", "results/bootstrap_ranking", "results/bootstrap_null_validation")


def source_hashes():
    files = sorted(Path("ehs_risk_sem").glob("*.py")) + [Path("simulations") / name for name in
        ("dgp.py", "inference_distributions.py", "study_05_inference_audit.py",
         "study_06_bootstrap_ranking.py", "study_08_bootstrap_null_validation.py",
         "study_09_bootstrap_weak_measurement.py")]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def choose_target(seconds):
    """A 25% scheduling/CPU allowance is fixed before pilot outcomes exist."""
    forecasts = {str(n): seconds / PILOT_REPS * n / 4 * 1.25 for n in (500, 250)}
    target = next((n for n in (500, 250) if forecasts[str(n)] <= 3600), None)
    return target, forecasts


def timing_pilot(out):
    out.mkdir(parents=True, exist_ok=True)
    destination = out / "timing_pilot.json"
    assert not destination.exists(), "Do not repeat the pilot to choose a favorable runtime"
    began = time.perf_counter()
    for rep in range(PILOT_REPS):
        bootstrap_replicate(CELL, STAGE, rep, PILOT_SEED, INNER_REPS)
    elapsed = time.perf_counter() - began
    target, forecasts = choose_target(elapsed)
    report = {"pilot_outer_reps": PILOT_REPS, "inner_reps": INNER_REPS,
              "pilot_seed": PILOT_SEED, "single_worker_seconds": elapsed,
              "four_worker_seconds_with_25_percent_allowance": forecasts,
              "selected_outer_reps": target,
              "outcomes": "Discarded; no rejection, fit-status or interval outcomes retained or inspected.",
              "source_sha256": source_hashes(),
              "recorded_utc": datetime.now(timezone.utc).isoformat()}
    destination.write_text(json.dumps(report, indent=2) + "\n")
    return report


def prepare(out):
    timing = json.loads((out / "timing_pilot.json").read_text())
    assert timing["source_sha256"] == source_hashes(), "Pilot and main implementation differ"
    target, forecasts = choose_target(timing["single_worker_seconds"])
    assert target is not None, "Even250 exceeds the prospective one-hour runtime budget; do not run"
    assert timing["selected_outer_reps"] == target
    boundaries = np.linspace(0, target, 5, dtype=int).tolist()
    plan = {
        "schema": "sem-bootstrap-weak-measurement-v1", "simulation_only": True,
        "motivation": "Single-condition implementation stress test selected from the original weakest-measurement cell (loading .50,N385), which had measurement nonconvergence; not selected using new outcomes.",
        "primary_estimand": "All-attempt null rejection probability of the399-resample percentile procedure.",
        "secondary_estimands": ["Naive and bootstrap-SE normal null rejection", "Computable-conditional rejection, separately from all-attempt rates", "Paired percentile-minus-naive and normal-minus-naive rejection differences", "Outer and inner computability, propriety and convergence accounting"],
        "pair": CELL.pair, "loading": CELL.loading, "multiplier": CELL.multiplier,
        "n": CELL.n, "stage": STAGE, "null_coefficients": [.375, .375, -.25, -.20],
        "predictor_correlation": .35, "indicators_per_factor": 3,
        "outer_reps": target, "inner_reps": INNER_REPS, "seed": MAIN_SEED, "pilot_seed": PILOT_SEED,
        "chunks": list(map(list, zip(boundaries[:-1], boundaries[1:]))),
        "sample_size_rule": "Before inspecting outcomes:500 if four-worker forecast with25% allowance<=60min; otherwise250 if<=60min; otherwise no run. No outcome-dependent extension or early stopping.",
        "timing_pilot_sha256": hashlib.sha256((out / "timing_pilot.json").read_bytes()).hexdigest(),
        "forecast_seconds": forecasts[str(target)],
        "estimator": "Unchanged study06.bootstrap_replicate, public two-step fit,399 whole-row resamples; only unused inner global fit diagnostics suppressed.",
        "interval_rule": "Frozen study06 rule: at least95% finite refits (380 of399) for computable percentile/normal intervals; NumPy linear quantiles, strict exclusion of zero.",
        "failure_rule": "Failed/noncomputable tests count as nonrejections in all-attempt rates. Conditional denominators include only the relevant computable tests. Improper and nonconverged finite fits remain included, with separate flags; no filtered estimator is substituted.",
        "uncertainty": "Pointwise95% Wilson and exact Clopper-Pearson outer-survey intervals; paired20,000-multinomial-resample intervals. Inner fit counts are descriptive nested counts, not independent trials for binomial intervals.",
        "interpretation": "One exploratory Gaussian null stress condition; no power, equivalence or universal calibration claims. No pooling or paired comparison with separate loading.75 experiments.",
        "previous_studies": list(PREVIOUS), "source_sha256": source_hashes(),
        "numpy": np.__version__, "pandas": pd.__version__,
    }
    destination = out / "protocol.json"
    if destination.exists():
        old = json.loads(destination.read_text())
        assert {k: old[k] for k in plan} == plan, "Prospective protocol changed"
        return old
    plan["recorded_before_main_computation_utc"] = datetime.now(timezone.utc).isoformat()
    destination.write_text(json.dumps(plan, indent=2) + "\n")
    return plan


def one_replication(rep):
    row = bootstrap_replicate(CELL, STAGE, rep, MAIN_SEED, INNER_REPS)
    # The frozen routine skips inner fits if the outer estimate is noncomputable.
    for key, default in (("bootstrap_draws", []), ("bootstrap_failed", 0),
                         ("bootstrap_improper", 0), ("bootstrap_nonconverged", 0)):
        row.setdefault(key, default)
    row["bootstrap_attempted"] = len(row["bootstrap_draws"]) + row["bootstrap_failed"]
    return row


def run_chunk(args):
    start, stop, out = args
    path = Path(out) / f"chunk_{start:04d}_{stop:04d}.jsonl"
    prior = [json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
    assert len(prior) <= stop - start
    for i, row in enumerate(prior):
        assert row["rep"] == start + i and row["seed"] == seed_for(MAIN_SEED, CELL, STAGE, start + i)
    began = time.perf_counter()
    with path.open("a") as stream:
        for rep in range(start + len(prior), stop):
            stream.write(json.dumps(one_replication(rep), allow_nan=False) + "\n")
            stream.flush()
            if (rep - start + 1) % 10 == 0:
                print(json.dumps({"chunk_start": start, "completed": rep - start + 1,
                                  "seconds": time.perf_counter() - began}), flush=True)
    return {"start": start, "stop": stop, "new_reps": stop - start - len(prior),
            "seconds": time.perf_counter() - began}


def rate_record(method, conditioning, values):
    result = {"method": method, "conditioning": conditioning}
    result.update(rate_fields("rejection_rate", values))
    n, k = result["rejection_rate_denominator"], result["rejection_rate_count"]
    lo, hi = exact_binomial_interval(k, n) if n else (None, None)
    result.update({"exact_cp_low": lo, "exact_cp_high": hi})
    return {key: None if isinstance(value, (float, np.floating)) and not np.isfinite(value) else value
            for key, value in result.items()}


def summaries(rows):
    records = []
    for method, flag, valid in (("naive", "observed_reject", "observed_valid"),
                               ("percentile", "percentile_reject", "bootstrap_valid"),
                               ("normal", "normal_reject", "bootstrap_valid")):
        records.append(rate_record(method, "all_attempts", [r[flag] for r in rows]))
        records.append(rate_record(method, "computable_only", [r[flag] for r in rows if r[valid]]))
    paired = {}
    for method in ("percentile", "normal"):
        values = paired_summary([r["observed_reject"] for r in rows], [r[method + "_reject"] for r in rows])
        names = {"b999_minus_b399": "bootstrap_minus_naive", "b999_only_reject": "bootstrap_only_reject",
                 "b399_only_reject": "naive_only_reject"}
        paired[method] = {names.get(k, k): v for k, v in values.items()}
    accounting = {"outer_attempts": len(rows), "inner_planned": len(rows) * INNER_REPS,
                  "inner_attempted": sum(r["bootstrap_attempted"] for r in rows),
                  "inner_computable": sum(len(r["bootstrap_draws"]) for r in rows),
                  "invalid_intervals": sum(not r["bootstrap_valid"] for r in rows)}
    for field in ("observed_completed", "observed_valid", "observed_proper", "observed_converged",
                  "bootstrap_valid", "bootstrap_failed", "bootstrap_improper", "bootstrap_nonconverged"):
        accounting[field] = sum(r[field] for r in rows)
    for field in ("bootstrap_failed", "bootstrap_improper", "bootstrap_nonconverged"):
        accounting["outer_with_" + field] = rate_fields("rate", [r[field] > 0 for r in rows])
    return records, paired, accounting


def verify_seed_separation(rows):
    previous = set()
    for directory in PREVIOUS:
        for path in Path(directory).rglob("*.jsonl"):
            for line in path.read_text().splitlines():
                row = json.loads(line)
                for key in ("seed", "bootstrap_seed"):
                    if key in row:
                        previous.add(row[key])
    for rep in range(PILOT_REPS):
        seed = seed_for(PILOT_SEED, CELL, STAGE, rep)
        previous.add(seed)
        previous.add(np.random.SeedSequence([seed, 606]).generate_state(1, dtype=np.uint64)[0].item())
    current = [r[key] for r in rows for key in ("seed", "bootstrap_seed") if key in r]
    assert len(current) == len(set(current)) and not previous.intersection(current)
    return {"previous_stream_ids_checked": len(previous), "new_stream_ids": len(current),
            "no_overlap_with_previous_studies_or_pilot": True}


def summarize(out):
    plan = prepare(out)
    paths = sorted(out.glob("chunk_*.jsonl"))
    rows = [json.loads(line) for p in paths for line in p.read_text().splitlines()]
    assert len(rows) == plan["outer_reps"] and sorted(r["rep"] for r in rows) == list(range(plan["outer_reps"]))
    separation = verify_seed_separation(rows)
    records, paired, accounting = summaries(rows)
    pd.DataFrame(records).to_csv(out / "summary.csv", index=False)
    for name, value in (("paired.json", paired), ("accounting.json", accounting)):
        (out / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    report = {"status": "passed", **separation, "outer_reps": len(rows),
              "checkpoint_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              "summary_sha256": hashlib.sha256((out / "summary.csv").read_bytes()).hexdigest()}
    (out / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("results/bootstrap_weak_measurement"))
    parser.add_argument("--workers", type=int, default=4)
    actions = parser.add_mutually_exclusive_group(required=True)
    for action in ("pilot", "prepare", "run", "summarize-only"):
        actions.add_argument("--" + action, action="store_true")
    args = parser.parse_args()
    if args.pilot:
        print(json.dumps(timing_pilot(args.out), indent=2))
        return
    if args.prepare:
        print(json.dumps(prepare(args.out), indent=2))
        return
    assert (args.out / "protocol.json").exists(), "Record prospective protocol before main computation"
    plan = prepare(args.out)
    if args.run:
        jobs = [(a, b, str(args.out)) for a, b in plan["chunks"]]
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(run_chunk, jobs):
                print(json.dumps(result), flush=True)
    summarize(args.out)


if __name__ == "__main__":
    main()
