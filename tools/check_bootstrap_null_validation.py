"""Verify the complete independent null validation without rerunning SEM fits."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulations.study_05_inference_audit import Cell, PAIRS, STAGES, rate_fields, seed_for
from simulations.study_08_bootstrap_null_validation import (
    CELL, INNER_REPS, MAIN_SEED, OUTER_REPS, STAGE, exact_binomial_interval, prefix_interval,
)


def check():
    folder = ROOT / "results/bootstrap_null_validation"
    protocol = json.loads((folder / "protocol.json").read_text())
    assert protocol["outer_reps"] == OUTER_REPS == 1000
    assert protocol["inner_reps"] == INNER_REPS == 999
    assert protocol["seed"] == MAIN_SEED
    assert protocol["n"] == CELL.n and protocol["loading"] == CELL.loading
    assert protocol["pair"] == CELL.pair and protocol["stage"] == STAGE
    assert hashlib.sha256((ROOT / "simulations/study_08_bootstrap_null_validation.py").read_bytes()).hexdigest() == protocol["source_sha256"]
    for name, expected in protocol["dependency_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    paths = sorted(folder.glob("chunk_*.jsonl"))
    rows = [json.loads(line) for path in paths for line in path.read_text().splitlines()]
    assert len(paths) == 4 and len(rows) == OUTER_REPS
    assert sorted(row["rep"] for row in rows) == list(range(OUTER_REPS))
    seeds = set()
    for row in rows:
        assert row["seed"] == seed_for(MAIN_SEED, CELL, STAGE, row["rep"])
        assert row["seed"] not in seeds
        seeds.add(row["seed"])
        assert row["true_difference"] == 0 and row["n"] == 6884 and row["loading"] == .75
        assert row["pair"] == "negative" and row["stage"] == STAGE and row["multiplier"] == 1.
        if row["observed_valid"]:
            expected_bootstrap_seed = np.random.SeedSequence([row["seed"], 808]).generate_state(1, dtype=np.uint64)[0].item()
            assert row["bootstrap_seed"] == expected_bootstrap_seed
            assert len(row["bootstrap_draws"]) == INNER_REPS
            assert all(value is None or np.isfinite(value) for value in row["bootstrap_draws"])
            assert row["bootstrap_failed"] == sum(value is None for value in row["bootstrap_draws"])
            for count in (399, 999):
                expected = prefix_interval(row["observed_estimate"], row["bootstrap_draws"], 0., count)
                actual = row[f"b{count}"]
                assert set(actual) == set(expected)
                for key, value in expected.items():
                    if isinstance(value, float):
                        np.testing.assert_allclose(actual[key], value, rtol=1e-12, atol=1e-12)
                    else:
                        assert actual[key] == value
        else:
            assert not row["b399"]["percentile_reject"] and not row["b999"]["percentile_reject"]
    # Recreate all original stream identities from manifests, so this check also
    # works in the follow-up bundle without duplicating 54,000 original records.
    main = json.loads((ROOT / "results/inference_audit/manifest.json").read_text())
    previous = {seed_for(main["seed"], Cell(c["pair"], c["loading"], c["multiplier"]), stage, rep)
                for c in main["cells"] for stage in STAGES for rep in range(main["reps_per_stage_cell"])}
    boot = json.loads((ROOT / "results/bootstrap_ranking/manifest.json").read_text())
    previous.update(seed_for(boot["seed"], Cell(pair, boot["loading"], 1.), stage, rep)
                    for pair in PAIRS for stage in ("null_validation", "alternative")
                    for rep in range(boot["outer_reps"]))
    assert not previous.intersection(seeds)
    summary = pd.read_csv(folder / "summary.csv")
    assert set(summary.resamples) == {399, 999} and (summary.attempts == 1000).all()
    for _, record in summary.iterrows():
        assert record.n == 6884 and record.loading == .75 and record.pair == "negative"
        block = [row[f"b{int(record.resamples)}"] for row in rows]
        for name, flag in [("false_positive_rate", "percentile_reject"),
                           ("normal_false_positive_rate", "normal_reject"),
                           ("coverage_rate", "percentile_coverage"), ("valid_rate", "valid")]:
            expected = rate_fields(name, [row[flag] for row in block])
            for key, value in expected.items():
                np.testing.assert_allclose(record[key], value, rtol=1e-12, atol=1e-12)
        exact_low, exact_high = exact_binomial_interval(sum(row["percentile_reject"] for row in block), 1000)
        np.testing.assert_allclose([record.exact_cp_low, record.exact_cp_high],
                                   [exact_low, exact_high], rtol=1e-12, atol=1e-12)
    paired = json.loads((folder / "paired.json").read_text())
    a = np.array([row["b399"]["percentile_reject"] for row in rows], bool)
    b = np.array([row["b999"]["percentile_reject"] for row in rows], bool)
    expected_counts = {"neither_reject": np.sum(~a & ~b), "b399_only_reject": np.sum(a & ~b),
                       "b999_only_reject": np.sum(~a & b), "both_reject": np.sum(a & b)}
    assert all(paired[key] == value for key, value in expected_counts.items())
    delta = b.astype(float) - a.astype(float)
    np.testing.assert_allclose(paired["b999_minus_b399"], delta.mean(), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(paired["paired_mcse"], delta.std(ddof=1) / np.sqrt(1000), rtol=1e-12, atol=1e-12)
    assert -1 <= paired["paired_bootstrap_low"] <= paired["b999_minus_b399"] <= paired["paired_bootstrap_high"] <= 1
    assert paired["attempts"] == OUTER_REPS
    verification = json.loads((folder / "verification.json").read_text())
    expected_verification = {
        "status": "passed", "outer_reps": len(rows),
        "bootstrap_refits": sum(len(row["bootstrap_draws"]) for row in rows),
        "no_seed_overlap_with_original_studies": True,
        "checkpoint_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
        "summary_sha256": hashlib.sha256((folder / "summary.csv").read_bytes()).hexdigest(),
    }
    for key in ("bootstrap_failed", "bootstrap_improper", "bootstrap_nonconverged"):
        expected_verification[key] = sum(row[key] for row in rows)
    for label, field in (("outer_valid", "observed_valid"), ("outer_proper", "observed_proper"),
                         ("outer_converged", "observed_converged")):
        expected_verification[label] = sum(row[field] for row in rows)
    assert verification == expected_verification, "Saved verification disagrees with raw checkpoints"
    report = {"status": "passed", "checks": ["prospective source and dependency hashes", "exact fixed sample size", "fresh deterministic seeds with no original-study overlap", "all prefix intervals recomputed", "failure positions retained", "summary rates and paired differences recomputed"],
              "outer_reps": len(rows), "bootstrap_refits": sum(len(r["bootstrap_draws"]) for r in rows),
              "summary_sha256": hashlib.sha256((folder / "summary.csv").read_bytes()).hexdigest()}
    (folder / "recomputed_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    print(json.dumps(check(), indent=2))
