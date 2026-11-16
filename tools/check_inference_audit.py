"""Verify complete simulation checkpoints, seeds, summaries and source identity."""
from pathlib import Path
import hashlib
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulations.study_05_inference_audit import Cell, seed_for
from simulations.inference_distributions import student_t_critical_975


def check(directory, expected_files, expected_reps, bootstrap=False, seen_seeds=None):
    manifest = json.loads((directory / "manifest.json").read_text())
    seed = manifest["seed"]
    files = sorted(directory.glob("*.jsonl"))
    assert len(files) == expected_files, (directory, len(files))
    seeds = set() if seen_seeds is None else seen_seeds
    counts = {"attempts": 0, "completed": 0, "valid": 0, "proper": 0, "converged": 0,
              "bootstrap_refits": 0, "bootstrap_failed": 0, "bootstrap_improper": 0}
    hashes = {}
    for path in files:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert len(rows) == expected_reps, (path, len(rows))
        for rep, row in enumerate(rows):
            cell = Cell(row["pair"], row["loading"], row["multiplier"])
            assert row["rep"] == rep
            assert row["n"] == cell.n
            assert row["seed"] == seed_for(seed, cell, row["stage"], rep)
            assert row["seed"] not in seeds
            seeds.add(row["seed"])
            counts["attempts"] += 1
            for status in ("completed", "valid", "proper", "converged"):
                counts[status] += row["observed_" + status]
            assert np.isfinite(row["oracle_estimate"])
            assert row["oracle_se"] > 0
            assert row["oracle_reject"] == (abs(row["oracle_z"]) > student_t_critical_975(row["n"] - 5))
            if row["observed_valid"]:
                assert np.isfinite(row["observed_z"])
                assert row["observed_se"] > 0
            else:
                assert not row["observed_reject"]
            if bootstrap and row["observed_valid"]:
                assert len(row["bootstrap_draws"]) + row["bootstrap_failed"] == row["n_boot"]
                counts["bootstrap_refits"] += row["n_boot"]
                counts["bootstrap_failed"] += row["bootstrap_failed"]
                counts["bootstrap_improper"] += row["bootstrap_improper"]
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    summary = pd.read_csv(directory / "summary.csv")
    assert (summary.attempts == expected_reps).all()
    assert len(summary) == (expected_files if bootstrap else expected_files * 2 // 3)
    for column in summary:
        if column.endswith("_denominator"):
            assert (summary[column].dropna() <= expected_reps).all()
    source = ROOT / "simulations" / ("study_06_bootstrap_ranking.py" if bootstrap else "study_05_inference_audit.py")
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest["source_sha256"]
    dependencies = manifest.get("dependencies_sha256", manifest.get("dependency_sha256", {}))
    for filename, digest in dependencies.items():
        assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == digest, filename
    report = {"status": "passed", "checks": ["complete cell counts", "deterministic seeds", "no duplicate seeds",
              "finite valid statistics", "oracle decisions recomputed with Student-t critical values", "explicit failed-fit accounting", "summary denominators", "source hashes"],
              "counts": counts, "checkpoint_sha256": hashes,
              "summary_sha256": hashlib.sha256((directory / "summary.csv").read_bytes()).hexdigest()}
    (directory / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


if __name__ == "__main__":
    all_seeds = set()
    primary = check(ROOT / "results/inference_audit", 54, 1000, seen_seeds=all_seeds)
    print("Primary:", primary["counts"])
    bootstrap = check(ROOT / "results/bootstrap_ranking", 4, 300, bootstrap=True, seen_seeds=all_seeds)
    print("Bootstrap:", bootstrap["counts"])
