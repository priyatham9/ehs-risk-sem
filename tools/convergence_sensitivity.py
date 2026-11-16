"""Check whether flagged measurement nonconvergence drives the primary result."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulations.study_05_inference_audit import calibrated_uncertainty, rate_fields


def main():
    folder = ROOT / "results/inference_audit"
    main_table = pd.read_csv(folder / "summary.csv")
    affected = main_table.loc[main_table.observed_converged_rate < 1, "cell"].unique()
    rows = []
    for cell in affected:
        cal = pd.read_json(folder / f"{cell}_null_calibration.jsonl", lines=True)
        calibrated_z = cal.loc[cal.observed_converged & cal.observed_valid, "observed_z"].abs().to_numpy()
        cutoff = np.quantile(calibrated_z, .95, method="higher")
        for stage in ("null_validation", "alternative"):
            full = pd.read_json(folder / f"{cell}_{stage}.jsonl", lines=True)
            converged = full.loc[full.observed_converged & full.observed_valid]
            z = converged.observed_z.abs().to_numpy()
            row = {"cell": cell, "stage": stage, "sensitivity": "exclude nonconverged measurement fits from both calibration and evaluation",
                   "original_attempts": len(full), "retained_evaluation": len(converged),
                   "retained_calibration": len(calibrated_z), "critical_z": cutoff,
                   "original_calibrated_rejection": main_table.loc[(main_table.cell == cell) & (main_table.stage == stage), "observed_calibrated_rejection_rate"].iloc[0]}
            row.update(rate_fields("calibrated_rejection", z > cutoff))
            low, high = calibrated_uncertainty(calibrated_z, z, np.random.default_rng(571 if stage == "alternative" else 572))
            row.update({"two_sample_low": low, "two_sample_high": high})
            rows.append(row)
    result = pd.DataFrame(rows)
    result.to_csv(folder / "convergence_sensitivity.csv", index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
