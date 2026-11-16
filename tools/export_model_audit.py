"""Export only completed, verified model-audit tables for the public workbench.

The primary table is required. Bootstrap cells are never exported partially:
until all four cells and their summary pass verification, rows=[] and complete
is false. The JSON contains simulation/theory summaries, never real records.
"""
from pathlib import Path
import hashlib
import json
import shutil
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.check_inference_audit import check
from simulations.study_08_bootstrap_null_validation import binomial_check


def read_json(path):
    return json.loads(path.read_text())


def records(path):
    return json.loads(pd.read_csv(path).to_json(orient="records", double_precision=15))


def export(destination=None):
    primary = ROOT / "results/inference_audit"
    bootstrap = ROOT / "results/bootstrap_ranking"
    fisher = ROOT / "results/information_bound"
    validation = ROOT / "results/bootstrap_null_validation"
    check(primary, 54, 1000)
    bootstrap_complete = (bootstrap / "summary.csv").exists()
    if bootstrap_complete:
        check(bootstrap, 4, 300, bootstrap=True)
        original_null = pd.read_csv(bootstrap / "summary.csv")
        original_null = original_null[(original_null.pair == "negative") & (original_null.stage == "null_validation")].iloc[0]
        calibration_check = binomial_check(int(original_null.percentile_reject_rate_count), int(original_null.attempts))
        calibration_check.update({"wilson_low": float(original_null.percentile_reject_rate_low),
                                 "wilson_high": float(original_null.percentile_reject_rate_high),
                                 "bonferroni_two_original_null_conditions_p": min(1., 2 * calibration_check["two_sided_probability_ordering_p"]),
                                 "interpretation": "Exploratory calibration concern: Wilson narrowly excludes5%, exact equal-tailed interval includes5%; no proof of miscalibration or certified remedy. The multiplicity calculation illustrates adjustment across two original null conditions, not all analyses."})
        (bootstrap / "calibration_check.json").write_text(json.dumps(calibration_check, indent=2) + "\n")
    validation_complete = (validation / "summary.csv").exists() and (validation / "verification.json").exists()
    if validation_complete:
        from tools.check_bootstrap_null_validation import check as check_validation
        check_validation()
        verification = read_json(validation / "verification.json")
        assert verification["status"] == "passed" and verification["outer_reps"] == 1000
        assert hashlib.sha256((validation / "summary.csv").read_bytes()).hexdigest() == verification["summary_sha256"]
    sources = [primary / "summary.csv", primary / "manifest.json", fisher / "summary.csv",
               fisher / "manifest.json", fisher / "marker_validation.csv",
               ROOT / "simulations/study_05_inference_audit.py",
               ROOT / "simulations/study_06_bootstrap_ranking.py",
               ROOT / "simulations/study_07_information_bound.py",
               ROOT / "simulations/validate_information_bound.py",
               primary / "convergence_sensitivity.csv",
               ROOT / "simulations/inference_distributions.py",
               primary / "distribution_migration.json", bootstrap / "distribution_migration.json"]
    if bootstrap_complete:
        sources += [bootstrap / "summary.csv", bootstrap / "manifest.json", bootstrap / "calibration_check.json"]
    if (validation / "protocol.json").exists():
        sources += [validation / "protocol.json", ROOT / "simulations/study_08_bootstrap_null_validation.py"]
    if validation_complete:
        sources += [validation / "summary.csv", validation / "paired.json", validation / "verification.json", validation / "recomputed_verification.json"]
    payload = {
        "schema": "sem-model-audit-v1", "simulation_only": True,
        "complete": bootstrap_complete,
        "main": {"complete": True, "manifest": read_json(primary / "manifest.json"),
                 "rows": records(primary / "summary.csv"),
                 "verification": read_json(primary / "verification.json")["counts"],
                 "convergence_sensitivity": records(primary / "convergence_sensitivity.csv")},
        "fisher": {"complete": True, "manifest": read_json(fisher / "manifest.json"),
                   "rows": records(fisher / "summary.csv"),
                   "validation": records(fisher / "marker_validation.csv")},
        "bootstrap": {"complete": bootstrap_complete,
                      "manifest": read_json(bootstrap / "manifest.json") if (bootstrap / "manifest.json").exists() else None,
                      "rows": records(bootstrap / "summary.csv") if bootstrap_complete else [],
                      "calibration_check": calibration_check if bootstrap_complete else None},
        "bootstrap_validation": {"complete": validation_complete,
                      "manifest": read_json(validation / "protocol.json") if (validation / "protocol.json").exists() else None,
                      "rows": records(validation / "summary.csv") if validation_complete else [],
                      "paired": read_json(validation / "paired.json") if validation_complete else None,
                      "verification": read_json(validation / "verification.json") if validation_complete else None},
        "reproducibility": {"required": ["Python >=3.9", "numpy", "pandas"],
            "optional_figures": "matplotlib",
            "primary_distribution_migration": read_json(primary / "distribution_migration.json"),
            "bootstrap_distribution_migration": read_json(bootstrap / "distribution_migration.json")},
        "assumptions": {
            "alternative_coefficients": [.45, .30, -.25, -.20],
            "predictor_correlation": .35, "indicators_per_factor": 3,
            "latent_factors": 5, "indicator_errors": "independent normal",
            "latent_distribution": "normal, population unit variances",
            "null_positive_coefficients": [.375, .375, -.25, -.20],
            "null_negative_coefficients": [.45, .30, -.225, -.225],
            "test": "two-sided, nominal alpha .05; no multiplicity adjustment",
            "calibration": "95th percentile of absolute naive z over independent null draws, numpy quantile(method='higher')",
            "primary_replicates_per_cell_per_stage": 1000,
            "bootstrap_outer_replicates_per_cell": 300,
            "bootstrap_inner_resamples": 399,
            "failures": "all-attempt rejection denominators; failed/nonfinite fits count as non-rejections; conditional rates separately labeled",
        },
        "source_sha256": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sources},
        "interpretation": {
            "main": "Independent null-calibration, held-out-null, and alternative simulations. Naive rejection under the alternative is not calibrated power.",
            "calibrated": "DGP-specific simulated null cutoff: a diagnostic, not an implementable real-data correction. Display two-sample Monte Carlo bands, which include cutoff uncertainty.",
            "oracle": "OLS with intercept on unobserved population-unit latent scores; exact t critical value. This oracle knows the latent units.",
            "fisher": "Asymptotic efficient Gaussian Wald benchmark for a freely estimated congeneric 40-parameter covariance model; not a universal or finite-sample power ceiling.",
            "bootstrap": "Implementable row-resampling contrast intervals refit all measurement and structural parameters. Only unused fit-index diagnostics are omitted within resamples; paired-seed equivalence is tested.",
            "scope": "Five normal latent variables, four correlated predictors (rho=.35), three independent-error reflective indicators per factor. No workplace observations, missingness, nonnormality, cross-loadings or correlated errors.",
            "intervals": "Pointwise 95% Monte Carlo intervals. Wilson intervals for rates; two-sample resampling intervals for calibrated diagnostic rates. No simultaneous coverage claim.",
        },
    }
    destination = destination or ROOT / "docs/model-audit-data.json"
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(destination)
    downloads = ROOT / "docs/model-audit-downloads"
    downloads.mkdir(exist_ok=True)
    for folder, names in [(primary, ["summary.csv", "threshold_comparison.csv", "convergence_sensitivity.csv",
                                    "figure_alternative.png", "figure_alternative.pdf",
                                    "figure_null_validation.png", "figure_null_validation.pdf"]),
                          (fisher, ["summary.csv", "marker_validation.csv"])]:
        for name in names:
            source = folder / name
            if source.exists():
                shutil.copyfile(source, downloads / f"{folder.name}_{name}")
    if bootstrap_complete:
        shutil.copyfile(bootstrap / "summary.csv", downloads / "bootstrap_ranking_summary.csv")
        shutil.copyfile(bootstrap / "calibration_check.json", downloads / "bootstrap_original_calibration_check.json")
    if (validation / "protocol.json").exists():
        shutil.copyfile(validation / "protocol.json", downloads / "bootstrap_null_validation_protocol.json")
    if validation_complete:
        for name in ("summary.csv", "paired.json", "verification.json"):
            shutil.copyfile(validation / name, downloads / f"bootstrap_null_validation_{name}")
    return payload


if __name__ == "__main__":
    result = export()
    print(json.dumps({"primary_rows": len(result["main"]["rows"]),
                      "bootstrap_complete": result["bootstrap"]["complete"],
                      "bootstrap_rows": len(result["bootstrap"]["rows"])}))
