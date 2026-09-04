"""Study 1. How much data does it take to recover the coefficients?

SIMULATION. All data are generated from a known model. Nothing here is an
empirical claim about workplace injuries.

Three questions, which safety-science papers usually collapse into one:

1. Is the estimator unbiased, and at what sample size does that become visible
   through the noise?
2. Do the analytic standard errors have nominal coverage? They should not: they
   condition on an estimated latent correlation matrix as though it were known.
   The study measures how far off they are and checks that a nonparametric
   bootstrap fixes it.
3. What sample size is needed to say that one coefficient is larger than
   another? This is the claim that gets made whenever four paths are reported
   and one is described as the dominant driver, and it needs far more data than
   estimating the paths themselves.

Run: ``python3 simulations/study_01_sample_size.py``
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ehs_risk_sem.compat import silence_accelerate_matmul_warnings
from ehs_risk_sem.model import ModelSpec, bootstrap_paths, fit
from ehs_risk_sem.power import (
    equicorrelated_matrix,
    min_n_for_path_se,
    min_n_for_rmsea_power,
    min_n_to_distinguish,
    se_of_difference,
)
from ehs_risk_sem.special import norm_ppf
from simulations.dgp import generate_dataset, make_four_factor_model

TRUE_BETA = np.array([0.45, 0.30, -0.25, -0.20])
SAMPLE_SIZES = (100, 200, 500, 1000, 2000, 5000)


def _spec(model) -> ModelSpec:
    return ModelSpec(
        latents={n: model.indicator_names[n] for n in model.latent_names},
        paths=model.paths(),
    )


def recovery_study(
    sample_sizes=SAMPLE_SIZES, n_reps: int = 300, seed: int = 20260903
) -> pd.DataFrame:
    """Bias, empirical variability, analytic-SE accuracy and coverage by N."""
    model = make_four_factor_model(beta=tuple(TRUE_BETA))
    spec = _spec(model)
    predictors = model.paths()["Risk"]
    z = norm_ppf(0.975)

    rows: List[dict] = []
    for n in sample_sizes:
        betas = np.zeros((n_reps, 4))
        ses = np.zeros((n_reps, 4))
        covered = np.zeros((n_reps, 4))
        improper = 0
        failed = 0
        for r in range(n_reps):
            data, _ = generate_dataset(model, n, seed=seed + 1000 * n + r)
            try:
                res = fit(spec, data)
            except Exception:  # noqa: BLE001
                failed += 1
                continue
            est = res.paths["Risk"]
            betas[r] = est.beta
            ses[r] = est.se
            lo = est.beta - z * est.se
            hi = est.beta + z * est.se
            covered[r] = (lo <= TRUE_BETA) & (TRUE_BETA <= hi)
            if res.phi_repaired or est.r_squared > 1.0:
                improper += 1

        used = n_reps - failed
        for j, name in enumerate(predictors):
            emp_sd = float(np.std(betas[:used, j], ddof=1))
            rows.append(
                {
                    "n": n,
                    "predictor": name,
                    "true_beta": float(TRUE_BETA[j]),
                    "mean_estimate": float(np.mean(betas[:used, j])),
                    "bias": float(np.mean(betas[:used, j]) - TRUE_BETA[j]),
                    "empirical_sd": emp_sd,
                    "mean_analytic_se": float(np.mean(ses[:used, j])),
                    "se_ratio_analytic_over_empirical": float(
                        np.mean(ses[:used, j]) / emp_sd
                    )
                    if emp_sd > 0
                    else np.nan,
                    "coverage_95_analytic": float(np.mean(covered[:used, j])),
                    "rmse": float(
                        np.sqrt(np.mean((betas[:used, j] - TRUE_BETA[j]) ** 2))
                    ),
                    "improper_solutions": improper,
                    "failed_fits": failed,
                    "n_reps": used,
                }
            )
    return pd.DataFrame(rows)


def bootstrap_coverage_study(
    n: int = 500, n_reps: int = 40, n_boot: int = 199, seed: int = 4242
) -> pd.DataFrame:
    """Compare analytic and bootstrap interval coverage at one sample size.

    Smaller ``n_reps`` than the recovery study because each replication refits
    the model ``n_boot`` times. The Monte Carlo error on a coverage estimate
    from 40 replications is about 6 percentage points, which is reported
    alongside the estimate so the comparison is not overread.
    """
    model = make_four_factor_model(beta=tuple(TRUE_BETA))
    spec = _spec(model)
    predictors = model.paths()["Risk"]
    z = norm_ppf(0.975)

    analytic_hits = np.zeros(4)
    boot_hits = np.zeros(4)
    analytic_width = np.zeros(4)
    boot_width = np.zeros(4)
    used = 0
    for r in range(n_reps):
        data, _ = generate_dataset(model, n, seed=seed + r)
        try:
            res = fit(spec, data)
            bs = bootstrap_paths(spec, data, n_boot=n_boot, seed=seed + 100000 + r)
        except Exception:  # noqa: BLE001
            continue
        used += 1
        est = res.paths["Risk"]
        for j, name in enumerate(predictors):
            lo_a, hi_a = est.beta[j] - z * est.se[j], est.beta[j] + z * est.se[j]
            analytic_hits[j] += float(lo_a <= TRUE_BETA[j] <= hi_a)
            analytic_width[j] += hi_a - lo_a
            row = bs[(bs["outcome"] == "Risk") & (bs["predictor"] == name)].iloc[0]
            boot_hits[j] += float(row["ci_low"] <= TRUE_BETA[j] <= row["ci_high"])
            boot_width[j] += row["ci_high"] - row["ci_low"]

    mc_se = np.sqrt(0.95 * 0.05 / max(used, 1))
    return pd.DataFrame(
        {
            "predictor": list(predictors),
            "true_beta": TRUE_BETA,
            "coverage_analytic": analytic_hits / max(used, 1),
            "coverage_bootstrap": boot_hits / max(used, 1),
            "mean_width_analytic": analytic_width / max(used, 1),
            "mean_width_bootstrap": boot_width / max(used, 1),
            "n_reps": used,
            "monte_carlo_se_of_coverage": mc_se,
        }
    )


def distinguishability_table(
    exogenous_correlation: float = 0.35, r_squared: float = 0.288
) -> pd.DataFrame:
    """Sample size needed to tell each pair of coefficients apart.

    Uses the exact covariance between two coefficients in the same standardized
    regression, so the correlation among predictors is accounted for rather
    than ignored.
    """
    corr = equicorrelated_matrix(4, exogenous_correlation)
    names = [
        "UnsafeActs",
        "OperationalStress",
        "SystemCondition",
        "SafetyResponseCapability",
    ]
    rows = []
    for i in range(4):
        for j in range(i + 1, 4):
            n_needed = min_n_to_distinguish(
                TRUE_BETA[i], TRUE_BETA[j], corr, r_squared, i=i, j=j
            )
            rows.append(
                {
                    "coefficient_a": names[i],
                    "beta_a": float(TRUE_BETA[i]),
                    "coefficient_b": names[j],
                    "beta_b": float(TRUE_BETA[j]),
                    "absolute_difference": abs(float(TRUE_BETA[i] - TRUE_BETA[j])),
                    "n_for_80pct_power": n_needed,
                    "se_of_difference_at_n_250": se_of_difference(
                        corr, r_squared, 250, i, j
                    ),
                }
            )
    return pd.DataFrame(rows)


def requirement_comparison(df_model: int = 80, r_squared: float = 0.288) -> pd.DataFrame:
    """The three sample-size requirements side by side."""
    corr = equicorrelated_matrix(4, 0.35)
    inv = np.linalg.inv(corr)
    vif = float(inv[0, 0])
    rows = [
        {
            "requirement": "global fit: 80% power for the test of close fit",
            "basis": f"MacCallum-style RMSEA power, df = {df_model}",
            "n_required": min_n_for_rmsea_power(df_model),
        },
        {
            "requirement": "path precision: SE = 0.05 on a single coefficient",
            "basis": f"standardized regression SE, R^2 = {r_squared:.3f}, VIF = {vif:.2f}",
            "n_required": min_n_for_path_se(0.05, r_squared, vif, 4),
        },
        {
            "requirement": "path precision: SE = 0.025 on a single coefficient",
            "basis": f"standardized regression SE, R^2 = {r_squared:.3f}, VIF = {vif:.2f}",
            "n_required": min_n_for_path_se(0.025, r_squared, vif, 4),
        },
        {
            "requirement": "distinguish beta = 0.45 from beta = 0.30 at 80% power",
            "basis": "z test on the coefficient difference, correlated predictors",
            "n_required": min_n_to_distinguish(0.45, 0.30, corr, r_squared, 0, 1),
        },
        {
            "requirement": "distinguish beta = -0.25 from beta = -0.20 at 80% power",
            "basis": "z test on the coefficient difference, correlated predictors",
            "n_required": min_n_to_distinguish(-0.25, -0.20, corr, r_squared, 2, 3),
        },
    ]
    return pd.DataFrame(rows)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=300)
    parser.add_argument("--boot-reps", type=int, default=40)
    parser.add_argument("--n-boot", type=int, default=199)
    parser.add_argument("--skip-bootstrap", action="store_true")
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args(argv)

    silence_accelerate_matmul_warnings()
    os.makedirs(args.outdir, exist_ok=True)
    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 40)

    print(__doc__)
    print("=" * 78)
    print("SIMULATED DATA. Generating model:")
    print(f"  four correlated exogenous latents (r = 0.35), loadings 0.75, 3 indicators each")
    print(f"  true standardized paths: {dict(zip(['UnsafeActs','OperationalStress','SystemCondition','SafetyResponseCapability'], TRUE_BETA))}")
    print("=" * 78)

    print("\n1. The three sample-size requirements are not the same number\n")
    req = requirement_comparison()
    print(req.to_string(index=False))
    req.to_csv(os.path.join(args.outdir, "study01_requirements.csv"), index=False)

    print("\n2. Recovery of the generating coefficients by sample size\n")
    rec = recovery_study(n_reps=args.reps)
    print(
        rec[
            [
                "n",
                "predictor",
                "true_beta",
                "mean_estimate",
                "bias",
                "empirical_sd",
                "mean_analytic_se",
                "se_ratio_analytic_over_empirical",
                "coverage_95_analytic",
            ]
        ].to_string(index=False)
    )
    rec.to_csv(os.path.join(args.outdir, "study01_recovery.csv"), index=False)

    worst = rec.groupby("n")["coverage_95_analytic"].min()
    print("\n   Minimum coverage of the nominal 95% analytic interval, by N:")
    for n, cov in worst.items():
        print(f"     n = {n:>5}: {cov:.3f}")
    print(
        "   The analytic interval conditions on an estimated latent correlation\n"
        "   matrix. Its coverage does not approach 0.95 as N grows, because the\n"
        "   omitted source of uncertainty does not vanish relative to the rest."
    )

    print("\n3. Sample size needed to tell two coefficients apart\n")
    dist = distinguishability_table()
    print(dist.to_string(index=False))
    dist.to_csv(os.path.join(args.outdir, "study01_distinguishability.csv"), index=False)

    if not args.skip_bootstrap:
        print("\n4. Analytic against bootstrap interval coverage at n = 500\n")
        boot = bootstrap_coverage_study(n_reps=args.boot_reps, n_boot=args.n_boot)
        print(boot.to_string(index=False))
        boot.to_csv(os.path.join(args.outdir, "study01_bootstrap.csv"), index=False)

    print(
        "\nWhat this study does not show: that the generating model is the right\n"
        "model. Recovery of a parameter from data generated by that same model is\n"
        "a check on the estimator, not evidence about safety."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
