"""Build the repository methods note directly from checked audit outputs."""
from pathlib import Path
import sys
import json

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.check_inference_audit import check


def percent(value):
    return f"{100 * value:.1f}%"


def build():
    folder = ROOT / "results/inference_audit"
    check(folder, 54, 1000)
    summary = pd.read_csv(folder / "summary.csv")
    bound = pd.read_csv(ROOT / "results/information_bound/summary.csv")
    rows = []
    for pair, label in [("positive", ".15 contrast"), ("negative", "−.05 contrast")]:
        for loading in (.5, .75, .9):
            alt = summary[(summary.pair == pair) & (summary.loading == loading) & (summary.multiplier == 1) & (summary.stage == "alternative")].iloc[0]
            null = summary[(summary.cell == alt.cell) & (summary.stage == "null_validation")].iloc[0]
            interval = f"{percent(alt.calibrated_two_sample_bootstrap_low)}–{percent(alt.calibrated_two_sample_bootstrap_high)}"
            rows.append(f"| {label} | {loading:.2f} | {int(alt.n):,} | {percent(alt.oracle_rejection_rate)} | {percent(alt.observed_naive_rejection_rate)} | {percent(null.observed_naive_rejection_rate)} | {percent(alt.observed_calibrated_rejection_rate)} ({interval}) | {percent(null.observed_calibrated_rejection_rate)} |")
    table = "\n".join(rows)
    bound_rows = []
    for pair, label in [("positive", ".15 contrast"), ("negative", "−.05 contrast")]:
        for loading in (.5, .75, .9):
            row = bound[(bound.pair == pair) & (bound.loading == loading) & (bound.stage == "alternative")].iloc[0]
            bound_rows.append(f"| {label} | {loading:.2f} | {percent(row.asymptotic_power_at_advertised_n)} | {int(row.asymptotic_n_for_80pct_power):,} | {row.variance_inflation_vs_standardized_latent:.2f}× |")
    fisher_table = "\n".join(bound_rows)
    boot_folder = ROOT / "results/bootstrap_ranking"
    if (boot_folder / "summary.csv").exists():
        check(boot_folder, 4, 300, bootstrap=True)
        boot = pd.read_csv(boot_folder / "summary.csv")
        boot_rows = []
        for pair, label in [("positive", ".15 contrast"), ("negative", "−.05 contrast")]:
            alt = boot[(boot.pair == pair) & (boot.stage == "alternative")].iloc[0]
            null = boot[(boot.pair == pair) & (boot.stage == "null_validation")].iloc[0]
            boot_rows.append(f"| {label} | {int(alt.n):,} | {percent(alt.percentile_reject_rate)} ({percent(alt.percentile_reject_rate_low)}–{percent(alt.percentile_reject_rate_high)}) | {percent(null.percentile_reject_rate)} ({percent(null.percentile_reject_rate_low)}–{percent(null.percentile_reject_rate_high)}) | {percent(alt.percentile_coverage_rate)} |")
        boot_text = """All four cells are complete. These are independent datasets from the main simulation. The intervals in parentheses are pointwise 95% Wilson Monte Carlo intervals across 300 outer surveys.

| Contrast | N | Percentile-bootstrap rejection under alternative | Held-out null false positives | Coverage of true alternative contrast |
|---|---:|---:|---:|---:|
""" + "\n".join(boot_rows)
    else:
        boot_text = "**Follow-up pending.** No partial bootstrap rejection or coverage estimates are published. The public data marks this block incomplete until all four 300-survey cells and their summary pass verification."
    validation_folder = ROOT / "results/bootstrap_null_validation"
    validation_text = "The independent follow-up is pending. No partial validation rates are published."
    if (validation_folder / "summary.csv").exists():
        from tools.check_bootstrap_null_validation import check as check_validation
        check_validation()
        validation_table = pd.read_csv(validation_folder / "summary.csv")
        validation_lines = ["| Inner resamples | Null rejection | 95% Wilson interval | 95% exact interval |", "|---:|---:|---:|---:|"]
        for _, row in validation_table.iterrows():
            validation_lines.append(f"| {int(row.resamples)} | {percent(row.false_positive_rate)} | {percent(row.false_positive_rate_low)}–{percent(row.false_positive_rate_high)} | {percent(row.exact_cp_low)}–{percent(row.exact_cp_high)} |")
        paired = json.loads((validation_folder / "paired.json").read_text())
        validation_text = "\n".join(validation_lines) + f"\n\nPaired B999−B399 difference: {100*paired['b999_minus_b399']:.2f} percentage points (95% paired resampling interval {100*paired['paired_bootstrap_low']:.2f} to {100*paired['paired_bootstrap_high']:.2f})."
        validation_text += f" Only B999 rejected in {paired['b999_only_reject']} surveys; only B399 rejected in {paired['b399_only_reject']}."
        if paired['b999_only_reject'] + paired['b399_only_reject'] == 0:
            validation_text += " The degenerate empirical paired interval reflects zero observed discordance, not proof of identical population rejection probabilities."
        verification = json.loads((validation_folder / "verification.json").read_text())
        validation_text += f"\n\nAll-attempt denominators include every one of the {verification['outer_reps']:,} outer surveys. Outer estimates: {verification['outer_valid']:,} computable, {verification['outer_proper']:,} proper, and {verification['outer_converged']:,} converged. Of {verification['bootstrap_refits']:,} attempted inner refits, {verification['bootstrap_failed']:,} failed, {verification['bootstrap_improper']:,} were improper, and {verification['bootstrap_nonconverged']:,} did not converge. A computable interval is not necessarily based only on proper or converged refits."
        validation_text += "\n\nIntervals containing 5% indicate compatibility at this Monte Carlo precision; they do not establish equivalence or certify calibration. The follow-up tests one Gaussian null condition and does not estimate alternative power. [Separate replication bundle](model-audit-downloads/bootstrap-null-validation-replication.zip), [checksum](model-audit-downloads/bootstrap-null-validation-replication.zip.sha256), and [summary](model-audit-downloads/bootstrap_null_validation_summary.csv)."
    note = f"""# Can the advertised sample sizes establish coefficient differences?

**A controlled simulation audit of this repository's two-step SEM estimator. All observations are generated; none describe a real workforce.**

The sample-size formulas behind N=770 and N=6,884 use latent-regression uncertainty. The operational estimator first estimates measurement loadings and reliability from noisy indicators. This audit checks the resulting mismatch on the actual estimator, with separate checks of false positives, detection under alternatives, point-estimate ordering, and global fit.

The general issue of uncertainty from estimated factors is established in related methods work, including [A Small Sample Correction for Factor Score Regression](https://pmc.ncbi.nlm.nih.gov/articles/PMC10177321/). This audit's contribution is the controlled falsification of these particular advertised thresholds and an independently verified information calculation. The repository's disattenuated-composite two-step estimator is not assumed to be identical to other factor-score regression procedures.

## Design fixed before the main run

The generating paths are `(0.45, 0.30, −0.25, −0.20)`, four equicorrelated normal predictors with correlation .35, and one normal outcome factor. All factors have population variance one. Each factor has three reflective indicators with common loading .50, .75, or .90 and independent normal errors. The fitted graph and measurement model are correctly specified. No missingness, selection, cross-loadings, correlated errors, or nonnormality is introduced.

- Positive contrast: `β₀−β₁=.15`; N=385, 770, 1,540. Its null uses `(.375, .375, −.25, −.20)`.
- Negative contrast: `β₂−β₃=−.05`; N=3,442, 6,884, 13,768. Its null uses `(.45, .30, −.225, −.225)`.
- Each null preserves the tested pair's mean and leaves the other two paths unchanged. Outcome disturbance variance is recomputed from that model; the null does not reuse the alternative's R².
- Eighteen cells each contain 1,000 independent null-calibration draws, 1,000 independent null-validation draws, and 1,000 independent alternative draws: **54,000 full estimator fits**.
- Every observed-data draw is paired with oracle OLS on that draw's unobserved latent scores. Streams are deterministic and distinct across cells, stages, and replications.

Tests are two-sided at nominal α=.05. These are pointwise tests; there is no multiple-comparison correction. The independent latent-unit oracle uses an intercept and Student-t reference with N−5 degrees of freedom. The two-step test reconstructs the full regression covariance for the contrast, including covariance between its two coefficients; it does not add their marginal variances as though they were independent.

## What the advertised thresholds deliver

The uncalibrated two-step alternative rejection rate must be read together with its null false-positive rate. A procedure that rejects nearly half of true nulls does not have a valid 5%-level power claim.

The calibrated column uses the 95th percentile of absolute naive z values from the separate null-calibration sample (`quantile(method='higher')`). Its false-positive rate is then evaluated on independent null-validation draws. This is a **DGP-specific simulation diagnostic**, not a correction available to an analyst with unknown population parameters. Its uncertainty interval resamples both independent calibration and evaluation samples, including uncertainty in the simulated cutoff.

| Contrast | Loading | Advertised N | Oracle power | Uncalibrated rejection | Uncalibrated null false positives | Calibrated diagnostic rejection (95% MC interval) | Held-out calibrated false positives |
|---|---:|---:|---:|---:|---:|---:|---:|
{table}

At loading .75, both threshold cells have conventional fit in all 1,000 alternative draws. Yet calibrated evidence for a nonzero contrast appears in only about half the draws. This does **not** mean only half of point estimates order the coefficients correctly: correct signs occur in 96.3% and 97.1% of these draws. Point ordering and evidence that the population coefficients differ are distinct quantities.

Increasing N does not cure the naive uncertainty formula. For the negative-pair null with loading .50, false positives are 46.4%, 47.9%, and 47.5% at N=3,442, 6,884, and 13,768. At loading .75 they are 19.5%, 19.0%, and 17.8%. The larger sample makes estimates tighter while leaving the missing measurement uncertainty unaccounted for.

At loading .50 and the advertised thresholds, the fraction of conventionally well-fitting alternative draws with the wrong coefficient ordering is 16.0% for the positive pair and 14.3% for the negative pair. Conventional fit is a report of covariance discrepancy, not a certificate of contrast precision or causal interpretation.

## Is this merely an inefficient two-step estimator?

A separate expected Gaussian Fisher-information calculation uses the corresponding freely estimated congeneric covariance model: 15 loadings, 15 uniquenesses, and ten latent correlations, with latent variances fixed to one. There are 40 free covariance parameters and 80 model degrees of freedom. The structural regression is saturated among the five factors; its standardized contrast is a differentiable transformation of the ten correlations.

For covariance Σ and parameter derivatives Σₐ, information per observation is `Iₐᵦ = tr(Σ⁻¹ΣₐΣ⁻¹Σᵦ)/2`. For contrast gradient g, the asymptotic efficient variance is `g′I⁻¹g/N`. Loading and uniqueness nuisance parameters remain in the matrix inversion. Expected 80% Wald sample sizes use `(z_.975 + z_.80)² × variance_per_observation / difference²`.

| Contrast | Loading | Asymptotic efficient Wald power at advertised N | Approximate N for 80% | Variance relative to standardized latent oracle |
|---|---:|---:|---:|---:|
{fisher_table}

At loading .75 the efficient benchmark itself is about 49%, with approximate 80% sample sizes of 1,611 and 14,524. This supports measurement-information loss as the main explanation, not merely two-step inefficiency. It is an **asymptotic benchmark under this freely estimated congeneric model**, not a universal finite-sample power ceiling; known measurement parameters or a different instrument change the calculation.

The paired raw-unit oracle OLS knows the population units of the latent variables. Randomly standardizing observed latent samples is a different uncertainty problem. The information output therefore also includes a separate standardized-latent benchmark, retaining latent standard-deviation nuisance parameters, so the scale distinction is visible rather than concealed.

Two independent parameterizations validate the calculation. Analytic derivatives under unit latent variances are compared with numerical derivatives under marker identification (five fixed first loadings, ten free remaining loadings, 15 uniquenesses, and 15 free latent variances/covariances). Across eight contrast/loading cases the information has rank 40 and the relative variance differences are below 1.2×10⁻⁹. Additional tests check all covariance derivatives, contrast gradients, direct-trace information, and the near-perfect-measurement limit.

## Implementable bootstrap follow-up

The follow-up uses loading .75 at N=770 and N=6,884, each under its null and alternative. For each of 300 independent outer surveys per cell, 399 nonparametric resamples draw complete rows and refit all measurement and structural parameters. Contrasts are calculated *within* resamples so coefficient covariance is preserved. Percentile and bootstrap-SE normal intervals are both retained; no truth-dependent null cutoff is used.

Only unused global-fit diagnostics are omitted inside bootstrap refits. Outer/base fits retain them. Matched-seed tests establish bitwise identical contrast draws and failure, improper, and convergence counts when this diagnostic computation is omitted. The optimization saves repeated RMSEA confidence-limit calculations; it changes neither the estimator nor its resampling distribution.

{boot_text}

With 300 outer surveys, Monte Carlo SE is about 1.3 percentage points for a 5% rate and 2.9 points for a 50% rate. With 399 inner resamples, a 2.5% tail contains about ten draws. These limits are reported rather than treated as exact calibration guarantees.

The negative-null percentile result is 23/300 (7.7%). Its Wilson 95% interval is 5.16%–11.24%, but the exact equal-tailed Clopper–Pearson interval is 4.92%–11.28%, which includes the nominal 5%. The exact upper-tail p-value is .0292 and the probability-ordering two-sided p-value is .0452; a conservative adjustment across the two original null conditions gives .0903. The exact interval is equal-tailed, while the exact two-sided test orders outcomes by their probability, so their 5% decisions need not coincide. These exploratory calculations motivate a calibration concern, not proof of miscalibration and not a claim that bootstrap is a certified remedy.

## Independent null-only follow-up

A separate prospective protocol, recorded before its main computation, responds to that concern. It fixes 1,000 **fresh-seed** outer null surveys at N=6,884 and loading .75, with 999 inner resamples each. The first 399 attempted resamples and all 999 are compared within each survey to inspect finite-bootstrap variability. The primary estimand is the B999 percentile interval's null rejection rate; the paired B999−B399 difference is secondary. Original 300-survey results remain separate and are never pooled or overwritten.

The target is fixed at 1,000 surveys with no outcome-dependent early stopping. This is an explicitly exploratory follow-up prompted by the original result, not a retrospective claim of preregistration for the original study. At a 5% null rate, Monte Carlo SE is approximately .7 percentage points. [Prospective protocol](model-audit-downloads/bootstrap_null_validation_protocol.json).

{validation_text}

## Failures, convergence, and uncertainty

All 54,000 main fits completed and produced finite tests; all were proper under the recorded Heywood, positive-definiteness-repair, and R² checks. Measurement iterations converged in 53,871. The 129 flagged fits occur only at loading .50, N=385: 44 calibration, 41 validation, and 44 alternative fits. They are retained and flagged in the primary analysis.

Excluding nonconverged fits from both calibration and evaluation changes that cell's alternative diagnostic rejection from 10.9% to 10.98%, and null false positives from 4.8% to 4.797%. Neither advertised-threshold headline is affected. `convergence_sensitivity.csv` retains the denominators and uncertainty intervals.

The code uses explicit fit-status masks. Failures never become zero coefficient estimates. Main all-attempt rejection denominators count a failed/nonfinite test as a non-rejection; valid-fit conditional rates are separately labeled. Wilson intervals quantify pointwise Monte Carlo rate uncertainty, and the calibrated diagnostic also has a two-sample resampling interval for cutoff uncertainty. These are not confidence intervals for real workplace effects.

## Reproduction and provenance

The core audit needs Python ≥3.9, numpy, and pandas. Matplotlib is optional for exported figures. Run from the repository root:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m simulations.study_05_inference_audit --out results/inference_audit --reps 1000 --workers 2
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m simulations.study_06_bootstrap_ranking --out results/bootstrap_ranking --reps 300 --n-boot 399 --workers 4
OPENBLAS_NUM_THREADS=1 python -m simulations.study_07_information_bound
OPENBLAS_NUM_THREADS=1 python -m simulations.validate_information_bound
OPENBLAS_NUM_THREADS=1 python tools/convergence_sensitivity.py
OPENBLAS_NUM_THREADS=1 python tools/check_inference_audit.py
OPENBLAS_NUM_THREADS=1 python tools/export_model_audit.py
OPENBLAS_NUM_THREADS=1 python tools/plot_inference_audit.py
OPENBLAS_NUM_THREADS=1 python tools/build_model_audit_note.py
```

Checkpoint streams retain every replicate and seed. Manifests and verification reports record code and output SHA-256 digests. The exporter refuses an incomplete primary table and exposes no partial bootstrap estimates. Existing checkpoints resume without duplicating rows.

Initial reference distribution calls used SciPy, which conflicts with the repository's dependency policy. Equivalent standard-library incomplete-beta/Student-t and normal functions replaced those calls. All 54,000 saved primary oracle decisions and all 532 completed bootstrap decisions at migration were verified unchanged; critical values differ by at most 1.05×10⁻¹² on the design grid. Fisher outputs changed by at most 2.22×10⁻¹⁶, with identical sample-size recommendations. Original execution manifests/source snapshots and explicit migration records are retained. The estimator and generated draws were unchanged, and the new audit runs with SciPy imports blocked.

Source artifacts: [primary summary](model-audit-downloads/inference_audit_summary.csv), [Fisher benchmark](model-audit-downloads/information_bound_summary.csv), [independent parameterization check](model-audit-downloads/information_bound_marker_validation.csv), and [public workbench data](model-audit-data.json). The [interactive audit](model-audit.html) presents the same checked tables.
"""
    path = ROOT / "docs/model-audit-methods.md"
    path.write_text(note)
    return path


if __name__ == "__main__":
    print(build())
