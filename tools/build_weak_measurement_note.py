"""Build the separate stress-study note from protocol and verified complete outputs."""
from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.check_bootstrap_weak_measurement import check


def pct(value):
    return 'unavailable' if pd.isna(value) else f'{100 * value:.2f}%'


def build(check_only=False):
    folder = ROOT / 'results/bootstrap_weak_measurement'
    plan = json.loads((folder / 'protocol.json').read_text())
    timing = json.loads((folder / 'timing_pilot.json').read_text())
    result = '**Pending fixed-sample computation. No partial outcome rates are reported.**'
    if (folder / 'summary.csv').exists():
        check()
        rows = pd.read_csv(folder / 'summary.csv')
        lines = ['| Method | Denominator | Rejections / surveys | Null rejection | 95% Wilson MC | 95% exact MC |',
                 '|---|---|---:|---:|---:|---:|']
        for _, row in rows.iterrows():
            denominator = 'All attempted surveys' if row.conditioning == 'all_attempts' else 'Computable tests only'
            lines.append(f'| {row.method} | {denominator} | {int(row.rejection_rate_count)} / {int(row.rejection_rate_denominator)} | {pct(row.rejection_rate)} | {pct(row.rejection_rate_low)}–{pct(row.rejection_rate_high)} | {pct(row.exact_cp_low)}–{pct(row.exact_cp_high)} |')
        result = '\n'.join(lines)
        paired = json.loads((folder / 'paired.json').read_text())
        result += '\n\nThe comparisons below are paired within the same generated surveys. Differences are bootstrap minus naive rejection, using all attempted surveys.\n\n'
        result += '| Method | Difference, percentage points | 95% paired MC interval | Bootstrap only / naive only / both |\n|---|---:|---:|---:|\n'
        for method, row in paired.items():
            result += f"| {method} | {100*row['bootstrap_minus_naive']:.2f} | {100*row['paired_bootstrap_low']:.2f} to {100*row['paired_bootstrap_high']:.2f} | {row['bootstrap_only_reject']} / {row['naive_only_reject']} / {row['both_reject']} |\n"
            if row['bootstrap_only_reject'] + row['naive_only_reject'] == 0:
                result += '\nA degenerate empirical paired interval with zero observed discordance does not prove population equality.\n'
        a = json.loads((folder / 'accounting.json').read_text())
        result += f"\nOuter surveys: {a['outer_attempts']:,} attempted; {a['observed_completed']:,} completed fits; {a['observed_valid']:,} computable naive tests; {a['observed_proper']:,} proper fits; {a['observed_converged']:,} converged fits. Bootstrap intervals were computable for {a['bootstrap_valid']:,} surveys and invalid or skipped for {a['invalid_intervals']:,}.\n"
        result += f"\nOf {a['inner_planned']:,} planned inner resamples, {a['inner_attempted']:,} were attempted and {a['inner_computable']:,} produced finite contrasts. There were {a['bootstrap_failed']:,} failed refits, {a['bootstrap_improper']:,} improper computable refits and {a['bootstrap_nonconverged']:,} nonconverged computable refits. Improper and nonconverged flags can overlap. They are retained in the unchanged procedure.\n"
        for field, label in [('bootstrap_failed', 'at least one failed refit'), ('bootstrap_improper', 'at least one improper computable refit'), ('bootstrap_nonconverged', 'at least one nonconverged computable refit')]:
            incidence = a['outer_with_' + field]
            result += f"\nSurveys with {label}: {incidence['rate_count']} / {incidence['rate_denominator']} ({pct(incidence['rate'])}; outer-survey Wilson interval {pct(incidence['rate_low'])}–{pct(incidence['rate_high'])}).\n"
    note = f"""# Does the row bootstrap remain usable with weak measurement?

This is an isolated implementation stress test of the repository's unchanged two-step estimator and row-bootstrap procedure. All observations are simulated. It adds one null condition and makes no alternative-power claim.

## Why this condition

The original fixed audit included loading .50 and N=385 for the positive contrast. That cell produced measurement nonconvergence in the original study. This follow-up selects that existing condition to ask whether a procedure tested at loading .75 remains computationally and inferentially usable under jointly lower loading and smaller N. It is not a search across new conditions for unfavorable results.

The protocol was recorded at **{plan['recorded_before_main_computation_utc']}**, before main computation. A separate five-survey timing-only pilot took {timing['single_worker_seconds']:.3f} seconds for 399 resamples per survey; its outcome values were discarded and not inspected. The prospective runtime-only rule chose **{plan['outer_reps']} outer surveys**, with **399 inner resamples each** and four workers. The forecast included 25% scheduling allowance and was about {plan['forecast_seconds']/60:.1f} minutes. The sample size is fixed, with no outcome-dependent stopping or extension.

## Generating model and unchanged procedure

The condition is the positive-pair equality null: coefficients [.375, .375, −.25, −.20], loading .50, N=385, predictor correlation .35, five Gaussian latent variables and three independent-error indicators per factor. There is no missingness, cross-loading, correlated measurement error or workplace observation.

The implementation calls the frozen `study_06_bootstrap_ranking.bootstrap_replicate` function. Whole survey rows are resampled, and the full measurement and structural estimator is refitted. Only unused global-fit diagnostics are suppressed inside resamples; outer fits retain them. Contrasts are computed within each resample so coefficient covariance is preserved.

Bootstrap computation is attempted only when the outer contrast estimate and naive standard error are finite and the standard error is positive. A percentile or bootstrap-SE normal interval is computable when at least 380 of 399 attempted inner refits produce finite contrasts. The percentile interval uses NumPy's linear 2.5%/97.5% quantiles. Rejection requires the interval strictly to exclude zero. Finite improper and nonconverged fits remain included, exactly as in the frozen procedure; they are not silently filtered or replaced.

Here “proper” means no flagged Heywood estimate, no latent-correlation positive-definiteness repair and no structural R² above one, using the repository's existing checks. It is not a claim that every model assumption holds.

## Outcomes and Monte Carlo uncertainty

The primary outcome is percentile null rejection over all attempted outer surveys. Naive and normal-bootstrap rejection are secondary. All-attempt denominators count skipped or invalid tests as nonrejections. Computable-only denominators condition on the relevant procedure producing a computable test and can describe a selected subset. A low all-attempt rate driven by failures is not evidence of calibration.

Pointwise Wilson and exact Clopper–Pearson 95% Monte Carlo intervals describe variability over independent generated outer surveys. At a 5% rate, 500 surveys give approximately one percentage point of Monte Carlo standard error. Paired rejection differences use 20,000 multinomial resamples of the within-survey decision table. Inner failure/status counts are nested within surveys and are descriptive; they are not treated as independent binomial trials.

{result}

## Interpretation limits

This is one exploratory Gaussian null stress condition. A rate interval containing 5% does not establish equivalence or certify calibration. Reduced rejection relative to a severely miscalibrated naive test does not by itself establish satisfactory calibration. Computational completion, propriety and convergence describe different properties from the rejection rate.

The loading .75 studies remain separate. Their surveys are not paired with these surveys, and their outcomes are not pooled with this experiment. Both loading and sample size differ, so a cross-study difference would not isolate a loading effect. The new outer and bootstrap stream IDs are checked against all previous recorded streams and the separate timing pilot. A saved-draw verifier independently recomputes interval endpoints, rejection decisions, conditional denominators, paired tables and status totals.

## Reproduce

```sh
OPENBLAS_NUM_THREADS=1 python tools/check_bootstrap_weak_measurement.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m simulations.study_09_bootstrap_weak_measurement --run --workers 4
python -m unittest discover -s tests -p 'test_bootstrap_weak_measurement.py' -v
```

The run command resumes the fixed, hash-bound protocol and checkpoints. For a new run directory, first execute `--out PATH --pilot`, inspect only its timing, then `--out PATH --prepare`; the runtime-only sample-size rule is part of the recorded design. Exact continuation uses the reference NumPy/Pandas runtime in the protocol. The estimator needs NumPy and Pandas; SciPy is not a runtime dependency.

[Prospective protocol](weak-measurement-downloads/protocol.json) · [Public data](weak-measurement-audit-data.json) · [Separate replication bundle](weak-measurement-downloads/replication.zip) · [SHA-256](weak-measurement-downloads/replication.zip.sha256)
"""
    target = ROOT / 'docs/weak-measurement-methods.md'
    if check_only:
        assert target.read_text() == note, 'Weak-measurement methods note is stale'
    else:
        target.write_text(note)
    print(target)


if __name__ == '__main__':
    build(check_only='--check' in sys.argv)
