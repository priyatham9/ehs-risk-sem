# Does the row bootstrap remain usable with weak measurement?

This is an isolated implementation stress test of the repository's unchanged two-step estimator and row-bootstrap procedure. All observations are simulated. It adds one null condition and makes no alternative-power claim.

## Why this condition

The original fixed audit included loading .50 and N=385 for the positive contrast. That cell produced measurement nonconvergence in the original study. This follow-up selects that existing condition to ask whether a procedure tested at loading .75 remains computationally and inferentially usable under jointly lower loading and smaller N. It is not a search across new conditions for unfavorable results.

The protocol was recorded at **2026-11-16T15:17:17.649714+00:00**, before main computation. A separate five-survey timing-only pilot took 36.119 seconds for 399 resamples per survey; its outcome values were discarded and not inspected. The prospective runtime-only rule chose **500 outer surveys**, with **399 inner resamples each** and four workers. The forecast included 25% scheduling allowance and was about 18.8 minutes. The sample size is fixed, with no outcome-dependent stopping or extension.

## Generating model and unchanged procedure

The condition is the positive-pair equality null: coefficients [.375, .375, −.25, −.20], loading .50, N=385, predictor correlation .35, five Gaussian latent variables and three independent-error indicators per factor. There is no missingness, cross-loading, correlated measurement error or workplace observation.

The implementation calls the frozen `study_06_bootstrap_ranking.bootstrap_replicate` function. Whole survey rows are resampled, and the full measurement and structural estimator is refitted. Only unused global-fit diagnostics are suppressed inside resamples; outer fits retain them. Contrasts are computed within each resample so coefficient covariance is preserved.

Bootstrap computation is attempted only when the outer contrast estimate and naive standard error are finite and the standard error is positive. A percentile or bootstrap-SE normal interval is computable when at least 380 of 399 attempted inner refits produce finite contrasts. The percentile interval uses NumPy's linear 2.5%/97.5% quantiles. Rejection requires the interval strictly to exclude zero. Finite improper and nonconverged fits remain included, exactly as in the frozen procedure; they are not silently filtered or replaced.

Here “proper” means no flagged Heywood estimate, no latent-correlation positive-definiteness repair and no structural R² above one, using the repository's existing checks. It is not a claim that every model assumption holds.

## Outcomes and Monte Carlo uncertainty

The primary outcome is percentile null rejection over all attempted outer surveys. Naive and normal-bootstrap rejection are secondary. All-attempt denominators count skipped or invalid tests as nonrejections. Computable-only denominators condition on the relevant procedure producing a computable test and can describe a selected subset. A low all-attempt rate driven by failures is not evidence of calibration.

Pointwise Wilson and exact Clopper–Pearson 95% Monte Carlo intervals describe variability over independent generated outer surveys. At a 5% rate, 500 surveys give approximately one percentage point of Monte Carlo standard error. Paired rejection differences use 20,000 multinomial resamples of the within-survey decision table. Inner failure/status counts are nested within surveys and are descriptive; they are not treated as independent binomial trials.

| Method | Denominator | Rejections / surveys | Null rejection | 95% Wilson MC | 95% exact MC |
|---|---|---:|---:|---:|---:|
| naive | All attempted surveys | 233 / 500 | 46.60% | 42.27%–50.98% | 42.16%–51.08% |
| naive | Computable tests only | 233 / 500 | 46.60% | 42.27%–50.98% | 42.16%–51.08% |
| percentile | All attempted surveys | 26 / 500 | 5.20% | 3.57%–7.51% | 3.42%–7.53% |
| percentile | Computable tests only | 26 / 500 | 5.20% | 3.57%–7.51% | 3.42%–7.53% |
| normal | All attempted surveys | 18 / 500 | 3.60% | 2.29%–5.62% | 2.15%–5.63% |
| normal | Computable tests only | 18 / 500 | 3.60% | 2.29%–5.62% | 2.15%–5.63% |

The comparisons below are paired within the same generated surveys. Differences are bootstrap minus naive rejection, using all attempted surveys.

| Method | Difference, percentage points | 95% paired MC interval | Bootstrap only / naive only / both |
|---|---:|---:|---:|
| percentile | -41.40 | -45.80 to -37.00 | 0 / 207 / 26 |
| normal | -43.00 | -47.40 to -38.60 | 0 / 215 / 18 |

Outer surveys: 500 attempted; 500 completed fits; 500 computable naive tests; 500 proper fits; 474 converged fits. Bootstrap intervals were computable for 500 surveys and invalid or skipped for 0.

Of 199,500 planned inner resamples, 199,500 were attempted and 199,500 produced finite contrasts. There were 0 failed refits, 5,715 improper computable refits and 61,722 nonconverged computable refits. Improper and nonconverged flags can overlap. They are retained in the unchanged procedure.

Surveys with at least one failed refit: 0 / 500 (0.00%; outer-survey Wilson interval 0.00%–0.76%).

Surveys with at least one improper computable refit: 451 / 500 (90.20%; outer-survey Wilson interval 87.28%–92.51%).

Surveys with at least one nonconverged computable refit: 500 / 500 (100.00%; outer-survey Wilson interval 99.24%–100.00%).


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
