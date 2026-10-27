# Good fit, imprecise paths: when structural equation models mislead at site-level sample sizes

Priyatham Chimmani

Methods note, ehs-risk-sem repository, version 0.1.0. All data simulated.

## Abstract

Weighted risk formulas of the form
`Risk = 0.45*UnsafeActs + 0.30*OperationalStress - 0.25*SystemCondition - 0.20*SafetyResponseCapability`
circulate in practitioner writing on safety analytics, presented as the output
of a structural equation model (SEM). This note reports simulation studies,
run with an open numpy SEM estimator, that ask what sample size such a model
needs before its coefficients mean anything. For a five-latent model with
three indicators per latent (df = 80), the test of close fit reaches 80% power
at N = 153. Estimating a single path to a standard error of 0.05 needs
N = 369. Telling a path of 0.45 from one of 0.30 needs N = 770, and telling
-0.25 from -0.20 needs N = 6,884. At N = 200 the empirical standard deviation
of a path estimate is about 0.10, so a 95% interval runs roughly 0.20 either
side of the estimate. Analytic standard errors from the two-step estimator are
too narrow at every sample size studied, with 95% coverage between 0.76 and
0.86; a percentile bootstrap restored coverage to 0.925 to 0.975 at N = 500.
Separate scenarios with reversed causal direction or an omitted common cause
meet every conventional fit cutoff while returning the wrong answer. The
practical message for safety teams is that passing fit cutoffs says little
about whether path coefficients can be ranked or used as weights. Every
number comes from generated data; no real injury data were analysed.

## 1. Problem

Safety analytics writing increasingly presents risk scores as weighted sums
of latent constructs, with weights attributed to an SEM. The formula above is
a representative example. A reader is invited to conclude that unsafe acts
are the dominant driver of risk, that operational stress matters more than
system condition, and that the weights can be applied to new data to score
crews or sites.

Each of those conclusions requires more than a well-fitting model. Ranking
two drivers requires the difference between their coefficients to be
estimated precisely. Applying weights requires the coefficients themselves to
be estimated precisely. Neither is checked by CFI, RMSEA or SRMR, which
summarise how well the model reproduces the indicator covariance matrix, not
how precisely any single parameter is known.

Sample size is where this matters most for safety teams. A model fitted to
one site's workforce survey, or to a small company's records, has at most a
few hundred rows when the workforce numbers in the hundreds. Published
rules of thumb for SEM sample size do not transfer across models: Wolf,
Harrington, Clark and Miller (2013) found requirements ranging from 30 to 460
cases across the model types they simulated, depending on the number of
factors, indicators, loadings and the size of the effects. The question for a
given model has to be answered for that model. This note answers it for the
model that the circulating formula implies.

## 2. Simulation design

All code is in `simulations/` and the estimator is in `ehs_risk_sem/`. The full
run is `python3 simulations/run_all.py`, which writes the CSV files in
`results/` that every number in Section 3 is read from.

**Estimator.** Limited-information, two-step estimation following Anderson and
Gerbing (1988): each latent's indicators are fitted as a congeneric one-factor
block by principal-axis factoring; latent correlations are obtained by
disattenuating composite correlations by composite reliability (with
projection to the nearest positive-definite matrix when needed, which is
reported); structural paths are solved by regression on the latent
correlation matrix. Fit indices (chi-square, CFI, RMSEA, SRMR) are evaluated
at the two-step estimates. This is not full-information maximum likelihood,
and Section 5 discusses the consequence.

**Study 1, generating model** (`simulations/dgp.py`, `make_four_factor_model`).
Four correlated exogenous latents (pairwise correlation 0.35) predict one
endogenous latent, Risk. Every latent has three indicators with standardized
loadings of 0.75. True standardized paths are 0.45, 0.30, -0.25 and -0.20, the
values in the circulating formula. The implied R-squared for Risk is 0.288 and
the model has df = 80.

**Study 1, analyses** (`simulations/study_01_sample_size.py`).

- Recovery: N = 100, 200, 500, 1,000, 2,000 and 5,000, with 300 replications
  per N. Replication r at sample size N uses seed 20260903 + 1000*N + r.
  Recorded per path: bias, empirical standard deviation of the estimates, mean
  analytic standard error, and 95% analytic interval coverage.
- Bootstrap: N = 500, 40 replications (seeds 4242 + r), each with 199
  nonparametric bootstrap resamples (seeds 4242 + 100000 + r) and percentile
  intervals.
- Analytic sample-size requirements: power of the test of close fit (RMSEA
  0.05 against 0.08, alpha 0.05) by the method of MacCallum, Browne and
  Sugawara (1996); N for a target standard error on one standardized path;
  and N for 80% power on a z test of the difference between two paths,
  using the covariance between the two coefficients implied by the correlated
  predictors.

**Study 3, misspecification** (`simulations/study_03_misspecification.py`).
Single large samples, so that the problem shown is not imprecision:
equivalent models (N = 3,000, seed 1, true path 0.50), reverse causation
(N = 3,000, seed 2, true Injuries to Climate path -0.55), and an omitted common
cause (N = 4,000, seed 3, true direct effect 0.20, confounder effects 0.60 and
0.50). Each latent has three indicators.

**Study 4, coefficient roundness** (`simulations/study_04_coefficient_interpretation.py`).
1,000 replications at N = 500 from the Study 1 model, recording the sum of
absolute estimated coefficients.

## 3. Results

Tables in this section are generated from `results/` by
`tools/methods_note_tables.py`, and `tests/test_methods_note.py` fails if they
drift from the committed CSV files.

### 3.1 The sample size that passes fit is not the sample size that supports the paths

Source: `results/study01_requirements.csv`.

<!-- BEGIN GENERATED: requirements -->
| Requirement | Basis | N required |
| --- | --- | --- |
| global fit: 80% power for the test of close fit | MacCallum-style RMSEA power, df = 80 | 153 |
| path precision: SE = 0.05 on a single coefficient | standardized regression SE, R^2 = 0.288, VIF = 1.28 | 369 |
| path precision: SE = 0.025 on a single coefficient | standardized regression SE, R^2 = 0.288, VIF = 1.28 | 1,459 |
| distinguish beta = 0.45 from beta = 0.30 at 80% power | z test on the coefficient difference, correlated predictors | 770 |
| distinguish beta = -0.25 from beta = -0.20 at 80% power | z test on the coefficient difference, correlated predictors | 6,884 |
<!-- END GENERATED: requirements -->

The fit test is the easiest requirement to meet. A team with N = 200 has more
than the 153 needed for the close-fit test and less than a third of the 770
needed to say that the first path is larger than the second.

### 3.2 Precision of the path estimates by sample size

Source: `results/study01_recovery.csv`. Ranges are across the four paths.

<!-- BEGIN GENERATED: precision -->
| N | Replications | Largest abs. bias | Empirical SD of path estimates | Approx. 95% half-width (1.96 x largest SD) | Analytic SE / empirical SD | Analytic 95% CI coverage | Improper or failed fits |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 100 | 300 | 0.0074 | 0.142 to 0.152 | 0.30 | 0.63 to 0.67 | 0.78 to 0.82 | 0 |
| 200 | 300 | 0.0069 | 0.093 to 0.101 | 0.20 | 0.67 to 0.73 | 0.79 to 0.85 | 0 |
| 500 | 300 | 0.0032 | 0.056 to 0.067 | 0.13 | 0.64 to 0.76 | 0.76 to 0.86 | 0 |
| 1,000 | 300 | 0.0016 | 0.041 to 0.044 | 0.09 | 0.69 to 0.74 | 0.81 to 0.86 | 0 |
| 2,000 | 300 | 0.0027 | 0.029 to 0.032 | 0.06 | 0.67 to 0.74 | 0.80 to 0.84 | 0 |
| 5,000 | 300 | 0.0014 | 0.019 to 0.020 | 0.04 | 0.69 to 0.72 | 0.82 to 0.86 | 0 |
<!-- END GENERATED: precision -->

Three things follow. First, the estimator is close to unbiased at every N, and
it produced no failed fits or improper solutions in this model. The problem is
variance, not bias. Second, at N = 100 an estimate of the 0.45 path is
typically within about 0.30 of the truth, so the same data could plausibly
return anything from about 0.15 to 0.75; at N = 200 the half-width is about
0.20 and at N = 500 about 0.13. Third, the analytic standard errors are
between 63% and 76% of the empirical variability, and nominal 95% intervals
cover the true value only 0.76 to 0.86 of the time, without improving as N
grows. The analytic standard errors treat the estimated latent correlation
matrix as known, and that omitted source of uncertainty does not shrink
relative to the rest.

Study 1 did not record fit indices in each replication, so the contrast
between fit and precision in this section is between the N that the close-fit
test needs (Section 3.1) and the precision observed at that N. The next
subsection shows recorded fit indices directly.

### 3.3 Ranking the drivers

Source: `results/study01_distinguishability.csv`.

<!-- BEGIN GENERATED: distinguish -->
| Pair (generating values) | Difference | SE of difference at N = 250 | N for 80% power |
| --- | --- | --- | --- |
| UnsafeActs (+0.45) vs OperationalStress (+0.30) | 0.15 | 0.095 | 770 |
| UnsafeActs (+0.45) vs SystemCondition (-0.25) | 0.70 | 0.095 | 41 |
| UnsafeActs (+0.45) vs SafetyResponseCapability (-0.20) | 0.65 | 0.095 | 46 |
| OperationalStress (+0.30) vs SystemCondition (-0.25) | 0.55 | 0.095 | 62 |
| OperationalStress (+0.30) vs SafetyResponseCapability (-0.20) | 0.50 | 0.095 | 74 |
| SystemCondition (-0.25) vs SafetyResponseCapability (-0.20) | 0.05 | 0.095 | 6,884 |
<!-- END GENERATED: distinguish -->

Telling a positive path from a negative one is cheap: no more than 74
observations for any such pair here. Ranking two paths of the same sign is
expensive. At N = 250 the standard error of any difference is 0.095, larger
than the 0.05 gap between the two negative paths and close to two thirds of
the 0.15 gap between the two positive ones.

### 3.4 Bootstrap intervals

Source: `results/study01_bootstrap.csv`, N = 500, 40 replications.

<!-- BEGIN GENERATED: bootstrap -->
| Path | Analytic coverage | Bootstrap coverage | Analytic width | Bootstrap width |
| --- | --- | --- | --- | --- |
| UnsafeActs (+0.45) | 0.850 | 0.925 | 0.168 | 0.227 |
| OperationalStress (+0.30) | 0.850 | 0.975 | 0.169 | 0.235 |
| SystemCondition (-0.25) | 0.775 | 0.950 | 0.169 | 0.235 |
| SafetyResponseCapability (-0.20) | 0.825 | 0.925 | 0.170 | 0.241 |
<!-- END GENERATED: bootstrap -->

Bootstrap coverage ranged from 0.925 to 0.975, against 0.775 to 0.850 for the
analytic interval. With 40 replications the Monte Carlo standard error of a
coverage estimate is about 3.4 percentage points, so the bootstrap is clearly
better than the analytic interval, but these runs cannot show whether it
reaches exactly 0.95. The bootstrap interval is 35% to 42% wider than the
analytic one, which is the width the data actually support.

### 3.5 Fit indices that pass while the answer is wrong

Sources: `results/study03_equivalent.csv`, `results/study03_reverse.csv`,
`results/study03_confounder.csv`. "Meets cutoffs" applies CFI at least 0.95,
RMSEA at most 0.06 and SRMR at most 0.08, the values associated with Hu and
Bentler (1999).

<!-- BEGIN GENERATED: fit_vs_truth -->
| Scenario | CFI | RMSEA | SRMR | Meets cutoffs | What the estimate says |
| --- | --- | --- | --- | --- | --- |
| 3A: Effect -> Cause fitted to Cause -> Effect data | 1.000 | 0.000 | 0.008 | True | beta = 0.493, direction wrong |
| 3B: Climate -> Injuries fitted to Injuries -> Climate data | 1.000 | 0.004 | 0.007 | True | beta = -0.570 (SE 0.015), direction wrong |
| 3C: common cause omitted | 1.000 | 0.000 | not recorded | True | beta = 0.495 vs generating 0.200 (bias 0.295) |
<!-- END GENERATED: fit_vs_truth -->

These samples are large (N = 3,000 to 4,000), so the estimates are precise,
and they are still wrong. In 3A the correct and reversed models imply the same
covariance matrix and return identical fit, the equivalent-models problem
described by MacCallum, Wegener, Uchino and Fabrigar (1993). In 3B the data
were generated with injuries lowering safety climate, the direction that Beus,
Payne, Bergman and Arthur (2010) found to be the stronger one in their
meta-analysis, and a model with the arrow reversed fits essentially perfectly.
In 3C a variable that was never measured cannot degrade fit indices computed
from the variables that were. More data does not fix any of these. At small N
the same failures occur with wider intervals on top.

### 3.6 What estimated coefficients look like

Source: `results/study04_roundness.csv`. Across 1,000 replications at N = 500
from a model whose true paths are exactly 0.45, 0.30, -0.25 and -0.20, the sum
of absolute estimated coefficients had a median of 1.19 and a 5th to 95th
percentile range of 0.99 to 1.41. Estimates scatter; a published vector whose
absolute values sum to exactly 1.20 and fall on round twentieths reads more
like a chosen weighting than an estimate.

## 4. Practical guidance for safety teams

The numbers below apply to the model simulated here: five latents, three
indicators each, loadings of 0.75, predictor correlations of 0.35 and
R-squared of 0.288. Weaker loadings, fewer indicators or a smaller R-squared
raise every requirement. The pattern, not the exact numbers, is what
transfers. Wolf et al. (2013) make the same point about rules of thumb.

**Minimum sample sizes these simulations support**

- Below about 150: do not fit a model of this size. The test of close fit
  itself is underpowered, and at N = 100 single-path estimates are uncertain
  by about 0.30 either way.
- About 150 to 370: fit statistics can be reported and paths can be signed
  (positive or negative), but no single path is known to within a standard
  error of 0.05. Do not publish the coefficients as weights.
- About 370 to 770: single paths are estimated to a standard error of about
  0.05, but paths of the same sign differing by 0.15 cannot be ranked with
  80% power.
- About 770 or more: a difference of 0.15 between two paths can be detected
  with 80% power. A difference of 0.05 needs about 6,884.
- None of these sample sizes protects against reversed direction or omitted
  common causes (Section 3.5).

**What to report with any SEM of safety data**

1. N after any exclusions, the number of indicators per latent, and the
   model degrees of freedom.
2. CFI, RMSEA with its confidence interval, and SRMR, described as measures of
   covariance reproduction, not of whether the paths are right. Marsh, Hau and
   Wen (2004) caution against treating cutoff values as golden rules.
3. A confidence interval for every path, from a bootstrap where the estimator's
   analytic standard errors have not been shown to be adequate.
4. For any claim that one driver matters more than another, a confidence
   interval for the difference between the two coefficients, not two separate
   intervals.
5. The assumed causal graph, the alternative directions and omitted variables
   that were considered, and why they were ruled out. Fit cannot rule them out.
6. A prospective sample-size calculation for the specific claim being made
   (fit, single-path precision, or ranking), not a generic rule of thumb.
   Kline (2023) covers the reporting of SEM results in detail.

## 5. Limitations

- Simulation only. No real injury, incident or survey data were analysed, and
  nothing here is a finding about workplace safety. The results describe the
  behaviour of an estimator on data from a known model.
- One structural model and one set of measurement conditions in Study 1.
  Different loadings, numbers of indicators, predictor correlations or
  R-squared would change every sample size in Section 3.
- The estimator is limited-information and two-step, not full-information
  maximum likelihood. Its chi-square is evaluated at the two-step estimates
  and its analytic standard errors are known to be too narrow; ML software
  such as lavaan or Mplus would give somewhat different fit values and
  standard errors. The sample-size requirements in Section 3.1 are analytic
  and do not depend on this estimator.
- Indicators are continuous and multivariate normal. Likert-type items,
  non-normality and missing data, all common in safety surveys, would
  generally make small-sample behaviour worse, but that was not simulated.
- The bootstrap comparison rests on 40 replications.
- Study 1 did not record fit indices per replication, so the contrast between
  fit and precision there is drawn between the required N for the fit test
  and the observed precision, not from fit and precision measured in the
  same replications.

## 6. References

Each entry below was checked against a publisher, index or author record.

- Anderson, J. C., & Gerbing, D. W. (1988). Structural equation modeling in
  practice: A review and recommended two-step approach. *Psychological
  Bulletin*, 103(3), 411-423.
- Beus, J. M., Payne, S. C., Bergman, M. E., & Arthur, W., Jr. (2010). Safety
  climate and injuries: An examination of theoretical and empirical
  relationships. *Journal of Applied Psychology*, 95(4), 713-727.
  https://doi.org/10.1037/a0019164
- Hu, L., & Bentler, P. M. (1999). Cutoff criteria for fit indexes in
  covariance structure analysis: Conventional criteria versus new
  alternatives. *Structural Equation Modeling*, 6(1), 1-55.
  https://doi.org/10.1080/10705519909540118
- Kline, R. B. (2023). *Principles and Practice of Structural Equation
  Modeling* (5th ed.). Guilford Press.
- MacCallum, R. C., Browne, M. W., & Sugawara, H. M. (1996). Power analysis
  and determination of sample size for covariance structure modeling.
  *Psychological Methods*, 1(2), 130-149.
- MacCallum, R. C., Wegener, D. T., Uchino, B. N., & Fabrigar, L. R. (1993).
  The problem of equivalent models in applications of covariance structure
  analysis. *Psychological Bulletin*, 114(1), 185-199.
  https://doi.org/10.1037/0033-2909.114.1.185
- Marsh, H. W., Hau, K.-T., & Wen, Z. (2004). In search of golden rules:
  Comment on hypothesis-testing approaches to setting cutoff values for fit
  indexes and dangers in overgeneralizing Hu and Bentler's (1999) findings.
  *Structural Equation Modeling*, 11(3), 320-341.
  https://doi.org/10.1207/s15328007sem1103_2
- Wolf, E. J., Harrington, K. M., Clark, S. L., & Miller, M. W. (2013). Sample
  size requirements for structural equation models: An evaluation of power,
  bias, and solution propriety. *Educational and Psychological Measurement*,
  73(6), 913-934. https://doi.org/10.1177/0013164413495237

## 7. Suggested citation

Chimmani, P. (2026). *Good fit, imprecise paths: When structural equation
models mislead at site-level sample sizes* (Methods note, ehs-risk-sem
version 0.1.0). https://github.com/priyatham9/ehs-risk-sem/blob/main/docs/methods-note.md
