#!/usr/bin/env python3
"""Data expectations and declared pipeline lineage for the committed simulation outputs (stdlib only).

A plain-Python take on the Foundry practice of named data expectations and pipeline lineage,
applied to this repository's committed files: the 23 simulation result tables in ``results/``,
the three synthetic datasets in ``synthetic/`` and the methods note. It reads only committed
files and is independent of and not affiliated with Palantir. Every number checked here comes
from a simulation with a known generating model; none is an empirical finding about workplaces.

Two parts.

* ``run_checks``: named expectations (schema, value domains, identities that must hold between
  columns, identities that must hold between files, monotone trends, null-free numerics), each
  with an observed value, a limit and a pass/warn/fail status.
* ``validate_lineage``: ``tools/lineage.json`` declares stages (script, function, inputs,
  outputs). The check confirms paths exist, functions are defined (via ast, nothing imported),
  every committed generated file is placed in the graph, no artifact has two producers and
  there is no cycle.

Usage::

    python3 tools/health.py            # print a summary
    python3 tools/health.py --write    # refresh health/*.csv, docs/health.md, docs/health.svg
    python3 tools/health.py --check    # exit 1 if a check fails, lineage is broken or outputs are stale
"""

from __future__ import annotations

import ast
import csv
import glob
import io
import json
import math
import os
import re
import sys
from collections import Counter
from typing import Dict, List

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPEC = os.path.join(ROOT, "tools", "lineage.json")
OUT_DIR = os.path.join(ROOT, "health")
DOC = os.path.join(ROOT, "docs", "health.md")
SVG = os.path.join(ROOT, "docs", "health.svg")
CHECKS_CSV = os.path.join(OUT_DIR, "health_checks.csv")
LINEAGE_CSV = os.path.join(OUT_DIR, "lineage_stages.csv")
COLUMNS_CSV = os.path.join(OUT_DIR, "health_columns.csv")

GROUPS = ("study01", "study02", "study03", "study04", "planner", "cutoffs", "synthetic")

# Exact column lists of every committed table.
SCHEMA = {
    "study01_bootstrap.csv": "predictor,true_beta,coverage_analytic,coverage_bootstrap,mean_width_analytic,mean_width_bootstrap,n_reps,monte_carlo_se_of_coverage",
    "study01_distinguishability.csv": "coefficient_a,beta_a,coefficient_b,beta_b,absolute_difference,n_for_80pct_power,se_of_difference_at_n_250",
    "study01_recovery.csv": "n,predictor,true_beta,mean_estimate,bias,empirical_sd,mean_analytic_se,se_ratio_analytic_over_empirical,coverage_95_analytic,rmse,improper_solutions,failed_fits,n_reps",
    "study01_requirements.csv": "requirement,basis,n_required",
    "study02_alert_burden.csv": "shift_hours,base_rate_per_shift,worker_shifts_per_event,sensitivity,specificity,ppv,alerts_per_true_event",
    "study02_discrimination.csv": "target_base_rate,realized_base_rate,events_in_training,auc,brier,brier_uncertainty_share,calibration_intercept,calibration_slope,ece,ppv_at_top_5pct,alerts_per_true_event_at_top_5pct",
    "study02_imbalance.csv": "training_sample,train_n,train_event_fraction,auc_test,mean_predicted_test,observed_rate_test,overprediction_ratio,calibration_slope,brier,ppv_at_top_5pct",
    "study02_king_zeng.csv": "estimator,mean_intercept,mean_predicted_probability,relative_error_vs_truth,n_usable_reps",
    "study02_sizing.csv": "unit_of_analysis,base_rate,relative_precision_target,events_required,units_required",
    "study02_zero_inflation.csv": "structural_zero_fraction,observed_zero_fraction,poisson_expected_zero_fraction,excess_zeros,mean_count,variance_count,variance_to_mean_ratio,ols_slope_on_latent_1",
    "study03_confounder.csv": "fitted_model,beta_predictor_on_outcome,true_direct_effect,bias,chi_square,df,cfi,rmsea,meets_conventional_cutoffs",
    "study03_equivalent.csv": "fitted_model,beta,se,p_value,chi_square,df,cfi,rmsea,srmr,meets_conventional_cutoffs",
    "study03_formative.csv": "quantity,value,note",
    "study03_mediator.csv": "quantity,estimate,generating_value",
    "study03_reverse.csv": "generating_structure,fitted_structure,estimated_beta,se,p_value,chi_square,df,cfi,rmsea,srmr,meets_conventional_cutoffs",
    "study04_ambiguity.csv": "world,fitted_coefficients,effect_of_intervening_on_UnsafeActs,note",
    "study04_invariance.csv": "quantity,value",
    "study04_raw_scale.csv": "site,crew,score_raw_inputs,score_z_inputs,ranking_on_z_inputs",
    "study04_roundness.csv": "quantity,median,p05,p95",
    "planner_loading050.csv": "n,reps,completed_fits,proper_solution_rate,conventional_fit_rate,loading_sd,loading_mean_abs_error,path_sd,true_betas",
    "planner_loading075.csv": "n,reps,completed_fits,proper_solution_rate,conventional_fit_rate,loading_sd,loading_mean_abs_error,path_sd,true_betas",
    "fit_index_cutoffs_n1200.csv": "index,n_reps_used,mean,sd,p05,p50,p95,tail,cutoff,fixed_cutoff,fixed_cutoff_failure_rate,observed,observed_percentile",
    "fit_index_cutoffs_n200.csv": "index,n_reps_used,mean,sd,p05,p50,p95,tail,cutoff,fixed_cutoff,fixed_cutoff_failure_rate,observed,observed_percentile",
}
# Columns that are free text or lists rather than numbers.
TEXT_COLS = {"predictor", "coefficient_a", "coefficient_b", "requirement", "basis", "estimator", "training_sample",
             "unit_of_analysis", "fitted_model", "quantity", "note", "generating_structure", "fitted_structure",
             "world", "fitted_coefficients", "site", "crew", "ranking_on_z_inputs", "true_betas", "index", "tail",
             "meets_conventional_cutoffs", "value", "estimate", "generating_value", "median", "p05", "p95"}
# Files whose "value" style columns mix numbers and lists, so numeric parsing is checked cell by cell.
MIXED_FILES = {"study03_formative.csv", "study03_mediator.csv", "study04_invariance.csv", "study04_roundness.csv"}

# Ceiling (share of empty cells, percent) for columns that are allowed to be empty.
EMPTY_ALLOWED = {
    ("fit_index_cutoffs_n1200.csv", "fixed_cutoff"): "only CFI, RMSEA and SRMR have a fixed rule of thumb",
    ("fit_index_cutoffs_n1200.csv", "fixed_cutoff_failure_rate"): "empty wherever there is no fixed cutoff",
    ("fit_index_cutoffs_n200.csv", "fixed_cutoff"): "only CFI, RMSEA and SRMR have a fixed rule of thumb",
    ("fit_index_cutoffs_n200.csv", "fixed_cutoff_failure_rate"): "empty wherever there is no fixed cutoff",
    ("fit_index_cutoffs_n200.csv", "observed"): "no observed model was supplied for the N = 200 run",
    ("fit_index_cutoffs_n200.csv", "observed_percentile"): "no observed model was supplied for the N = 200 run",
    ("study02_king_zeng.csv", "mean_intercept"): "the generating-model row has no fitted intercept",
    ("study03_formative.csv", "note"): "some rows carry no note",
    ("study04_roundness.csv", "p05"): "the last two rows are single values",
    ("study04_roundness.csv", "p95"): "the last two rows are single values",
}


def read_results(name: str) -> List[Dict[str, str]]:
    with open(os.path.join(ROOT, "results", name), encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def read_header(path: str) -> List[str]:
    with open(path, encoding="utf-8", newline="") as fh:
        return next(csv.reader(fh))


def read_synthetic(name: str):
    """(comment lines, header, number of data rows, rows as dicts of strings)."""
    comments, body = [], []
    with open(os.path.join(ROOT, "synthetic", name), encoding="utf-8", newline="") as fh:
        for line in fh:
            (comments if line.startswith("#") else body).append(line)
    rows = list(csv.DictReader(io.StringIO("".join(body))))
    return comments, list(rows[0].keys()), len(rows), rows


def load() -> Dict[str, object]:
    d: Dict[str, object] = {"tables": {n: read_results(n) for n in SCHEMA}}
    d["headers"] = {n: read_header(os.path.join(ROOT, "results", n)) for n in SCHEMA}
    d["extra_results"] = sorted(set(os.path.basename(p) for p in glob.glob(os.path.join(ROOT, "results", "*.csv"))) - set(SCHEMA))
    d["synthetic"] = {n: read_synthetic(n) for n in ("survey_style.csv", "rare_event.csv", "zero_inflated_counts.csv")}
    with open(os.path.join(ROOT, "docs", "methods-note.md"), encoding="utf-8") as fh:
        d["note"] = fh.read()
    return d


def num(v):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x if math.isfinite(x) else None


def f(rows, col):
    return [float(r[col]) for r in rows]


def close(a, b, tol=1e-9, rel=False):
    return abs(a - b) <= (tol * max(abs(a), abs(b), 1e-300) if rel else tol)


def group_of(name: str) -> str:
    if name.startswith("planner"):
        return "planner"
    if name.startswith("fit_index"):
        return "cutoffs"
    return name[:7]


def column_profile(name, rows):
    out = []
    n = len(rows)
    for c in rows[0].keys():
        vals = [r[c] for r in rows]
        empty = sum(1 for v in vals if v is None or str(v).strip() == "")
        out.append({"dataset": name, "column": c, "rows": n, "empty": empty,
                    "empty_pct": round(100.0 * empty / n, 2),
                    "distinct": len({v for v in vals if str(v).strip() != ""})})
    return out


def profiles(data) -> List[Dict[str, object]]:
    out = []
    for name in SCHEMA:
        out += column_profile(name, data["tables"][name])
    return out


def _row(cid, dataset, rule, observed, limit, ok, severity="error"):
    return {"id": cid, "dataset": dataset, "rule": rule, "observed": observed, "limit": limit,
            "status": "pass" if ok else ("fail" if severity == "error" else "warn"), "severity": severity}


def list_of(s):
    return [float(x) for x in re.findall(r"-?\d+\.?\d*", s)]


def run_checks(data) -> List[Dict[str, object]]:
    T = data["tables"]
    out: List[Dict[str, object]] = []

    def add(*a, **k):
        out.append(_row("R%02d" % (len(out) + 1), *a, **k))

    def zero(dataset, rule, bad, severity="error"):
        add(dataset, rule, bad, 0, bad == 0, severity)

    # ---- structure of the whole results directory
    add("results", "no results CSV outside the declared schema", len(data["extra_results"]), 0, not data["extra_results"])
    for g in GROUPS[:6]:
        names = [n for n in SCHEMA if group_of(n) == g]
        bad = sum(1 for n in names if data["headers"][n] != SCHEMA[n].split(","))
        add(g, "%s: columns equal the declared schema (%d files)" % (g, len(names)), bad, 0, bad == 0)
    for g in GROUPS[:6]:
        names = [n for n in SCHEMA if group_of(n) == g]
        bad = 0
        for n in names:
            for r in T[n]:
                for c, v in r.items():
                    if c in TEXT_COLS and n not in MIXED_FILES:
                        continue
                    if n in MIXED_FILES:
                        continue
                    if num(v) is None and not (n, c) in EMPTY_ALLOWED:
                        bad += 1
        add(g, "%s: numeric columns parse as finite numbers" % g, bad, 0, bad == 0)
    bad = 0
    for n in SCHEMA:
        for c in data["headers"][n]:
            share = 100.0 * sum(1 for r in T[n] if str(r[c]).strip() == "") / len(T[n])
            if share > 0 and (n, c) not in EMPTY_ALLOWED:
                bad += 1
    add("results", "empty cells only in the %d columns that may be empty" % len(EMPTY_ALLOWED), bad, 0, bad == 0)

    # ---- study 1: sample size
    rec, req, dist, boot = T["study01_recovery.csv"], T["study01_requirements.csv"], T["study01_distinguishability.csv"], T["study01_bootstrap.csv"]
    ns = sorted({int(r["n"]) for r in rec})
    preds = sorted({r["predictor"] for r in rec})
    add("study01", "recovery has one row per N and predictor", len(rec), len(ns) * len(preds), len(rec) == len(ns) * len(preds))
    add("study01", "every recovery row used the same replication count", len({r["n_reps"] for r in rec}), 1, len({r["n_reps"] for r in rec}) == 1)
    zero("study01", "no improper solutions or failed fits in the recovery grid",
         sum(int(r["improper_solutions"]) + int(r["failed_fits"]) for r in rec))
    add("study01", "coverage and the SE ratio are positive and coverage is a probability",
        sum(1 for r in rec if not (0 <= float(r["coverage_95_analytic"]) <= 1 and float(r["se_ratio_analytic_over_empirical"]) > 0)), 0,
        all(0 <= float(r["coverage_95_analytic"]) <= 1 and float(r["se_ratio_analytic_over_empirical"]) > 0 for r in rec))
    b = sum(1 for r in rec if not close(float(r["bias"]), float(r["mean_estimate"]) - float(r["true_beta"]), 1e-9))
    zero("study01", "bias equals mean estimate minus true beta", b)
    b = sum(1 for r in rec if not close(float(r["se_ratio_analytic_over_empirical"]),
                                        float(r["mean_analytic_se"]) / float(r["empirical_sd"]), 1e-9))
    zero("study01", "SE ratio equals mean analytic SE over empirical SD", b)
    b = sum(1 for r in rec if abs(float(r["rmse"]) - math.hypot(float(r["bias"]), float(r["empirical_sd"]))) > 0.02 * float(r["rmse"]))
    zero("study01", "RMSE is within 2% of sqrt(bias^2 + SD^2)", b)
    nonmono = 0
    for p in preds:
        sds = [float(r["empirical_sd"]) for r in sorted((r for r in rec if r["predictor"] == p), key=lambda r: int(r["n"]))]
        nonmono += sum(1 for a, c in zip(sds, sds[1:]) if c >= a)
    zero("study01", "empirical SD falls at every step up in N, for each predictor", nonmono)
    se = {int(r["n"]): float(r["mean_analytic_se"]) for r in rec if r["predictor"] == preds[0]}
    ratio = se[min(ns)] / se[max(ns)]
    expect = math.sqrt(max(ns) / min(ns))
    add("study01", "analytic SE scales as 1/sqrt(N) between the smallest and largest N (within 5%)", round(ratio, 3),
        "%.3f +/- 5%%" % expect, abs(ratio / expect - 1) <= 0.05)
    mx_bias = max(abs(float(r["bias"])) for r in rec)
    add("study01", "absolute bias stays below 0.01 everywhere (point estimates are not the problem)", round(mx_bias, 4), 0.01, mx_bias < 0.01)
    min_ratio = min(float(r["se_ratio_analytic_over_empirical"]) for r in rec)
    add("study01", "analytic SE is not below 90% of the empirical SD in any row (lowest ratio shown)", round(min_ratio, 3), ">= 0.9 for all", min(float(r["se_ratio_analytic_over_empirical"]) for r in rec) >= 0.9, "warn")
    cmin = min(float(r["coverage_95_analytic"]) for r in rec)
    add("study01", "analytic 95% coverage is at least 0.93 in every row", round(cmin, 3), 0.93, cmin >= 0.93, "warn")
    rq = {r["requirement"]: int(r["n_required"]) for r in req}
    add("study01", "requirements has five rows with positive integer N", len(req), 5, len(req) == 5 and all(v > 0 for v in rq.values()))
    se05 = [v for k, v in rq.items() if "SE = 0.05" in k][0]
    se025 = [v for k, v in rq.items() if "SE = 0.025" in k][0]
    add("study01", "halving the target SE needs about 4x the sample (3.8x to 4.2x)", round(se025 / se05, 2), "3.8 to 4.2", 3.8 <= se025 / se05 <= 4.2)
    bad = sum(1 for r in dist if not close(float(r["absolute_difference"]), abs(float(r["beta_a"]) - float(r["beta_b"])), 1e-9))
    zero("study01", "absolute difference equals |beta_a - beta_b|", bad)
    srt = sorted(dist, key=lambda r: -float(r["absolute_difference"]))
    mono = sum(1 for a, c in zip(srt, srt[1:]) if int(c["n_for_80pct_power"]) < int(a["n_for_80pct_power"]))
    zero("study01", "required N for 80% power rises as the coefficient gap shrinks", mono)
    ses = f(dist, "se_of_difference_at_n_250")
    add("study01", "SE of a coefficient difference at N = 250 is the same for every pair (1e-9)", "%.2e" % (max(ses) - min(ses)), 1e-9, max(ses) - min(ses) < 1e-9)
    ds = {(r["beta_a"], r["beta_b"]): int(r["n_for_80pct_power"]) for r in dist}
    add("study01", "required N for the two quoted pairs agrees between requirements and distinguishability",
        "%d / %d" % (ds[("0.45", "0.3")], ds[("-0.25", "-0.2")]), "%d / %d" % (rq[[k for k in rq if "beta = 0.45" in k][0]], rq[[k for k in rq if "beta = -0.25" in k][0]]),
        ds[("0.45", "0.3")] == rq[[k for k in rq if "beta = 0.45" in k][0]] and ds[("-0.25", "-0.2")] == rq[[k for k in rq if "beta = -0.25" in k][0]])
    truth = {r["predictor"]: float(r["true_beta"]) for r in rec}
    betas = {r["coefficient_a"]: float(r["beta_a"]) for r in dist}
    betas.update({r["coefficient_b"]: float(r["beta_b"]) for r in dist})
    add("study01", "true betas agree between recovery, distinguishability and bootstrap",
        sum(1 for p, v in betas.items() if not close(truth[p], v)) + sum(1 for r in boot if not close(truth[r["predictor"]], float(r["true_beta"]))), 0,
        all(close(truth[p], v) for p, v in betas.items()) and all(close(truth[r["predictor"]], float(r["true_beta"])) for r in boot))
    add("study01", "bootstrap intervals are wider than analytic intervals for every predictor",
        sum(1 for r in boot if float(r["mean_width_bootstrap"]) <= float(r["mean_width_analytic"])), 0,
        all(float(r["mean_width_bootstrap"]) > float(r["mean_width_analytic"]) for r in boot))
    bad = sum(1 for r in boot if abs(float(r["monte_carlo_se_of_coverage"]) - math.sqrt(0.95 * 0.05 / float(r["n_reps"]))) > 1e-3)
    zero("study01", "Monte Carlo SE of coverage equals sqrt(0.95 * 0.05 / reps) within 0.001", bad)

    # ---- study 2: rare events
    ab, dc, im, kz, sz, zi = (T[k] for k in ("study02_alert_burden.csv", "study02_discrimination.csv", "study02_imbalance.csv",
                                             "study02_king_zeng.csv", "study02_sizing.csv", "study02_zero_inflation.csv"))
    bad = 0
    for r in ab:
        p, se_, sp = float(r["base_rate_per_shift"]), float(r["sensitivity"]), float(r["specificity"])
        ppv = se_ * p / (se_ * p + (1 - sp) * (1 - p))
        bad += (not close(ppv, float(r["ppv"]), 1e-9, rel=True)) or (not close(1 / ppv, float(r["alerts_per_true_event"]), 1e-9, rel=True))
    zero("study02", "PPV and alerts per true event recompute from Bayes' rule", bad)
    bad = sum(1 for r in ab if not close(1 / float(r["base_rate_per_shift"]), float(r["worker_shifts_per_event"]), 1e-9, rel=True))
    zero("study02", "worker-shifts per event equals 1 / base rate", bad)
    bad = sum(1 for r in ab if abs(float(r["base_rate_per_shift"]) * 2000.0 / float(r["shift_hours"]) - 0.023) > 1e-9)
    zero("study02", "per-shift rate times shifts per 2,000-hour year equals the 2.3 per 100 FTE annual rate", bad)
    bad = sum(1 for r in sz if not close(float(r["units_required"]), float(r["events_required"]) / float(r["base_rate"]), 1e-9, rel=True))
    zero("study02", "units required equals events required over base rate", bad)
    bad = sum(1 for r in sz if abs(float(r["events_required"]) - (1.959964 / float(r["relative_precision_target"])) ** 2) > 1e-3 * float(r["events_required"]))
    zero("study02", "events required equals (1.96 / relative precision)^2 within 0.1%", bad)
    bad = sum(1 for r in dc if abs(float(r["realized_base_rate"]) - float(r["target_base_rate"])) > 0.005 * max(1.0, float(r["target_base_rate"]) * 10))
    zero("study02", "realised base rate is within 0.005 (scaled) of target in every discrimination row", bad)
    bad = sum(1 for r in dc if not (0.5 < float(r["auc"]) < 1) or not close(1 / float(r["ppv_at_top_5pct"]), float(r["alerts_per_true_event_at_top_5pct"]), 1e-3, rel=True))
    zero("study02", "AUC lies in (0.5, 1) and alerts per true event is 1 / PPV at the top 5%", bad)
    bad = sum(1 for r in im if not close(float(r["mean_predicted_test"]) / float(r["observed_rate_test"]), float(r["overprediction_ratio"]), 1e-3, rel=True))
    zero("study02", "over-prediction ratio equals mean predicted over observed rate", bad)
    add("study02", "every imbalance row is scored on the same test set (identical observed rate)", len({r["observed_rate_test"] for r in im}), 1, len({r["observed_rate_test"] for r in im}) == 1)
    us = [r for r in im if "undersampling" in r["training_sample"]]
    add("study02", "undersampled training set is balanced (event fraction within 0.01 of 0.5)", [round(float(r["train_event_fraction"]), 3) for r in us], "0.5 +/- 0.01",
        bool(us) and all(abs(float(r["train_event_fraction"]) - 0.5) < 0.01 for r in us))
    mle = [r for r in kz if r["estimator"] == "maximum likelihood"][0]
    kzr = [r for r in kz if r["estimator"].startswith("King-Zeng")][0]
    gen = [r for r in kz if r["estimator"] == "generating model"][0]
    truth_p = float(gen["mean_predicted_probability"])
    bad = sum(1 for r in (mle, kzr) if not close(float(r["relative_error_vs_truth"]), float(r["mean_predicted_probability"]) / truth_p - 1, 1e-6))
    zero("study02", "relative error vs truth recomputes from the generating-model row for ML and King-Zeng", bad)
    add("study02", "ML, King-Zeng and generating-model rows used the same positive number of usable replications", sorted({r["n_usable_reps"] for r in kz}), "one positive value",
        len({r["n_usable_reps"] for r in kz}) == 1 and int(kz[0]["n_usable_reps"]) > 0)
    bad = sum(1 for r in zi if not close(float(r["excess_zeros"]), float(r["observed_zero_fraction"]) - float(r["poisson_expected_zero_fraction"]), 1e-9)
              or not close(float(r["variance_to_mean_ratio"]), float(r["variance_count"]) / float(r["mean_count"]), 1e-9, rel=True))
    zero("study02", "excess zeros and variance-to-mean ratio recompute from their columns", bad)
    base = [r for r in zi if float(r["structural_zero_fraction"]) == 0.0]
    add("study02", "with no structural zeros the excess-zero fraction is below 0.01 (the null behaves)", [round(float(r["excess_zeros"]), 4) for r in base], 0.01,
        bool(base) and all(abs(float(r["excess_zeros"])) < 0.01 for r in base))

    # ---- study 3: misspecification
    cf, eq, rv, md, fm = (T[k] for k in ("study03_confounder.csv", "study03_equivalent.csv", "study03_reverse.csv", "study03_mediator.csv", "study03_formative.csv"))
    bad = sum(1 for r in cf if not close(float(r["bias"]), float(r["beta_predictor_on_outcome"]) - float(r["true_direct_effect"]), 1e-9))
    zero("study03", "confounder bias equals fitted beta minus true direct effect", bad)
    inc = [r for r in cf if "included" in r["fitted_model"]][0]
    omi = [r for r in cf if "omitted" in r["fitted_model"]][0]
    add("study03", "omitting the confounder biases the path more than including it", "%.3f vs %.3f" % (abs(float(omi["bias"])), abs(float(inc["bias"]))), "omitted > included",
        abs(float(omi["bias"])) > abs(float(inc["bias"])))
    add("study03", "the omitted-confounder model still meets conventional fit cutoffs (the study's premise)", omi["meets_conventional_cutoffs"], "True", omi["meets_conventional_cutoffs"] == "True")
    keys = ("beta", "chi_square", "df", "cfi", "rmsea")
    add("study03", "the two equivalent models report identical beta and fit", sum(1 for k in keys if not close(float(eq[0][k]), float(eq[1][k]), 1e-9)), 0,
        len(eq) == 2 and all(close(float(eq[0][k]), float(eq[1][k]), 1e-9) for k in keys))
    bad = sum(1 for r in eq + rv if not (0 <= float(r["p_value"]) <= 1 and float(r["rmsea"]) >= 0 and float(r["cfi"]) <= 1 + 1e-9))
    zero("study03", "p-values lie in [0, 1], RMSEA is non-negative and CFI is at most 1", bad)
    m = {r["quantity"]: r for r in md}
    d_ = float(m["direct effect of X on Y (mediator in model)"]["estimate"])
    i_ = float(m["indirect effect X -> M -> Y"]["estimate"])
    t_ = float(m["total effect of X on Y (path tracing)"]["estimate"])
    add("study03", "estimated direct plus indirect effect equals the total effect", round(d_ + i_, 6), round(t_, 6), close(d_ + i_, t_, 1e-6))
    gd, gi, gt = (float(m[k]["generating_value"]) for k in ("direct effect of X on Y (mediator in model)", "indirect effect X -> M -> Y", "total effect of X on Y (path tracing)"))
    add("study03", "generating direct plus indirect effect equals the generating total", round(gd + gi, 6), round(gt, 6), close(gd + gi, gt, 1e-6))
    add("study03", "omitting the mediator returns the total effect", m["effect of X on Y (mediator omitted)"]["estimate"] == m["total effect of X on Y (path tracing)"]["estimate"], True,
        close(float(m["effect of X on Y (mediator omitted)"]["estimate"]), t_, 1e-9))
    fmv = {r["quantity"]: float(r["value"]) for r in fm if num(r["value"]) is not None}
    add("study03", "formative block: omega is below 0.5 and the disattenuated correlation exceeds 1 (the failure being shown)",
        "%.3f / %.3f" % (fmv["composite reliability (omega) of the formative block"], fmv["disattenuated latent correlation"]), "< 0.5 / > 1",
        fmv["composite reliability (omega) of the formative block"] < 0.5 and fmv["disattenuated latent correlation"] > 1)

    # ---- study 4: coefficient interpretation
    rd, am, iv, rs = (T[k] for k in ("study04_roundness.csv", "study04_ambiguity.csv", "study04_invariance.csv", "study04_raw_scale.csv"))
    r0 = rd[0]
    add("study04", "rounding distance is at most 0.025 and p05 <= median <= p95", [round(float(r0["p05"]), 4), round(float(r0["median"]), 4), round(float(r0["p95"]), 4)], "p05 <= median <= p95 <= 0.025",
        float(r0["p05"]) <= float(r0["median"]) <= float(r0["p95"]) <= 0.025 + 1e-12)
    rv_ = {r["quantity"]: r for r in rd}
    obs_k = [k for k in rv_ if k.startswith("observed fraction")][0]
    exp_k = [k for k in rv_ if k.startswith("expected fraction")][0]
    add("study04", "expected chance fraction of grid-like estimates equals (0.01 / 0.05)^4", round(float(rv_[exp_k]["median"]), 6), round((0.01 / 0.05) ** 4, 6),
        close(float(rv_[exp_k]["median"]), (0.01 / 0.05) ** 4, 1e-9))
    add("study04", "observed grid-like fraction is a probability", float(rv_[obs_k]["median"]), "[0, 1]", 0 <= float(rv_[obs_k]["median"]) <= 1)
    add("study04", "every ambiguity world reports four fitted coefficients", [len(list_of(r["fitted_coefficients"])) for r in am], "4 each", all(len(list_of(r["fitted_coefficients"])) == 4 for r in am))
    iq = {r["quantity"]: r["value"] for r in iv}
    def cr(lams):
        s = sum(lams)
        return s * s / (s * s + sum(1 - l_ * l_ for l_ in lams))
    la, lb = list_of(iq["Site A loadings"]), list_of(iq["Site B loadings"])
    add("study04", "composite reliabilities recompute from the listed loadings (within 0.002)", "%.3f / %.3f" % (cr(la), cr(lb)),
        "%s / %s" % (iq["Site A composite reliability"], iq["Site B composite reliability"]),
        abs(cr(la) - float(iq["Site A composite reliability"])) < 0.002 and abs(cr(lb) - float(iq["Site B composite reliability"])) < 0.002)
    add("study04", "Site B (one weak loading) has lower reliability than Site A", "%s vs %s" % (iq["Site B composite reliability"], iq["Site A composite reliability"]), "B < A",
        float(iq["Site B composite reliability"]) < float(iq["Site A composite reliability"]))
    shares = [float(v) for k, v in iq.items() if k.startswith("share") or k.startswith("correlation")]
    add("study04", "shares and correlations in the invariance table lie in [-1, 1]", len(shares), "all in range", all(-1 <= s <= 1 for s in shares))
    add("study04", "raw-scale table has two sites by two crews", len(rs), 4, len(rs) == 4 and len({r["site"] for r in rs}) == 2)

    # ---- planner
    p5, p75 = T["planner_loading050.csv"], T["planner_loading075.csv"]
    for name, rows in (("planner_loading050.csv", p5), ("planner_loading075.csv", p75)):
        tag = "loading 0.50" if "050" in name else "loading 0.75"
        bad = sum(1 for r in rows if r["reps"] != r["completed_fits"] or not (0 <= float(r["proper_solution_rate"]) <= 1 and 0 <= float(r["conventional_fit_rate"]) <= 1))
        zero("planner", "%s: rates are probabilities and every replication completed" % tag, bad)
        sorted_rows = sorted(rows, key=lambda r: int(r["n"]))
        for col, direction in (("path_sd", -1), ("loading_sd", -1), ("proper_solution_rate", 1), ("conventional_fit_rate", 1)):
            v = [float(r[col]) for r in sorted_rows]
            bad = sum(1 for a, c in zip(v, v[1:]) if (c - a) * direction < 0)
            zero("planner", "%s: %s %s with N" % (tag, col, "falls" if direction < 0 else "does not fall"), bad)
    add("planner", "both loading levels use the same N grid", [int(r["n"]) for r in p5], [int(r["n"]) for r in p75], [r["n"] for r in p5] == [r["n"] for r in p75])
    bad = sum(1 for a, c in zip(p5, p75) if float(c["path_sd"]) >= float(a["path_sd"]))
    zero("planner", "stronger loadings (0.75) give a smaller path SD than weak loadings (0.50) at every N", bad)
    tb = {r["true_betas"] for r in p5 + p75}
    add("planner", "planner true betas equal the study 1 true betas", sorted(tb), " ".join(str(truth[p]) for p in ("UnsafeActs", "OperationalStress", "SystemCondition", "SafetyResponseCapability")),
        len(tb) == 1 and [float(x) for x in next(iter(tb)).split()] == [truth[p] for p in ("UnsafeActs", "OperationalStress", "SystemCondition", "SafetyResponseCapability")])
    low = min(float(r["proper_solution_rate"]) for r in p5)
    add("planner", "weak-loading planner run has at least 95% proper solutions at every N", round(low, 3), 0.95, low >= 0.95, "warn")

    # ---- fit-index cutoffs
    for name in ("fit_index_cutoffs_n1200.csv", "fit_index_cutoffs_n200.csv"):
        rows = T[name]
        tag = "n1200" if "1200" in name else "n200"
        add("cutoffs", "%s: five indices (chi_square, cfi, tli, rmsea, srmr)" % tag, [r["index"] for r in rows], "5 indices", [r["index"] for r in rows] == ["chi_square", "cfi", "tli", "rmsea", "srmr"])
        bad = sum(1 for r in rows if not (float(r["p05"]) <= float(r["p50"]) <= float(r["p95"])))
        zero("cutoffs", "%s: p05 <= p50 <= p95 for every index" % tag, bad)
        bad = 0
        for r in rows:
            want = float(r["p05"]) if r["index"] in ("cfi", "tli") else float(r["p95"])
            bad += not close(float(r["cutoff"]), want, 1e-9)
        zero("cutoffs", "%s: cutoff is the lower 5%% quantile for CFI and TLI and the upper 95%% quantile otherwise" % tag, bad)
        bad = sum(1 for r in rows if r["fixed_cutoff_failure_rate"] and not 0 <= float(r["fixed_cutoff_failure_rate"]) <= 1)
        zero("cutoffs", "%s: fixed-cutoff failure rates are probabilities" % tag, bad)
        add("cutoffs", "%s: replications used per index are equal and positive" % tag, sorted({r["n_reps_used"] for r in rows}), "one positive value", len({r["n_reps_used"] for r in rows}) == 1 and int(rows[0]["n_reps_used"]) > 0)
    obs = [r for r in T["fit_index_cutoffs_n1200.csv"] if r["observed"]]
    bad = 0
    for r in obs:
        lower = r["index"] in ("cfi", "tli")
        beyond = float(r["observed"]) < float(r["cutoff"]) if lower else float(r["observed"]) > float(r["cutoff"])
        pct = float(r["observed_percentile"])
        bad += beyond != (pct < 0.05 if lower else pct > 0.95)
    zero("cutoffs", "n1200: an observed value past the cutoff has a percentile past 5% / 95%", bad)

    # ---- synthetic datasets and note
    spec_rows = {"survey_style.csv": (1200, 20260903), "rare_event.csv": (25000, 424242), "zero_inflated_counts.csv": (8000, 771)}
    for name, (n, seed) in spec_rows.items():
        com, hdr, nrows, rows = data["synthetic"][name]
        txt = "\n".join(com)
        add("synthetic", "%s: labelled synthetic, header states n = %d and seed = %d, and %d rows are present" % (name, n, seed, n),
            "%s rows; n=%s; seed=%s" % (nrows, ("n = %d" % n) in txt, ("seed = %d" % seed) in txt), "%d rows" % n,
            "SYNTHETIC DATA" in txt and ("n = %d," % n) in txt and ("seed = %d" % seed) in txt and nrows == n)
        mis = sum(1 for r in rows for v in r.values() if num(v) is None)
        add("synthetic", "%s: every data cell is a finite number" % name, mis, 0, mis == 0)
    rare = data["synthetic"]["rare_event.csv"][3]
    rate = sum(int(r["recordable"]) for r in rare) / len(rare)
    add("synthetic", "rare-event outcome rate is within 10% of the stated 0.005", round(rate, 5), "0.005 +/- 10%", abs(rate / 0.005 - 1) <= 0.10)
    pr = max(float(r["true_probability"]) for r in rare)
    add("synthetic", "generating probabilities in the rare-event file lie in (0, 1)", round(pr, 4), "< 1", 0 < min(float(r["true_probability"]) for r in rare) and pr < 1)
    zin = data["synthetic"]["zero_inflated_counts.csv"][3]
    zshare = sum(1 for r in zin if int(r["recordable_count"]) == 0) / len(zin)
    add("synthetic", "zero-inflated counts: observed zero share is above the 0.38 structural-zero rate", round(zshare, 3), "> 0.38", zshare > 0.38)
    add("methods note", "methods note states that it uses simulated data", "simulat" in data["note"].lower(), True, "simulat" in data["note"].lower())
    return out


# ---------------------------------------------------------------- lineage

def load_spec():
    with open(SPEC, encoding="utf-8") as fh:
        return json.load(fh)


def _matches(root, pat):
    return sorted(os.path.relpath(p, root).replace(os.sep, "/") for p in glob.glob(os.path.join(root, pat), recursive=True) if os.path.isfile(p))


def _top_level_names(path):
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    return {n.name for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}


def validate_lineage(spec, root=ROOT):
    problems = []
    stages, arts = spec["stages"], spec["artifacts"]
    declared = set()
    for aid, a in arts.items():
        for pat in a["paths"]:
            hits = _matches(root, pat)
            if not hits:
                problems.append("artifact %s: nothing matches %s" % (aid, pat))
            declared.update(hits)
    producer, seen = {}, set()
    for st in stages:
        sid = st["id"]
        if sid in seen:
            problems.append("duplicate stage id: " + sid)
        seen.add(sid)
        fp = os.path.join(root, st["script"])
        if not os.path.exists(fp):
            problems.append("stage %s: script not found: %s" % (sid, st["script"]))
        elif st["function"] not in _top_level_names(fp):
            problems.append("stage %s: function %s not defined in %s" % (sid, st["function"], st["script"]))
        for ref in st["inputs"] + st["outputs"]:
            if ref not in arts:
                problems.append("stage %s: undeclared artifact %s" % (sid, ref))
        for o in st["outputs"]:
            if o in producer:
                problems.append("artifact %s produced by both %s and %s" % (o, producer[o], sid))
            producer[o] = sid
    deps = {s["id"]: {producer[a] for a in s["inputs"] if a in producer} for s in stages}
    rem = dict(deps)
    while rem:
        ready = [s for s, d in rem.items() if not (d & set(rem))]
        if not ready:
            problems.append("cycle among stages: " + ", ".join(sorted(rem)))
            break
        for s in ready:
            del rem[s]
    for aid, a in arts.items():
        if aid not in producer and not a.get("source", False):
            problems.append("artifact %s: committed file with no producing stage and not marked source" % aid)
    for pat in spec["must_be_declared"]:
        for rel in _matches(root, pat):
            if rel not in declared:
                problems.append("committed file not declared in lineage: " + rel)
    return problems


def topo(stages):
    producer = {a: s["id"] for s in stages for a in s["outputs"]}
    by_id = {s["id"]: s for s in stages}
    deps = {i: {producer[a] for a in s["inputs"] if a in producer} for i, s in by_id.items()}
    order, done = [], set()
    while len(done) < len(by_id):
        for i in by_id:
            if i not in done and deps[i] <= done:
                order.append(by_id[i])
                done.add(i)
                break
        else:
            break
    return order


def lineage_rows(spec):
    producer = {a: s["id"] for s in spec["stages"] for a in s["outputs"]}
    rows = []
    for n, st in enumerate(topo(spec["stages"]), 1):
        ups = sorted({producer[a] for a in st["inputs"] if a in producer})
        rows.append({"order": n, "stage": st["id"], "label": st["label"], "script": st["script"],
                     "command": st["command"], "inputs": ";".join(st["inputs"]),
                     "outputs": ";".join(st["outputs"]), "upstream_stages": ";".join(ups)})
    return rows


# ---------------------------------------------------------------- rendering

def csv_text(rows) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(rows[0]), lineterminator="\n")
    w.writeheader()
    w.writerows(rows)
    return buf.getvalue()


def datasets_in_order(checks):
    order = ["results", "study01", "study02", "study03", "study04", "planner", "cutoffs", "synthetic", "methods note"]
    return [d for d in order if any(c["dataset"] == d for c in checks)]


def render_svg(checks) -> str:
    order = datasets_in_order(checks)
    colour = {"pass": "#2f855a", "warn": "#b7791f", "fail": "#c53030"}
    W, left, right, top, rh = 760, 150, 60, 70, 30
    H = top + rh * len(order) + 24
    mx = max(Counter(c["dataset"] for c in checks).values())
    sx = lambda v: left + (W - left - right) * v / mx  # noqa: E731
    o = ['<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="%d" height="%d" role="img" '
         'aria-labelledby="t d" font-family="system-ui, sans-serif" font-size="12">' % (W, H, W, H),
         '<title id="t">Data expectations by group in the ehs-risk-sem repository</title>',
         '<desc id="d">Stacked horizontal bars: number of named expectations per group of committed files, split into pass, warn and fail.</desc>',
         '<rect width="%d" height="%d" fill="#ffffff"/>' % (W, H),
         '<text x="16" y="24" font-size="15" font-weight="600" fill="#1a1a1a">Named data expectations by group of files</text>',
         '<text x="16" y="42" fill="#555">Green passes, amber warns (reported, not blocking), red fails. Committed files only.</text>']
    for i, ds in enumerate(order):
        y = top + rh * i
        o.append('<text x="%d" y="%.1f" text-anchor="end" fill="#1a1a1a">%s</text>' % (left - 8, y + 17, ds))
        x = left
        for st in ("pass", "warn", "fail"):
            n = sum(1 for c in checks if c["dataset"] == ds and c["status"] == st)
            if n:
                w = sx(n) - left
                o.append('<rect x="%.1f" y="%.1f" width="%.1f" height="20" fill="%s"/>' % (x, y + 4, w, colour[st]))
                o.append('<text x="%.1f" y="%.1f" text-anchor="middle" fill="#ffffff" font-weight="600">%d</text>' % (x + w / 2, y + 18, n))
                x += w
        tot = sum(1 for c in checks if c["dataset"] == ds)
        o.append('<text x="%.1f" y="%.1f" fill="#1a1a1a">%d</text>' % (x + 6, y + 18, tot))
    o.append("</svg>")
    return "\n".join(o) + "\n"


def render_doc(checks, spec, prof) -> str:
    n = len(checks)
    npass = sum(1 for c in checks if c["status"] == "pass")
    warns = [c for c in checks if c["status"] == "warn"]
    fails = [c for c in checks if c["status"] == "fail"]
    arts, stages = spec["artifacts"], topo(spec["stages"])
    L = ["# Data health and pipeline lineage", "",
         "Generated by `python3 tools/health.py --write`; do not edit by hand. `tests/test_health.py` fails if this file, "
         "`docs/health.svg` or `health/*.csv` differ from the generator output.", "",
         "A Foundry-style practice in plain Python on committed files: every rule below is a named expectation with an "
         "observed value, a limit and a status, and the build pipeline is declared as stages with inputs and outputs and "
         "checked against the repository. Independent of and not affiliated with Palantir. Every table checked here comes "
         "from a simulation with a known generating model; none is a measurement of real workplaces.", "",
         "## Result", "",
         "%d expectations: %d pass, %d warn, %d fail. Rows: `health/health_checks.csv`. Empty-cell profile of every column of "
         "every results table: `health/health_columns.csv`." % (n, npass, len(warns), len(fails)), "",
         "![Expectations by group](health.svg)", "",
         "| Group | Expectations | Pass | Warn | Fail |", "|---|---|---|---|---|"]
    for ds in datasets_in_order(checks):
        cs = [c for c in checks if c["dataset"] == ds]
        L.append("| %s | %d | %d | %d | %d |" % (ds, len(cs), sum(c["status"] == "pass" for c in cs), sum(c["status"] == "warn" for c in cs), sum(c["status"] == "fail" for c in cs)))
    L += ["", "## What the warnings say", ""]
    if warns:
        for c in warns:
            L.append("- %s `%s`: %s. Observed %s, limit %s." % (c["id"], c["dataset"], c["rule"], c["observed"], c["limit"]))
    else:
        L.append("None.")
    if fails:
        L += ["", "## Failures", ""]
        for c in fails:
            L.append("- %s `%s`: %s. Observed %s, limit %s." % (c["id"], c["dataset"], c["rule"], c["observed"], c["limit"]))
    L += ["", "Warnings are the repository's own findings restated as data limits: they describe results the methods note and the planner already "
          "reports (the analytic standard errors understate the sampling spread, and a weak-loading survey needs a larger N to "
          "give a proper solution). They are reported so a reader sees the limit and they do not block CI. A new "
          "error-severity failure does.", "",
          "## What this does and does not show", "",
          "- The expectations check that each committed table has its declared columns, plausible values, and the identities "
          "that must hold inside and between files (for example PPV recomputed from Bayes' rule, bias equal to estimate minus "
          "truth, the same true betas in four files). They do not re-run the simulations or check that a simulation is "
          "correct: that is the estimator-recovery tests.",
          "- Tolerances and the 0.93 coverage and 0.95 proper-solution thresholds are judgement, set from conventional targets.",
          "- The lineage is declared, then validated (paths, functions via ast, undeclared outputs, double producers, cycles); "
          "it is not discovered by tracing execution.", "",
          "## Pipeline lineage", "",
          "%d stages, %d artifacts. Hexagons are hand-written sources, rectangles are committed generated files." % (len(stages), len(arts)),
          "", "```mermaid", "flowchart LR"]
    for aid, a in arts.items():
        lab = a["label"].replace('"', "'")
        wrap = '{{"%s"}}' if a.get("source") else '["%s"]'
        L.append("  A_%s%s" % (aid, wrap % lab))
    for s in stages:
        L.append('  S_%s("%s")' % (s["id"], s["label"].replace('"', "'")))
    for s in stages:
        for a in s["inputs"]:
            L.append("  A_%s --> S_%s" % (a, s["id"]))
        for a in s["outputs"]:
            L.append("  S_%s --> A_%s" % (s["id"], a))
    L += ["```", "", "| # | Stage | Command | Reads | Writes |", "|---|---|---|---|---|"]
    for i, s in enumerate(stages, 1):
        L.append("| %d | %s | `%s` | %s | %s |" % (i, s["label"], s["command"],
                                                   ", ".join(arts[a]["label"] for a in s["inputs"]) or "none",
                                                   ", ".join(arts[a]["label"] for a in s["outputs"])))
    L += ["", "## Reproduce", "", "```", "python3 tools/health.py --write", "python3 tools/health.py --check", "```", ""]
    return "\n".join(L)


def outputs():
    data = load()
    checks = run_checks(data)
    spec = load_spec()
    prof = profiles(data)
    return {
        CHECKS_CSV: csv_text(checks),
        COLUMNS_CSV: csv_text(prof),
        LINEAGE_CSV: csv_text(lineage_rows(spec)),
        SVG: render_svg(checks),
        DOC: render_doc(checks, spec, prof),
    }, checks, validate_lineage(spec)


def main(argv) -> int:
    outs, checks, problems = outputs()
    errors = [c for c in checks if c["status"] == "fail"]
    warns = [c for c in checks if c["status"] == "warn"]
    print("%d expectations, %d fail, %d warn; lineage problems: %d" % (len(checks), len(errors), len(warns), len(problems)))
    for c in errors + warns:
        print("  %s %s %s: %s (observed %s, limit %s)" % (c["status"], c["id"], c["dataset"], c["rule"], c["observed"], c["limit"]))
    for p in problems:
        print("  lineage:", p)
    if "--write" in argv:
        os.makedirs(OUT_DIR, exist_ok=True)
        for path, text in outs.items():
            with open(path, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(text)
        print("wrote %d files" % len(outs))
    if "--check" in argv:
        stale = []
        for path, text in outs.items():
            if not os.path.exists(path) or open(path, encoding="utf-8", newline="").read() != text:
                stale.append(os.path.relpath(path, ROOT))
        for s in stale:
            print("  stale:", s)
        return 1 if errors or problems or stale else 0
    return 1 if errors or problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
