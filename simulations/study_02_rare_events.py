"""Study 2. What happens when the outcome is an actual injury.

SIMULATION. All data are generated from a known model.

Safety-climate SEMs are usually fitted with a survey-scale outcome, where about
half the sample is above the mean and everything behaves. An operational risk
score has to predict a recordable case on a worker-shift, where the base rate
is on the order of 1e-4. This study shows what changes.

Sections:

1. Base-rate arithmetic. No simulation needed; it is arithmetic, and it decides
   whether an alerting system can work before any model is fitted.
2. Discrimination against calibration as the base rate falls. The same
   generating model, the same predictor strength, four base rates.
3. What a latent risk score is missing. A score with no link function has no
   calibration to assess; adding one is a modelling step with consequences.
4. Class-imbalance "corrections". Undersampling, oversampling and a SMOTE-style
   interpolation are applied to the rarest condition. Discrimination is
   unchanged and calibration is destroyed. This reproduces, on generated data,
   the result that van den Goorbergh et al. (2022) and Carriero et al. (2025)
   established by simulation for clinical prediction models.
5. Zero-inflated counts. Observed zeros are a mixture of "nothing happened" and
   "nothing was recorded", and a model that ignores the mixture misstates both
   the rate and the covariate effects.

Run: ``python3 simulations/study_02_rare_events.py``
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ehs_risk_sem.calibration import (
    assess_calibration,
    auc,
    brier_decomposition,
    calibration_intercept_slope,
    confusion_at_threshold,
    decision_curve,
)
from ehs_risk_sem.compat import silence_accelerate_matmul_warnings
from ehs_risk_sem.glm import king_zeng_correction, logistic_irls, predict_proba
from ehs_risk_sem.rare_events import (
    BLS_PRIVATE_TRC_RATE_2024,
    BLS_SOURCE_NOTE,
    ExposureModel,
    alerts_per_true_event,
    worker_shifts_for_calibration,
)
from simulations.dgp import (
    generate_binary_outcome,
    generate_dataset,
    generate_zero_inflated_count,
    make_four_factor_model,
)

BASE_RATES = (0.30, 0.05, 0.005, 0.0005)
OUTCOME_COEFS = (0.6, 0.4, -0.3, -0.25)


def base_rate_arithmetic() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Alert burden and calibration sample size at realistic injury rates.

    Not a simulation. The incidence rate used as the default is recorded in
    ``ehs_risk_sem.rare_events`` and flagged there as not independently
    re-verified.
    """
    rows = []
    for shift_hours in (8.0, 12.0):
        model = ExposureModel(trir=BLS_PRIVATE_TRC_RATE_2024, shift_hours=shift_hours)
        for sens, spec in ((0.80, 0.95), (0.80, 0.99), (0.90, 0.999), (0.95, 0.9999)):
            burden = model.alert_burden(sens, spec)
            rows.append(
                {
                    "shift_hours": shift_hours,
                    "base_rate_per_shift": burden["base_rate_per_shift"],
                    "worker_shifts_per_event": burden["shifts_per_event"],
                    "sensitivity": sens,
                    "specificity": spec,
                    "ppv": burden["ppv"],
                    "alerts_per_true_event": burden["alerts_per_true_event"],
                }
            )
    burden_df = pd.DataFrame(rows)

    sizing = []
    for unit, p in (
        ("worker-shift (8 h)", ExposureModel(shift_hours=8.0).per_shift()),
        ("worker-shift (12 h)", ExposureModel(shift_hours=12.0).per_shift()),
        ("worker-year", ExposureModel().per_worker_year()),
    ):
        for rel in (0.10, 0.25):
            need = worker_shifts_for_calibration(rel_precision=rel)
            n = need["events_required_approx"] / p
            sizing.append(
                {
                    "unit_of_analysis": unit,
                    "base_rate": p,
                    "relative_precision_target": rel,
                    "events_required": need["events_required_approx"],
                    "units_required": n,
                }
            )
    return burden_df, pd.DataFrame(sizing)


def discrimination_versus_calibration(
    n: int = 200_000, seed: int = 11
) -> pd.DataFrame:
    """Fit the same model at four base rates and compare what the metrics say."""
    model = make_four_factor_model()
    rng = np.random.default_rng(seed)
    _, eta = generate_dataset(model, n, seed=seed)
    x = eta[:, :4]

    rows = []
    for base in BASE_RATES:
        y, p_true = generate_binary_outcome(x, OUTCOME_COEFS, base, rng)
        n_train = n // 2
        fit_obj = logistic_irls(x[:n_train], y[:n_train])
        p_hat = predict_proba(x[n_train:], fit_obj.coef)
        y_test = y[n_train:]

        cal = assess_calibration(y_test, p_hat, n_bins=10)
        conf = confusion_at_threshold(y_test, p_hat, float(np.quantile(p_hat, 0.95)))
        rows.append(
            {
                "target_base_rate": base,
                "realized_base_rate": float(np.mean(y)),
                "events_in_training": int(np.sum(y[:n_train])),
                "auc": cal.auc,
                "brier": cal.brier["brier"],
                "brier_uncertainty_share": cal.brier["uncertainty"] / cal.brier["brier"]
                if cal.brier["brier"] > 0
                else np.nan,
                "calibration_intercept": cal.intercept_slope["calibration_intercept"],
                "calibration_slope": cal.intercept_slope["calibration_slope"],
                "ece": cal.ece["ece"],
                "ppv_at_top_5pct": conf["ppv"],
                "alerts_per_true_event_at_top_5pct": conf["alerts_per_true_event"],
            }
        )
    return pd.DataFrame(rows)


def king_zeng_demo(
    n: int = 5000, base_rate: float = 0.005, n_reps: int = 200, seed: int = 77
) -> pd.DataFrame:
    """Bias in the maximum-likelihood intercept at a low base rate, and its correction."""
    model = make_four_factor_model()
    rng = np.random.default_rng(seed)
    _, eta = generate_dataset(model, n, seed=seed)
    x = eta[:, :4]

    ml_int, kz_int, ml_mean_p, kz_mean_p, truth = [], [], [], [], []
    for r in range(n_reps):
        rep_rng = np.random.default_rng(seed + 1000 + r)
        y, p_true = generate_binary_outcome(x, OUTCOME_COEFS, base_rate, rep_rng)
        if y.sum() < 5:
            continue
        f = logistic_irls(x, y)
        if f.separation_warning is not None:
            continue
        corrected = king_zeng_correction(x, y, f)
        ml_int.append(f.coef[0])
        kz_int.append(corrected[0])
        # Compare marginal predicted probabilities, not sigmoid(intercept): the
        # link is nonlinear, so the value at x = 0 is not the population mean.
        ml_mean_p.append(float(np.mean(predict_proba(x, f.coef))))
        kz_mean_p.append(float(np.mean(predict_proba(x, corrected))))
        truth.append(float(np.mean(p_true)))

    n_used = len(ml_int)
    return pd.DataFrame(
        [
            {
                "estimator": "maximum likelihood",
                "mean_intercept": float(np.mean(ml_int)),
                "mean_predicted_probability": float(np.mean(ml_mean_p)),
                "relative_error_vs_truth": float(
                    np.mean(ml_mean_p) / np.mean(truth) - 1.0
                ),
                "n_usable_reps": n_used,
            },
            {
                "estimator": "King-Zeng bias corrected",
                "mean_intercept": float(np.mean(kz_int)),
                "mean_predicted_probability": float(np.mean(kz_mean_p)),
                "relative_error_vs_truth": float(
                    np.mean(kz_mean_p) / np.mean(truth) - 1.0
                ),
                "n_usable_reps": n_used,
            },
            {
                "estimator": "generating model",
                "mean_intercept": np.nan,
                "mean_predicted_probability": float(np.mean(truth)),
                "relative_error_vs_truth": 0.0,
                "n_usable_reps": n_used,
            },
        ]
    )


def _smote_like(
    x_min: np.ndarray, n_new: int, k: int, rng: np.random.Generator
) -> np.ndarray:
    """Minimal SMOTE-style synthetic minority oversampling.

    Interpolates between a minority point and one of its k nearest minority
    neighbours. Implemented here only so that the harm demonstration below uses
    the actual mechanism rather than a caricature of it.
    """
    n_min = x_min.shape[0]
    if n_min < 2:
        return np.empty((0, x_min.shape[1]))
    k = min(k, n_min - 1)
    d = np.linalg.norm(x_min[:, None, :] - x_min[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    neighbours = np.argsort(d, axis=1)[:, :k]
    idx = rng.integers(0, n_min, size=n_new)
    pick = neighbours[idx, rng.integers(0, k, size=n_new)]
    gap = rng.random((n_new, 1))
    return x_min[idx] + gap * (x_min[pick] - x_min[idx])


def imbalance_correction_harm(
    n: int = 200_000, base_rate: float = 0.005, seed: int = 909
) -> pd.DataFrame:
    """Apply three imbalance corrections and measure what they change.

    The comparison holds the model, the features and the test set fixed. Only
    the training sample is altered.
    """
    model = make_four_factor_model()
    rng = np.random.default_rng(seed)
    _, eta = generate_dataset(model, n, seed=seed)
    x = eta[:, :4]
    y, _ = generate_binary_outcome(x, OUTCOME_COEFS, base_rate, rng)

    n_train = n // 2
    x_tr, y_tr = x[:n_train], y[:n_train]
    x_te, y_te = x[n_train:], y[n_train:]

    pos = np.where(y_tr == 1)[0]
    neg = np.where(y_tr == 0)[0]

    variants = {}
    variants["none (as collected)"] = (x_tr, y_tr)

    keep_neg = rng.choice(neg, size=min(len(pos) * 1, len(neg)), replace=False)
    idx = np.concatenate([pos, keep_neg])
    variants["random undersampling of majority"] = (x_tr[idx], y_tr[idx])

    up = rng.choice(pos, size=len(neg), replace=True)
    idx = np.concatenate([neg, up])
    variants["random oversampling of minority"] = (x_tr[idx], y_tr[idx])

    synth = _smote_like(x_tr[pos], n_new=len(neg) - len(pos), k=5, rng=rng)
    x_sm = np.vstack([x_tr, synth])
    y_sm = np.concatenate([y_tr, np.ones(synth.shape[0])])
    variants["SMOTE-style interpolation"] = (x_sm, y_sm)

    rows = []
    for label, (xt, yt) in variants.items():
        f = logistic_irls(xt, yt)
        p_hat = predict_proba(x_te, f.coef)
        cs = calibration_intercept_slope(y_te, p_hat)
        bd = brier_decomposition(y_te, p_hat)
        conf = confusion_at_threshold(y_te, p_hat, float(np.quantile(p_hat, 0.95)))
        rows.append(
            {
                "training_sample": label,
                "train_n": int(xt.shape[0]),
                "train_event_fraction": float(np.mean(yt)),
                "auc_test": auc(y_te, p_hat),
                "mean_predicted_test": float(np.mean(p_hat)),
                "observed_rate_test": float(np.mean(y_te)),
                "overprediction_ratio": float(np.mean(p_hat) / np.mean(y_te)),
                "calibration_slope": cs["calibration_slope"],
                "brier": bd["brier"],
                "ppv_at_top_5pct": conf["ppv"],
            }
        )
    return pd.DataFrame(rows)


def zero_inflation_demo(n: int = 50_000, seed: int = 313) -> pd.DataFrame:
    """A zero-inflated count outcome analysed as though it were not."""
    model = make_four_factor_model()
    rng = np.random.default_rng(seed)
    _, eta = generate_dataset(model, n, seed=seed)
    x = eta[:, :4]

    rows = []
    for zi in (0.0, 0.20, 0.38):
        counts, mu, _ = generate_zero_inflated_count(
            x,
            count_coefficients=(0.35, 0.20, -0.15, -0.10),
            zero_coefficients=(-0.30, 0.0, 0.0, 0.25),
            base_mean=1.5,
            zero_inflation=zi if zi > 0 else 1e-9,
            rng=rng,
        )
        observed_zero = float(np.mean(counts == 0))
        poisson_zero = float(np.mean(np.exp(-mu)))
        design = np.column_stack([np.ones(n), x])
        coef, *_ = np.linalg.lstsq(design, counts, rcond=None)
        rows.append(
            {
                "structural_zero_fraction": zi,
                "observed_zero_fraction": observed_zero,
                "poisson_expected_zero_fraction": poisson_zero,
                "excess_zeros": observed_zero - poisson_zero,
                "mean_count": float(np.mean(counts)),
                "variance_count": float(np.var(counts, ddof=1)),
                "variance_to_mean_ratio": float(
                    np.var(counts, ddof=1) / max(np.mean(counts), 1e-9)
                ),
                "ols_slope_on_latent_1": float(coef[1]),
            }
        )
    return pd.DataFrame(rows)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=200_000)
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args(argv)

    silence_accelerate_matmul_warnings()
    os.makedirs(args.outdir, exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.float_format", lambda v: f"{v:,.6g}")

    print(__doc__)
    print("=" * 78)

    print("\n1. Base-rate arithmetic (not a simulation)\n")
    print(f"   {BLS_SOURCE_NOTE}\n")
    print(ExposureModel().summary())
    burden, sizing = base_rate_arithmetic()
    print("\n   Alert burden:\n")
    print(burden.to_string(index=False))
    print("\n   Sample size to estimate the base rate to a relative precision:\n")
    print(sizing.to_string(index=False))
    burden.to_csv(os.path.join(args.outdir, "study02_alert_burden.csv"), index=False)
    sizing.to_csv(os.path.join(args.outdir, "study02_sizing.csv"), index=False)

    print("\n2. Discrimination against calibration, same model, four base rates\n")
    dvc = discrimination_versus_calibration(n=args.n)
    print(dvc.to_string(index=False))
    dvc.to_csv(os.path.join(args.outdir, "study02_discrimination.csv"), index=False)
    print(
        "\n   AUC is nearly flat across base rates. The Brier score falls by more\n"
        "   than two orders of magnitude, almost entirely because its uncertainty\n"
        "   term is a property of the outcome. Neither number tells an operations\n"
        "   manager what fraction of alerts will be real."
    )

    print("\n3. Maximum-likelihood intercept bias at a low base rate\n")
    kz = king_zeng_demo()
    print(kz.to_string(index=False))
    kz.to_csv(os.path.join(args.outdir, "study02_king_zeng.csv"), index=False)

    print("\n4. Class-imbalance corrections applied to the same training data\n")
    harm = imbalance_correction_harm(n=args.n)
    print(harm.to_string(index=False))
    harm.to_csv(os.path.join(args.outdir, "study02_imbalance.csv"), index=False)
    print(
        "\n   Every correction leaves AUC essentially unchanged and multiplies the\n"
        "   predicted probabilities by one to two orders of magnitude. A model\n"
        "   trained this way ranks as well as before and reports risks that are\n"
        "   wrong by a factor of a hundred."
    )

    print("\n5. Zero-inflated counts\n")
    zi = zero_inflation_demo()
    print(zi.to_string(index=False))
    zi.to_csv(os.path.join(args.outdir, "study02_zero_inflation.csv"), index=False)

    print(
        "\n   A spike of zeros in reported counts is consistent with two very\n"
        "   different worlds: nothing happened, or nothing was recorded. The\n"
        "   observed data cannot separate them without an assumption, and the\n"
        "   variance-to-mean ratio above shows how far from Poisson the result is."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
