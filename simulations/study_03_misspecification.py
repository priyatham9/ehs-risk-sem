"""Study 3. How badly a wrong model misleads, and how well it fits while doing it.

SIMULATION. Every dataset is generated from a model whose structure is known,
then analysed with a model whose structure is wrong in one specific way. The
question in each case is the same: does the fit tell you?

Five demonstrations:

A. **Equivalent models.** Generate from ``X -> Y``. Fit ``X -> Y`` and ``Y -> X``.
   The two produce identical chi-square, CFI, RMSEA and SRMR, to numerical
   precision, because they imply the same covariance matrix. No fit statistic
   can prefer one.

B. **Reverse causation.** Generate a world where injuries degrade safety
   climate. Fit the model everyone fits, climate predicting injuries. It
   returns a large, tightly bounded, confidently signed coefficient, and it
   fits.

C. **Omitted common cause.** A confounder drives both a predictor and the
   outcome. Omit it. The coefficient on the predictor absorbs the confounding
   and the model still fits, because the confounder's indicators are not in the
   covariance matrix being reproduced.

D. **Conditioning on a mediator.** Generate ``X -> M -> Y`` plus ``X -> Y``.
   Fit ``Y`` on both ``X`` and ``M``. The coefficient on ``X`` is the direct
   effect with the mediated pathway conditioned away, and it is not the
   quantity a reader interprets it as. Presenting several such coefficients as
   comparable contributions is the Table 2 fallacy.

E. **Formative construct modelled reflectively.** Generate a construct that its
   indicators constitute rather than reflect. Fit a reflective measurement
   model. Reliability collapses, the disattenuation over-corrects, and the
   structural coefficient is wrong.

Run: ``python3 simulations/study_03_misspecification.py``
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ehs_risk_sem.compat import silence_accelerate_matmul_warnings
from ehs_risk_sem.model import ModelSpec, fit
from simulations.dgp import (
    TrueModel,
    generate_dataset,
    generate_formative_block,
    generate_indicators,
    generate_latent,
    make_four_factor_model,
)


def _two_factor_model(beta: float = 0.50, loading: float = 0.80) -> TrueModel:
    """A minimal generating model: ``Cause -> Effect``, three indicators each."""
    names = ["Cause", "Effect"]
    b = np.zeros((2, 2))
    b[1, 0] = beta
    return TrueModel(
        latent_names=names,
        loadings={n: np.full(3, loading) for n in names},
        beta=b,
        exogenous_corr=np.array([[1.0]]),
        indicator_names={"Cause": [f"c{i+1}" for i in range(3)],
                         "Effect": [f"e{i+1}" for i in range(3)]},
    )


def demo_a_equivalent_models(n: int = 3000, seed: int = 1) -> pd.DataFrame:
    """Fit both directions to data generated in one direction."""
    truth = _two_factor_model()
    data, _ = generate_dataset(truth, n, seed=seed)

    forward = ModelSpec(
        latents={"Cause": ["c1", "c2", "c3"], "Effect": ["e1", "e2", "e3"]},
        paths={"Effect": ["Cause"]},
    )
    reverse = ModelSpec(
        latents={"Cause": ["c1", "c2", "c3"], "Effect": ["e1", "e2", "e3"]},
        paths={"Cause": ["Effect"]},
    )

    rows = []
    for label, spec, target in (
        ("Cause -> Effect (correct direction)", forward, "Effect"),
        ("Effect -> Cause (reversed)", reverse, "Cause"),
    ):
        res = fit(spec, data)
        est = res.paths[target]
        f = res.fit_indices
        rows.append(
            {
                "fitted_model": label,
                "beta": float(est.beta[0]),
                "se": float(est.se[0]),
                "p_value": float(est.p_values()[0]),
                "chi_square": f.chi_square,
                "df": f.df,
                "cfi": f.cfi,
                "rmsea": f.rmsea,
                "srmr": f.srmr,
                "meets_conventional_cutoffs": f.meets_conventional_cutoffs(),
            }
        )
    out = pd.DataFrame(rows)
    return out


def demo_b_reverse_causation(n: int = 3000, seed: int = 2) -> pd.DataFrame:
    """Injuries degrade climate; the analyst fits climate predicting injuries."""
    truth = TrueModel(
        latent_names=["Injuries", "Climate"],
        loadings={"Injuries": np.full(3, 0.78), "Climate": np.full(3, 0.78)},
        beta=np.array([[0.0, 0.0], [-0.55, 0.0]]),
        exogenous_corr=np.array([[1.0]]),
        indicator_names={
            "Injuries": ["inj_1", "inj_2", "inj_3"],
            "Climate": ["cli_1", "cli_2", "cli_3"],
        },
    )
    data, _ = generate_dataset(truth, n, seed=seed)

    fitted = ModelSpec(
        latents={
            "Climate": ["cli_1", "cli_2", "cli_3"],
            "Injuries": ["inj_1", "inj_2", "inj_3"],
        },
        paths={"Injuries": ["Climate"]},
    )
    res = fit(fitted, data)
    est = res.paths["Injuries"]
    f = res.fit_indices
    return pd.DataFrame(
        [
            {
                "generating_structure": "Injuries -> Climate, beta = -0.55",
                "fitted_structure": "Climate -> Injuries",
                "estimated_beta": float(est.beta[0]),
                "se": float(est.se[0]),
                "p_value": float(est.p_values()[0]),
                "chi_square": f.chi_square,
                "df": f.df,
                "cfi": f.cfi,
                "rmsea": f.rmsea,
                "srmr": f.srmr,
                "meets_conventional_cutoffs": f.meets_conventional_cutoffs(),
            }
        ]
    )


def demo_c_omitted_confounder(n: int = 4000, seed: int = 3) -> pd.DataFrame:
    """A common cause of predictor and outcome, present in the world, absent from the model."""
    truth = TrueModel(
        latent_names=["Confounder", "Predictor", "Outcome"],
        loadings={
            "Confounder": np.full(3, 0.78),
            "Predictor": np.full(3, 0.78),
            "Outcome": np.full(3, 0.78),
        },
        beta=np.array(
            [
                [0.0, 0.0, 0.0],
                [0.60, 0.0, 0.0],
                [0.50, 0.20, 0.0],
            ]
        ),
        exogenous_corr=np.array([[1.0]]),
        indicator_names={
            "Confounder": ["z1", "z2", "z3"],
            "Predictor": ["x1", "x2", "x3"],
            "Outcome": ["y1", "y2", "y3"],
        },
    )
    data, _ = generate_dataset(truth, n, seed=seed)

    rows = []
    full = ModelSpec(
        latents={
            "Confounder": ["z1", "z2", "z3"],
            "Predictor": ["x1", "x2", "x3"],
            "Outcome": ["y1", "y2", "y3"],
        },
        paths={"Predictor": ["Confounder"], "Outcome": ["Confounder", "Predictor"]},
    )
    res_full = fit(full, data)
    b_full = res_full.paths["Outcome"]
    j = b_full.predictors.index("Predictor")
    rows.append(
        {
            "fitted_model": "confounder included",
            "beta_predictor_on_outcome": float(b_full.beta[j]),
            "true_direct_effect": 0.20,
            "bias": float(b_full.beta[j]) - 0.20,
            "chi_square": res_full.fit_indices.chi_square,
            "df": res_full.fit_indices.df,
            "cfi": res_full.fit_indices.cfi,
            "rmsea": res_full.fit_indices.rmsea,
            "meets_conventional_cutoffs": res_full.fit_indices.meets_conventional_cutoffs(),
        }
    )

    reduced = ModelSpec(
        latents={"Predictor": ["x1", "x2", "x3"], "Outcome": ["y1", "y2", "y3"]},
        paths={"Outcome": ["Predictor"]},
    )
    res_red = fit(reduced, data[["x1", "x2", "x3", "y1", "y2", "y3"]])
    b_red = res_red.paths["Outcome"]
    rows.append(
        {
            "fitted_model": "confounder omitted",
            "beta_predictor_on_outcome": float(b_red.beta[0]),
            "true_direct_effect": 0.20,
            "bias": float(b_red.beta[0]) - 0.20,
            "chi_square": res_red.fit_indices.chi_square,
            "df": res_red.fit_indices.df,
            "cfi": res_red.fit_indices.cfi,
            "rmsea": res_red.fit_indices.rmsea,
            "meets_conventional_cutoffs": res_red.fit_indices.meets_conventional_cutoffs(),
        }
    )
    return pd.DataFrame(rows)


def demo_d_mediator(n: int = 4000, seed: int = 4) -> pd.DataFrame:
    """Direct effect against total effect when a mediator is in the model."""
    direct = 0.20
    a, b = 0.55, 0.45
    truth = TrueModel(
        latent_names=["X", "M", "Y"],
        loadings={k: np.full(3, 0.78) for k in ("X", "M", "Y")},
        beta=np.array(
            [
                [0.0, 0.0, 0.0],
                [a, 0.0, 0.0],
                [direct, b, 0.0],
            ]
        ),
        exogenous_corr=np.array([[1.0]]),
        indicator_names={
            "X": ["x1", "x2", "x3"],
            "M": ["m1", "m2", "m3"],
            "Y": ["y1", "y2", "y3"],
        },
    )
    data, _ = generate_dataset(truth, n, seed=seed)

    with_m = ModelSpec(
        latents={
            "X": ["x1", "x2", "x3"],
            "M": ["m1", "m2", "m3"],
            "Y": ["y1", "y2", "y3"],
        },
        paths={"M": ["X"], "Y": ["X", "M"]},
    )
    res = fit(with_m, data)
    est = res.paths["Y"]
    jx = est.predictors.index("X")
    names = res.latent_names()
    total = res.effects["total"][names.index("Y"), names.index("X")]

    without_m = ModelSpec(
        latents={"X": ["x1", "x2", "x3"], "Y": ["y1", "y2", "y3"]},
        paths={"Y": ["X"]},
    )
    res2 = fit(without_m, data[["x1", "x2", "x3", "y1", "y2", "y3"]])

    return pd.DataFrame(
        [
            {
                "quantity": "direct effect of X on Y (mediator in model)",
                "estimate": float(est.beta[jx]),
                "generating_value": direct,
            },
            {
                "quantity": "total effect of X on Y (path tracing)",
                "estimate": float(total),
                "generating_value": direct + a * b,
            },
            {
                "quantity": "effect of X on Y (mediator omitted)",
                "estimate": float(res2.paths["Y"].beta[0]),
                "generating_value": direct + a * b,
            },
            {
                "quantity": "indirect effect X -> M -> Y",
                "estimate": float(total - est.beta[jx]),
                "generating_value": a * b,
            },
        ]
    )


def demo_e_formative(n: int = 4000, seed: int = 5) -> pd.DataFrame:
    """A formative construct fitted with a reflective measurement model."""
    rng = np.random.default_rng(seed)
    construct, formative_indicators = generate_formative_block(
        n, n_indicators=3, weights=(0.6, 0.5, 0.4), rng=rng, indicator_correlation=0.05
    )

    true_beta = 0.40
    outcome_latent = true_beta * construct + np.sqrt(1 - true_beta**2) * rng.standard_normal(n)
    outcome_loadings = np.full(3, 0.80)
    outcome_cols = {
        f"y{i+1}": outcome_loadings[i] * outcome_latent
        + np.sqrt(1 - outcome_loadings[i] ** 2) * rng.standard_normal(n)
        for i in range(3)
    }
    data = pd.concat([formative_indicators, pd.DataFrame(outcome_cols)], axis=1)

    spec = ModelSpec(
        latents={
            "SystemCondition": ["form_1", "form_2", "form_3"],
            "Outcome": ["y1", "y2", "y3"],
        },
        paths={"Outcome": ["SystemCondition"]},
    )
    res = fit(spec, data)
    sol = res.measurement["SystemCondition"]
    est = res.paths["Outcome"]

    # Reference point: regress the outcome composite on the construct that was
    # actually generated. Not available in real data; shown here to say what the
    # reflective model got wrong.
    y_comp = data[["y1", "y2", "y3"]].to_numpy().mean(axis=1)
    y_comp = (y_comp - y_comp.mean()) / y_comp.std(ddof=1)
    oracle = float(np.corrcoef(construct, y_comp)[0, 1] / np.sqrt(0.8574))

    return pd.DataFrame(
        [
            {
                "quantity": "mean within-block indicator correlation",
                "value": float(
                    np.mean(
                        np.abs(
                            np.corrcoef(formative_indicators.to_numpy().T)[
                                np.triu_indices(3, k=1)
                            ]
                        )
                    )
                ),
                "note": "formative indicators need not correlate",
            },
            {
                "quantity": "composite reliability (omega) of the formative block",
                "value": sol.omega,
                "note": "meaningless for a formative construct, but the code will compute it",
            },
            {
                "quantity": "observed composite correlation",
                "value": float(res.composite_correlations[0, 1]),
                "note": "before correction for attenuation",
            },
            {
                "quantity": "disattenuated latent correlation",
                "value": float(res.phi_estimated[0, 1]),
                "note": "dividing by a reliability of 0.14 over-corrects without limit",
            },
            {
                "quantity": "latent correlation matrix required repair",
                "value": float(res.phi_repaired),
                "note": "1 = the disattenuated matrix was indefinite and was projected",
            },
            {
                "quantity": "estimated beta, reflective model",
                "value": float(est.beta[0]),
                "note": "",
            },
            {
                "quantity": "generating beta",
                "value": true_beta,
                "note": "",
            },
            {
                "quantity": "beta using the generated construct directly",
                "value": oracle,
                "note": "not available in real data; shown as a reference point",
            },
            {
                "quantity": "model chi-square",
                "value": res.fit_indices.chi_square,
                "note": f"df = {res.fit_indices.df}, CFI = {res.fit_indices.cfi:.3f}, "
                f"RMSEA = {res.fit_indices.rmsea:.3f}",
            },
        ]
    )


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outdir", default="results")
    args = parser.parse_args(argv)

    silence_accelerate_matmul_warnings()
    os.makedirs(args.outdir, exist_ok=True)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 40)

    print(__doc__)
    print("=" * 78)

    print("\nA. Equivalent models: the same covariance matrix, opposite arrows\n")
    a = demo_a_equivalent_models()
    print(a.to_string(index=False))
    a.to_csv(os.path.join(args.outdir, "study03_equivalent.csv"), index=False)
    same = np.allclose(a["chi_square"].to_numpy(), a["chi_square"].to_numpy()[0], atol=1e-6)
    print(f"\n   chi-square identical to 1e-6 across the two directions: {same}")
    print(
        "   Any procedure that selects a causal direction on fit alone is\n"
        "   selecting on nothing. The direction has to come from outside the data."
    )

    print("\nB. Reverse causation, fitted the conventional way\n")
    b = demo_b_reverse_causation()
    print(b.to_string(index=False))
    b.to_csv(os.path.join(args.outdir, "study03_reverse.csv"), index=False)
    print(
        "\n   This is not hypothetical in safety research. Beus, Payne, Bergman &\n"
        "   Arthur (2010) report that injuries predicted subsequent safety climate\n"
        "   more strongly than climate predicted subsequent injuries."
    )

    print("\nC. Omitted common cause\n")
    c = demo_c_omitted_confounder()
    print(c.to_string(index=False))
    c.to_csv(os.path.join(args.outdir, "study03_confounder.csv"), index=False)
    print(
        "\n   Both models fit. The one missing the confounder reports a coefficient\n"
        "   several times the generating direct effect. Fit indices are computed\n"
        "   from the covariance matrix of the indicators that are in the model, so\n"
        "   an omitted variable leaves no trace in them."
    )

    print("\nD. Direct effect, total effect, and what conditioning on a mediator does\n")
    d = demo_d_mediator()
    print(d.to_string(index=False))
    d.to_csv(os.path.join(args.outdir, "study03_mediator.csv"), index=False)
    print(
        "\n   The coefficient on X changes by more than a factor of two depending on\n"
        "   whether M is in the model, and both numbers are correct estimates of\n"
        "   different quantities. A table of coefficients that does not say which\n"
        "   is which cannot be interpreted."
    )

    print("\nE. Formative construct fitted as reflective\n")
    e = demo_e_formative()
    print(e.to_string(index=False))
    e.to_csv(os.path.join(args.outdir, "study03_formative.csv"), index=False)
    print(
        "\n   Equipment age, overdue preventive maintenance and alarm rate jointly\n"
        "   constitute system condition; they do not reflect a common cause and\n"
        "   have no reason to correlate. Fitting a reflective model to them is a\n"
        "   specification error, and the reliability estimate that the code\n"
        "   happily returns is not a reliability."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
